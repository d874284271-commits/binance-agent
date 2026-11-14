from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import Field

from binance_agent.agents.reading import (
    coverage_for,
    document_view,
    mark_batch,
    pending_chunks,
    verification_documents,
)
from binance_agent.agents.research import _sanitize_research, build_news_research_graph
from binance_agent.infra.settings import ResearchConfig
from binance_agent.intel.evidence import evidence_document
from test_news_research_graph import ResearchModel, initial, model_for


def test_deduplicated_body_keeps_each_sources_metadata_through_reading_and_verification() -> None:
    docs = [
        evidence_document(
            url=f"https://example.invalid/{i}",
            source=f"Source {i}",
            title=f"Title {i}",
            text="Application is pending, not approved.",
            level="body",
            published_at=datetime(year, 1, 1, tzinfo=timezone.utc),
        )
        for i, year in enumerate([2025, 2026])
    ]
    views = document_view(docs)
    assert len(views) == 1
    expected = [doc.model_dump(mode="json", exclude={"text"}) for doc in docs]
    for source, metadata in zip(views[0]["sources"], expected, strict=True):
        assert {key: source.get(key) for key in metadata} == metadata
    records = {doc.evidence_id: coverage_for(doc) for doc in docs}
    chunks = pending_chunks(views, records)
    mark_batch(records, chunks, "processed")
    records = {doc.evidence_id: coverage_for(doc, records[doc.evidence_id]) for doc in docs}
    verified = verification_documents(docs, records)
    assert len(chunks) == len(verified) == 1
    assert chunks[0]["sources"] == verified[0]["sources"] == views[0]["sources"]
    assert all(record["complete"] for record in records.values())


def test_interval_union_and_duplicate_sources() -> None:
    docs = [
        evidence_document(url=f"https://example.org/{i}", text="x" * 14000, level="body")
        for i in range(2)
    ]
    records = {d.evidence_id: coverage_for(d) for d in docs}
    views = document_view(docs)
    assert len(views) == 1
    assert len(views[0]["sources"]) == 2
    chunks = pending_chunks(views, records)
    mark_batch(records, chunks, "submitted")
    assert not coverage_for(docs[0], records[docs[0].evidence_id])["complete"]
    mark_batch(records, chunks, "processed")
    for doc in docs:
        coverage = coverage_for(doc, records[doc.evidence_id])
        assert coverage["complete"]
        assert coverage["covered_chars"] == 14000
        assert coverage["processed_intervals"] == [[0, 14000]]
        assert coverage["verified_intervals"] == []


class BatchModel(ResearchModel):
    batches: list[dict[str, Any]] = Field(default_factory=list)
    checks: list[dict[str, Any]] = Field(default_factory=list)
    empty_second: bool = True
    fail_second: bool = False
    negate_tail: bool = False
    cancel_verifier_once: bool = False

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        prompt = str(messages[0].content)
        payload = json.loads(messages[-1].content)
        if "金融事件研究者" in prompt:
            self.batches.append(payload)
            if len(self.batches) > 1 and self.fail_second:
                raise ValueError("fixture researcher failure")
            doc = payload["documents"][0]
            facts = (
                []
                if len(self.batches) > 1 and self.empty_second
                else [
                    {
                        "statement": "Bitcoin ETF approved.",
                        "quote": doc["text"][:30],
                        "evidence_id": doc["evidence_id"],
                        "confirmation_status": "confirmed_in_source",
                    }
                ]
            )
            return AIMessage(
                content=json.dumps(
                    {"facts": facts, "gaps": [], "conflicts": [], "stance": "bullish"}
                )
            )
        if "verdicts" in prompt:
            self.checks.append(payload)
            if self.cancel_verifier_once:
                self.cancel_verifier_once = False
                raise asyncio.CancelledError()
            reject = self.negate_tail and any(
                "NOT APPROVED" in d["text"] for d in payload["documents"]
            )
            return AIMessage(
                content=json.dumps(
                    {
                        "verdicts": [
                            {"index": i, "supported": not reject}
                            for i in range(len(payload["facts"]))
                        ]
                    }
                )
            )
        return await super().ainvoke(messages, *args, **kwargs)


def small_config(_: str) -> dict[str, Any]:
    return {
        "max_retries": 0,
        "context_policy": {"window_tokens": 40000, "trigger_tokens": 32000, "target_tokens": 24000},
    }


async def run_batches(
    model: BatchModel, text: str, *, rounds: int = 2
) -> tuple[dict[str, Any], list[str]]:
    fetched: list[str] = []

    async def fetch(url: str) -> Any:
        fetched.append(url)
        return evidence_document(url=url, text=text, level="body")

    state = initial()
    state["intel_snapshot"]["news_items"][0]["summary"] = ""
    graph = build_news_research_graph(
        model_for(model),
        config=ResearchConfig(enabled=True, max_rounds=rounds),
        runtime_config_for=small_config,
        fetch=fetch,
    )
    return await graph.ainvoke(state), fetched


async def test_second_batch_empty_facts_preserves_and_rechecks_first() -> None:
    model = BatchModel(responses=["unused"])
    result, calls = await run_batches(model, "Bitcoin ETF approved. " + "x" * 13970)
    assert len(calls) == 1
    assert len(model.batches) == 2
    assert len(model.checks) == 2
    assert model.batches[1]["existing_facts"]
    assert result["news_research"]["facts"]
    assert result["news_research"]["reading_complete"]


async def test_new_tail_negates_old_fact_even_without_new_facts() -> None:
    model = BatchModel(responses=["unused"], negate_tail=True)
    result, _ = await run_batches(model, "Bitcoin ETF approved. " + "x" * 13900 + " NOT APPROVED")
    assert len(model.batches) == 2
    assert len(model.checks) == 2
    assert result["news_research"]["facts"] == []
    assert result["news_research"]["critical_unresolved"]


@pytest.mark.parametrize("fail_second,rounds", [(True, 2), (False, 1)])
async def test_empty_model_gaps_never_clear_unread_intervals(
    fail_second: bool, rounds: int
) -> None:
    model = BatchModel(responses=["unused"], fail_second=fail_second)
    result, _ = await run_batches(model, "Bitcoin ETF approved. " + "x" * 30000, rounds=rounds)
    research = result["news_research"]
    assert research["facts"]
    assert not research["reading_complete"]
    assert research["critical_unresolved"]
    assert any("正文阅读覆盖缺口" in gap for gap in research["gaps"])


async def test_checkpoint_after_research_resumes_directly_at_verifier() -> None:
    model = BatchModel(responses=["unused"], cancel_verifier_once=True)

    async def fetch(url: str) -> Any:
        return evidence_document(url=url, text="Bitcoin ETF approved.", level="body")

    graph = build_news_research_graph(
        model_for(model), config=ResearchConfig(enabled=True), fetch=fetch
    )
    graph.checkpointer = MemorySaver()
    cfg = {"configurable": {"thread_id": "interrupted-research"}}
    with pytest.raises(asyncio.CancelledError):
        await graph.ainvoke(initial(), cfg)
    assert len(model.batches) == 1
    result = await graph.ainvoke(None, cfg)
    assert len(model.batches) == 1
    assert result["news_research"]["facts"]


def test_legacy_full_saved_text_does_not_prove_reading() -> None:
    doc = evidence_document(url="https://example.org", text="full body", level="body")
    research = _sanitize_research(
        {
            "tasks": [
                {
                    "task_id": "legacy",
                    "critical": True,
                    "status": "completed",
                    "facts": [],
                    "document_coverage": [{"evidence_id": doc.evidence_id, "complete": True}],
                }
            ]
        },
        [doc],
    )
    assert not research["reading_complete"]
    assert research["document_coverage"][0]["status"] == "unknown"
    assert research["critical_unresolved"]


async def test_replay_without_saved_result_never_invokes_model() -> None:
    model = BatchModel(responses=["unused"])
    graph = build_news_research_graph(model_for(model), config=ResearchConfig(enabled=True))
    result = await graph.ainvoke({**initial(), "research_replay": True})
    assert not model.batches and not model.seen_payloads
    assert result["news_research"]["legacy_no_original_evidence"]


def test_equal_summary_and_body_do_not_share_reading_coverage() -> None:
    body = evidence_document(url="https://example.org/body", text="same text", level="body")
    summary = evidence_document(
        url="https://example.org/summary", text="same text", level="summary"
    )
    views = document_view([summary, body])
    assert len(views) == 2
    records = {doc.evidence_id: coverage_for(doc) for doc in [body, summary]}
    batch = [chunk for chunk in pending_chunks(views, records) if chunk["level"] == "summary"]
    mark_batch(records, batch, "processed")
    assert coverage_for(summary, records[summary.evidence_id])["complete"]
    assert not coverage_for(body, records[body.evidence_id])["complete"]


async def test_later_fact_is_rechecked_against_distant_earlier_negation() -> None:
    model = BatchModel(responses=["unused"], empty_second=False, negate_tail=True)
    result, _ = await run_batches(
        model, "Bitcoin ETF approved. " + "x" * 2500 + " NOT APPROVED " + "x" * 11400
    )
    assert len(model.batches) == 2
    assert len(model.checks) == 2
    assert any("NOT APPROVED" in doc["text"] for doc in model.checks[1]["documents"])
    assert result["news_research"]["facts"] == []
    assert result["news_research"]["critical_unresolved"]


async def test_qualification_fields_are_preserved_whole() -> None:
    class QualifiedModel(BatchModel):
        async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
            response = await super().ainvoke(messages, *args, **kwargs)
            if "金融事件研究者" in str(messages[0].content):
                data = json.loads(response.content)
                for fact in data["facts"]:
                    fact.update(
                        scope=["condition " + "x" * 510 + " except jurisdiction X"],
                        numbers=[f"{i} units" for i in range(12)],
                        uncertainty=["limited approval; not effective until September"],
                    )
                response.content = json.dumps(data)
            return response

    model = QualifiedModel(responses=["unused"])
    result, _ = await run_batches(model, "Bitcoin ETF approved.")
    fact = result["news_research"]["facts"][0]
    assert fact["scope"][0].endswith("except jurisdiction X")
    assert len(fact["numbers"]) == 12
    assert fact["uncertainty"] == ["limited approval; not effective until September"]
    assert model.checks[0]["facts"][0]["scope"] == fact["scope"]


async def test_research_compaction_updates_are_output_for_later_nodes(monkeypatch: Any) -> None:
    from binance_agent.agents.runtime import AgentRuntime

    original = AgentRuntime.invoke_structured

    async def wrapped(self: Any, agent: str, *args: Any, **kwargs: Any) -> Any:
        result = await original(self, agent, *args, **kwargs)
        result.context_updates[f"cache:{agent}"] = {"fixture": True}
        return result

    monkeypatch.setattr(AgentRuntime, "invoke_structured", wrapped)
    result, _ = await run_batches(BatchModel(responses=["unused"]), "Bitcoin ETF approved.")
    assert "cache:research_supervisor" in result["context_compactions"]
    assert any(key.startswith("cache:news_researcher:") for key in result["context_compactions"])
    assert any(key.startswith("cache:evidence_verifier:") for key in result["context_compactions"])


@pytest.mark.parametrize("has_saved_verdict", [True, False])
async def test_final_gate_replay_never_calls_model_and_preserves_only_matching_support(
    has_saved_verdict: bool,
) -> None:
    from binance_agent.agents.research import finalize_evidence_advice
    from binance_agent.agents.state import StrategyAdvice

    model = BatchModel(responses=["unused"])
    state, _ = await run_batches(model, "Bitcoin ETF approved.")
    state["final_advice"] = StrategyAdvice(
        symbol="BTCUSDT", direction="long", decision_type="trade", rationale="Bitcoin ETF approved."
    )
    live = await finalize_evidence_advice(state, llm=model)
    assert live["news_research"]["final_verification"]["request_supported"]
    assert live["final_advice"].decision_type == "trade"
    gate_payload = json.loads(model.seen_payloads[-1])
    assert {"document_coverage", "gaps", "conflicts", "reading_complete"} <= set(gate_payload)
    if not has_saved_verdict:
        live["news_research"].pop("final_verification")

    class ForbiddenModel(ResearchModel):
        async def ainvoke(self, *args: Any, **kwargs: Any) -> AIMessage:
            raise AssertionError("final replay must not invoke any model")

    replay = await finalize_evidence_advice(
        {**state, **live, "research_replay": True}, llm=ForbiddenModel(responses=["unused"])
    )
    assert replay["runtime_events"] == []
    if has_saved_verdict:
        assert replay["final_advice"].decision_type == "trade"
        assert replay["news_research"]["final_verification"]["status"] == "supported"
    else:
        assert replay["final_advice"].decision_type == "watchlist"
        assert replay["final_advice"].evidence_status == "replay_verification_unknown"


async def test_final_gate_unknown_replay_keeps_no_trade_and_clears_prices() -> None:
    from binance_agent.agents.research import finalize_evidence_advice
    from binance_agent.agents.state import StrategyAdvice

    model = ResearchModel(responses=["unused"])
    advice = StrategyAdvice(
        symbol="BTCUSDT",
        direction="neutral",
        decision_type="no_trade",
        entry_zone="1",
        stop_loss="0",
        targets=["2"],
    )
    result = await finalize_evidence_advice(
        {"research_replay": True, "final_advice": advice}, llm=model
    )
    assert not model.seen_payloads
    assert result["final_advice"].decision_type == "no_trade"
    assert result["final_advice"].entry_zone is None
    assert result["final_advice"].targets == []


@pytest.mark.parametrize(
    "invalid",
    [
        {"error": "unable to read"},
        {"facts": [], "gaps": None, "conflicts": []},
        {"facts": ["not a fact"], "gaps": [], "conflicts": []},
        {"facts": [{}], "gaps": [], "conflicts": []},
    ],
)
async def test_second_round_schema_error_cannot_mark_reading_complete(
    invalid: dict[str, Any],
) -> None:
    class SchemaErrorModel(BatchModel):
        async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
            response = await super().ainvoke(messages, *args, **kwargs)
            if "金融事件研究者" in str(messages[0].content) and len(self.batches) == 2:
                return AIMessage(content=json.dumps(invalid))
            return response

    result, _ = await run_batches(
        SchemaErrorModel(responses=["unused"]), "Bitcoin ETF approved. " + "x" * 13970
    )
    research = result["news_research"]
    assert research["facts"]
    assert not research["reading_complete"]
    assert research["critical_unresolved"]
    assert research["tasks"][0]["system_gaps"]["research_result"]


@pytest.mark.parametrize("always_fail", [False, True])
async def test_provisional_facts_survive_failed_verifier_until_actual_success(
    always_fail: bool,
) -> None:
    class VerifierFailureModel(BatchModel):
        attempts: int = 0

        async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
            if "verdicts" in str(messages[0].content):
                self.attempts += 1
                if self.attempts == 1 or always_fail:
                    return AIMessage(content=json.dumps({"error": "temporarily unable to verify"}))
            return await super().ainvoke(messages, *args, **kwargs)

    model = VerifierFailureModel(responses=["unused"])
    result, _ = await run_batches(model, "Bitcoin ETF approved. " + "x" * 13970)
    task = result["news_research"]["tasks"][0]
    assert model.attempts == 2
    assert model.batches[1]["provisional_facts"]
    if always_fail:
        assert task["provisional_facts"]
        assert task["system_gaps"]["verification"]
        assert task["status"] == "unresolved"
    else:
        assert task["facts"]
        assert not task["provisional_facts"]
        assert "verification" not in task["system_gaps"]
        assert task["status"] == "completed"


async def test_attempted_failed_source_is_not_hidden_by_another_successful_body() -> None:
    model = ResearchModel(responses=["unused"])
    state = initial()
    state["intel_snapshot"]["news_items"].append(
        {**state["intel_snapshot"]["news_items"][0], "url": "https://news.example.com/failed"}
    )

    async def fetch(url: str) -> Any:
        return evidence_document(
            url=url,
            text=""
            if url.endswith("failed")
            else "Bitcoin ETF application is pending, not approved.",
            level="body",
            status="unavailable" if url.endswith("failed") else "available",
        )

    graph = build_news_research_graph(
        model_for(model),
        config=ResearchConfig(enabled=True, max_documents=4, max_rounds=1),
        fetch=fetch,
    )
    result = await graph.ainvoke(state)
    research = result["news_research"]
    assert research["facts"]
    assert research["critical_unresolved"]
    assert any(gap["url"].endswith("failed") for gap in research["capture_gaps"])
    assert any("正文抓取缺口" in gap for gap in research["gaps"])


async def test_final_gate_downgrade_replay_is_idempotent_and_keeps_actual_request_hash() -> None:
    import hashlib

    from binance_agent.agents.context import stable_json
    from binance_agent.agents.research import finalize_evidence_advice
    from binance_agent.agents.state import StrategyAdvice

    model = BatchModel(responses=["unused"])
    state, _ = await run_batches(model, "Bitcoin ETF approved. " + "x" * 13970, rounds=1)
    state["final_advice"] = StrategyAdvice(
        symbol="BTCUSDT", direction="long", decision_type="trade", rationale="Bitcoin ETF approved."
    )
    live = await finalize_evidence_advice(state, llm=model)
    request = json.loads(model.seen_payloads[-1])
    record = live["news_research"]["final_verification"]
    assert (
        record["request_input_sha256"] == hashlib.sha256(stable_json(request).encode()).hexdigest()
    )
    assert record["publication_status"] == "deterministic_gate"
    # v4 retains verified facts during a conservative downgrade. Their bindings
    # must describe the delivered text, rather than the rejected candidate.
    assert live["final_advice"].claims
    assert all(
        claim["statement"] in live["final_advice"].rationale
        for claim in live["final_advice"].claims
    )
    calls = len(model.seen_payloads)
    replay = await finalize_evidence_advice({**state, **live, "research_replay": True}, llm=model)
    again = await finalize_evidence_advice({**state, **replay, "research_replay": True}, llm=model)
    assert len(model.seen_payloads) == calls
    assert replay["final_advice"].model_dump() == live["final_advice"].model_dump()
    assert again["final_advice"].model_dump() == live["final_advice"].model_dump()
    assert replay["news_research"]["final_verification"] == record
    assert again["news_research"]["final_verification"] == record


@pytest.mark.parametrize("damage", ["text", "status"])
def test_coverage_cannot_upgrade_corrupted_or_unavailable_saved_body(damage: str) -> None:
    doc = evidence_document(url="https://example.org/read", text="original body", level="body")
    record = coverage_for(doc)
    record["processed_intervals"] = [[0, len(doc.text)]]
    record = coverage_for(doc, record)
    assert record["complete"]
    damaged = doc.model_copy(
        update={"text": "tampered body"} if damage == "text" else {"status": "failed"}
    )
    record = coverage_for(damaged, record)
    assert record["status"] == "unknown"
    assert not record["complete"]
    record["processed_intervals"] = [[0, len(damaged.text)]]
    assert not coverage_for(damaged, record)["complete"]


def test_forged_hash_does_not_coalesce_different_body_text() -> None:
    first = evidence_document(url="https://example.org/first", text="original", level="body")
    second = evidence_document(
        url="https://example.org/second", text="different", level="body"
    ).model_copy(update={"content_sha256": first.content_sha256})
    views = document_view([first, second])
    assert len(views) == 2
    assert any(not view["integrity_valid"] for view in views)


def test_corrupted_unreferenced_read_document_keeps_critical_task_unresolved() -> None:
    from binance_agent.intel.models import FactCard

    first = evidence_document(
        url="https://example.org/referenced", text="Bitcoin ETF approved.", level="body"
    )
    other = evidence_document(
        url="https://example.org/unreferenced", text="Additional qualifications.", level="body"
    )
    records = []
    for doc in [first, other]:
        record = coverage_for(doc)
        record["processed_intervals"] = [[0, len(doc.text)]]
        records.append(coverage_for(doc, record))
    fact = FactCard(
        fact_id="fact",
        statement=first.text,
        evidence_id=first.evidence_id,
        quote=first.text,
        support="model_supported",
        confirmation_status="confirmed_in_source",
    ).model_dump(mode="json")
    saved = {
        "tasks": [
            {
                "task_id": "critical",
                "critical": True,
                "status": "completed",
                "facts": [fact],
                "document_coverage": records,
            }
        ]
    }
    corrupted = other.model_copy(update={"text": "Different qualifications."})
    research = _sanitize_research(saved, [first, corrupted])
    assert research["facts"]
    assert not research["reading_complete"]
    assert research["critical_unresolved"]
    assert any("content_hash_mismatch" in gap for gap in research["gaps"])


async def test_unknown_body_reservation_is_not_silently_refetched() -> None:
    pending = evidence_document(
        url="https://news.example.com/etf", level="body", status="unavailable"
    )
    pending.fetch_error = "interrupted_or_pending"

    async def fetch(url: str) -> Any:
        pytest.fail("unknown prior request must remain visible, not silently resend")

    graph = build_news_research_graph(
        model_for(ResearchModel(responses=["unused"])),
        config=ResearchConfig(enabled=True),
        fetch=fetch,
        load=lambda _: [pending],
    )
    result = await graph.ainvoke({**initial(), "research_run_id": "restored"})
    assert result["news_research"]["critical_unresolved"]
    assert all(
        doc["status"] != "available"
        for doc in result["news_research"]["documents"]
        if doc["level"] == "body"
    )


def test_task_diagnostics_keep_material_scope_without_changing_task_audit() -> None:
    from binance_agent.intel.models import FactCard

    documents = [
        evidence_document(
            url=f"https://authority.example/{identity}",
            title=f"Announcement {identity}",
            text=f"Authority {identity} issued its announcement.",
            level="body",
        )
        for identity in ("A", "B")
    ]
    tasks = []
    for identity, doc in zip(("A", "B"), documents, strict=True):
        coverage = coverage_for(doc)
        coverage["processed_intervals"] = [[0, len(doc.text)]]
        coverage["verified_intervals"] = [[0, len(doc.text)]]
        tasks.append(
            {
                "task_id": identity,
                "items": [{"url": doc.url, "title": doc.title}],
                "critical": True,
                "status": "completed",
                "facts": [
                    FactCard(
                        fact_id=identity,
                        statement=doc.text,
                        quote=doc.text,
                        evidence_id=doc.evidence_id,
                        confirmation_status="confirmed_in_source",
                        support="model_supported",
                    ).model_dump(mode="json")
                ],
                "document_coverage": [coverage],
                "gaps": ["所给材料未包含B公告"] if identity == "A" else [],
                "conflicts": ["其他公告的状态未确认"] if identity == "A" else [],
            }
        )
    research = {
        "tasks": tasks,
        "gaps": tasks[0]["gaps"],
        "conflicts": tasks[0]["conflicts"],
    }
    result = _sanitize_research(research, documents)
    assert result["tasks"][0]["gaps"] == ["所给材料未包含B公告"]
    assert result["tasks"][0]["conflicts"] == ["其他公告的状态未确认"]
    assert "所给材料未包含B公告" not in result["gaps"]
    assert all("任务 A" in note and documents[0].url in note for note in result["gaps"])
    assert all("任务 A" in note and "局部诊断" in note for note in result["conflicts"])
    assert result["facts"][1]["statement"] == documents[1].text
    assert result["critical_unresolved"]
    again = _sanitize_research(result, documents)
    assert again["gaps"] == result["gaps"]
    assert again["conflicts"] == result["conflicts"]


async def test_final_publication_payload_binds_and_hashes_all_narrative_fields() -> None:
    import hashlib

    from binance_agent.agents.context import stable_json
    from binance_agent.agents.research import finalize_evidence_advice
    from binance_agent.agents.state import StrategyAdvice

    model = BatchModel(responses=["unused"])
    state, _ = await run_batches(model, "Bitcoin ETF approved.")
    state["final_advice"] = StrategyAdvice(
        symbol="BTCUSDT",
        direction="neutral",
        decision_type="no_trade",
        rationale="保持观望。",
        no_trade_reason="Bitcoin ETF approved.",
        confidence=0.2,
    )
    result = await finalize_evidence_advice(state, llm=model)
    payload = json.loads(model.seen_payloads[-1])
    assert payload["no_trade_reason"] == "Bitcoin ETF approved."
    assert "evidence_gaps" in payload
    assert result["final_advice"].claims
    record = result["news_research"]["final_verification"]
    assert record["version"] == "final-evidence-v4"
    assert (
        record["request_input_sha256"] == hashlib.sha256(stable_json(payload).encode()).hexdigest()
    )
    assert record["publication_status"] == "verified_request"


@pytest.mark.parametrize("unknown", [False, True])
async def test_final_verifier_failure_clears_every_unsafe_field_but_keeps_verified_facts(
    unknown: bool,
) -> None:
    from binance_agent.agents.research import finalize_evidence_advice
    from binance_agent.agents.state import StrategyAdvice

    model = BatchModel(responses=["unused"])
    state, _ = await run_batches(model, "Bitcoin ETF approved.")
    unsafe = "所给全部材料均未包含任何批准公告"
    state["news_research"]["tasks"][0]["gaps"] = [unsafe]
    state["news_research"]["gaps"] = [unsafe]
    state["final_advice"] = StrategyAdvice(
        symbol="BTCUSDT",
        direction="neutral",
        decision_type="no_trade",
        rationale=unsafe,
        no_trade_reason=unsafe,
        risk_warnings=[unsafe],
        reentry_conditions=[unsafe],
        confidence=0.2,
    )

    class RejectingModel(ResearchModel):
        async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
            if unknown:
                raise TimeoutError("offline verifier failure")
            return AIMessage(content='{"supported":false,"reason":"与事实卡矛盾"}')

    result = await finalize_evidence_advice(
        state, llm=RejectingModel(responses=["unused"]), runtime_cfg={"max_retries": 0}
    )
    advice = result["final_advice"]
    published = [
        advice.rationale,
        advice.no_trade_reason or "",
        *advice.risk_warnings,
        *advice.reentry_conditions,
        *advice.evidence_gaps,
    ]
    assert all(unsafe not in value for value in published)
    assert "Bitcoin ETF approved." in advice.rationale
    assert advice.decision_type == "no_trade" and advice.direction == "neutral"
    assert advice.entry_zone is None and advice.targets == []
    assert advice.confidence <= 0.2
    assert result["news_research"]["tasks"][0]["gaps"] == [unsafe]
    assert result["news_research"]["final_verification"]["status"] == (
        "unknown" if unknown else "unsupported"
    )


async def test_legacy_final_v3_verdict_is_not_reused_and_v4_replay_is_idempotent() -> None:
    from binance_agent.agents.research import finalize_evidence_advice
    from binance_agent.agents.state import StrategyAdvice

    model = BatchModel(responses=["unused"])
    state, _ = await run_batches(model, "Bitcoin ETF approved.")
    state["final_advice"] = StrategyAdvice(
        symbol="BTCUSDT", direction="long", decision_type="trade", rationale="Bitcoin ETF approved."
    )
    live = await finalize_evidence_advice(state, llm=model)
    # Even an otherwise matching publication hash cannot upgrade the old contract.
    live["news_research"]["final_verification"]["version"] = "final-evidence-v3"

    class ForbiddenModel(ResearchModel):
        async def ainvoke(self, *args: Any, **kwargs: Any) -> AIMessage:
            raise AssertionError("replay must not invoke models")

    forbidden = ForbiddenModel(responses=["unused"])
    replay = await finalize_evidence_advice(
        {**state, **live, "research_replay": True}, llm=forbidden
    )
    assert not replay["runtime_events"]
    assert replay["final_advice"].decision_type == "watchlist"
    record = replay["news_research"]["final_verification"]
    assert record["version"] == "final-evidence-v4" and record["status"] == "unknown"
    again = await finalize_evidence_advice(
        {**state, **replay, "research_replay": True}, llm=forbidden
    )
    assert again["final_advice"].model_dump() == replay["final_advice"].model_dump()
    assert again["news_research"]["final_verification"] == record
