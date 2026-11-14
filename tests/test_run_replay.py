"""Run Replay / Evaluation Harness:把历史 run 变成回归评估资产。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from binance_agent.app.run_replay import (
    build_replay_state,
    check_replay_invariants,
    compare_final_advice,
    replay_recent_runs,
    replay_run,
    replay_runs,
)
from binance_agent.app.run_store import FileRunStore


class _Graph:
    def __init__(self, final_advice: dict[str, Any]) -> None:
        self.final_advice = final_advice
        self.received_state: dict[str, Any] | None = None

    async def astream(self, state: dict[str, Any], *, stream_mode: str) -> Any:
        assert stream_mode == "updates"
        self.received_state = state
        yield {
            "strategy_synth": {
                "final_advice": self.final_advice,
                "runtime_events": [{"agent": "strategy_synth", "ok": True}],
            }
        }


class _BatchGraph:
    async def astream(self, state: dict[str, Any], *, stream_mode: str) -> Any:
        assert stream_mode == "updates"
        final = (
            _trade_advice(confidence=0.8)
            if not state["market_snapshot"].get("ticker_24h", {}).get("lastPrice")
            else _no_trade_advice(confidence=0.2)
        )
        yield {"strategy_synth": {"final_advice": final}}


class _PlannerGraph:
    async def astream(self, state: dict[str, Any], *, stream_mode: str) -> Any:
        assert stream_mode == "updates"
        yield {
            "strategy_planner": {
                "planner_steps": [{"step": 1, "action": "llm_action_planner"}],
                "planner_decision": {
                    "decision_type": "no_trade",
                    "mode": "llm_shadow",
                    "enforced": False,
                    "affects_final_advice": False,
                    "action_status": "accepted",
                    "reason": "shadow only",
                },
            }
        }
        yield {"strategy_synth": {"final_advice": _no_trade_advice(confidence=0.2)}}


def test_build_replay_state_from_saved_run(tmp_path: Path) -> None:
    store, run_id = _saved_run(tmp_path, market_snapshot=_complete_market())
    record = store.get(run_id)
    assert record is not None

    state = build_replay_state(record, max_debate_rounds=3)

    assert state["symbol"] == "BTCUSDT"
    assert state["market"] == "fapi"
    assert state["market_snapshot"]["ticker_24h"]["lastPrice"] == "100"
    assert state["intel_snapshot"]["source_status"]["rss"] == "ok"
    assert state["max_debate_rounds"] == 3
    assert state["analyst_reports"] == []
    assert state["runtime_events"] == []


def test_build_replay_state_disables_snapshot_freshness() -> None:
    state = build_replay_state(
        {
            "symbol": "BTCUSDT",
            "market": "fapi",
            "market_snapshot": {"fetched_at": "2020-01-01T00:00:00+00:00"},
            "intel_snapshot": {},
        }
    )

    assert state["snapshot_freshness_enabled"] is False
    assert state["preflight_refetch_enabled"] is False
    assert state["latest_price"] is None


def test_replay_uses_preflight_promoted_canonical_snapshots(tmp_path: Path) -> None:
    store, run_id = _saved_run(tmp_path, market_snapshot=_complete_market())
    store.append_event(
        run_id,
        "node",
        {
            "node": "data_preflight",
            "update": {
                "market_snapshot": {
                    **_complete_market(),
                    "ticker_24h": {
                        **_complete_market()["ticker_24h"],
                        "lastPrice": "123",
                    },
                },
                "preflight_report": {
                    "status": "ready",
                    "effective_snapshot_version": "sha256:new",
                    "final_quality": {"level": "ok", "score": 1.0},
                },
            },
        },
    )
    record = store.get(run_id)
    assert record is not None

    state = build_replay_state(record)

    assert state["market_snapshot"]["ticker_24h"]["lastPrice"] == "123"
    assert state["preflight_report"] is None


def test_build_replay_state_covers_all_agent_state_keys(tmp_path: Path) -> None:
    # #7 回归:replay 初始 state 必须与 AgentState/生产 /analyze state 同构,
    # 否则 planner 开启时回放路径(缺 tool_events/planner_steps/planner_decision)与生产漂移。
    from binance_agent.agents.state import AgentState

    store, run_id = _saved_run(tmp_path, market_snapshot=_complete_market())
    record = store.get(run_id)
    assert record is not None

    state = build_replay_state(record)
    missing = set(AgentState.__annotations__) - set(state)
    assert missing == set(), f"replay state 缺少 AgentState 键: {missing}"
    assert state["tool_events"] == []
    assert state["planner_steps"] == []
    assert state["planner_decision"] is None


async def test_replay_run_invokes_graph_with_saved_snapshots(tmp_path: Path) -> None:
    store, run_id = _saved_run(tmp_path, market_snapshot=_complete_market())
    graph = _Graph(_no_trade_advice(confidence=0.2))

    result = await replay_run(store, run_id, graph, max_debate_rounds=2)

    assert graph.received_state is not None
    assert graph.received_state["market_snapshot"]["ticker_24h"]["lastPrice"] == "100"
    assert graph.received_state["intel_snapshot"]["source_status"]["rss"] == "ok"
    assert result.run_id == run_id
    assert result.replay_final_advice is not None
    assert result.replay_final_advice["decision_type"] == "no_trade"
    assert result.node_outputs[0]["node"] == "strategy_synth"


async def test_application_graph_replay_does_not_invoke_any_model(tmp_path: Path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from binance_agent.agents.graph import build_graph

    class ForbiddenModel(FakeListChatModel):
        async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("replay must not call models, including compaction")

    store, run_id = _saved_run(tmp_path, market_snapshot=_complete_market())
    graph = build_graph(lambda _: ForbiddenModel(responses=["unused"]))
    result = await replay_run(store, run_id, graph)
    assert [node["node"] for node in result.node_outputs] == ["replay_saved"]
    assert result.replay_final_advice is not None
    assert result.replay_final_advice["decision_type"] == "no_trade"


async def test_replay_runs_evaluates_batch_without_aborting_on_missing_run(
    tmp_path: Path,
) -> None:
    store, stable_run = _saved_run(tmp_path, market_snapshot=_complete_market())
    _, risky_run = _saved_run(
        tmp_path,
        market_snapshot={"klines": {}, "ticker_24h": {}, "errors": ["ticker: boom"]},
        data_quality={"level": "critical", "trade_blocked": True, "score": 0.1},
    )

    result = await replay_runs(
        store,
        [stable_run, "missing-run", risky_run],
        _BatchGraph(),
        max_debate_rounds=2,
    )

    assert result.missing_run_ids == ["missing-run"]
    assert [item.run_id for item in result.results] == [stable_run, risky_run]
    assert result.summary["requested_runs"] == 3
    assert result.summary["replayed_runs"] == 2
    assert result.summary["missing_runs"] == 1
    assert result.summary["runs_with_advice_diffs"] == 1
    assert result.summary["runs_with_invariant_violations"] == 1
    assert result.summary["final_decision_changes"] == 1


async def test_replay_runs_summarizes_planner_rollout_signals(tmp_path: Path) -> None:
    store, run_id = _saved_run(tmp_path, market_snapshot=_complete_market())

    result = await replay_runs(store, [run_id], _PlannerGraph())

    planner_decision = result.results[0].replay_planner_decision
    planner_steps = result.results[0].replay_planner_steps
    assert planner_decision is not None
    assert planner_steps is not None
    assert planner_decision["mode"] == "llm_shadow"
    assert planner_steps[0]["action"] == "llm_action_planner"
    assert result.summary["planner_runs"] == 1
    assert result.summary["planner_modes"] == {"llm_shadow": 1}
    assert result.summary["planner_actions"] == {"no_trade": 1}
    assert result.summary["planner_action_status"] == {"accepted": 1}
    assert result.summary["planner_enforced_runs"] == 0
    assert result.summary["planner_affects_final_advice_runs"] == 0


async def test_replay_recent_runs_uses_completed_records_in_recent_order(tmp_path: Path) -> None:
    store, older_run = _saved_run(tmp_path, market_snapshot=_complete_market())
    _, newer_run = _saved_run(tmp_path, market_snapshot=_complete_market())
    failed_run = store.create(symbol="ETHUSDT", market="fapi")
    store.fail(failed_run, "boom")

    result = await replay_recent_runs(store, _BatchGraph(), limit=3)

    assert [item.run_id for item in result.results] == [newer_run, older_run]
    assert result.summary["requested_runs"] == 2
    assert result.summary["replayed_runs"] == 2


def test_compare_final_advice_ignores_text_but_reports_material_changes() -> None:
    original = _no_trade_advice(confidence=0.2, rationale="old text")
    replayed = {**original, "rationale": "new text"}
    assert compare_final_advice(original, replayed) == []

    diffs = compare_final_advice(original, _trade_advice(confidence=0.8))
    assert any("decision_type" in diff for diff in diffs)
    assert any("direction" in diff for diff in diffs)


def test_invariants_require_no_trade_when_critical_market_missing(tmp_path: Path) -> None:
    store, run_id = _saved_run(
        tmp_path,
        market_snapshot={"klines": {}, "ticker_24h": {}, "errors": ["klines[1h]: boom"]},
        data_quality={"level": "critical", "trade_blocked": True, "score": 0.2},
    )
    record = store.get(run_id)
    assert record is not None

    violations = check_replay_invariants(record, _trade_advice(confidence=0.8))

    assert any("关键行情缺失" in violation for violation in violations)


def test_invariants_require_risk_veto_to_remain_no_trade(tmp_path: Path) -> None:
    store, run_id = _saved_run(
        tmp_path,
        market_snapshot=_complete_market(),
        risk_gate={"veto": True, "reason": "long 止损方向错误"},
    )
    record = store.get(run_id)
    assert record is not None

    violations = check_replay_invariants(record, _trade_advice(confidence=0.7))

    assert any("RiskGate veto" in violation for violation in violations)


def test_invariants_enforce_confidence_caps_and_analyst_disagreement(tmp_path: Path) -> None:
    store, run_id = _saved_run(
        tmp_path,
        market_snapshot=_complete_market(),
        data_quality={"level": "degraded", "trade_blocked": False, "score": 0.62},
        analyst_reports=[
            {"analyst": "technical", "stance": "bullish", "confidence": 0.7},
            {"analyst": "funding", "stance": "bearish", "confidence": 0.7},
            {"analyst": "news_sentiment", "stance": "neutral", "confidence": 0.6},
        ],
    )
    record = store.get(run_id)
    assert record is not None

    violations = check_replay_invariants(record, _trade_advice(confidence=0.9))

    assert any("数据质量" in violation and "confidence" in violation for violation in violations)
    assert any("分析师分歧" in violation for violation in violations)


def test_invariants_penalize_strategy_runtime_repair_and_fallback(tmp_path: Path) -> None:
    store, run_id = _saved_run(
        tmp_path,
        market_snapshot=_complete_market(),
        runtime_events=[
            {"agent": "strategy_synth", "ok": False, "error_category": "timeout"},
            {"agent": "strategy_synth", "ok": True, "repaired": True},
        ],
    )
    record = store.get(run_id)
    assert record is not None

    violations = check_replay_invariants(record, _no_trade_advice(confidence=0.8))

    assert any("runtime fallback" in violation for violation in violations)
    assert any("JSON repair" in violation for violation in violations)


def _saved_run(
    tmp_path: Path,
    *,
    market_snapshot: dict[str, Any],
    data_quality: dict[str, Any] | None = None,
    risk_gate: dict[str, Any] | None = None,
    analyst_reports: list[dict[str, Any]] | None = None,
    runtime_events: list[dict[str, Any]] | None = None,
) -> tuple[FileRunStore, str]:
    store = FileRunStore(tmp_path / "runs.jsonl")
    run_id = store.create(symbol="BTCUSDT", market="fapi")
    store.append_event(run_id, "market", market_snapshot)
    store.append_event(run_id, "intel", _intel())
    if analyst_reports:
        store.append_event(
            run_id,
            "node",
            {"node": "technical", "update": {"analyst_reports": analyst_reports}},
        )
    if data_quality:
        store.append_event(
            run_id,
            "node",
            {
                "node": "risk_manager",
                "update": {
                    "risk_assessment": {
                        "level": "low",
                        "warnings": [],
                        "data_quality": data_quality,
                    }
                },
            },
        )
    if runtime_events or risk_gate:
        store.append_event(
            run_id,
            "node",
            {
                "node": "strategy_synth",
                "update": {
                    "runtime_events": runtime_events or [],
                    "risk_gate_result": risk_gate or {"veto": False},
                },
            },
        )
    store.complete(run_id, _no_trade_advice(confidence=0.2))
    return store, run_id


def _complete_market() -> dict[str, Any]:
    rows = [{"high": 102, "low": 98, "close": 100, "volume": 10} for _ in range(35)]
    return {
        "market": "fapi",
        "klines": {"1h": rows, "4h": rows},
        "ticker_24h": {
            "lastPrice": "100",
            "highPrice": "105",
            "lowPrice": "95",
            "volume": "1000",
            "quoteVolume": "100000",
        },
        "errors": [],
    }


def _intel() -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "source_status": {"rss": "ok", "reddit": "ok"},
        "news_items": [{"title": "headline"}],
        "macro_events": [],
        "sentiment_features": {},
    }


def _no_trade_advice(*, confidence: float, rationale: str = "观望") -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "direction": "neutral",
        "decision_type": "no_trade",
        "entry_zone": None,
        "stop_loss": None,
        "targets": [],
        "confidence": confidence,
        "rationale": rationale,
        "no_trade_reason": "数据不足",
        "reentry_conditions": ["等待数据恢复"],
    }


def _trade_advice(*, confidence: float) -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "direction": "long",
        "decision_type": "trade",
        "entry_zone": "100",
        "stop_loss": "95",
        "targets": ["115"],
        "confidence": confidence,
        "rationale": "trade",
    }
