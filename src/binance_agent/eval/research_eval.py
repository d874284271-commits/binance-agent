"""离线新闻研究评测工具。

此模块只生成冻结实验清单及汇总外部 runner 的 JSONL 结果。它不调用模型、搜索
或新闻网站, 因此不能把合成 fixture 的通过率表述为模型效果。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

INPUT_MODES = ("title", "summary", "evidence")
WORKFLOWS = ("single", "multi_no_debate", "multi_debate")
SCHEMA_VERSION = 1


def derived_summary_source(case_id: str, article: Mapping[str, Any]) -> dict[str, Any] | None:
    """A case-wide summary is derived input, never one publisher's original text."""
    summary = article.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return None
    return {
        "source_id": f"case_summary:{case_id}",
        "url": f"https://fixture.invalid/corpus-summary/{quote(case_id, safe='')}",
        "title": str(article.get("title") or "Case summary"),
        "source": "Composite corpus summary (derived; not an original source)",
        "source_kind": "unknown",
        "published_at": None,
        "summary": summary,
        "extraction_kind": "derived_case_summary/v2",
    }


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Validate legacy fixtures or frozen historical sources without cross-split leakage."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(cases, list):
        raise ValueError("cases JSON 必须包含 cases 数组")
    identifiers = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if (
        len(identifiers) != len(cases)
        or len(set(identifiers)) != len(cases)
        or any(not isinstance(value, str) or not value for value in identifiers)
    ):
        raise ValueError("每个 case 必须有唯一 case_id")
    historical = isinstance(raw, dict) and raw.get("corpus_kind") == "historical_sources"
    if historical and raw.get("schema_version") != 2:
        raise ValueError("historical_sources requires schema_version 2")
    splits_by_event: dict[str, set[str]] = {}
    splits_by_source: dict[str, set[str]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case 必须为对象")
        if case.get("split") not in {"dev", "holdout"}:
            raise ValueError("case split 必须为 dev 或 holdout")
        event_id = case.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("case 必须有 event_id")
        event_ids = case.get("event_ids", [event_id])
        if not isinstance(event_ids, list) or not event_ids or event_id not in event_ids:
            raise ValueError("event_ids must include event_id")
        for event in event_ids:
            if not isinstance(event, str) or not event:
                raise ValueError("event_ids must contain nonempty strings")
            splits_by_event.setdefault(event, set()).add(str(case["split"]))
        if historical:
            _validate_historical_case(case)
            case["corpus_kind"] = "historical_sources"
            for source in case["sources"]:
                for identity in (source["url"], source["content_sha256"]):
                    splits_by_source.setdefault(identity, set()).add(case["split"])
    if any(len(splits) > 1 for splits in splits_by_event.values()):
        raise ValueError("同一 event_id 不得跨 dev/holdout, 防止泄漏")
    if any(len(splits) > 1 for splits in splits_by_source.values()):
        raise ValueError("同一 source 不得跨 dev/holdout")
    return cases


def build_manifest(
    cases: Iterable[Mapping[str, Any]],
    *,
    input_mode: str,
    workflow: str,
    model_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze one experimental cell without altering the other variable."""
    _validate_cell(input_mode, workflow)
    frozen_cases = [_case_for_mode(case, input_mode) for case in cases]
    historical = any(c.get("corpus_kind") == "historical_sources" for c in frozen_cases)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_kind": "historical_sources" if historical else "offline_fixture",
        "input_mode": input_mode,
        "workflow": workflow,
        "model_config": dict(model_config or {}),
        "case_count": len(frozen_cases),
        "input_fingerprint": _fingerprint(frozen_cases),
        "cases": frozen_cases,
        "limitations": [
            "frozen source extracts do not establish complete webpage coverage or financial returns"
            if historical
            else "synthetic fixtures are engineering regression data, not real-news performance evidence",
            "human source verification remains required before claiming factual-model quality",
        ],
    }


def build_comparison_manifests(
    cases: Iterable[Mapping[str, Any]], *, model_config: Mapping[str, Any] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Build two non-confounded comparisons.

    Content comparison fixes the workflow at ``multi_no_debate``. Workflow comparison
    fixes the input at ``evidence``. Callers must not aggregate the two groups together.
    """
    frozen = [dict(case) for case in cases]
    return {
        "input_mode_comparison": [
            build_manifest(
                frozen, input_mode=mode, workflow="multi_no_debate", model_config=model_config
            )
            for mode in INPUT_MODES
        ],
        "workflow_comparison": [
            build_manifest(
                frozen, input_mode="evidence", workflow=workflow, model_config=model_config
            )
            for workflow in WORKFLOWS
        ],
    }


async def run_fixture_manifest(
    manifest: Mapping[str, Any],
    runner: Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Run an injected async runner for offline wiring tests only.

    The runner is deliberately supplied by the caller; this package has no model runner
    and no budget authority.
    """
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest 缺少 cases")
    responses = await asyncio.gather(*(runner(case) for case in cases))
    return [
        {
            "case_id": case["case_id"],
            "input_mode": manifest["input_mode"],
            "workflow": manifest["workflow"],
            **dict(response),
        }
        for case, response in zip(cases, responses, strict=True)
    ]


def score_jsonl(
    cases: Iterable[Mapping[str, Any]], results: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score one manifest cell; denominators include unsupported and failed outputs."""
    case_map = {str(case["case_id"]): dict(case) for case in cases}
    result_map: dict[str, dict[str, Any]] = {}
    for row in results:
        case_id = str(row.get("case_id"))
        if case_id in result_map:
            raise ValueError("duplicate case result; select one attempt explicitly")
        result_map[case_id] = dict(row)
    rows: list[dict[str, Any]] = []
    for case_id, case in case_map.items():
        result = result_map.get(case_id)
        if result is None:
            rows.append(_missing_row(case))
            continue
        rows.append(_score_case(case, result))
    report = _summarize(rows, extra_results=max(0, len(result_map) - len(case_map)))
    if any(case.get("corpus_kind") == "historical_sources" for case in case_map.values()):
        report["interpretation"] = (
            "真实来源的关键词诊断;不构成语义质量验收。必须另行导入独立原文评审。"
        )
    return report


def score_jsonl_file(cases_path: str | Path, results_path: str | Path) -> dict[str, Any]:
    """Read newline-delimited JSON runner output and produce a transparent summary."""
    results: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(results_path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_no} 行无效") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL 第 {line_no} 行必须是对象")
            results.append(item)
    return score_jsonl(load_cases(cases_path), results)


def _case_for_mode(case: Mapping[str, Any], mode: str) -> dict[str, Any]:
    # Explicit allowlist: newly added labels cannot silently leak to the model.
    copied = {
        key: case[key]
        for key in (
            "case_id",
            "event_id",
            "event_ids",
            "split",
            "symbol",
            "analysis_at",
            "question",
            "corpus_kind",
        )
        if key in case
    }
    article = case.get("article") or {}
    research_input = {"title": article.get("title", "")}
    if mode != "title":
        research_input["summary"] = article.get("summary", "")
    if mode == "evidence":
        research_input["body"] = article.get("body", "")
    if case.get("corpus_kind") == "historical_sources":
        if mode == "evidence":
            # Full-source arms use the original bodies, not an additional curated hint.
            research_input.pop("summary", None)
        research_input["sources"] = [
            {
                key: source[key]
                for key in (
                    "source_id",
                    "url",
                    "title",
                    "source",
                    "source_kind",
                    "published_at",
                    "fetched_at",
                    "content_sha256",
                    "status",
                    "extraction_kind",
                    "body",
                )
                if key in source and (mode == "evidence" or key != "body")
            }
            for source in case["sources"]
        ]
    copied["research_input"] = research_input
    return copied


def _timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("source timestamps must have timezone")
    return parsed


def _validate_historical_case(case: Mapping[str, Any]) -> None:
    cutoff = _timestamp(case.get("analysis_at"))
    if not case.get("question") or not case.get("symbol"):
        raise ValueError("historical case requires question and symbol")
    sources = case.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("historical case requires frozen sources")
    source_map = {}
    for source in sources:
        source_id = source.get("source_id")
        if not source_id or source_id in source_map or not source.get("url"):
            raise ValueError("invalid or duplicate source_id/url")
        if _timestamp(source.get("published_at")) > cutoff:
            raise ValueError("source published after analysis_at")
        _timestamp(source.get("fetched_at"))
        body = source.get("body")
        if not isinstance(body, str) or not body or source.get("status") != "available":
            raise ValueError("source requires available frozen body")
        if hashlib.sha256(body.encode()).hexdigest() != source.get("content_sha256"):
            raise ValueError("frozen source hash mismatch")
        source_map[source_id] = source
    facts = case.get("expected_facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError("historical case requires expected facts")
    for fact in facts:
        source = source_map.get(fact.get("source_id"))
        quote = fact.get("quote")
        if not source or not isinstance(quote, str) or not quote or quote not in source["body"]:
            raise ValueError("expected fact quote not found in frozen source")
        if not fact.get("statement") or not isinstance(fact.get("critical"), bool):
            raise ValueError("invalid expected fact statement/critical")


def _score_case(case: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    status = str(result.get("status", "completed"))
    output = _text(result.get("output", result.get("conclusion", "")))
    required = [str(term) for term in case.get("required_terms", [])]
    forbidden = [str(term) for term in case.get("forbidden_assertions", [])]
    expected_facts = case.get("expected_facts", [])
    facts_ok = sum(1 for fact in expected_facts if _contains_all(output, _fact_terms(fact)))
    return {
        "case_id": str(case["case_id"]),
        "status": status,
        "required_total": len(required),
        "required_supported": sum(1 for term in required if term.casefold() in output.casefold()),
        "facts_total": len(expected_facts),
        "facts_supported": facts_ok,
        "forbidden_assertions": [
            term for term in forbidden if term.casefold() in output.casefold()
        ],
        "critical_unresolved": bool(case.get("critical_unresolved")),
        "unresolved_acknowledged": _acknowledges_unresolved(output),
        "token": _number_or_none(result.get("token")),
        "latency_ms": _number_or_none(result.get("latency_ms")),
        "cost_usd": _number_or_none(result.get("cost_usd")),
    }


def _summarize(rows: list[dict[str, Any]], *, extra_results: int) -> dict[str, Any]:
    total = len(rows)
    required_total = sum(row["required_total"] for row in rows)
    facts_total = sum(row["facts_total"] for row in rows)
    unsupported = sum(row["status"] == "unsupported" for row in rows)
    missing = sum(row["status"] == "missing" for row in rows)
    forbidden = sum(len(row["forbidden_assertions"]) for row in rows)
    critical_total = sum(row["critical_unresolved"] for row in rows)
    critical_ack = sum(
        row["critical_unresolved"] and row["unresolved_acknowledged"] for row in rows
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "case_denominator": total,
        "result_rows": len(rows),
        "extra_result_rows": extra_results,
        "unsupported_count": unsupported,
        "missing_result_count": missing,
        "required_term_coverage": _rate(
            sum(row["required_supported"] for row in rows), required_total
        ),
        "fact_coverage": _rate(sum(row["facts_supported"] for row in rows), facts_total),
        "forbidden_assertion_count": forbidden,
        "critical_unresolved_acknowledgement": _rate(critical_ack, critical_total),
        "metrics": {
            metric: _usage_summary(rows, metric) for metric in ("token", "latency_ms", "cost_usd")
        },
        "rows": rows,
        "interpretation": (
            "仅为离线 fixture 评分; 不构成真实新闻、模型事实性或投资表现的成功声明。"
            "人工原文核对尚待完成。"
        ),
    }


def _missing_row(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(case["case_id"]),
        "status": "missing",
        "required_total": len(case.get("required_terms", [])),
        "required_supported": 0,
        "facts_total": len(case.get("expected_facts", [])),
        "facts_supported": 0,
        "forbidden_assertions": [],
        "critical_unresolved": bool(case.get("critical_unresolved")),
        "unresolved_acknowledged": False,
        "token": None,
        "latency_ms": None,
        "cost_usd": None,
    }


def _usage_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [value for row in rows if (value := row[key]) is not None]
    return {
        "known_count": len(values),
        "unknown_count": len(rows) - len(values),
        "total": sum(values) if values else None,
    }


def _validate_cell(input_mode: str, workflow: str) -> None:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"未知 input_mode: {input_mode}")
    if workflow not in WORKFLOWS:
        raise ValueError(f"未知 workflow: {workflow}")


def _fingerprint(cases: list[dict[str, Any]]) -> str:
    payload = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _text(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        if isinstance(value, (dict, list))
        else str(value or "")
    )


def _fact_terms(fact: Any) -> list[str]:
    if isinstance(fact, dict):
        return [str(value) for value in fact.get("terms", [])]
    return [str(fact)]


def _contains_all(text: str, terms: list[str]) -> bool:
    return bool(terms) and all(term.casefold() in text.casefold() for term in terms)


def _acknowledges_unresolved(text: str) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in ("未核实", "不确定", "无法确认", "watchlist", "观望"))


def _number_or_none(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None
