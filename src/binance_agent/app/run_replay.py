"""Run Replay / Evaluation Harness.

基于 RunStore 中保存的 snapshot 重放 graph,并用业务不变量做回归检查。
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from ..agents.budget import BudgetPersistenceError
from ..agents.state import StrategyAdvice
from .run_store import FileRunStore
from .streaming import jsonable


@asynccontextmanager
async def _closing_stream(stream: Any):
    try:
        yield stream
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()


@dataclass
class ReplayResult:
    run_id: str
    original_record: dict[str, Any]
    replay_state: dict[str, Any]
    node_outputs: list[dict[str, Any]] = field(default_factory=list)
    original_final_advice: dict[str, Any] | None = None
    replay_final_advice: dict[str, Any] | None = None
    replay_planner_steps: list[dict[str, Any]] = field(default_factory=list)
    replay_planner_decision: dict[str, Any] | None = None
    advice_diffs: list[str] = field(default_factory=list)
    invariant_violations: list[str] = field(default_factory=list)


@dataclass
class ReplayBatchResult:
    run_ids: list[str]
    results: list[ReplayResult] = field(default_factory=list)
    missing_run_ids: list[str] = field(default_factory=list)
    failed_runs: list[dict[str, str]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def build_replay_state(record: dict[str, Any], *, max_debate_rounds: int = 2) -> dict[str, Any]:
    """从 run record 中恢复 graph 初始 state。

    保存的最终建议供应用图的离线回放分支使用;证据闸门仅确定性复核保存材料。
    注入的评估图仍可用快照检查规则,但应用图不会在 replay 重新请求模型。
    """

    return {
        "mode": record.get("mode"),
        "target": record.get("target"),
        "title": record.get("title"),
        "final_report": record.get("report"),
        "discovery": {},
        "candidate_advice": (
            StrategyAdvice.model_validate(record["final_advice"])
            if record.get("final_advice")
            else None
        ),
        "research_replay": True,
        "research_legacy": not bool((record.get("news_research") or {}).get("version")),
        "research_run_id": "",
        "input_revision": record.get("input_revision", 0),
        "research_demo": bool((record.get("news_research") or {}).get("synthetic", False)),
        "research_budget": {},
        "context_compactions": record.get("context_compactions") or {},
        "evidence_documents": [],
        "research_question": (record.get("research_input") or {}).get("research_question", ""),
        "analysis_at": (record.get("news_research") or {}).get("analysis_at")
        or record.get("created_at"),
        "news_research": record.get("news_research") or {},
        "symbol": str(record.get("symbol") or "").upper(),
        "market": record.get("market") or "fapi",
        "market_snapshot": record.get("market_snapshot") or {},
        "preflight_refetch_enabled": False,
        "snapshot_freshness_enabled": False,
        "snapshot_freshness_now": None,
        "latest_price": None,
        "intel_snapshot": record.get("intel_snapshot") or {},
        "analyst_reports": [],
        "debate_turns": [],
        "runtime_events": [],
        "tool_events": [],
        "planner_steps": [],
        "planner_decision": None,
        "debate_round": 0,
        "max_debate_rounds": max_debate_rounds,
        "debate_stop_reason": None,
        "research_summary": None,
        # 重放使用已提升的 canonical snapshots,但让当前 preflight 规则重新产生报告。
        "preflight_report": None,
        "debate_grounding": None,
        "risk_assessment": None,
        "risk_gate_result": None,
        "final_advice": None,
    }


async def replay_run(
    store: FileRunStore,
    run_id: str,
    graph: Any,
    *,
    max_debate_rounds: int = 2,
    config: dict[str, Any] | None = None,
    record: dict[str, Any] | None = None,
) -> ReplayResult:
    """用保存快照和证据驱动调用者提供的图,收集节点并对比最终建议及不变量。

    不追加新的 RunStore 运行记录;是否产生模型调用由传入图决定,生产离线回放须传保存结果图。
    """
    if record is None:
        record = await asyncio.to_thread(store.get, run_id)
    if record is None:
        raise ValueError(f"run not found: {run_id}")

    replay_state = build_replay_state(record, max_debate_rounds=max_debate_rounds)
    evidence = await asyncio.to_thread(store.load_evidence, run_id)
    replay_state["evidence_documents"] = [d.model_dump(mode="json") for d in evidence]
    node_outputs: list[dict[str, Any]] = []
    replay_final: dict[str, Any] | None = None
    replay_planner_steps: list[dict[str, Any]] = []
    replay_planner_decision: dict[str, Any] | None = None

    async with _closing_stream(
        _graph_astream(graph, replay_state, config=config)
    ) as owned_stream_2:
        async for update in owned_stream_2:
            for node, raw_output in update.items():
                payload = jsonable(raw_output)
                node_outputs.append({"node": node, "update": payload})
                if not isinstance(payload, dict):
                    continue
                planner_steps = payload.get("planner_steps") or []
                replay_planner_steps.extend(
                    step for step in planner_steps if isinstance(step, dict)
                )
                if payload.get("planner_decision") is not None:
                    replay_planner_decision = _as_dict(payload.get("planner_decision"))
                if payload.get("final_advice") is not None:
                    replay_final = _as_dict(payload.get("final_advice"))

    original_final = _as_dict(record.get("final_advice"))
    return ReplayResult(
        run_id=run_id,
        original_record=record,
        replay_state=replay_state,
        node_outputs=node_outputs,
        original_final_advice=original_final,
        replay_final_advice=replay_final,
        replay_planner_steps=replay_planner_steps,
        replay_planner_decision=replay_planner_decision,
        advice_diffs=compare_final_advice(original_final, replay_final),
        invariant_violations=check_replay_invariants(record, replay_final),
    )


async def replay_runs(
    store: FileRunStore,
    run_ids: Iterable[str],
    graph: Any,
    *,
    max_debate_rounds: int = 2,
    graph_config_for: Callable[[str], dict[str, Any]] | None = None,
) -> ReplayBatchResult:
    """依次回放指定 ID,分别记录缺失、失败和成功;单个失败不会吞掉其他运行的结果。"""
    ids = [str(run_id) for run_id in run_ids]
    results: list[ReplayResult] = []
    missing_run_ids: list[str] = []
    failed_runs: list[dict[str, str]] = []

    records = {str(record["run_id"]): record for record in await asyncio.to_thread(store.list_all)}
    for run_id in ids:
        if run_id not in records:
            missing_run_ids.append(run_id)
            continue
        try:
            results.append(
                await replay_run(
                    store,
                    run_id,
                    graph,
                    max_debate_rounds=max_debate_rounds,
                    config=graph_config_for(run_id) if graph_config_for else None,
                    record=records[run_id],
                )
            )
        except BudgetPersistenceError:
            raise
        except Exception as exc:  # one bad replay should not invalidate the whole batch
            failed_runs.append({"run_id": run_id, "error": str(exc)[:300]})

    return ReplayBatchResult(
        run_ids=ids,
        results=results,
        missing_run_ids=missing_run_ids,
        failed_runs=failed_runs,
        summary=summarize_replay_results(
            results,
            missing_run_ids=missing_run_ids,
            failed_runs=failed_runs,
        ),
    )


async def replay_recent_runs(
    store: FileRunStore,
    graph: Any,
    *,
    limit: int = 20,
    max_debate_rounds: int = 2,
    completed_only: bool = True,
    graph_config_for: Callable[[str], dict[str, Any]] | None = None,
) -> ReplayBatchResult:
    records = await asyncio.to_thread(store.list_recent, limit=limit)
    if completed_only:
        records = [record for record in records if record.get("status") == "completed"]
    run_ids = [str(record.get("run_id")) for record in records if record.get("run_id")]
    return await replay_runs(
        store,
        run_ids,
        graph,
        max_debate_rounds=max_debate_rounds,
        graph_config_for=graph_config_for,
    )


def summarize_replay_results(
    results: list[ReplayResult],
    *,
    missing_run_ids: list[str] | None = None,
    failed_runs: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    missing = missing_run_ids or []
    failed = failed_runs or []
    summary: dict[str, Any] = {
        "requested_runs": len(results) + len(missing) + len(failed),
        "replayed_runs": len(results),
        "missing_runs": len(missing),
        "failed_runs": len(failed),
        "runs_with_advice_diffs": 0,
        "advice_diff_count": 0,
        "final_decision_changes": 0,
        "runs_with_invariant_violations": 0,
        "invariant_violation_count": 0,
        "planner_runs": 0,
        "planner_modes": {},
        "planner_actions": {},
        "planner_action_status": {},
        "planner_enforced_runs": 0,
        "planner_affects_final_advice_runs": 0,
    }

    for result in results:
        if result.advice_diffs:
            summary["runs_with_advice_diffs"] += 1
            summary["advice_diff_count"] += len(result.advice_diffs)
        if any(diff.startswith("decision_type changed:") for diff in result.advice_diffs):
            summary["final_decision_changes"] += 1
        if result.invariant_violations:
            summary["runs_with_invariant_violations"] += 1
            summary["invariant_violation_count"] += len(result.invariant_violations)

        decision = result.replay_planner_decision
        if not isinstance(decision, dict):
            continue
        summary["planner_runs"] += 1
        _bump(summary["planner_modes"], str(decision.get("mode") or "unknown"))
        _bump(summary["planner_actions"], str(decision.get("decision_type") or "unknown"))
        _bump(summary["planner_action_status"], str(decision.get("action_status") or "unknown"))
        if decision.get("enforced") is True:
            summary["planner_enforced_runs"] += 1
        if decision.get("affects_final_advice") is True:
            summary["planner_affects_final_advice_runs"] += 1

    return summary


def compare_final_advice(
    original: Any,
    replayed: Any,
    *,
    confidence_tolerance: float = 0.15,
) -> list[str]:
    """比较结构化策略差异,忽略 LLM 文本类字段。"""

    left = _as_dict(original)
    right = _as_dict(replayed)
    if left is None or right is None:
        return ["final_advice missing in original or replay"]

    diffs: list[str] = []
    for field_name in ("decision_type", "direction", "entry_zone", "stop_loss", "targets"):
        if left.get(field_name) != right.get(field_name):
            diffs.append(
                f"{field_name} changed: {left.get(field_name)!r} -> {right.get(field_name)!r}"
            )

    left_conf = _float(left.get("confidence"))
    right_conf = _float(right.get("confidence"))
    if (
        left_conf is not None
        and right_conf is not None
        and abs(left_conf - right_conf) > confidence_tolerance
    ):
        diffs.append(f"confidence changed: {left_conf:.2f} -> {right_conf:.2f}")
    return diffs


def check_replay_invariants(record: dict[str, Any], replay_advice: Any) -> list[str]:
    advice = _as_dict(replay_advice) or {}
    violations: list[str] = []
    decision_type = str(advice.get("decision_type") or "")
    direction = str(advice.get("direction") or "")
    confidence = _float(advice.get("confidence")) or 0.0

    data_quality = _data_quality(record)
    if _critical_market_missing(record, data_quality) and not _is_no_trade(advice):
        violations.append("关键行情缺失仍必须 no_trade")
    if data_quality.get("trade_blocked") and confidence > 0.25:
        violations.append("数据质量 trade_blocked 时 confidence 必须 <= 0.25")
    if str(data_quality.get("level") or "") == "degraded" and confidence > 0.55:
        violations.append("数据质量 degraded 时 confidence 必须 <= 0.55")

    risk_gate = _risk_gate(record)
    if risk_gate.get("veto") and not _is_no_trade(advice):
        violations.append("RiskGate veto 后仍必须 no_trade")
    if risk_gate.get("veto") and confidence > 0.25:
        violations.append("RiskGate veto 后 confidence 必须 <= 0.25")
    if risk_gate.get("confidence_cap") is not None:
        cap = _float(risk_gate.get("confidence_cap"))
        if cap is not None and confidence > cap:
            violations.append(f"RiskGate confidence cap {cap:.2f} 被突破")

    if _analysts_contested(record) and decision_type == "trade" and confidence > 0.55:
        violations.append("分析师分歧时不能高置信单边 trade")

    runtime_events = _runtime_events(record)
    if (
        any(
            event.get("agent") == "strategy_synth" and not event.get("ok", True)
            for event in runtime_events
        )
        and confidence > 0.25
    ):
        violations.append("strategy_synth runtime fallback 后 confidence 必须 <= 0.25")
    if (
        any(
            event.get("agent") == "strategy_synth" and event.get("repaired")
            for event in runtime_events
        )
        and confidence > 0.65
    ):
        violations.append("strategy_synth JSON repair 后 confidence 必须 <= 0.65")

    if decision_type == "no_trade":
        if direction != "neutral":
            violations.append("no_trade 必须 direction=neutral")
        if advice.get("entry_zone") or advice.get("stop_loss") or advice.get("targets"):
            violations.append("no_trade 不允许输出具体交易价位")
        if not advice.get("no_trade_reason"):
            violations.append("no_trade_reason 不能为空")
        if not advice.get("reentry_conditions"):
            violations.append("reentry_conditions 不能为空")

    return violations


def _critical_market_missing(record: dict[str, Any], data_quality: dict[str, Any]) -> bool:
    if data_quality.get("trade_blocked") or str(data_quality.get("level") or "") == "critical":
        return True
    market = record.get("market_snapshot") or {}
    ticker = market.get("ticker_24h") or {}
    klines = market.get("klines") or {}
    missing_price = _float(ticker.get("lastPrice")) is None
    missing_core_klines = not klines.get("1h") or not klines.get("4h")
    return missing_price or missing_core_klines


def _data_quality(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("data_quality_report")
    return raw if isinstance(raw, dict) else {}


def _risk_gate(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("risk_gate_result")
    return raw if isinstance(raw, dict) else {}


def _runtime_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    events = record.get("runtime_events") or []
    return [event for event in events if isinstance(event, dict)]


def _analysts_contested(record: dict[str, Any]) -> bool:
    stances: list[str] = []
    for node in record.get("node_outputs") or []:
        update = node.get("update") if isinstance(node, dict) else None
        if not isinstance(update, dict):
            continue
        for report in update.get("analyst_reports") or []:
            if isinstance(report, dict):
                stances.append(str(report.get("stance") or "neutral"))
    return "bullish" in stances and "bearish" in stances


def _is_no_trade(advice: dict[str, Any]) -> bool:
    return advice.get("decision_type") == "no_trade" and advice.get("direction") == "neutral"


def _as_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, StrategyAdvice):
        return value.model_dump(mode="json")
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else None
    if isinstance(value, dict):
        return value
    return None


async def _graph_astream(
    graph: Any,
    state: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    try:
        params = inspect.signature(graph.astream).parameters
    except (TypeError, ValueError):
        accepts_config = True
    else:
        accepts_config = "config" in params or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )
    if accepts_config:
        async with _closing_stream(
            graph.astream(state, config=config or {}, stream_mode="updates")
        ) as owned_stream_1:
            async for update in owned_stream_1:
                yield update
    else:
        async with _closing_stream(graph.astream(state, stream_mode="updates")) as owned_stream_0:
            async for update in owned_stream_0:
                yield update


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
