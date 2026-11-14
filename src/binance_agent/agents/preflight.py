"""确定性数据预检:在任何 LLM fan-out 前建立唯一 canonical snapshots。

调用方只需传入 graph state 与可选 refetch 依赖;本模块隐藏质检、目标推断、每类一次预算、
工具失败降级、快照提升和 deterministic no-trade 的全部实现细节。
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .state import AgentState, StrategyAdvice
from .tools import ToolInvocationResult, ToolRegistry, build_default_tool_registry
from .validation import DataQualityReport, build_data_quality_report

logger = logging.getLogger(__name__)

ToolContextProvider = dict[str, Any] | Callable[[], dict[str, Any]] | None


async def run_data_preflight(
    state: AgentState | dict[str, Any],
    *,
    tool_registry: ToolRegistry | None = None,
    tool_context: ToolContextProvider = None,
    refetch_enabled: bool = True,
    analysis_clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """在任何分析 LLM 调用前建立 canonical 快照,返回 LangGraph state update。

    按缺失类别最多补抓一次,成功数据替换该类快照;补抓失败或最终质量阻断时
    返回确定性 no_trade,ready 时不生成 final_advice。analysis_at 在允许补抓
    完成后冻结,注入时钟必须带时区,后续研究和回放复用该时点。
    """

    market_snapshot = dict(state.get("market_snapshot") or {})
    intel_snapshot = dict(state.get("intel_snapshot") or {})
    effective_refetch_enabled = bool(state.get("preflight_refetch_enabled", refetch_enabled))
    initial_quality = build_data_quality_report(market_snapshot, intel_snapshot)
    requested = _refetch_targets(initial_quality)
    attempts: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    failures: list[str] = []
    updates: dict[str, Any] = {}

    if effective_refetch_enabled and requested:
        context = _resolve_tool_context(tool_context)
        capabilities = _refetch_capabilities(context, str(state.get("market") or "fapi"))
        registry = tool_registry if tool_registry is not None else build_default_tool_registry()
        for kind in requested:  # requested 去重且顺序固定;每类在整个节点内最多一次。
            result = await _refetch_once(
                registry=registry,
                kind=kind,
                symbol=str(state.get("symbol") or "").upper(),
                market=str(state.get("market") or "fapi"),
                context=capabilities.get(kind),
            )
            attempts.append(result["attempt"])
            if result["tool_event"] is not None:
                tool_events.append(result["tool_event"])
            fresh = result["snapshot"]
            if fresh is not None:
                if kind == "market":
                    market_snapshot = fresh
                    updates["market_snapshot"] = fresh
                else:
                    intel_snapshot = fresh
                    updates["intel_snapshot"] = fresh
            else:
                failures.append(str(result["attempt"]["reason"]))

    final_quality = build_data_quality_report(market_snapshot, intel_snapshot)
    blocked = _quality_blocked(final_quality) or bool(failures)
    reason = _blocked_reason(final_quality, failures) if blocked else "数据预检通过"
    # The timestamp is frozen only after all permitted preflight refetches.
    # Downstream research and replay can therefore distinguish the analysis
    # snapshot from a provider's later collection time.
    frozen = analysis_clock() if analysis_clock is not None else datetime.now(timezone.utc)
    if frozen.tzinfo is None or frozen.utcoffset() is None:
        raise ValueError("analysis clock must return a timezone-aware datetime")
    frozen_at = frozen.isoformat()
    report = {
        "status": "blocked" if blocked else "ready",
        "reason": reason,
        "refetch_enabled": effective_refetch_enabled,
        "requested_refetches": requested,
        "refetch_attempts": attempts,
        "initial_quality": initial_quality.model_dump(mode="json"),
        "final_quality": final_quality.model_dump(mode="json"),
        "effective_snapshot_version": _snapshot_version(market_snapshot, intel_snapshot),
        "analysis_at": frozen_at,
    }
    updates["preflight_report"] = report
    updates["analysis_at"] = frozen_at
    if tool_events:
        updates["tool_events"] = tool_events
    if blocked:
        updates["final_advice"] = _blocked_advice(
            symbol=str(state.get("symbol") or "").upper(),
            reason=reason,
            quality=final_quality,
        )
    return updates


def preflight_allows_analysis(state: AgentState) -> bool:
    """只有明确 ready 的预检允许分析;缺失、未知或 blocked 状态均拒绝。"""
    report = state.get("preflight_report") or {}
    return report.get("status") == "ready"


def _resolve_tool_context(provider: ToolContextProvider) -> dict[str, Any]:
    if provider is None:
        return {}
    if callable(provider):
        try:
            return dict(provider() or {})
        except Exception:
            logger.warning("preflight tool_context provider 失败", exc_info=True)
            return {}
    return dict(provider)


def _refetch_capabilities(context: dict[str, Any], market: str) -> dict[str, dict[str, Any] | None]:
    client = (context.get("rest_clients") or {}).get(market)
    market_context: dict[str, Any] | None = None
    if client is not None:
        market_context = {"rest_client": client}
        if context.get("build_market_snapshot") is not None:
            market_context["build_market_snapshot"] = context["build_market_snapshot"]
    aggregator = context.get("aggregator")
    intel_context = {"aggregator": aggregator} if aggregator is not None else None
    return {"market": market_context, "intel": intel_context}


def _refetch_targets(report: DataQualityReport) -> list[str]:
    targets: list[str] = []
    if report.missing_market_fields or report.market_error_count:
        targets.append("market")
    if (
        report.failed_sources
        or report.source_success_rate < 0.8
        or any(
            token in reason
            for reason in report.reasons
            for token in ("无资讯源状态", "资讯/情绪数据为空")
        )
    ):
        targets.append("intel")
    return targets


async def _refetch_once(
    *,
    registry: ToolRegistry,
    kind: str,
    symbol: str,
    market: str,
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    tool = "market_snapshot" if kind == "market" else "intel_snapshot"
    if context is None:
        return {
            "attempt": {
                "kind": kind,
                "tool": tool,
                "ok": False,
                "reason": f"{kind} refetch dependency unavailable",
            },
            "tool_event": None,
            "snapshot": None,
        }
    payload = {"symbol": symbol, "market": market} if kind == "market" else {"symbol": symbol}
    try:
        raw_result = await registry.invoke(tool, payload, context=context)
    except Exception as exc:
        return {
            "attempt": {
                "kind": kind,
                "tool": tool,
                "ok": False,
                "reason": f"{tool} tool failed: {str(exc)[:200]}",
            },
            "tool_event": {
                "tool": tool,
                "ok": False,
                "input": payload,
                "output": None,
                "errors": [{"category": "execution", "message": str(exc)[:200]}],
            },
            "snapshot": None,
        }

    event = _tool_event(raw_result)
    output = event.get("output") if isinstance(event.get("output"), dict) else {}
    key = "market_snapshot" if kind == "market" else "intel_snapshot"
    fresh = output.get(key) if isinstance(output, dict) else None
    ok = bool(event.get("ok")) and isinstance(fresh, dict) and bool(fresh)
    reason = "refetch succeeded" if ok else _tool_failure_reason(event, tool)
    return {
        "attempt": {"kind": kind, "tool": tool, "ok": ok, "reason": reason},
        "tool_event": event,
        "snapshot": fresh if ok else None,
    }


def _tool_event(result: Any) -> dict[str, Any]:
    if isinstance(result, ToolInvocationResult):
        return result.model_dump(mode="json")
    if hasattr(result, "model_dump"):
        dumped = result.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    return dict(result) if isinstance(result, dict) else {}


def _tool_failure_reason(event: dict[str, Any], tool: str) -> str:
    errors = event.get("errors") or []
    if errors and isinstance(errors[0], dict) and errors[0].get("message"):
        return f"{tool} tool failed: {str(errors[0]['message'])[:200]}"
    return f"{tool} tool failed or returned an empty snapshot"


def _quality_blocked(report: DataQualityReport) -> bool:
    return report.trade_blocked or report.level == "critical"


def _blocked_reason(report: DataQualityReport, failures: list[str]) -> str:
    details = [*failures]
    if _quality_blocked(report):
        details.extend(report.reasons or ["关键行情数据缺失"])
    return "数据预检阻断:" + "; ".join(_dedup(details))


def _blocked_advice(*, symbol: str, reason: str, quality: DataQualityReport) -> StrategyAdvice:
    warnings = [f"[数据预检] {item}" for item in quality.reasons]
    return StrategyAdvice(
        symbol=symbol,
        direction="neutral",
        decision_type="no_trade",
        confidence=min(0.25, max(0.0, quality.score)),
        rationale="数据预检未通过,未启动任何 LLM 分析节点。",
        risk_warnings=warnings,
        no_trade_reason=reason,
        reentry_conditions=["等待关键行情与资讯源恢复后重新评估"],
    )


def _snapshot_version(market_snapshot: dict[str, Any], intel_snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"market_snapshot": market_snapshot, "intel_snapshot": intel_snapshot},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()[:16]


def _dedup(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))
