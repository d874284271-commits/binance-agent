"""分析师节点:消费 state 快照 → 调用 LLM → 解析为结构化 AnalystReport。

节点依赖 BaseChatModel(而非 registry),便于图构建时按 agent 绑定模型、测试时注入假模型。
LLM 输出按 JSON 解析,失败则降级为 neutral 报告(不阻断流程)。
"""

from __future__ import annotations

import html
import json
import logging
import math
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..indicators import ta
from . import prompts
from .confidence import calibrate_confidence
from .context_views import narrative_envelope, news_evidence_view
from .output_validation import analyst_output, debate_output, planner_output, strategy_output
from .preflight import _snapshot_version
from .referee import ground_debate, new_verified_sources_in_round
from .runtime import AgentRuntime, RuntimeResult
from .state import AgentState, AnalystReport, DebateClaim, DebateTurn, StrategyAdvice
from .tools import ToolInvocationResult, ToolRegistry, build_default_tool_registry
from .validation import (
    DataQualityReport,
    apply_quality_and_risk_gates,
    build_data_quality_report,
    evaluate_risk_gate,
    validate_advice,
)

logger = logging.getLogger(__name__)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]*>")
_STANCES = {"bullish", "bearish", "neutral"}
_DIRECTIONS = {"long", "short", "neutral"}
# intel/providers/binance_deriv.py 写入 sentiment_features 的这些键与 funding 分析师同源
# (Binance fapi);情绪分析师不再消费,避免同一数据被两位分析师重复计票。
_MARKET_DERIVED_SENTIMENT_KEYS = frozenset({"funding_rate", "long_short_ratio", "mark_price"})
_FORBIDDEN_PLANNER_FIELDS = {
    "direction",
    "entry",
    "entry_price",
    "entry_zone",
    "leverage",
    "order",
    "position",
    "position_size",
    "price",
    "side",
    "stop",
    "stop_loss",
    "take_profit",
    "target",
    "targets",
}


class StrategyPlannerAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "no_trade",
        "watchlist",
        "continue_to_synth",
        "rerun_quality_check",
    ]
    reason: str = ""
    reentry_conditions: list[str] = Field(default_factory=list)
    confidence_cap: float | None = None


def _strip_think(text: str) -> str:
    """剥离 thinking 模型的 <think>...</think> 推理块,只留最终答案。"""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # 残留未配对的收尾标签 → 取其后内容
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        raw = content
    elif isinstance(content, list):
        raw = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    else:
        raw = str(content)
    return _strip_think(raw)


def _report_from_data(data: dict[str, Any], analyst: str) -> AnalystReport:
    stance = data.get("stance", "neutral")
    if not isinstance(stance, str) or stance not in _STANCES:
        stance = "neutral"
    try:
        confidence = float(data.get("confidence", 0.5))
    except (ValueError, TypeError, OverflowError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    key_points = [str(p) for p in (data.get("key_points") or [])][:8]
    return AnalystReport(
        analyst=analyst,
        stance=cast(Literal["bullish", "bearish", "neutral"], stance),
        confidence=confidence,
        summary=str(data.get("summary", ""))[:1000],
        key_points=key_points,
    )


def _parse_claims(raw_claims: Any) -> list[DebateClaim]:
    if not isinstance(raw_claims, list):
        return []

    claims: list[DebateClaim] = []
    for raw in raw_claims:
        if len(claims) >= 5:
            break
        if not isinstance(raw, dict):
            continue

        statement = str(raw.get("statement") or "").strip()[:300]
        evidence = str(raw.get("evidence") or "").strip()[:500]
        if not statement or not evidence:
            continue

        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(confidence):
            continue

        source_value = raw.get("source_value")
        if source_value is not None and not isinstance(source_value, (str, int, float, bool)):
            source_value = None

        claims.append(
            DebateClaim(
                statement=statement,
                evidence=evidence,
                source_path=str(raw.get("source_path") or "").strip()[:200] or None,
                source_value=source_value,
                # 旧模型/历史 fixture 的 source_ref 仅用于读取迁移;新 prompt 不再生成。
                source_ref=str(raw.get("source_ref") or "").strip()[:120] or None,
                confidence=max(0.0, min(1.0, confidence)),
            )
        )
    return claims


def _runtime_event(agent: str, result: RuntimeResult) -> dict[str, Any]:
    return {
        "agent": agent,
        "ok": result.ok,
        "latency_ms": result.latency_ms,
        "attempts": result.attempts,
        "repaired": result.repaired,
        "model_index": result.model_index,
        "endpoint": result.endpoint,
        "token_usage": result.token_usage,
        "error_category": result.error_category,
        "errors": [error.model_dump(mode="json") for error in result.errors],
        "context_usage": result.context_usage,
    }


def _runtime_kwargs(
    runtime_cfg: dict[str, Any] | None,
    state: AgentState | None = None,
) -> tuple[list[BaseChatModel] | None, dict[str, Any]]:
    cfg = dict(runtime_cfg or {})
    fallbacks = cfg.pop("fallbacks", None)
    if state is not None:
        cfg["context_cache"] = state.get("context_compactions", {})
    return fallbacks, cfg


async def _run_analyst(
    llm: BaseChatModel,
    system_prompt: str,
    payload: dict[str, Any],
    analyst: str,
    runtime_cfg: dict[str, Any] | None = None,
    *,
    data_sources: list[str],
) -> tuple[AnalystReport, dict[str, Any]]:
    fallbacks, kwargs = _runtime_kwargs(runtime_cfg)
    result = await AgentRuntime(llm, fallbacks=fallbacks, **kwargs).invoke_structured(
        analyst,
        system_prompt,
        payload,
        output_validator=analyst_output,
        fallback_data={
            "stance": "neutral",
            "confidence": 0.2,
            "summary": "LLM 调用失败或输出不可解析,降级为 neutral",
            "key_points": [],
        },
    )
    report = _report_from_data(result.data, analyst)
    report.data_sources = data_sources
    return report, _runtime_event(analyst, result)


def _kline_arrays(rows: list[dict[str, Any]]) -> tuple[list[float], list[float], list[float]]:
    """从已通过上游校验的 K 线提取等长 high/low/close 数组,不补齐或跳过坏行。"""
    highs = [float(r["high"]) for r in rows]
    lows = [float(r["low"]) for r in rows]
    closes = [float(r["close"]) for r in rows]
    return highs, lows, closes


async def technical_analyst(
    state: AgentState, llm: BaseChatModel, runtime_cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """从 canonical 行情计算多周期指标并调用技术分析模型,返回报告与 runtime 事件。

    每周期少于 30 根 K 线时不计算该周期,不额外补抓;模型失败由统一分析师
    调用层降级,数据来源由代码填写而非模型自报。
    """
    snapshot = state["market_snapshot"]
    klines = snapshot.get("klines", {})
    summaries: dict[str, Any] = {}
    # 多周期综合走势:遍历快照全部周期(5m/1h/4h/1d),各计指标交 LLM 综合
    for interval in ("5m", "1h", "4h", "1d"):
        rows = klines.get(interval) or []
        if len(rows) >= 30:
            highs, lows, closes = _kline_arrays(rows)
            summaries[interval] = ta.summarize(highs, lows, closes)
    payload = {
        "symbol": state["symbol"],
        "indicators": summaries,
        "ticker_24h": snapshot.get("ticker_24h", {}),
    }
    report, runtime_event = await _run_analyst(
        llm,
        prompts.TECHNICAL,
        payload,
        "technical",
        runtime_cfg,
        data_sources=["market.klines", "market.ticker_24h"],
    )
    return {"analyst_reports": [report], "runtime_events": [runtime_event]}


async def funding_analyst(
    state: AgentState, llm: BaseChatModel, runtime_cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """汇总合约资金费率、未平仓量变化和多空比,调用资金面模型并返回 trace。

    现货没有衍生品数据时直接返回中性报告及 skipped 事件,不消耗模型请求。
    未平仓量变化使用百分数,原始资金费率保留比率单位。
    """
    if state.get("market") == "api":  # 现货无资金费率/持仓量/多空比 → 跳过,不调 LLM
        return {
            "analyst_reports": [
                AnalystReport(
                    analyst="funding",
                    stance="neutral",
                    confidence=0.3,
                    summary="现货无衍生品资金面数据,跳过资金面分析",
                )
            ],
            # 与 _runtime_event 结构对齐:跳过也记一条事件,置信度校准/trace 不漏该 agent
            "runtime_events": [
                {
                    "agent": "funding",
                    "ok": True,
                    "latency_ms": 0.0,
                    "attempts": 0,
                    "repaired": False,
                    "model_index": 0,
                    "error_category": None,
                    "errors": [],
                    "skipped": "spot_market_no_derivatives_data",
                }
            ],
        }
    snapshot = state["market_snapshot"]
    oi_hist = snapshot.get("open_interest_hist") or []
    oi_change_pct: float | None = None
    if len(oi_hist) >= 2:
        try:
            first = float(oi_hist[0].get("sumOpenInterest"))
            last = float(oi_hist[-1].get("sumOpenInterest"))
            if first:
                oi_change_pct = round((last - first) / first * 100, 2)
        except (TypeError, ValueError):
            oi_change_pct = None
    lsr = snapshot.get("long_short_ratio") or []
    payload = {
        "symbol": state["symbol"],
        "funding": snapshot.get("funding", {}),
        "open_interest": snapshot.get("open_interest"),
        "open_interest_change_pct": oi_change_pct,
        "long_short_ratio_latest": (lsr[-1] if lsr else None),
        "long_short_ratio_series": [r.get("long_short_ratio") for r in lsr],
    }
    report, runtime_event = await _run_analyst(
        llm,
        prompts.FUNDING,
        payload,
        "funding",
        runtime_cfg,
        data_sources=["market.funding", "market.open_interest", "market.long_short_ratio"],
    )
    return {"analyst_reports": [report], "runtime_events": [runtime_event]}


async def news_sentiment_analyst(
    state: AgentState, llm: BaseChatModel, runtime_cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """在未启用正文研究图时用现有情绪特征与至多 20 条不可信标题生成报告。

    剔除与行情重复的情绪特征,不抓正文,也不把标题线索标成已核验新闻事实。
    """
    intel = state["intel_snapshot"]
    news = intel.get("news_items") or []
    features = {
        key: value
        for key, value in (intel.get("sentiment_features") or {}).items()
        if key not in _MARKET_DERIVED_SENTIMENT_KEYS
    }
    payload = {
        "symbol": state["symbol"],
        "sentiment_features": features,
        "macro_events_count": len(intel.get("macro_events") or []),
        "untrusted_headlines": [
            {
                "title": _sanitize_untrusted_text(n.get("title"), max_chars=300),
                "sentiment": _sanitize_untrusted_text(n.get("sentiment"), max_chars=40),
                "source": _sanitize_untrusted_text(n.get("source"), max_chars=80),
                "source_id": _sanitize_untrusted_text(n.get("source_id"), max_chars=80),
                "source_kind": _sanitize_untrusted_text(n.get("source_kind"), max_chars=30),
                "source_refs": n.get("source_refs") or [],
            }
            for n in news[:20]
        ],
    }
    report, runtime_event = await _run_analyst(
        llm,
        prompts.NEWS_SENTIMENT,
        payload,
        "news_sentiment",
        runtime_cfg,
        data_sources=[f"intel.{key}" for key in sorted(features)]
        + (["intel.headlines"] if news else []),
    )
    return {"analyst_reports": [report], "runtime_events": [runtime_event]}


def _sanitize_untrusted_text(value: Any, *, max_chars: int) -> str:
    """解码实体、去标签和控制字符后折叠空白,按字符数限制进入模型的外部短文本。

    这是提示输入清理,不是 HTML 输出转义,也不能保证模型抵抗提示注入。
    """
    text = html.unescape(str(value or ""))
    text = _HTML_TAG_RE.sub(" ", text)
    text = "".join(" " if unicodedata.category(char).startswith("C") else char for char in text)
    return " ".join(text.split())[:max_chars]


def _parse_event_time(value: Any) -> datetime | None:
    """解析事件时间并统一补足 UTC 时区;无法解析返回 None,不推测事件发生时间。"""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def _parse_snapshot_time(value: Any) -> datetime | None:
    return _parse_event_time(value)


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _evaluate_snapshot_freshness(
    *,
    enabled: bool,
    snapshot_time: datetime | None,
    now: datetime,
    snapshot_price: float | None,
    latest_price: float | None,
    staleness_threshold_seconds: int = 300,
    price_deviation_threshold: float = 0.005,
) -> dict[str, Any]:
    if not enabled:
        return {"stale": False, "action": "proceed", "reason": ""}
    if snapshot_time is None:
        return {"stale": True, "action": "no_trade", "reason": "快照缺失时间戳"}
    if snapshot_time.tzinfo is None:
        snapshot_time = snapshot_time.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    elapsed = (now - snapshot_time).total_seconds()
    if elapsed > staleness_threshold_seconds:
        return {
            "stale": True,
            "action": "no_trade",
            "reason": f"快照已过期 {int(elapsed)}秒(>{staleness_threshold_seconds}秒)",
        }

    if snapshot_price and latest_price:
        deviation = abs(latest_price - snapshot_price) / snapshot_price
        if deviation > price_deviation_threshold:
            return {
                "stale": True,
                "action": "watchlist",
                "reason": f"最新价相对快照价偏离 {deviation:.2%}(>{price_deviation_threshold:.2%})",
            }

    return {"stale": False, "action": "proceed", "reason": ""}


def _debate_payload(state: AgentState) -> dict[str, Any]:
    snapshot = state["market_snapshot"]
    klines = snapshot.get("klines") or {}
    technicals: dict[str, Any] = {}
    for interval in ("1h", "4h"):
        rows = klines.get(interval) or []
        if len(rows) < 30:
            continue
        try:
            highs, lows, closes = _kline_arrays(rows)
        except (KeyError, TypeError, ValueError):
            continue
        technicals[interval] = ta.summarize(highs, lows, closes)

    # 上一轮双方 turn 全文(argument + claims):让辩手能针对对手论点交锋;
    # 只传上一轮而非全部历史,控制上下文膨胀
    round_no = state.get("debate_round", 0)
    completed_turns = [t for t in state.get("debate_turns") or [] if t.round < round_no]
    prior_round_turns = [
        turn.model_dump(mode="json")
        for turn in _sorted_debate_turns([t for t in completed_turns if t.round == round_no - 1])
    ]
    # 全部已完成轮次引用过的路径:后续轮须引入其中没有的新证据,否则判定辩论收敛
    previously_cited_paths = sorted(
        {c.source_path for t in completed_turns for c in t.structured_claims if c.source_path}
    )

    return {
        "analyst_reports": _analyst_reports_namespace(state["analyst_reports"]),
        "news_evidence": news_evidence_view(state.get("news_research", {})),
        "prior_round_turns": prior_round_turns,
        "previously_cited_paths": previously_cited_paths,
        "snapshot_data": {
            "ticker_24h": snapshot.get("ticker_24h", {}),
            "funding": snapshot.get("funding", {}),
            "open_interest": snapshot.get("open_interest"),
            "open_interest_hist_tail": (snapshot.get("open_interest_hist") or [])[-3:],
            "long_short_ratio_tail": [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in (snapshot.get("long_short_ratio") or [])[-3:]
            ],
            "technicals": technicals,
            "fetched_at": snapshot.get("fetched_at"),
            "errors": snapshot.get("errors") or [],
        },
    }


def _analyst_reports_namespace(reports: list[AnalystReport]) -> dict[str, dict[str, Any]]:
    # data_sources 是系统元数据而非市场证据:排除后辩手无法引用它凑出「已核验数据点」
    return {
        report.analyst: report.model_dump(mode="json", exclude={"data_sources"})
        for report in reports
    }


async def bull_debater(
    state: AgentState, llm: BaseChatModel, runtime_cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """基于同一 canonical 快照生成本轮多方论据及结构化引用,不发起补抓。

    模型失败返回空引用的降级论据;论点仍需后续 referee 核验,不能因生成成功
    就视为独立证据。
    """
    round_no = state["debate_round"]
    fallbacks, kwargs = _runtime_kwargs(runtime_cfg, state)
    result = await AgentRuntime(llm, fallbacks=fallbacks, **kwargs).invoke_structured(
        "bull_debater",
        prompts.BULL_STRUCTURED,
        narrative_envelope(_debate_payload(state)),
        output_validator=debate_output,
        fallback_data={
            "argument": "LLM 调用失败,多方辩论降级为空论据",
            "claims": [],
        },
    )
    argument = str(result.data.get("argument") or result.text or "")
    claims = _parse_claims(result.data.get("claims"))
    return {
        "debate_turns": [
            DebateTurn(
                side="bull",
                round=round_no,
                argument=argument,
                structured_claims=claims,
                snapshot_ref_at=_parse_snapshot_time(state["market_snapshot"].get("fetched_at")),
            )
        ],
        "runtime_events": [_runtime_event("bull_debater", result)],
        "context_compactions": result.context_updates,
    }


async def bear_debater(
    state: AgentState, llm: BaseChatModel, runtime_cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """基于同一 canonical 快照生成本轮空方论据及结构化引用,不发起补抓。

    模型失败返回空引用的降级论据,并保存 runtime 与上下文压缩记录供回放。
    """
    round_no = state["debate_round"]
    fallbacks, kwargs = _runtime_kwargs(runtime_cfg, state)
    result = await AgentRuntime(llm, fallbacks=fallbacks, **kwargs).invoke_structured(
        "bear_debater",
        prompts.BEAR_STRUCTURED,
        narrative_envelope(_debate_payload(state)),
        output_validator=debate_output,
        fallback_data={
            "argument": "LLM 调用失败,空方辩论降级为空论据",
            "claims": [],
        },
    )
    argument = str(result.data.get("argument") or result.text or "")
    claims = _parse_claims(result.data.get("claims"))
    return {
        "debate_turns": [
            DebateTurn(
                side="bear",
                round=round_no,
                argument=argument,
                structured_claims=claims,
                snapshot_ref_at=_parse_snapshot_time(state["market_snapshot"].get("fetched_at")),
            )
        ],
        "runtime_events": [_runtime_event("bear_debater", result)],
        "context_compactions": result.context_updates,
    }


def debate_controller(state: AgentState) -> dict[str, Any]:
    """推进轮次并判定收敛:达上限,或一轮反驳后上一轮没有引入任何新的已核验数据点。"""
    rounds_done = state["debate_round"] + 1
    update: dict[str, Any] = {"debate_round": rounds_done}
    if rounds_done >= state["max_debate_rounds"]:
        update["debate_stop_reason"] = "max_rounds"
    elif rounds_done >= 2 and not new_verified_sources_in_round(
        state["debate_turns"], _referee_namespace(state), rounds_done - 1
    ):
        update["debate_stop_reason"] = "no_new_evidence"
    return update


def _sorted_debate_turns(turns: list[DebateTurn]) -> list[DebateTurn]:
    side_order = {"bull": 0, "bear": 1}
    return sorted(turns, key=lambda turn: (turn.round, side_order.get(turn.side, 99)))


def _referee_namespace(state: AgentState) -> dict[str, Any]:
    """复用 _debate_payload 的 snapshot_data,确保核验的正是辩手所见,不产生漂移。"""
    return {
        "snapshot_data": _debate_payload(state)["snapshot_data"],
        "analyst_reports": _analyst_reports_namespace(state["analyst_reports"]),
    }


def debate_referee(state: AgentState) -> dict[str, Any]:
    """辩论收敛后核验 claim 的结构化标量引用,产出证据完整性报告。

    纯函数、无 LLM(模板同 risk_manager);只写独立字段,**不返回 debate_turns**——该字段是
    operator.add 累加 reducer,返回会把已有轮次重复追加。
    """
    turns = state["debate_turns"]
    grounding = ground_debate(turns, _referee_namespace(state))
    grounding["snapshot_version"] = _snapshot_version(
        state["market_snapshot"], state["intel_snapshot"]
    )
    grounding["convergence"] = {
        "rounds_run": len({turn.round for turn in turns}),
        "max_rounds": state["max_debate_rounds"],
        "stop_reason": (state.get("debate_stop_reason") or "max_rounds") if turns else "not_run",
    }
    return {"debate_grounding": grounding}


def _grounded_debate_context(state: AgentState) -> dict[str, Any]:
    """Rebuild positional claim verdicts against the saved canonical namespace.

    Old checkpoints may have truncated statements and no claim IDs; never join
    them by text or source path. Missing canonical inputs stay explicitly unknown.
    """
    turns = _sorted_debate_turns(state["debate_turns"])
    try:
        grounding = ground_debate(turns, _referee_namespace(state))
    except (KeyError, TypeError, ValueError):
        grounding = {
            "claims": [
                {
                    "claim_id": f"{turn.side}:{turn.round}:{index}",
                    "claim_index": index,
                    "side": turn.side,
                    "round": turn.round,
                    "statement": claim.statement,
                    "source_path": claim.source_path,
                    "source_value": claim.source_value,
                    "observed": None,
                    "verdict": "unknown",
                }
                for turn in turns
                for index, claim in enumerate(turn.structured_claims)
            ]
        }
    version = (
        _snapshot_version(state["market_snapshot"], state["intel_snapshot"])
        if isinstance(state.get("market_snapshot"), dict)
        and isinstance(state.get("intel_snapshot"), dict)
        else None
    )
    for claim in grounding["claims"]:
        claim["snapshot_version"] = version
    return {
        "debate_turns": [
            {
                "side": turn.side,
                "round": turn.round,
                "argument": turn.argument,
                "claim_ids": [
                    f"{turn.side}:{turn.round}:{i}" for i in range(len(turn.structured_claims))
                ],
                "snapshot_ref_at": turn.snapshot_ref_at,
            }
            for turn in turns
        ],
        "claim_verdicts": grounding["claims"],
        "evidence_verified_ratio": grounding.get("verified_ratio", 0.0),
        "evidence_missing": grounding.get("missing_count", 0),
        "evidence_mismatch": grounding.get("mismatch_count", 0),
        "evidence_invalid": grounding.get("invalid_count", 0),
        "evidence_by_side": grounding.get("by_side", {}),
        "shared_sources": grounding.get("shared_sources", []),
        "snapshot_version": version,
    }


async def research_manager(
    state: AgentState, llm: BaseChatModel, runtime_cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """结合分析报告、新闻证据视图和已核验辩论结果生成研究总结。

    总结属于可压缩模型论述,不改变原始证据;调用失败保留观望文本及失败 trace。
    """
    payload = {
        "analyst_reports": [r.model_dump(mode="json") for r in state["analyst_reports"]],
        "news_evidence": news_evidence_view(state.get("news_research", {})),
        **_grounded_debate_context(state),
    }
    fallbacks, kwargs = _runtime_kwargs(runtime_cfg, state)
    result = await AgentRuntime(llm, fallbacks=fallbacks, **kwargs).invoke_text(
        "research_manager",
        prompts.RESEARCH_MANAGER,
        narrative_envelope(payload),
        fallback_text="LLM 调用失败,研究结论降级:证据不足,倾向观望。",
    )
    return {
        "research_summary": result.text,
        "runtime_events": [_runtime_event("research_manager", result)],
        "context_compactions": result.context_updates,
    }


def _quality_blocked(quality_report: dict[str, Any]) -> bool:
    return bool(quality_report.get("trade_blocked")) or (
        str(quality_report.get("level") or "") == "critical"
    )


async def strategy_planner(
    state: AgentState,
    *,
    tool_registry: ToolRegistry | None = None,
    max_steps: int = 3,
    llm: BaseChatModel | None = None,
    enable_llm_actions: bool = False,
    planner_mode: str = "deterministic",
    planner_enforced: bool = True,
    runtime_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在研究和最终风控之间执行有限规划,记录步骤但不生成最终交易价位。

    步数限制在 1..8;可建议 no_trade/watchlist 或继续合成,shadow 是否影响
    最终结果由 planner_enforced 表示。补抓仅属于 data_preflight,本节点不改快照。
    """

    registry = tool_registry or build_default_tool_registry()
    max_steps = max(1, min(8, max_steps))
    steps: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []

    def done(result: dict[str, Any], action_status: str | None = None) -> dict[str, Any]:
        return _annotate_planner_result(
            result,
            mode=planner_mode,
            enforced=planner_enforced,
            action_status=action_status,
        )

    quality = await _planner_tool_call(
        registry,
        "data_quality_check",
        {
            "market_snapshot": state["market_snapshot"],
            "intel_snapshot": state["intel_snapshot"],
        },
        step_no=1,
        thought="检查行情与资讯数据质量,决定是否允许继续策略规划",
        steps=steps,
        tool_events=tool_events,
    )
    quality_report = (quality.get("output") or {}).get("data_quality_report") or {}
    if _quality_blocked(quality_report):
        reason = "数据质量不足:" + "; ".join(quality_report.get("reasons") or ["关键数据缺失"])
        steps[-1]["decision"] = "no_trade"
        return done(
            {
                "planner_steps": steps,
                "planner_decision": {
                    "decision_type": "no_trade",
                    "reason": reason,
                    "confidence_cap": 0.25,
                    "reentry_conditions": ["等待关键行情 K 线、ticker 与资讯源状态恢复后重新评估"],
                },
                "tool_events": tool_events,
            }
        )
    if not quality.get("ok", False):
        reason = _tool_error_reason(quality)
        steps[-1]["decision"] = "no_trade"
        return done(
            {
                "planner_steps": steps,
                "planner_decision": {
                    "decision_type": "no_trade",
                    "reason": reason,
                    "confidence_cap": 0.25,
                    "reentry_conditions": ["等待 planner 工具恢复后重新评估"],
                },
                "tool_events": tool_events,
            },
            action_status="fallback",
        )

    if len(steps) >= max_steps:
        steps[-1]["decision"] = "watchlist"
        return done(_planner_max_steps_result(steps, tool_events), action_status="fallback")

    indicators = await _planner_tool_call(
        registry,
        "technical_indicators",
        {
            "market_snapshot": state["market_snapshot"],
            "intervals": ["1h", "4h"],
        },
        step_no=len(steps) + 1,
        thought="检查 1h/4h 技术指标是否足够支撑继续策略合成",
        steps=steps,
        tool_events=tool_events,
    )
    if not indicators.get("ok", False):
        reason = _tool_error_reason(indicators)
        steps[-1]["decision"] = "no_trade"
        return done(
            {
                "planner_steps": steps,
                "planner_decision": {
                    "decision_type": "no_trade",
                    "reason": reason,
                    "confidence_cap": 0.25,
                    "reentry_conditions": ["等待 technical_indicators 工具恢复后重新评估"],
                },
                "tool_events": tool_events,
            },
            action_status="fallback",
        )

    missing = (indicators.get("output") or {}).get("missing_intervals") or []
    if missing:
        steps[-1]["decision"] = "watchlist"
        return done(
            {
                "planner_steps": steps,
                "planner_decision": {
                    "decision_type": "watchlist",
                    "reason": "关键技术指标缺失:" + ", ".join(str(item) for item in missing),
                    "confidence_cap": 0.45,
                    "reentry_conditions": ["等待 1h/4h K 线足够计算关键指标后重新评估"],
                },
                "tool_events": tool_events,
            }
        )

    if enable_llm_actions:
        if len(steps) >= max_steps:
            steps[-1]["decision"] = "watchlist"
            return done(_planner_max_steps_result(steps, tool_events), action_status="fallback")
        if llm is None:
            steps[-1]["decision"] = "watchlist"
            return done(
                {
                    "planner_steps": steps,
                    "planner_decision": {
                        "decision_type": "watchlist",
                        "reason": "strategy_planner LLM 未配置,降级观望",
                        "confidence_cap": 0.35,
                        "reentry_conditions": ["配置 strategy_planner LLM 后重新评估"],
                        "source": "llm_action_planner",
                    },
                    "tool_events": tool_events,
                },
                action_status="fallback",
            )
        result = await _run_strategy_action_planner(
            state,
            llm=llm,
            registry=registry,
            max_steps=max_steps,
            steps=steps,
            tool_events=tool_events,
            runtime_cfg=runtime_cfg,
        )
        return done(result)

    steps[-1]["decision"] = "proceed"
    return done(
        {
            "planner_steps": steps,
            "planner_decision": {
                "decision_type": "proceed",
                "reason": "数据质量与关键指标足够,允许进入策略合成",
                "confidence_cap": None,
                "reentry_conditions": [],
            },
            "tool_events": tool_events,
        }
    )


def risk_manager(state: AgentState) -> dict[str, Any]:
    """以确定性规则检查当前 UTC 未来 24 小时宏观事件、资金费率和账户拥挤度。

    不调用 LLM;宏观事件格式或重要性未知时保守置为高风险,返回风险等级、
    警告及数据质量报告,具体建议降级由合成阶段的硬闸门执行。
    """
    intel = state["intel_snapshot"]
    market = state["market_snapshot"]
    warnings: list[str] = []
    now = datetime.now(timezone.utc)

    for event in intel.get("macro_events") or []:
        if not isinstance(event, dict):
            warnings.append("宏观事件风险未知:事件数据格式无效,禁止开新单")
            continue
        event_time = _parse_event_time(event.get("time"))
        importance = event.get("importance")
        if (
            isinstance(importance, bool)
            or not isinstance(importance, int)
            or not 1 <= importance <= 3
        ):
            warnings.append("宏观事件风险未知:importance 无效,禁止开新单")
            continue
        if event_time and importance >= 3 and now <= event_time <= now + timedelta(hours=24):
            warnings.append(
                f"临近高重要性宏观事件:{event.get('name')} @ {event.get('time')},"
                "事件驱动波动风险高,建议降杠杆/观望"
            )

    funding = market.get("funding") or {}
    funding_rate = funding.get("last_funding_rate")
    if funding_rate is None:
        funding_rate = funding.get("funding_rate")
    try:
        if funding_rate is not None and abs(float(funding_rate)) >= 0.0005:
            warnings.append("资金费率偏高,杠杆多头拥挤,警惕回调")
    except (TypeError, ValueError):
        pass

    features = intel.get("sentiment_features") or {}
    long_short = features.get("long_short_ratio")
    try:
        if long_short is not None:
            ratio = float(long_short)
            if ratio >= 3.0:
                warnings.append("多空账户比极端偏多,情绪过热,注意反向风险")
            elif 0 < ratio <= 1 / 3.0:
                warnings.append("多空账户比极端偏空,空头拥挤,警惕轧空反弹")
    except (TypeError, ValueError):
        pass

    if any("宏观事件" in w for w in warnings):
        level = "high"
    elif warnings:
        level = "medium"
    else:
        level = "low"
    data_quality = build_data_quality_report(market, intel)
    return {
        "risk_assessment": {
            "level": level,
            "warnings": warnings,
            "data_quality": data_quality.model_dump(mode="json"),
        }
    }


def _opt_str(value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    return str(value)


def _parse_advice(text: str, symbol: str, risk_assessment: dict[str, Any] | None) -> StrategyAdvice:
    risk_warnings = list((risk_assessment or {}).get("warnings") or [])
    match = _JSON_RE.search(text or "")
    if match:
        try:
            data = json.loads(match.group(0))
        except (ValueError, TypeError) as exc:
            logger.warning("策略合成 JSON 解析失败: %s", exc)
        else:
            direction = data.get("direction", "neutral")
            if not isinstance(direction, str) or direction not in _DIRECTIONS:
                direction = "neutral"
            decision_type = data.get("decision_type")
            if not isinstance(decision_type, str) or decision_type not in {
                "trade",
                "no_trade",
                "watchlist",
            }:
                decision_type = "no_trade" if direction == "neutral" else "trade"
            try:
                confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            except (ValueError, TypeError):
                confidence = 0.5
            return StrategyAdvice(
                symbol=symbol,
                direction=cast(Literal["long", "short", "neutral"], direction),
                decision_type=cast(Literal["trade", "no_trade", "watchlist"], decision_type),
                entry_zone=_opt_str(data.get("entry_zone")),
                stop_loss=_opt_str(data.get("stop_loss")),
                targets=[str(t) for t in (data.get("targets") or [])][:5],
                confidence=confidence,
                rationale=str(data.get("rationale", ""))[:1500],
                risk_warnings=risk_warnings,
                no_trade_reason=_opt_str(data.get("no_trade_reason")),
                reentry_conditions=[str(c) for c in (data.get("reentry_conditions") or [])][:5],
            )
    return StrategyAdvice(
        symbol=symbol,
        direction="neutral",
        decision_type="no_trade",
        confidence=0.3,
        rationale=(text or "合成失败")[:300],
        risk_warnings=risk_warnings,
        no_trade_reason="策略合成失败或输出不可解析",
        reentry_conditions=["等待模型输出可解析策略后重新评估"],
    )


async def strategy_synth(
    state: AgentState,
    llm: BaseChatModel,
    runtime_cfg: dict[str, Any] | None = None,
    apply_planner_decision: bool = True,
    confidence_bucket_caps: dict[str, float] | None = None,
    confidence_policy_version: str = "confidence-policy/v2",
    evidence_mode: bool = False,
) -> dict[str, Any]:
    """生成策略候选后依次应用数据质量、风险、规划、新鲜度及置信度规则。

    模型失败降为 no_trade;返回 runtime 和闸门记录。证据模式只输出
    candidate_advice,必须再经 final_evidence_gate 才能作为最终建议发布;
    规则 confidence 不代表 Outcome 胜率或收益概率。
    """
    snapshot = state["market_snapshot"]
    # 关键价位:供 LLM 基于 ATR/布林算入场/止损/目标(避免凭空编价位)
    key_levels: dict[str, Any] = {}
    rows = (snapshot.get("klines") or {}).get("1h") or []
    if len(rows) >= 30:
        highs, lows, closes = _kline_arrays(rows)
        s = ta.summarize(highs, lows, closes)
        key_levels = {
            k: s.get(k) for k in ("close", "atr14", "boll_upper", "boll_middle", "boll_lower")
        }
    payload = {
        "symbol": state["symbol"],
        "last_price": (snapshot.get("ticker_24h") or {}).get("lastPrice"),
        "key_levels": key_levels,
        "analyst_reports": [r.model_dump(mode="json") for r in state["analyst_reports"]],
        **_grounded_debate_context(state),
        "research_summary": state.get("research_summary"),
        "news_evidence": news_evidence_view(state.get("news_research", {})),
        "risk_assessment": state.get("risk_assessment"),
        "planner_decision": state.get("planner_decision"),
    }
    fallbacks, kwargs = _runtime_kwargs(runtime_cfg, state)
    result = await AgentRuntime(llm, fallbacks=fallbacks, **kwargs).invoke_structured(
        "strategy_synth",
        prompts.STRATEGY_SYNTH,
        narrative_envelope(payload),
        output_validator=strategy_output,
        fallback_data={
            "direction": "neutral",
            "decision_type": "no_trade",
            "entry_zone": None,
            "stop_loss": None,
            "targets": [],
            "confidence": 0.2,
            "rationale": "策略合成 LLM 调用失败,降级为观望。",
            "no_trade_reason": "策略合成 LLM 调用失败",
            "reentry_conditions": ["等待模型服务恢复且输出可解析后重新评估"],
        },
    )
    advice = _parse_advice(
        json.dumps(result.data, ensure_ascii=False), state["symbol"], state.get("risk_assessment")
    )
    advice.model_confidence = advice.confidence
    checks = validate_advice(advice, state["market_snapshot"], state["analyst_reports"])
    if checks:
        advice.risk_warnings = [*advice.risk_warnings, *checks]
    risk_assessment = state.get("risk_assessment") or {}
    raw_quality = risk_assessment.get("data_quality")
    if raw_quality:
        data_quality = DataQualityReport.model_validate(raw_quality)
    else:
        data_quality = build_data_quality_report(state["market_snapshot"], state["intel_snapshot"])
    risk_gate = evaluate_risk_gate(
        advice, state["market_snapshot"], risk_assessment, state["analyst_reports"]
    )
    advice = apply_quality_and_risk_gates(
        advice,
        data_quality,
        risk_assessment,
        state["analyst_reports"],
        state["market_snapshot"],
        pre_computed_risk_gate=risk_gate,
    )
    if apply_planner_decision:
        advice = _apply_planner_decision(advice, state.get("planner_decision"))
    freshness = _evaluate_snapshot_freshness(
        enabled=bool(state.get("snapshot_freshness_enabled", False)),
        snapshot_time=_parse_snapshot_time(snapshot.get("fetched_at")),
        now=state.get("snapshot_freshness_now") or datetime.now(timezone.utc),
        snapshot_price=_as_float((snapshot.get("ticker_24h") or {}).get("lastPrice")),
        latest_price=_as_float(state.get("latest_price")),
    )
    if freshness["stale"]:
        reason = str(freshness["reason"])
        advice.risk_warnings = _dedup([*advice.risk_warnings, f"[staleness] {reason}"])
        advice.reentry_conditions = _dedup(
            [*advice.reentry_conditions, "等待新鲜行情快照后重新评估"]
        )
        if freshness["action"] == "no_trade":
            advice.direction = "neutral"
            advice.decision_type = "no_trade"
            advice.entry_zone = None
            advice.stop_loss = None
            advice.targets = []
            advice.no_trade_reason = reason
            advice.confidence = min(advice.confidence, 0.25)
        elif advice.decision_type != "no_trade":
            advice.decision_type = "watchlist"
            advice.confidence = min(advice.confidence, 0.35)
    runtime_event = _runtime_event("strategy_synth", result)
    calibration = calibrate_confidence(
        advice=advice,
        data_quality=data_quality,
        risk_gate=risk_gate,
        analyst_reports=state["analyst_reports"],
        runtime_events=[*state.get("runtime_events", []), runtime_event],
        risk_assessment=risk_assessment,
        debate_grounding=state.get("debate_grounding"),
        bucket_caps=confidence_bucket_caps,
        policy_version=confidence_policy_version,
    )
    advice.confidence = calibration.calibrated_confidence
    validation = (state.get("news_research") or {}).get("evidence_validation") or {}
    valid = validation.get("valid")
    invalid = validation.get("invalid")
    unresolved = validation.get("unresolved")
    denominator = sum(value for value in (valid, invalid, unresolved) if isinstance(value, int))
    advice.evidence_confidence = (
        valid / denominator if isinstance(valid, int) and denominator else None
    )
    advice.outcome_confidence = None
    advice.pre_policy_confidence = calibration.pre_policy_confidence
    advice.confidence_cap = calibration.confidence_cap
    advice.confidence_bucket = calibration.policy_bucket
    advice.confidence_policy_version = calibration.policy_version
    advice.confidence_reasons = calibration.confidence_reasons
    return {
        "candidate_advice" if evidence_mode else "final_advice": advice,
        "runtime_events": [runtime_event],
        "context_compactions": result.context_updates,
        "risk_gate_result": risk_gate.model_dump(mode="json"),
    }


async def _planner_tool_call(
    registry: ToolRegistry,
    name: str,
    payload: dict[str, Any],
    *,
    step_no: int,
    thought: str,
    steps: list[dict[str, Any]],
    tool_events: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        result = await registry.invoke(name, payload, context=context or {})
    except Exception as exc:
        event = {
            "tool": name,
            "ok": False,
            "latency_ms": 0.0,
            "input": payload,
            "output": None,
            "errors": [{"category": "execution", "message": str(exc)[:300]}],
        }
    else:
        event = _tool_result_payload(result)
    tool_events.append(event)
    steps.append(
        {
            "step": step_no,
            "thought": thought,
            "action": name,
            "observation": {
                "ok": bool(event.get("ok")),
                "output": event.get("output"),
                "errors": event.get("errors") or [],
            },
            "decision": "continue",
        }
    )
    return event


async def _run_strategy_action_planner(
    state: AgentState,
    *,
    llm: BaseChatModel,
    registry: ToolRegistry,
    max_steps: int,
    steps: list[dict[str, Any]],
    tool_events: list[dict[str, Any]],
    runtime_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    allowed_actions = ["no_trade", "watchlist", "continue_to_synth", "rerun_quality_check"]

    payload = {
        "symbol": state["symbol"],
        "market": state["market"],
        "research_summary": state.get("research_summary"),
        "analyst_reports": [r.model_dump() for r in state["analyst_reports"]],
        "risk_assessment": state.get("risk_assessment"),
        "planner_steps": steps,
        "allowed_actions": allowed_actions,
        "forbidden_outputs": sorted(_FORBIDDEN_PLANNER_FIELDS),
    }
    fallbacks, kwargs = _runtime_kwargs(runtime_cfg, state)
    result = await AgentRuntime(llm, fallbacks=fallbacks, **kwargs).invoke_structured(
        "strategy_planner",
        prompts.STRATEGY_ACTION_PLANNER,
        narrative_envelope(payload),
        output_validator=_validate_planner_output,
        fallback_data={
            "action": "watchlist",
            "reason": "LLM action planner 输出不可解析,降级观望",
            "reentry_conditions": ["等待 planner 输出合法 JSON action 后重新评估"],
            "confidence_cap": 0.35,
        },
    )
    runtime_event = _runtime_event("strategy_planner", result)
    action, violation = _coerce_planner_action(result.data)
    if result.error_category == "schema_invalid":
        violation = "非法 planner action: 输出未通过字段与权限校验"
    step = {
        "step": len(steps) + 1,
        "thought": "受限 LLM action planner 只判断流程去向,不生成交易价位",
        "action": "llm_action_planner",
        "observation": {
            "ok": result.ok and violation is None,
            "output": result.data,
            "errors": [error.model_dump(mode="json") for error in result.errors],
            "violation": violation,
        },
        "decision": "continue",
    }
    steps.append(step)

    if violation is not None:
        step["decision"] = "watchlist"
        return {
            "planner_steps": steps,
            "planner_decision": _planner_decision_payload(
                "watchlist",
                reason=violation,
                confidence_cap=0.35,
                reentry_conditions=["等待 planner 输出合法且不含交易价位的 action 后重新评估"],
            ),
            "tool_events": tool_events,
            "runtime_events": [runtime_event],
            "context_compactions": result.context_updates,
        }

    if not result.ok:
        step["decision"] = "watchlist"
        return {
            "planner_steps": steps,
            "planner_decision": _planner_decision_payload(
                "watchlist",
                reason=action.reason or "LLM action planner 失败,降级观望",
                confidence_cap=0.35,
                reentry_conditions=action.reentry_conditions
                or ["等待 planner 输出合法 JSON action 后重新评估"],
            ),
            "tool_events": tool_events,
            "runtime_events": [runtime_event],
            "context_compactions": result.context_updates,
        }

    if action.action == "rerun_quality_check":
        if len(steps) >= max_steps:
            step["decision"] = "watchlist"
            return {
                "planner_steps": steps,
                "planner_decision": _planner_decision_payload(
                    "watchlist",
                    reason="strategy_planner max_steps reached before rerun_quality_check",
                    confidence_cap=0.45,
                    reentry_conditions=["等待 planner 预算允许重新检查数据质量后再评估"],
                ),
                "tool_events": tool_events,
                "runtime_events": [runtime_event],
                "context_compactions": result.context_updates,
            }
        update = await _rerun_quality_and_decide(
            registry=registry,
            market_snapshot=state["market_snapshot"],
            intel_snapshot=state["intel_snapshot"],
            steps=steps,
            tool_events=tool_events,
            runtime_event=runtime_event,
            thought="LLM action planner 请求重新评估数据质量",
            proceed_reason=action.reason or "重新检查数据质量后允许进入策略合成",
        )
        update["context_compactions"] = result.context_updates
        return update

    if action.action == "continue_to_synth":
        step["decision"] = "proceed"
        return {
            "planner_steps": steps,
            "planner_decision": _planner_decision_payload(
                "proceed",
                reason=action.reason or "LLM action planner 允许进入策略合成",
                confidence_cap=action.confidence_cap,
                reentry_conditions=action.reentry_conditions,
            ),
            "tool_events": tool_events,
            "runtime_events": [runtime_event],
            "context_compactions": result.context_updates,
        }

    decision_type = "no_trade" if action.action == "no_trade" else "watchlist"
    step["decision"] = decision_type
    return {
        "planner_steps": steps,
        "planner_decision": _planner_decision_payload(
            decision_type,
            reason=action.reason or "LLM action planner 建议观望",
            confidence_cap=action.confidence_cap or (0.25 if decision_type == "no_trade" else 0.45),
            reentry_conditions=action.reentry_conditions or ["等待触发重新评估条件"],
        ),
        "tool_events": tool_events,
        "runtime_events": [runtime_event],
        "context_compactions": result.context_updates,
    }


async def _rerun_quality_and_decide(
    *,
    registry: ToolRegistry,
    market_snapshot: dict[str, Any],
    intel_snapshot: dict[str, Any],
    steps: list[dict[str, Any]],
    tool_events: list[dict[str, Any]],
    runtime_event: dict[str, Any],
    thought: str,
    proceed_reason: str,
) -> dict[str, Any]:
    """按受限 planner 请求重跑 data_quality_check 并据结果裁决。"""
    quality = await _planner_tool_call(
        registry,
        "data_quality_check",
        {
            "market_snapshot": market_snapshot,
            "intel_snapshot": intel_snapshot,
        },
        step_no=len(steps) + 1,
        thought=thought,
        steps=steps,
        tool_events=tool_events,
    )
    if not quality.get("ok", False):
        steps[-1]["decision"] = "no_trade"
        return {
            "planner_steps": steps,
            "planner_decision": _planner_decision_payload(
                "no_trade",
                reason=_tool_error_reason(quality),
                confidence_cap=0.25,
                reentry_conditions=["等待数据质量工具恢复后重新评估"],
            ),
            "tool_events": tool_events,
            "runtime_events": [runtime_event],
        }
    quality_report = (quality.get("output") or {}).get("data_quality_report") or {}
    if _quality_blocked(quality_report):
        steps[-1]["decision"] = "no_trade"
        return {
            "planner_steps": steps,
            "planner_decision": _planner_decision_payload(
                "no_trade",
                reason="数据质量不足:"
                + "; ".join(quality_report.get("reasons") or ["关键数据缺失"]),
                confidence_cap=0.25,
                reentry_conditions=["等待关键行情与资讯源恢复后重新评估"],
            ),
            "tool_events": tool_events,
            "runtime_events": [runtime_event],
        }
    steps[-1]["decision"] = "proceed"
    return {
        "planner_steps": steps,
        "planner_decision": _planner_decision_payload(
            "proceed",
            reason=proceed_reason,
            confidence_cap=None,
            reentry_conditions=[],
        ),
        "tool_events": tool_events,
        "runtime_events": [runtime_event],
    }


def _validate_planner_output(data: dict[str, Any]) -> None:
    planner_output(data)
    if _FORBIDDEN_PLANNER_FIELDS.intersection(data):
        raise ValueError("planner action contains unauthorized trade fields")


def _coerce_planner_action(data: dict[str, Any]) -> tuple[StrategyPlannerAction, str | None]:
    forbidden = sorted(_FORBIDDEN_PLANNER_FIELDS.intersection(data))
    if forbidden:
        return (
            StrategyPlannerAction(action="watchlist", reason="planner action 含越权交易字段"),
            "非法 planner action:包含越权交易字段 " + ", ".join(forbidden),
        )
    try:
        action = StrategyPlannerAction.model_validate(data)
    except ValidationError as exc:
        return (
            StrategyPlannerAction(action="watchlist", reason="planner action schema invalid"),
            "非法 planner action:" + str(exc.errors()[0].get("msg") or exc)[:200],
        )
    if action.confidence_cap is not None:
        action.confidence_cap = max(0.0, min(1.0, action.confidence_cap))
    action.reentry_conditions = [str(item)[:300] for item in action.reentry_conditions][:5]
    action.reason = action.reason[:500]
    return action, None


def _planner_decision_payload(
    decision_type: str,
    *,
    reason: str,
    confidence_cap: float | None,
    reentry_conditions: list[str],
) -> dict[str, Any]:
    return {
        "decision_type": decision_type,
        "reason": reason,
        "confidence_cap": confidence_cap,
        "reentry_conditions": reentry_conditions,
        "source": "llm_action_planner",
    }


def _annotate_planner_result(
    result: dict[str, Any], *, mode: str, enforced: bool, action_status: str | None = None
) -> dict[str, Any]:
    decision = result.get("planner_decision")
    if isinstance(decision, dict):
        status = action_status or _infer_planner_action_status(decision)
        decision.setdefault("mode", mode)
        decision.setdefault("enforced", enforced)
        decision.setdefault(
            "affects_final_advice",
            enforced and decision.get("decision_type") in {"no_trade", "watchlist"},
        )
        decision.setdefault("action_status", status)
    for step in result.get("planner_steps") or []:
        if isinstance(step, dict):
            step.setdefault("mode", mode)
    return result


def _infer_planner_action_status(decision: dict[str, Any]) -> str:
    reason = str(decision.get("reason") or "").lower()
    if "非法" in reason or "unauthorized" in reason or "schema invalid" in reason:
        return "rejected"
    if any(token in reason for token in ("失败", "降级", "max_steps", "tool", "恢复", "未配置")):
        return "fallback"
    return "accepted"


def _tool_result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, ToolInvocationResult):
        return result.model_dump(mode="json")
    if hasattr(result, "model_dump"):
        dumped = result.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    return result if isinstance(result, dict) else {}


def _tool_error_reason(event: dict[str, Any]) -> str:
    errors = event.get("errors") or []
    if errors and isinstance(errors[0], dict):
        return str(errors[0].get("message") or "planner tool failed")[:300]
    return "planner tool failed"


def _planner_max_steps_result(
    steps: list[dict[str, Any]],
    tool_events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "planner_steps": steps,
        "planner_decision": {
            "decision_type": "watchlist",
            "reason": "strategy_planner max_steps reached before completing checks",
            "confidence_cap": 0.45,
            "reentry_conditions": ["等待 planner 完成全部检查后重新评估"],
        },
        "tool_events": tool_events,
    }


def _apply_planner_decision(
    advice: StrategyAdvice,
    planner_decision: dict[str, Any] | None,
) -> StrategyAdvice:
    if not isinstance(planner_decision, dict):
        return advice
    decision_type = planner_decision.get("decision_type")
    if decision_type not in {"no_trade", "watchlist"}:
        return advice

    reason = str(planner_decision.get("reason") or "strategy_planner 建议观望")
    reentry = [str(item) for item in (planner_decision.get("reentry_conditions") or [])]
    cap = planner_decision.get("confidence_cap")
    try:
        confidence_cap = float(cap) if cap is not None else None
    except (TypeError, ValueError):
        confidence_cap = None

    if decision_type == "no_trade":
        advice.direction = "neutral"
        advice.decision_type = "no_trade"
        advice.entry_zone = None
        advice.stop_loss = None
        advice.targets = []
        advice.no_trade_reason = reason
        advice.reentry_conditions = _dedup([*advice.reentry_conditions, *reentry])
        advice.confidence = min(advice.confidence, confidence_cap or 0.25)
    elif advice.decision_type != "no_trade":
        advice.decision_type = "watchlist"
        advice.reentry_conditions = _dedup([*advice.reentry_conditions, *reentry])
        advice.confidence = min(advice.confidence, confidence_cap or 0.45)

    advice.risk_warnings = _dedup([*advice.risk_warnings, f"[planner] {reason}"])
    return advice


def _dedup(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out
