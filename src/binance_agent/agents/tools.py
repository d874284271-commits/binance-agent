"""有界 Agent runtime 使用的确定性工具注册表。

通过输入输出模型、超时和结构化错误暴露项目能力,调用方负责选择工具及记录 trace。
行情和资讯工具可以访问网络,其余校验工具读取传入快照,不让工具名称隐含额外权限。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..indicators import ta
from ..market.snapshot import build_market_snapshot
from .budget import BudgetPersistenceError
from .confidence import calibrate_confidence
from .state import AnalystReport, StrategyAdvice
from .validation import (
    DataQualityReport,
    RiskGateResult,
    build_data_quality_report,
    evaluate_risk_gate,
)

ToolContext = dict[str, Any]
ToolHandler = Callable[[BaseModel, ToolContext], Awaitable[BaseModel]]
ToolFallback = Callable[[BaseModel | None, str], BaseModel]


class ToolErrorInfo(BaseModel):
    category: str
    message: str


class ToolInvocationResult(BaseModel):
    tool: str
    ok: bool
    latency_ms: float
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)


class MarketSnapshotInput(BaseModel):
    symbol: str
    market: str = "fapi"


class MarketSnapshotOutput(BaseModel):
    market_snapshot: dict[str, Any]


class IntelSnapshotInput(BaseModel):
    symbol: str
    research_question: str = ""


class IntelSnapshotOutput(BaseModel):
    intel_snapshot: dict[str, Any]


class TechnicalIndicatorsInput(BaseModel):
    market_snapshot: dict[str, Any]
    intervals: list[str] = Field(default_factory=lambda: ["1h", "4h"])


class TechnicalIndicatorsOutput(BaseModel):
    indicators: dict[str, dict[str, Any]] = Field(default_factory=dict)
    missing_intervals: list[str] = Field(default_factory=list)


class DataQualityCheckInput(BaseModel):
    market_snapshot: dict[str, Any]
    intel_snapshot: dict[str, Any]


class DataQualityCheckOutput(BaseModel):
    data_quality_report: dict[str, Any]


class RiskGateCheckInput(BaseModel):
    advice: dict[str, Any]
    market_snapshot: dict[str, Any]
    risk_assessment: dict[str, Any] | None = None
    analyst_reports: list[dict[str, Any]] = Field(default_factory=list)


class RiskGateCheckOutput(BaseModel):
    risk_gate_result: dict[str, Any]


class ConfidenceCalibrationInput(BaseModel):
    advice: dict[str, Any]
    data_quality_report: dict[str, Any]
    risk_gate_result: dict[str, Any]
    analyst_reports: list[dict[str, Any]] = Field(default_factory=list)
    runtime_events: list[dict[str, Any]] = Field(default_factory=list)
    risk_assessment: dict[str, Any] | None = None


class ConfidenceCalibrationOutput(BaseModel):
    calibration: dict[str, Any]


class ToolDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: ToolHandler
    timeout_s: float | None = 8.0
    fallback: ToolFallback | None = None


class ToolRegistry:
    def __init__(self, tools: list[ToolDefinition]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> ToolDefinition:
        return self._tools[name]

    async def invoke(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        context: ToolContext | None = None,
    ) -> ToolInvocationResult:
        """校验输入,在工具时限内执行 handler,并将输出或错误封装为统一结果。

        普通异常返回 ok=False,可附工具自己的保守 fallback;未知工具名直接抛 KeyError。
        取消和 BudgetPersistenceError 不转成成功结果,预算持久化失败必须阻止后续消费。
        依赖由 context 注入,latency_ms 记录本次调用耗时,不保存业务状态。
        """
        started = time.perf_counter()
        tool = self.get(name)
        data: BaseModel | None = None
        try:
            data = tool.input_model.model_validate(payload)
            operation = tool.handler(data, context or {})
            output = (
                await operation
                if tool.timeout_s is None
                else await asyncio.wait_for(operation, timeout=tool.timeout_s)
            )
            output = tool.output_model.model_validate(output)
            return ToolInvocationResult(
                tool=name,
                ok=True,
                latency_ms=_elapsed_ms(started),
                input=data.model_dump(mode="json"),
                output=output.model_dump(mode="json"),
                errors=[],
            )
        except BudgetPersistenceError:
            raise
        except Exception as exc:
            category = _classify_error(exc)
            error = ToolErrorInfo(category=category, message=_error_message(exc, category))
            fallback_output = tool.fallback(data, str(exc)) if tool.fallback else None
            output_payload = (
                tool.output_model.model_validate(fallback_output).model_dump(mode="json")
                if fallback_output is not None
                else None
            )
            return ToolInvocationResult(
                tool=name,
                ok=False,
                latency_ms=_elapsed_ms(started),
                input=data.model_dump(mode="json") if data is not None else payload,
                output=output_payload,
                errors=[error.model_dump(mode="json")],
            )


def build_default_tool_registry() -> ToolRegistry:
    """装配行情、资讯、指标、数据质量、风控与置信度工具。

    只创建注册表,不发起请求;行情/资讯失败有明确空快照回退,资讯时限由聚合器管理。
    """
    return ToolRegistry(
        [
            ToolDefinition(
                name="market_snapshot",
                input_model=MarketSnapshotInput,
                output_model=MarketSnapshotOutput,
                handler=_market_snapshot_tool,
                fallback=_fallback_market_snapshot,
            ),
            ToolDefinition(
                name="intel_snapshot",
                input_model=IntelSnapshotInput,
                output_model=IntelSnapshotOutput,
                handler=_intel_snapshot_tool,
                timeout_s=None,
                fallback=_fallback_intel_snapshot,
            ),
            ToolDefinition(
                name="technical_indicators",
                input_model=TechnicalIndicatorsInput,
                output_model=TechnicalIndicatorsOutput,
                handler=_technical_indicators_tool,
                timeout_s=1.0,
            ),
            ToolDefinition(
                name="data_quality_check",
                input_model=DataQualityCheckInput,
                output_model=DataQualityCheckOutput,
                handler=_data_quality_tool,
                timeout_s=1.0,
            ),
            ToolDefinition(
                name="risk_gate_check",
                input_model=RiskGateCheckInput,
                output_model=RiskGateCheckOutput,
                handler=_risk_gate_tool,
                timeout_s=1.0,
            ),
            ToolDefinition(
                name="confidence_calibration",
                input_model=ConfidenceCalibrationInput,
                output_model=ConfidenceCalibrationOutput,
                handler=_confidence_tool,
                timeout_s=1.0,
            ),
        ]
    )


async def _market_snapshot_tool(
    request: BaseModel,
    context: ToolContext,
) -> MarketSnapshotOutput:
    data = _as_model(request, MarketSnapshotInput)
    client = context.get("rest_client")
    if client is None:
        raise ToolDependencyError("rest_client is required")
    builder = context.get("build_market_snapshot", build_market_snapshot)
    snapshot = await builder(client, data.symbol.upper())
    return MarketSnapshotOutput(market_snapshot=_model_dump(snapshot))


async def _intel_snapshot_tool(
    request: BaseModel,
    context: ToolContext,
) -> IntelSnapshotOutput:
    data = _as_model(request, IntelSnapshotInput)
    aggregator = context.get("aggregator")
    if aggregator is None:
        raise ToolDependencyError("aggregator is required")
    if data.research_question:
        try:
            snapshot = await aggregator.gather(
                data.symbol.upper(), research_question=data.research_question
            )
        except TypeError as exc:
            # Third-party/test aggregators may still expose the original
            # one-argument contract; built-in providers support the optional
            # question for relevance ranking before applying the news cap.
            if "research_question" not in str(exc):
                raise
            snapshot = await aggregator.gather(data.symbol.upper())
    else:
        snapshot = await aggregator.gather(data.symbol.upper())
    return IntelSnapshotOutput(intel_snapshot=_model_dump(snapshot))


async def _technical_indicators_tool(
    request: BaseModel,
    context: ToolContext,
) -> TechnicalIndicatorsOutput:
    data = _as_model(request, TechnicalIndicatorsInput)
    indicators: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    klines = data.market_snapshot.get("klines") or {}
    for interval in data.intervals:
        rows = klines.get(interval) or []
        if len(rows) < 30:
            missing.append(interval)
            continue
        try:
            highs = [float(_row_value(row, "high")) for row in rows]
            lows = [float(_row_value(row, "low")) for row in rows]
            closes = [float(_row_value(row, "close")) for row in rows]
        except (TypeError, ValueError, KeyError):
            missing.append(interval)
            continue
        indicators[interval] = ta.summarize(highs, lows, closes)
    return TechnicalIndicatorsOutput(indicators=indicators, missing_intervals=missing)


async def _data_quality_tool(
    request: BaseModel,
    context: ToolContext,
) -> DataQualityCheckOutput:
    data = _as_model(request, DataQualityCheckInput)
    report = build_data_quality_report(data.market_snapshot, data.intel_snapshot)
    return DataQualityCheckOutput(data_quality_report=report.model_dump(mode="json"))


async def _risk_gate_tool(
    request: BaseModel,
    context: ToolContext,
) -> RiskGateCheckOutput:
    data = _as_model(request, RiskGateCheckInput)
    advice = StrategyAdvice.model_validate(data.advice)
    reports = [AnalystReport.model_validate(report) for report in data.analyst_reports]
    result = evaluate_risk_gate(
        advice,
        data.market_snapshot,
        data.risk_assessment,
        reports,
    )
    return RiskGateCheckOutput(risk_gate_result=result.model_dump(mode="json"))


async def _confidence_tool(
    request: BaseModel,
    context: ToolContext,
) -> ConfidenceCalibrationOutput:
    data = _as_model(request, ConfidenceCalibrationInput)
    calibration = calibrate_confidence(
        advice=StrategyAdvice.model_validate(data.advice),
        data_quality=DataQualityReport.model_validate(data.data_quality_report),
        risk_gate=RiskGateResult.model_validate(data.risk_gate_result),
        analyst_reports=[AnalystReport.model_validate(report) for report in data.analyst_reports],
        runtime_events=data.runtime_events,
        risk_assessment=data.risk_assessment,
    )
    return ConfidenceCalibrationOutput(calibration=calibration.model_dump(mode="json"))


def _fallback_market_snapshot(data: BaseModel | None, message: str) -> MarketSnapshotOutput:
    request = _as_model(data, MarketSnapshotInput) if data is not None else None
    symbol = request.symbol.upper() if request is not None else ""
    market = request.market if request is not None else "fapi"
    return MarketSnapshotOutput(
        market_snapshot={
            "symbol": symbol,
            "market": market,
            "klines": {},
            "ticker_24h": {},
            "errors": [f"market_snapshot tool failed: {message}"],
        }
    )


def _fallback_intel_snapshot(data: BaseModel | None, message: str) -> IntelSnapshotOutput:
    request = _as_model(data, IntelSnapshotInput) if data is not None else None
    symbol = request.symbol.upper() if request is not None else ""
    return IntelSnapshotOutput(
        intel_snapshot={
            "symbol": symbol,
            "source_status": {"intel_snapshot": f"error: {message}"},
            "news_items": [],
            "macro_events": [],
            "sentiment_features": {},
        }
    )


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    return value if isinstance(value, dict) else {}


def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row[key]
    if hasattr(row, key):
        return getattr(row, key)
    if hasattr(row, "model_dump"):
        return row.model_dump(mode="json")[key]
    raise KeyError(key)


def _as_model(value: BaseModel | None, model: type[BaseModel]) -> Any:
    if isinstance(value, model):
        return value
    return model.model_validate(value)


def _classify_error(exc: BaseException) -> str:
    if isinstance(exc, ToolDependencyError):
        return "dependency"
    if isinstance(exc, TimeoutError | asyncio.TimeoutError):
        return "timeout"
    if isinstance(exc, ValidationError):
        return "validation"
    return "execution"


def _error_message(exc: BaseException, category: str) -> str:
    message = str(exc).strip()
    return message[:300] if message else category


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
