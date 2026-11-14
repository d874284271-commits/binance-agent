"""完全离线、可重复的演示依赖。"""

from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, ClassVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from .infra.settings import LlmConfig, LlmEndpoint
from .intel.evidence import evidence_document
from .intel.models import IntelSnapshot, NewsItem
from .llm.registry import LlmRegistry
from .market.snapshot import FundingInfo, Kline, MarketSnapshot

_DEMO_TIME = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
_PRICE = 100.0


class DemoRestClient:
    """只提供本地固定 ticker 的最小 REST client。"""

    def __init__(self, market: str) -> None:
        self.market = market
        self.has_derivatives = market == "fapi"

    async def ticker_24h(self, symbol: str) -> dict[str, str]:
        return _ticker(symbol)

    async def aclose(self) -> None:
        return None


DEMO_REQUEST_ID: ContextVar[str | None] = ContextVar("binance_agent_demo_request_id", default=None)


class DemoMarketData:
    """按请求上下文驱动三个固定 preflight 场景。"""

    def __init__(self) -> None:
        # The request id is set by the demo API stream before graph execution.
        # A normal mutable set is used so a refetch in a child task can update
        # the state created by the initial snapshot task on Python 3.10 too.
        self._awaiting_refetch: set[tuple[str, str, str]] = set()
        # 仅用于测试/trace 观测总调用量,不能参与场景决策。
        self._calls: dict[tuple[str, str], int] = {}

    async def build_snapshot(self, client: DemoRestClient, symbol: str) -> MarketSnapshot:
        symbol = symbol.upper()
        request_id = DEMO_REQUEST_ID.get()
        if request_id is None:
            task = asyncio.current_task()
            request_id = f"task:{id(task)}"
        key = (request_id, client.market, symbol)
        calls_key = (client.market, symbol)
        self._calls[calls_key] = self._calls.get(calls_key, 0) + 1
        if symbol == "BTCUSDT":
            return _complete_snapshot(symbol, client.market)
        if symbol in {"ETHUSDT", "SOLUSDT"}:
            if key not in self._awaiting_refetch:
                self._awaiting_refetch.add(key)
                return _incomplete_snapshot(symbol, client.market)

            self._awaiting_refetch.discard(key)
            if symbol == "ETHUSDT":
                return _complete_snapshot(symbol, client.market)
            raise RuntimeError("demo scripted market refetch failure")
        return _incomplete_snapshot(symbol, client.market)

    def calls_for(self, symbol: str, market: str = "fapi") -> int:
        return self._calls.get((market, symbol.upper()), 0)


class DemoIntelAggregator:
    _disabled: ClassVar[dict[str, str]] = {}

    async def gather(
        self, symbol: str | None, *, research_question: str = "", **_: Any
    ) -> IntelSnapshot:
        return IntelSnapshot(
            symbol=symbol.upper() if symbol else None,
            fetched_at=_DEMO_TIME,
            source_status={"demo_news": "ok", "demo_sentiment": "ok", "demo_macro": "ok"},
            sentiment_features={"fear_greed": 54, "long_short_ratio": 1.08},
            news_items=[
                NewsItem(
                    title=f"{symbol.upper() if symbol else 'Digital assets'} demo liquidity remains stable",
                    source="offline-demo",
                    url="demo://headline/1",
                    published_at=_DEMO_TIME,
                    sentiment="neutral",
                )
            ],
            macro_events=[],
        )


class DemoResearchContent:
    """Deterministic body reader used by the packaged demo research mode."""

    async def fetch(self, url: str, **_kwargs: Any) -> Any:
        symbol = url.rsplit("/", 1)[-1] or "BTCUSDT"
        title = f"{symbol} demo liquidity remains stable"
        text = (
            f"{symbol} demo liquidity remains stable. "
            "The fixed offline sample reports no material supply change."
        )
        return evidence_document(
            url=url,
            source="offline-demo",
            title=title,
            text=text,
            published_at=_DEMO_TIME,
            level="body",
            source_kind="media",
        )


class DemoHttpClient:
    async def get(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("demo profile is offline")

    async def aclose(self) -> None:
        return None


class DemoWsManager:
    def __init__(self, market: str) -> None:
        self.market = market

    async def price_stream(self, symbol: str, *, stop: asyncio.Event | None = None) -> Any:
        while stop is None or not stop.is_set():
            yield {
                "symbol": symbol.upper(),
                "market": self.market,
                "price": f"{_PRICE:.2f}",
                "mark_price": f"{_PRICE:.2f}" if self.market == "fapi" else None,
                "funding_rate": "0.0001" if self.market == "fapi" else None,
                "next_funding_time": None,
            }
            await asyncio.sleep(1)


class DemoOutcomeStore:
    """固定 outcome 数据集,让评估页和 proposal 在每次离线演示中一致。"""

    def __init__(self) -> None:
        self._records = _demo_outcome_records()

    def list_all(self) -> list[dict[str, Any]]:
        return deepcopy(self._records)

    def list_recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return list(reversed(self.list_all()))[: max(0, limit)]

    def tagged_run_ids(self) -> set[str]:
        return {str(record["run_id"]) for record in self._records}


def build_demo_registry(config: LlmConfig) -> LlmRegistry:
    return LlmRegistry(config, model_factory=_demo_model_factory)


def _demo_model_factory(_endpoint: LlmEndpoint, model: str) -> BaseChatModel:
    if model in {
        "demo-research_supervisor",
        "demo-news_researcher",
        "demo-evidence_verifier",
    }:
        return cast(BaseChatModel, _DemoResearchChatModel())
    response = _SCRIPTED_RESPONSES.get(model, _SCRIPTED_RESPONSES["demo-strategy_synth"])
    return cast(BaseChatModel, _DemoChatModel(response))


class _DemoChatModel:
    def __init__(self, response: str) -> None:
        self._response = response

    def bind(self, **_kwargs: Any) -> _DemoChatModel:
        return self

    async def ainvoke(self, _messages: Any, **_kwargs: Any) -> AIMessage:
        return AIMessage(
            content=self._response,
            usage_metadata={"input_tokens": 64, "output_tokens": 24, "total_tokens": 88},
        )


class _DemoResearchChatModel:
    """Small protocol-aware model so the frozen demo exercises evidence nodes."""

    def bind(self, **_kwargs: Any) -> _DemoResearchChatModel:
        return self

    async def ainvoke(self, messages: Any, **_kwargs: Any) -> AIMessage:
        prompt = str(messages[0].content)
        try:
            payload = json.loads(str(messages[-1].content))
        except (TypeError, ValueError):
            payload = {}
        if "queries" in prompt:
            data: dict[str, Any] = {"queries": [payload["question"][:500]], "time_scope": "未指定"}
        elif "recommendations" in prompt:
            data = {
                "summary": "合成示例:" + ";".join(f["statement"] for f in payload.get("facts", [])),
                "recommendations": ["仅展示研究流程"],
                "conditions": ["出现新资料时重新评估"],
                "risks": ["合成内容,不能用于决策"],
                "gaps": [],
            }
        elif isinstance(payload.get("candidates"), dict):
            candidate_ids = list(payload["candidates"].keys())[:1]
            data = {
                "tasks": [
                    {
                        "candidate_ids": candidate_ids,
                        "question": "核查离线正文是否支持标题所述事件",
                        "critical": False,
                        "completion": "正文证据可追溯且条件完整",
                    }
                ]
            }
        elif isinstance(payload.get("documents"), list) and "task" in payload:
            facts = []
            for document in payload["documents"]:
                if document.get("level") not in {"body", "summary"}:
                    continue
                text = str(document.get("text") or "")
                if text:
                    facts.append(
                        {
                            "statement": text,
                            "evidence_id": document.get("evidence_id", ""),
                            "quote": text,
                            "confirmation_status": "confirmed_in_source",
                        }
                    )
            data = {
                "facts": facts[:4],
                "stance": "neutral",
                "conflicts": [],
                "gaps": [] if facts else ["离线正文不可用"],
            }
        elif "verdicts" in prompt and isinstance(payload.get("facts"), list):
            data = {
                "verdicts": [
                    {"index": index, "supported": True} for index, _ in enumerate(payload["facts"])
                ]
            }
        elif "supported" in prompt:
            data = {"supported": True, "reason": "离线固定正文与引文一致"}
        else:
            data = {"follow_up": False, "missing": []}
        return AIMessage(
            content=json.dumps(data, ensure_ascii=False),
            usage_metadata={"input_tokens": 64, "output_tokens": 24, "total_tokens": 88},
        )


def _complete_snapshot(symbol: str, market: str) -> MarketSnapshot:
    rows = [
        Kline(
            open_time=1_700_000_000_000 + index * 60_000,
            open=98.0 + index * 0.05,
            high=101.5 + index * 0.05,
            low=96.5 + index * 0.05,
            close=99.0 + index * 0.05,
            volume=1_000.0 + index,
            close_time=1_700_000_059_999 + index * 60_000,
        )
        for index in range(36)
    ]
    return MarketSnapshot(
        symbol=symbol,
        market=market,
        fetched_at=_DEMO_TIME,
        klines={interval: list(rows) for interval in ("5m", "1h", "4h", "1d")},
        ticker_24h=_ticker(symbol),
        funding=FundingInfo(mark_price=_PRICE, last_funding_rate=0.0001),
        open_interest=12_500_000,
        errors=[],
    )


def _incomplete_snapshot(symbol: str, market: str) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        market=market,
        fetched_at=_DEMO_TIME,
        klines={},
        ticker_24h={},
        errors=["demo scripted initial snapshot is incomplete"],
    )


def _ticker(symbol: str) -> dict[str, str]:
    return {
        "symbol": symbol.upper(),
        "lastPrice": f"{_PRICE:.2f}",
        "priceChangePercent": "1.25",
        "highPrice": "104.00",
        "lowPrice": "96.00",
        "volume": "24000",
        "quoteVolume": "2400000",
    }


def _demo_outcome_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(24):
        records.append(
            _demo_outcome(
                outcome_id=f"demo-high-{index:02d}",
                confidence=0.85,
                outcome="target_hit" if index < 6 else "stop_hit",
            )
        )
    for index in range(24):
        records.append(
            _demo_outcome(
                outcome_id=f"demo-mid-{index:02d}",
                confidence=0.7,
                outcome="target_hit" if index < 17 else "stop_hit",
            )
        )
    for outcome, count in (("not_entered", 5), ("open", 3), ("unevaluable", 2)):
        for index in range(count):
            records.append(
                _demo_outcome(
                    outcome_id=f"demo-{outcome}-{index:02d}",
                    confidence=0.65,
                    outcome=outcome,
                )
            )
    return records


def _demo_outcome(*, outcome_id: str, confidence: float, outcome: str) -> dict[str, Any]:
    return {
        "outcome_id": outcome_id,
        "run_id": f"run-{outcome_id}",
        "symbol": "BTCUSDT",
        "market": "fapi",
        "direction": "long",
        "decision_type": "trade",
        "confidence": confidence,
        "model_confidence": confidence,
        "entry": 100.0,
        "stop": 95.0,
        "targets": [110.0],
        "outcome": outcome,
        "r_realized": 2.0 if outcome == "target_hit" else (-1.0 if outcome == "stop_hit" else None),
        "evaluation_started_at": "2026-01-01T00:00:00+00:00",
        "evaluation_ended_at": "2026-01-02T00:00:00+00:00",
        "entry_fill": 101.0,
        "target_evaluated": 110.0,
        "policy_version": "ohlc-conservative-v1",
        "created_at": "2026-01-02T00:00:00+00:00",
    }


_SCRIPTED_RESPONSES: dict[str, str] = {
    "demo-technical": (
        '{"stance":"bullish","confidence":0.72,"summary":"离线技术样本保持温和上行",'
        '"key_points":["1h 与 4h 数据完整","波动仍在预设区间"]}'
    ),
    "demo-funding": (
        '{"stance":"bullish","confidence":0.64,"summary":"资金费率温和,未见拥挤",'
        '"key_points":["funding=0.0001","持仓量稳定"]}'
    ),
    "demo-news_sentiment": (
        '{"stance":"neutral","confidence":0.58,"summary":"固定资讯样本偏中性",'
        '"key_points":["三路离线数据源可用"]}'
    ),
    "demo-bull_debater": (
        '{"argument":"完整行情支持谨慎偏多",'
        '"claims":[{"statement":"最新价处于固定演示区间",'
        '"evidence":"canonical ticker lastPrice",'
        '"source_path":"snapshot_data.ticker_24h.lastPrice",'
        '"source_value":"100.00","confidence":0.7}]}'
    ),
    "demo-bear_debater": (
        '{"argument":"资金费率虽温和,仍需保留止损",'
        '"claims":[{"statement":"当前资金费率为正",'
        '"evidence":"canonical funding rate",'
        '"source_path":"snapshot_data.funding.last_funding_rate",'
        '"source_value":0.0001,"confidence":0.62}]}'
    ),
    "demo-research_manager": "多空证据均可追溯,固定场景下采用小幅偏多并保留明确止损。",
    "demo-strategy_planner": (
        '{"decision_type":"trade","reason":"离线 shadow 仅记录,不覆盖最终建议"}'
    ),
    "demo-strategy_synth": (
        '{"direction":"long","decision_type":"trade","entry_zone":"99-101",'
        '"stop_loss":"95","targets":["110"],"confidence":0.68,'
        '"rationale":"固定行情完整且证据引用可核验",'
        '"risk_warnings":["离线场景仅用于演示工程链路"]}'
    ),
}
