"""Offline evidence-research demonstration: normal / misleading title / missing body.

Run: python -m binance_agent.research_demo (loopback port 8766).
All market data, articles and model responses are synthetic.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from .app.server import _build_runtime_graph, create_app
from .demo import _complete_snapshot, _demo_model_factory
from .infra.settings import ResearchConfig, Settings
from .intel.evidence import evidence_document
from .intel.models import IntelSnapshot, NewsItem
from .llm.registry import LlmRegistry

_SCENARIOS = {
    "independent": (
        "Digital asset regulation proposal remains pending",
        "The digital asset regulation proposal is pending. It has not been adopted.",
    ),
    "BTCUSDT": (
        "Bitcoin network upgrade completed",
        "The Bitcoin network upgrade completed. No supply change was announced.",
    ),
    "ETHUSDT": (
        "Ethereum ETF approved!",
        "The Ethereum ETF application was submitted. Approval has not been granted. "
        "The proposed annual management fee is 0.25%.",
    ),
    "SOLUSDT": ("Solana exchange hack reported", ""),
}


class ResearchDemoModel:
    def bind(self, **kwargs: Any) -> ResearchDemoModel:
        return self

    async def ainvoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        # Keep the offline collaboration controls observable without a paid endpoint.
        await asyncio.sleep(0.2)
        payload = json.loads(messages[-1].content)
        prompt = str(messages[0].content)
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
        elif "candidates" in payload:
            data = {
                "tasks": [
                    {
                        "candidate_ids": list(payload["candidates"]),
                        "question": "核查新闻内容与标题是否一致",
                        "critical": True,
                        "completion": "正文证据支持且状态清晰",
                    }
                ]
            }
        elif "documents" in payload and "task" in payload:
            docs = [d for d in payload["documents"] if d["level"] == "body"]
            # Deliberately scripted: demonstrate a new extraction from the same
            # cached article, without pretending this fixture evaluates a model.
            fee_question = any(
                term in payload.get("research_question", "").lower()
                for term in ("费率", "management fee")
            )
            statements = []
            for doc in docs:
                text = doc["text"]
                prefix, separator, fee = text.partition("The proposed annual management fee")
                statement = (separator + fee) if separator and fee_question else prefix.strip()
                statements.append((doc, statement))
            data = {
                "facts": [
                    {
                        "statement": statement,
                        "evidence_id": d["evidence_id"],
                        "quote": statement,
                        "confirmation_status": "confirmed_in_source",
                    }
                    for d, statement in statements
                ],
                "stance": "neutral",
                "conflicts": [],
                "gaps": [] if docs else ["没有取得正文,重大事件尚待核实"],
            }
        elif "verdicts" in prompt:
            data = {
                "verdicts": [{"index": i, "supported": True} for i in range(len(payload["facts"]))]
            }
        else:
            data = {"supported": True}
        return AIMessage(
            content=json.dumps(data, ensure_ascii=False),
            usage_metadata={"input_tokens": 64, "output_tokens": 24, "total_tokens": 88},
        )


class ResearchDemoIntel:
    def __init__(self) -> None:
        self._disabled: dict[str, str] = {}

    async def gather(
        self, symbol: str | None, *, research_question: str = "", **_kwargs: Any
    ) -> IntelSnapshot:
        title, _ = _SCENARIOS.get(symbol or "independent", _SCENARIOS["SOLUSDT"])
        return IntelSnapshot(
            symbol=symbol,
            source_status={"synthetic_fixture": "ok"},
            news_items=[
                NewsItem(
                    title=title,
                    source="Synthetic demo",
                    url=f"https://example.invalid/{symbol or 'independent'}",
                )
            ],
            sentiment_features={"fear_greed": {"value": 50}},
        )


class ResearchDemoContent:
    async def fetch(self, url: str) -> Any:
        symbol = url.rsplit("/", 1)[-1]
        title, text = _SCENARIOS.get(symbol, _SCENARIOS["SOLUSDT"])
        return evidence_document(
            url=url,
            title=title,
            source="Synthetic demo",
            text=text,
            level="body",
            status="available" if text else "unavailable",
        )


def create_research_demo_app() -> FastAPI:
    settings = Settings.model_validate(
        {
            "profile": "demo",
            "collaboration": {"enabled": True},
            "runtime": {"checkpoint_backend": "sqlite"},
            "research": {"enabled": True, "max_requests": 100, "deadline_s": 600},
        }
    )
    app = create_app(settings)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(instance: FastAPI) -> Any:
        async with original_lifespan(instance):
            settings.research = ResearchConfig(enabled=True, max_requests=100, deadline_s=600)
            for name in ("research_supervisor", "news_researcher", "evidence_verifier"):
                settings.llm.agent_models[name] = ("demo", f"evidence-{name}")

            def factory(endpoint: Any, model: str) -> BaseChatModel:
                if model.startswith("evidence-"):
                    return cast(BaseChatModel, ResearchDemoModel())
                return _demo_model_factory(endpoint, model)

            registry = LlmRegistry(settings.llm, model_factory=factory)
            instance.state.llm_registry = registry
            instance.state.aggregator = ResearchDemoIntel()
            instance.state.content_fetcher = ResearchDemoContent()

            async def market(client: Any, symbol: str) -> Any:
                return _complete_snapshot(symbol, client.market)

            instance.state.market_snapshot_builder = market
            instance.state.graph = _build_runtime_graph(settings, registry, instance)
            try:
                yield
            finally:
                await registry.aclose()

    app.router.lifespan_context = lifespan
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_research_demo_app(), host="127.0.0.1", port=8766)
