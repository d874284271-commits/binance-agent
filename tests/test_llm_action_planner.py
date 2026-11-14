"""Strictly bounded LLM action planner tests."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import MemorySaver

from binance_agent.agents.graph import build_graph
from binance_agent.agents.tools import build_default_tool_registry
from binance_agent.infra.settings import Settings


class _CountingToolRegistry:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._delegate = build_default_tool_registry()

    async def invoke(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append(name)
        return await self._delegate.invoke(name, payload, context=context)


class _FailingToolRegistry:
    async def invoke(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> Any:
        raise RuntimeError(f"{name} tool unavailable")


def _fakes(
    *,
    planner_response: str = '{"action":"continue_to_synth","reason":"ok"}',
    strategy_json: str | None = None,
) -> dict[str, BaseChatModel]:
    return {
        "technical": FakeListChatModel(
            responses=['{"stance":"bullish","confidence":0.8,"summary":"technical"}']
        ),
        "funding": FakeListChatModel(
            responses=['{"stance":"bullish","confidence":0.8,"summary":"funding"}']
        ),
        "news_sentiment": FakeListChatModel(
            responses=['{"stance":"bullish","confidence":0.8,"summary":"news"}']
        ),
        "bull_debater": FakeListChatModel(responses=["bull"]),
        "bear_debater": FakeListChatModel(responses=["bear"]),
        "research_manager": FakeListChatModel(responses=["research"]),
        "strategy_planner": FakeListChatModel(responses=[planner_response]),
        "strategy_synth": FakeListChatModel(
            responses=[
                strategy_json
                or (
                    '{"direction":"long","decision_type":"trade","entry_zone":"100",'
                    '"stop_loss":"95","targets":["115"],"confidence":0.85,'
                    '"rationale":"planner action test"}'
                )
            ]
        ),
    }


async def test_llm_action_planner_config_defaults_off() -> None:
    settings = Settings()

    assert settings.runtime.strategy_planner_llm_actions is False


async def test_llm_action_planner_disabled_preserves_deterministic_planner_behavior() -> None:
    graph = build_graph(
        lambda agent: _fakes(planner_response='{"action":"no_trade","reason":"llm says stop"}')[
            agent
        ],
        enable_strategy_planner=True,
        enable_strategy_planner_llm_actions=False,
    )

    result = await graph.ainvoke(_initial_state(complete_market=True))

    assert result["planner_decision"]["decision_type"] == "proceed"
    assert result["final_advice"].decision_type == "trade"
    assert not any(event.get("agent") == "strategy_planner" for event in result["runtime_events"])


async def test_llm_action_planner_no_trade_downgrades_final_strategy_without_prices() -> None:
    graph = build_graph(
        lambda agent: _fakes(
            planner_response=(
                '{"action":"no_trade","reason":"宏观事件前不追单",'
                '"reentry_conditions":["事件落地后重新评估"],"confidence_cap":0.2}'
            )
        )[agent],
        enable_strategy_planner=True,
        enable_strategy_planner_llm_actions=True,
    )

    result = await graph.ainvoke(_initial_state(complete_market=True))

    assert result["planner_decision"]["decision_type"] == "no_trade"
    assert result["planner_decision"]["source"] == "llm_action_planner"
    assert result["final_advice"].decision_type == "no_trade"
    assert result["final_advice"].direction == "neutral"
    assert result["final_advice"].entry_zone is None
    assert result["final_advice"].stop_loss is None
    assert result["final_advice"].targets == []
    assert "宏观事件前不追单" in result["final_advice"].no_trade_reason
    assert any(event.get("agent") == "strategy_planner" for event in result["runtime_events"])


async def test_llm_action_planner_shadow_records_without_changing_final_strategy() -> None:
    graph = build_graph(
        lambda agent: _fakes(
            planner_response=(
                '{"action":"no_trade","reason":"shadow says wait",'
                '"reentry_conditions":["observe only"],"confidence_cap":0.2}'
            )
        )[agent],
        strategy_planner_mode="llm_shadow",
    )

    result = await graph.ainvoke(_initial_state(complete_market=True))

    assert result["planner_decision"]["decision_type"] == "no_trade"
    assert result["planner_decision"]["mode"] == "llm_shadow"
    assert result["planner_decision"]["enforced"] is False
    assert result["planner_decision"]["action_status"] == "accepted"
    assert result["final_advice"].decision_type == "trade"
    assert result["final_advice"].entry_zone == "100"
    assert any(event.get("agent") == "strategy_planner" for event in result["runtime_events"])


async def test_llm_action_planner_enforce_mode_can_downgrade_final_strategy() -> None:
    graph = build_graph(
        lambda agent: _fakes(
            planner_response=(
                '{"action":"no_trade","reason":"enforce says wait",'
                '"reentry_conditions":["wait"],"confidence_cap":0.2}'
            )
        )[agent],
        strategy_planner_mode="llm_enforce",
    )

    result = await graph.ainvoke(_initial_state(complete_market=True))

    assert result["planner_decision"]["mode"] == "llm_enforce"
    assert result["planner_decision"]["enforced"] is True
    assert result["final_advice"].decision_type == "no_trade"


async def test_llm_action_planner_rejects_unauthorized_trade_payload() -> None:
    graph = build_graph(
        lambda agent: _fakes(
            planner_response=(
                '{"action":"buy","entry_zone":"100","stop_loss":"95",'
                '"targets":["115"],"reason":"越权交易建议"}'
            )
        )[agent],
        enable_strategy_planner=True,
        enable_strategy_planner_llm_actions=True,
    )

    result = await graph.ainvoke(_initial_state(complete_market=True))

    assert result["planner_decision"]["decision_type"] == "watchlist"
    assert "非法" in result["planner_decision"]["reason"]
    assert result["final_advice"].decision_type in {"watchlist", "no_trade"}
    assert any(
        "越权" in warning or "planner" in warning
        for warning in result["final_advice"].risk_warnings
    )


async def test_llm_action_planner_bad_json_does_not_fail_graph() -> None:
    graph = build_graph(
        lambda agent: _fakes(planner_response="not json at all")[agent],
        enable_strategy_planner=True,
        enable_strategy_planner_llm_actions=True,
    )

    result = await graph.ainvoke(_initial_state(complete_market=True))

    assert result["planner_decision"]["decision_type"] in {"watchlist", "no_trade"}
    assert result["final_advice"].decision_type in {"watchlist", "no_trade"}
    planner_runtime = next(
        event for event in result["runtime_events"] if event.get("agent") == "strategy_planner"
    )
    assert planner_runtime["ok"] is False
    assert planner_runtime["error_category"] == "json_parse"


async def test_llm_action_planner_continue_still_goes_through_risk_gate() -> None:
    graph = build_graph(
        lambda agent: _fakes(
            planner_response='{"action":"continue_to_synth","reason":"继续合成"}',
            strategy_json=(
                '{"direction":"long","decision_type":"trade","entry_zone":"100",'
                '"stop_loss":"105","targets":["115"],"confidence":0.85,'
                '"rationale":"bad stop should be vetoed"}'
            ),
        )[agent],
        enable_strategy_planner=True,
        enable_strategy_planner_llm_actions=True,
    )

    result = await graph.ainvoke(_initial_state(complete_market=True))

    assert result["planner_decision"]["decision_type"] == "proceed"
    assert result["risk_gate_result"]["veto"] is True
    assert result["final_advice"].decision_type == "no_trade"
    assert result["final_advice"].entry_zone is None
    assert "止损方向错误" in result["final_advice"].no_trade_reason


async def test_llm_action_planner_tool_failure_falls_back_without_graph_failure() -> None:
    graph = build_graph(
        lambda agent: _fakes(
            planner_response='{"action":"rerun_quality_check","reason":"重查质量"}'
        )[agent],
        enable_strategy_planner=True,
        enable_strategy_planner_llm_actions=True,
        planner_tool_registry=_FailingToolRegistry(),
    )

    result = await graph.ainvoke(_initial_state(complete_market=True))

    assert result["planner_decision"]["decision_type"] == "no_trade"
    assert "tool unavailable" in result["planner_decision"]["reason"]
    assert result["final_advice"].decision_type == "no_trade"


async def test_llm_action_planner_rerun_quality_respects_max_steps() -> None:
    registry = _CountingToolRegistry()
    graph = build_graph(
        lambda agent: _fakes(
            planner_response='{"action":"rerun_quality_check","reason":"重查质量"}'
        )[agent],
        enable_strategy_planner=True,
        enable_strategy_planner_llm_actions=True,
        planner_max_steps=3,
        planner_tool_registry=registry,
    )

    result = await graph.ainvoke(_initial_state(complete_market=True))

    assert len(result["planner_steps"]) == 3
    assert result["planner_decision"]["decision_type"] == "watchlist"
    assert "max_steps" in result["planner_decision"]["reason"]
    assert registry.calls == ["data_quality_check", "technical_indicators"]


async def test_llm_action_planner_resume_does_not_repeat_completed_step() -> None:
    registry = _CountingToolRegistry()
    graph = build_graph(
        lambda agent: _fakes(planner_response='{"action":"watchlist","reason":"等回调"}')[agent],
        checkpointer=MemorySaver(),
        interrupt_after=["strategy_planner"],
        enable_strategy_planner=True,
        enable_strategy_planner_llm_actions=True,
        planner_tool_registry=registry,
    )
    config = {"configurable": {"thread_id": "llm-action-planner-resume"}}

    updates = []
    async for update in graph.astream(
        _initial_state(complete_market=True),
        config=config,
        stream_mode="updates",
    ):
        updates.append(update)

    assert any("__interrupt__" in update for update in updates)
    calls_after_interrupt = list(registry.calls)
    result = await graph.ainvoke(None, config=config)

    assert result["final_advice"].decision_type in {"watchlist", "no_trade"}
    assert registry.calls == calls_after_interrupt


def _initial_state(*, complete_market: bool) -> dict[str, Any]:
    rows = [
        {"open": i, "high": i + 2, "low": i - 2, "close": i + 0.5, "volume": 10}
        for i in range(1, 50)
    ]
    market_snapshot = (
        {
            "klines": {"5m": rows, "1h": rows, "4h": rows, "1d": rows},
            "funding": {"last_funding_rate": 0.0001},
            "ticker_24h": {
                "lastPrice": "100",
                "priceChangePercent": "1.2",
                "highPrice": "105",
                "lowPrice": "95",
                "volume": "1000",
                "quoteVolume": "100000",
            },
            "errors": [],
        }
        if complete_market
        else {
            "klines": {},
            "funding": {"last_funding_rate": 0.0001},
            "ticker_24h": {},
            "errors": ["klines[1h]: boom"],
        }
    )
    return {
        "symbol": "BTCUSDT",
        "market": "fapi",
        "market_snapshot": market_snapshot,
        "intel_snapshot": {
            "sentiment_features": {"long_short_ratio": 1.2},
            "news_items": [{"title": "headline"}],
            "macro_events": [],
            "source_status": {"rss": "ok", "reddit": "ok"},
        },
        "analyst_reports": [],
        "debate_turns": [],
        "runtime_events": [],
        "tool_events": [],
        "planner_steps": [],
        "planner_decision": None,
        "debate_round": 0,
        "max_debate_rounds": 1,
        "research_summary": None,
        "risk_assessment": None,
        "risk_gate_result": None,
        "final_advice": None,
    }
