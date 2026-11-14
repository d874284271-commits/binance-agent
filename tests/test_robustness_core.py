from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage

from binance_agent.agents.budget import BudgetPersistenceError, RunBudget, run_budget_scope
from binance_agent.agents.nodes import technical_analyst
from binance_agent.agents.runtime import AgentRuntime
from binance_agent.infra.retry import parse_retry_after
from binance_agent.market.rest_client import BinanceRestClient


class CountingModel:
    def __init__(self, content: str = "{}") -> None:
        self.content = content
        self.calls = 0

    def bind(self, **kwargs: Any) -> CountingModel:
        return self

    async def ainvoke(self, *args: Any, **kwargs: Any) -> AIMessage:
        self.calls += 1
        return AIMessage(
            content=self.content,
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )


@pytest.mark.parametrize("raw", [True, False, "nan", "Infinity", "1e999", "-1", "bad"])
def test_invalid_retry_after(raw: Any) -> None:
    assert parse_retry_after(raw) is None


def test_retry_after_formats() -> None:
    assert parse_retry_after("2.5") == 2.5
    assert parse_retry_after("Thu Jan  1 00:00:10 1970", now=5) == 5
    assert parse_retry_after("Thu, 01 Jan 1970 00:00:10 GMT", now=5) == 5
    assert parse_retry_after("Thu, 01 Jan 1970 00:00:10 GMT", now=11) == 0


async def test_long_retry_after_is_not_retried_early() -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "3600"})

    client = BinanceRestClient()
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url="https://fixture.invalid", transport=httpx.MockTransport(respond)
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await asyncio.wait_for(client.premium_index("BTCUSDT"), 1)
    finally:
        await client.aclose()
    assert calls == 1


async def test_settle_failure_stops_all_further_consumption() -> None:
    notices = 0

    def notify(snapshot: dict[str, Any]) -> None:
        nonlocal notices
        notices += 1
        if notices == 3:
            raise OSError("disk full")

    model = CountingModel()
    budget = RunBudget(
        max_requests=10,
        on_change=notify,
        endpoint_prices={"primary": {"input_per_million": 1, "output_per_million": 2}},
    )
    with run_budget_scope(budget), pytest.raises(BudgetPersistenceError):
        await AgentRuntime(model, max_retries=2).invoke_structured(
            "technical", "sys", {}, fallback_data={}
        )
    assert model.calls == 1
    assert budget.cost_usd == 0.00014
    assert budget.unknown_cost_requests == 0
    assert budget.requests_settled == 1
    with pytest.raises(BudgetPersistenceError):
        await budget.reserve("primary", "later")


@pytest.mark.parametrize(
    "content",
    ["{}", '{"stance":["bullish"]}', '{"stance":"bullish","confidence":NaN,"summary":"x"}'],
)
async def test_bad_analyst_output_is_fallback_without_extra_calls(content: str) -> None:
    from test_agents_nodes import _state

    model = CountingModel(content)
    output = await technical_analyst(_state(), model)
    assert output["analyst_reports"][0].stance == "neutral"
    assert output["analyst_reports"][0].confidence <= 0.3
    assert output["runtime_events"][0]["ok"] is False
    assert model.calls == 1


@pytest.mark.parametrize("repair", [False, True])
async def test_huge_integer_confidence_is_schema_failure(repair: bool) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from binance_agent.agents.output_validation import analyst_output

    output = json.dumps({"stance": "bullish", "confidence": 10**400, "summary": "x"})
    model = FakeListChatModel(responses=(["not json", output] if repair else [output]))
    result = await AgentRuntime(model).invoke_structured(
        "technical", "sys", {}, fallback_data={}, output_validator=analyst_output
    )
    assert not result.ok
    assert result.error_category == "schema_invalid"


async def test_empty_repair_does_not_issue_another_request() -> None:
    class Model(CountingModel):
        async def ainvoke(self, *args: Any, **kwargs: Any) -> AIMessage:
            self.content = ["not json", "", '{"ok":true}'][self.calls]
            return await super().ainvoke(*args, **kwargs)

    model = Model()
    result = await AgentRuntime(model, max_retries=1).invoke_structured(
        "technical", "sys", {}, fallback_data={}
    )
    assert not result.ok
    assert result.error_category == "empty_response"
    assert model.calls == 2


@pytest.mark.parametrize(
    "content",
    [
        "",
        "   ",
        '{"stance":"bullish","summary":"x","confidence":true}',
        '{"stance":"bullish","summary":[],"confidence":0.9}',
        '{"stance":"bullish","summary":"x","confidence":Infinity}',
        '{"stance":"bullish","summary":"x"}',
    ],
)
async def test_invalid_analyst_variants_never_raise_or_repair(content: str) -> None:
    from test_agents_nodes import _state

    model = CountingModel(content)
    output = await technical_analyst(_state(), model)
    assert not output["runtime_events"][0]["ok"]
    assert output["analyst_reports"][0].stance == "neutral"
    assert model.calls == 1


@pytest.mark.parametrize(
    "content",
    [
        '{"direction":[],"decision_type":"trade","confidence":1}',
        '{"direction":"long","decision_type":{},"confidence":0.9}',
        '{"direction":"long","decision_type":"trade","confidence":true}',
        '{"direction":"neutral","decision_type":"no_trade","confidence":0.9,"entry_zone":"100"}',
    ],
)
async def test_invalid_strategy_is_neutral_without_price_levels(content: str) -> None:
    from binance_agent.agents.nodes import strategy_synth
    from test_agents_nodes import _state

    model = CountingModel(content)
    output = await strategy_synth(_state(), model)
    advice = output["final_advice"]
    assert advice.direction == "neutral" and advice.decision_type == "no_trade"
    assert advice.entry_zone is None and advice.stop_loss is None and advice.targets == []
    assert advice.confidence <= 0.25
    assert not output["runtime_events"][0]["ok"]
    assert model.calls == 1


@pytest.mark.parametrize("importance", [True, None, "high", [], -1, 4])
async def test_unknown_macro_importance_prevents_trade(importance: Any) -> None:
    from binance_agent.agents.nodes import risk_manager, strategy_synth
    from test_agents_nodes import _state

    state = _state()
    state["intel_snapshot"]["macro_events"] = [{"name": "CPI", "importance": importance}]
    state.update(risk_manager(state))
    model = CountingModel(
        json.dumps(
            {
                "direction": "long",
                "decision_type": "trade",
                "confidence": 0.9,
                "rationale": "x",
                "entry_zone": "99-101",
                "stop_loss": "95",
                "targets": ["110"],
            }
        )
    )
    output = await strategy_synth(state, model)
    assert output["final_advice"].decision_type == "no_trade"
    assert state["risk_assessment"]["level"] == "high"


async def test_rest_total_deadline_cancels_inflight_request(monkeypatch: Any) -> None:
    from types import SimpleNamespace

    from binance_agent.market import rest_client

    original = asyncio.wait_for
    cancelled = asyncio.Event()
    deadlines: list[float] = []

    async def deadline(awaitable: Any, timeout: float) -> Any:
        deadlines.append(timeout)
        return await original(awaitable, 0.02)

    async def get(*args: Any, **kwargs: Any) -> Any:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client = BinanceRestClient()
    await client.aclose()
    client._client = SimpleNamespace(get=get)
    monkeypatch.setattr(rest_client.asyncio, "wait_for", deadline)
    with pytest.raises(asyncio.TimeoutError):
        await client.premium_index("BTCUSDT")
    assert deadlines == [45.0]
    assert cancelled.is_set()


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        '[{"stance":"bullish","confidence":0.9,"summary":"x"}]',
        "null",
        "true",
        "42",
        '"text"',
        '```json\n[{"x":1}]\n```',
    ],
)
async def test_valid_json_with_wrong_root_is_not_repaired(content: str) -> None:
    model = CountingModel(content)
    result = await AgentRuntime(model).invoke_structured("technical", "sys", {}, fallback_data={})
    assert not result.ok and result.error_category == "schema_invalid"
    assert model.calls == 1
