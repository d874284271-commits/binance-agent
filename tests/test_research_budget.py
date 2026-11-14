from __future__ import annotations

import asyncio
import contextlib
from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from binance_agent.agents.budget import BudgetExceeded, EndpointPool, RunBudget, run_budget_scope
from binance_agent.agents.runtime import AgentRuntime, _retry_after_seconds


class _LLM:
    def __init__(self, responses: list[Any], delay: float = 0) -> None:
        self.responses = responses
        self.delay = delay
        self.calls = 0

    def bind(self, **kwargs: Any) -> _LLM:
        assert kwargs.get("max_tokens", 0) > 0
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return AIMessage(content=str(item))


async def test_budget_refuses_request_before_model_call() -> None:
    llm = _LLM(['{"stance":"bullish"}'])
    budget = RunBudget(max_requests=0)
    with run_budget_scope(budget):
        result = await AgentRuntime(cast(BaseChatModel, llm), max_retries=1).invoke_structured(
            "technical", "system", {}, fallback_data={"stance": "neutral"}
        )
    assert not result.ok
    assert llm.calls == 0
    assert result.errors[0].category == "budget"
    assert budget.requests_reserved == 0


async def test_repair_and_retry_each_reserve_a_request() -> None:
    llm = _LLM(["not json", '{"stance":"neutral"}'])
    budget = RunBudget(max_requests=2)
    with run_budget_scope(budget):
        result = await AgentRuntime(cast(BaseChatModel, llm), max_retries=0).invoke_structured(
            "technical", "system", {}, fallback_data={}
        )
    assert result.ok and result.repaired
    assert [item["kind"] for item in budget.attempts] == ["invoke", "repair"]
    assert budget.requests_reserved == 2
    assert budget.cost_usd is None  # no provider usage / price is never reported as free


async def test_shared_endpoint_pool_limits_parallel_runs() -> None:
    first = _LLM(['{"stance":"bullish"}'], delay=0.03)
    second = _LLM(['{"stance":"bearish"}'], delay=0.03)
    pool = EndpointPool(endpoint_concurrency=1)
    first_budget = RunBudget(max_requests=1, endpoint_pool=pool)
    second_budget = RunBudget(max_requests=1, endpoint_pool=pool)

    async def invoke(llm: _LLM, budget: RunBudget) -> Any:
        with run_budget_scope(budget):
            return await AgentRuntime(cast(BaseChatModel, llm), max_retries=0).invoke_structured(
                "technical", "system", {}, fallback_data={}
            )

    await asyncio.gather(invoke(first, first_budget), invoke(second, second_budget))
    assert max(first_budget.attempts[0]["wait_ms"], second_budget.attempts[0]["wait_ms"]) >= 20


async def test_snapshot_recovers_pending_as_unknown_and_callback_persists_changes() -> None:
    snapshots: list[dict[str, Any]] = []
    budget = RunBudget(max_requests=3, on_change=snapshots.append)
    reservation = await budget.reserve("primary", "technical")
    recovered = RunBudget.from_snapshot(budget.snapshot())
    assert recovered.attempts[0]["status"] == "unknown"
    assert recovered.cost_usd is None
    await budget.settle(reservation, outcome="cancelled", possibly_billed=True)
    assert len(snapshots) >= 2


async def test_cooldown_reservation_reuses_its_permit_without_deadlocking() -> None:
    budget = RunBudget(max_requests=2, endpoint_concurrency=1)
    first = await budget.reserve("primary", "technical")
    await budget.settle(first, outcome="error", retry_after_s=0.01)
    second = await asyncio.wait_for(budget.reserve("primary", "technical"), timeout=0.2)
    assert second.record["wait_ms"] >= 5
    await budget.settle(second, outcome="ok", usage={"input_tokens": 1, "output_tokens": 1})


async def test_cooldown_survives_checkpoint_restore() -> None:
    budget = RunBudget(max_requests=2, endpoint_concurrency=1)
    first = await budget.reserve("primary", "technical")
    await budget.settle(first, outcome="rate_limit", retry_after_s=0.05)
    snapshot = budget.snapshot()
    assert snapshot["endpoint_cooldown_remaining_s"]["primary"] > 0

    restored = RunBudget.from_snapshot(snapshot)
    started = asyncio.get_running_loop().time()
    second = await restored.reserve("primary", "technical")
    waited = asyncio.get_running_loop().time() - started
    await restored.settle(second, outcome="ok", usage={"input_tokens": 1, "output_tokens": 1})
    assert waited >= 0.03


async def test_cancelling_queued_admission_marks_it_not_sent_and_releases_capacity() -> None:
    pool = EndpointPool(endpoint_concurrency=1)
    holder_budget = RunBudget(max_requests=1, endpoint_pool=pool)
    waiting_budget = RunBudget(max_requests=1, endpoint_pool=pool)
    holder = await holder_budget.reserve("primary", "technical")
    waiting = asyncio.create_task(waiting_budget.reserve("primary", "funding"))
    await asyncio.sleep(0)
    waiting.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await waiting
    assert waiting_budget.attempts[0]["status"] == "not_sent"
    await holder_budget.settle(holder, outcome="ok", usage={"input_tokens": 1, "output_tokens": 1})
    follow_up_budget = RunBudget(max_requests=1, endpoint_pool=pool)
    follow_up = await asyncio.wait_for(follow_up_budget.reserve("primary", "funding"), timeout=0.1)
    await follow_up_budget.settle(
        follow_up, outcome="ok", usage={"input_tokens": 1, "output_tokens": 1}
    )


async def test_checkpoint_callback_failure_fails_closed_before_model_request() -> None:
    def fail_checkpoint(_: dict[str, Any]) -> None:
        raise OSError("store unavailable")

    llm = _LLM(['{"stance":"bullish"}'])
    budget = RunBudget(max_requests=1, on_change=fail_checkpoint)
    from binance_agent.agents.budget import BudgetPersistenceError

    with run_budget_scope(budget), pytest.raises(BudgetPersistenceError):
        await AgentRuntime(cast(BaseChatModel, llm), max_retries=0).invoke_structured(
            "technical", "system", {}, fallback_data={}
        )
    assert llm.calls == 0
    assert budget.requests_reserved == 1


def test_endpoint_pool_uses_per_endpoint_override() -> None:
    pool = EndpointPool(endpoint_concurrency=2, per_endpoint_concurrency={"slow": 1})
    assert pool.concurrency_for("slow") == 1
    assert pool.concurrency_for("other") == 2


def test_retry_after_accepts_seconds_and_http_date() -> None:
    class _Response:
        def __init__(self) -> None:
            self.headers = {"retry-after": "2"}

    class _Error(Exception):
        def __init__(self) -> None:
            self.response = _Response()

    assert _retry_after_seconds(_Error()) == 2


async def test_monetary_reservations_survive_unknown_outcome_and_restart() -> None:
    budget = RunBudget(
        max_requests=10,
        max_cost_usd=0.003,
        endpoint_prices={"one": {"input_per_million": 1, "output_per_million": 1}},
    )
    attempt = await budget.reserve(
        "one", "research", input_tokens_upper=1000, output_tokens_limit=1000
    )
    await budget.settle(attempt, outcome="timeout", possibly_billed=True)
    restored = RunBudget.from_snapshot(budget.snapshot())
    assert restored.cost_usd is None
    assert restored.cost_committed_usd == 0.002
    with pytest.raises(BudgetExceeded, match="monetary"):
        await restored.reserve("one", "retry", input_tokens_upper=1000, output_tokens_limit=1000)


async def test_paid_search_and_free_body_are_both_reserved() -> None:
    from binance_agent.agents.research import _call_tool

    calls = []

    async def tool() -> list[str]:
        calls.append("called")
        return ["result"]

    budget = RunBudget(max_requests=2, max_cost_usd=0.01)
    with run_budget_scope(budget):
        assert await _call_tool("public_html", "event-1", tool, fixed_cost_usd=0.0) == ["result"]
        assert await _call_tool("search", "event-1", tool, fixed_cost_usd=0.005) == ["result"]
        with pytest.raises(BudgetExceeded):
            await _call_tool("search", "event-1", tool, fixed_cost_usd=0.005)
    assert len(calls) == 2
    assert budget.cost_usd == 0.005
    assert budget.requests_reserved == 2


async def test_unpriced_search_cannot_consume_money_budget() -> None:
    from binance_agent.agents.research import _call_tool

    async def forbidden() -> None:
        raise AssertionError("must not call unpriced service")

    with (
        run_budget_scope(RunBudget(max_requests=1, max_cost_usd=0.01)),
        pytest.raises(BudgetExceeded, match="priced"),
    ):
        await _call_tool("search", "event-1", forbidden, fixed_cost_usd=None)


async def test_provider_usage_overrun_is_committed_before_next_request_and_after_restore() -> None:
    budget = RunBudget(
        max_requests=5,
        max_cost_usd=0.0005,
        endpoint_prices={"one": {"input_per_million": 1, "output_per_million": 1}},
    )
    attempt = await budget.reserve("one", "probe", input_tokens_upper=100, output_tokens_limit=100)
    assert budget.cost_committed_usd == pytest.approx(0.0002)
    await budget.settle(attempt, usage={"input_tokens": 100, "output_tokens": 1000})
    assert budget.cost_committed_usd == pytest.approx(0.0011)
    await budget.settle(attempt, usage={"input_tokens": 100, "output_tokens": 1000})
    assert budget.cost_committed_usd == pytest.approx(0.0011)
    restored = RunBudget.from_snapshot(budget.snapshot())
    assert restored.cost_committed_usd == pytest.approx(0.0011)
    with pytest.raises(BudgetExceeded, match="monetary"):
        await restored.reserve("one", "must_not_send", input_tokens_upper=1, output_tokens_limit=1)


def test_legacy_snapshot_with_known_overrun_is_reconciled_once() -> None:
    snapshot = {
        "max_cost_usd": 0.001,
        "cost_committed_usd": 0.0002,
        "endpoint_prices": {"one": {"input_per_million": 1, "output_per_million": 1}},
        "cost_usd": 0.0011,
        "attempts": [
            {"id": "legacy", "status": "ok", "reserved_cost_usd": 0.0002, "cost_usd": 0.0011}
        ],
    }
    restored = RunBudget.from_snapshot(snapshot)
    assert restored.cost_committed_usd == pytest.approx(0.0011)
    assert RunBudget.from_snapshot(restored.snapshot()).cost_committed_usd == pytest.approx(0.0011)


async def test_known_lower_usage_does_not_release_conservative_reservation() -> None:
    budget = RunBudget(
        max_cost_usd=0.003,
        endpoint_prices={"one": {"input_per_million": 1, "output_per_million": 1}},
    )
    attempt = await budget.reserve(
        "one", "probe", input_tokens_upper=1000, output_tokens_limit=1000
    )
    await budget.settle(attempt, usage={"input_tokens": 10, "output_tokens": 10})
    assert budget.cost_usd == pytest.approx(0.00002)
    assert budget.cost_committed_usd == pytest.approx(0.002)


async def test_queued_request_is_not_sent_after_prior_provider_overrun() -> None:
    budget = RunBudget(
        max_cost_usd=0.003,
        endpoint_concurrency=1,
        endpoint_prices={"one": {"input_per_million": 1, "output_per_million": 1}},
    )
    first = await budget.reserve("one", "first", input_tokens_upper=500, output_tokens_limit=500)
    waiting = asyncio.create_task(
        budget.reserve("one", "queued", input_tokens_upper=500, output_tokens_limit=500)
    )
    await asyncio.sleep(0)
    await budget.settle(first, usage={"input_tokens": 500, "output_tokens": 3500})
    with pytest.raises(BudgetExceeded, match="monetary"):
        await waiting
    assert budget.attempts[1]["status"] == "not_sent"
