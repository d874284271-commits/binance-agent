"""A failed SSE send must stop real graph fanout before surrendering admission."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from sse_starlette.sse import EventSourceResponse

from binance_agent.app.run_store import FileRunStore
from binance_agent.app.security import AnalysisLifecycleMiddleware, AnalysisLimiter
from binance_agent.app.server import _stream_graph_events
from binance_agent.app.streaming import ContextIterator


class State(TypedDict, total=False):
    fast: bool
    slow: bool


@pytest.mark.parametrize("cancel_request", [False, True])
async def test_graph_fanout_stops_before_admission_release(
    tmp_path: Path, cancel_request: bool
) -> None:
    slow_started = asyncio.Event()
    slow_stopped = asyncio.Event()
    allow_finish = asyncio.Event()
    first_update_sent = asyncio.Event()

    async def fast(state: State) -> State:
        await slow_started.wait()
        return {"fast": True}

    async def slow(state: State) -> State:
        slow_started.set()
        try:
            await allow_finish.wait()
            return {"slow": True}
        finally:
            slow_stopped.set()

    builder = StateGraph(State)
    builder.add_node("first", fast)
    builder.add_node("second", slow)
    for node in ("first", "second"):
        builder.add_edge(START, node)
        builder.add_edge(node, END)
    graph = builder.compile()
    store = FileRunStore(tmp_path / "runs.jsonl")
    run_id = store.create(symbol="BTCUSDT", market="fapi")
    limiter = AnalysisLimiter(1)
    released_after_stop: list[bool] = []
    original_release = limiter._release

    async def release() -> None:
        released_after_stop.append(slow_stopped.is_set())
        await original_release()

    limiter._release = release
    responses: list[Any] = []  # Keep iterators alive: GC must not perform essential cleanup.

    async def handler(scope: Any, receive: Any, send: Any) -> None:
        reservation = await limiter.try_acquire()
        assert reservation is not None
        stream = _stream_graph_events(
            graph=graph, state={}, run_id=run_id, run_store=store, checkpointer=None
        )
        driver = ContextIterator(stream)
        reservation.close_stream = driver.aclose
        response = EventSourceResponse(driver)
        responses.append(response)
        await response(scope, receive, send)

    async def receive() -> Any:
        await asyncio.Event().wait()

    async def send(message: Any) -> None:
        if message["type"] != "http.response.body":
            return
        assert b"event: node" in message["body"]
        first_update_sent.set()
        if cancel_request:
            await asyncio.Event().wait()
        raise OSError("send failed at first graph update")

    request = asyncio.create_task(
        AnalysisLifecycleMiddleware(handler)({"type": "http"}, receive, send)
    )
    try:
        await asyncio.wait_for(first_update_sent.wait(), 2)
        if cancel_request:
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        else:
            with pytest.raises(OSError, match="send failed"):
                await request
        assert limiter._active == 0
        assert released_after_stop == [True]
        assert slow_stopped.is_set()
        driver_task = responses[0].body_iterator._driver
        assert driver_task.done()
        if not driver_task.cancelled():
            driver_task.result()  # No cross-Context reset exception may hide in close().
    finally:
        allow_finish.set()
        if not request.done():
            request.cancel()
        await asyncio.gather(request, return_exceptions=True)
        for response in responses:
            with contextlib.suppress(Exception):
                await response.body_iterator.aclose()
        await asyncio.sleep(0)
