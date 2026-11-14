"""Canceling both a cleanup waiter and its child must terminate at loop shutdown."""

from __future__ import annotations

import subprocess
import sys

import pytest

from binance_agent.app.security import AnalysisLimiter


@pytest.mark.parametrize("resource", ["stream", "reservation"])
def test_asyncio_shutdown_cancels_cleanup_without_spinning(resource: str) -> None:
    script = """
import asyncio
import sys
from binance_agent.app.streaming import ContextIterator
from binance_agent.app.security import AnalysisLimiter

async def main():
    if sys.argv[1] == "stream":
        async def events():
            try:
                yield 1
            finally:
                await asyncio.sleep(0.05)
        resource = ContextIterator(events())
        await anext(resource)
        asyncio.create_task(resource.aclose())
    else:
        limiter = AnalysisLimiter(1)
        resource = await limiter.try_acquire()
        await limiter._lock.acquire()
        asyncio.create_task(resource.release())
    # Let both the waiter and its cleanup task start before runner shutdown.
    for _ in range(4):
        await asyncio.sleep(0)

asyncio.run(main())
print("shutdown complete")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, resource], capture_output=True, text=True, timeout=3
    )
    assert result.returncode == 0, result.stderr
    assert "shutdown complete" in result.stdout


async def test_cancelled_release_child_can_be_retried_without_double_release() -> None:
    import asyncio

    limiter = AnalysisLimiter(1)
    reservation = await limiter.try_acquire()
    assert reservation is not None
    await limiter._lock.acquire()
    waiter = asyncio.create_task(reservation.release())
    await asyncio.sleep(0)
    assert reservation._release_task is not None
    reservation._release_task.cancel()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not reservation._released
    assert limiter._active == 1
    limiter._lock.release()
    await asyncio.gather(reservation.release(), reservation.release())
    assert reservation._released
    assert limiter._active == 0


async def test_cancelled_stream_waiter_drains_generator_before_return() -> None:
    import asyncio

    from binance_agent.app.streaming import ContextIterator

    started, finish, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def events():
        try:
            yield 1
        finally:
            started.set()
            await finish.wait()
            stopped.set()

    stream = ContextIterator(events())
    await anext(stream)
    waiter = asyncio.create_task(stream.aclose())
    await started.wait()
    waiter.cancel()
    await asyncio.sleep(0)
    assert not waiter.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert stopped.is_set()
