"""Offline lifecycle regressions for partial initialization and resource ownership."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from binance_agent.app import server
from binance_agent.infra.settings import Settings
from binance_agent.infra.startup_errors import StartupError


class Resource:
    def __init__(self, name, closed, *, fail=False):
        self.name, self.closed, self.fail = name, closed, fail

    async def aclose(self):
        self.closed.append(self.name)
        if self.fail:
            raise OSError("close failed")


def demo_app(tmp_path, monkeypatch):
    monkeypatch.setenv("BINANCE_AGENT_DATA_DIR", str(tmp_path))
    return server.create_app(
        Settings.model_validate({"profile": "demo", "collaboration": {"enabled": False}})
    )


async def test_lifespan_cleanup_failure_does_not_skip_other_resources(tmp_path, monkeypatch):
    app = demo_app(tmp_path, monkeypatch)
    closed = []
    names = (
        "run_store",
        "rest_client",
        "spot_client",
        "intel_http",
        "llm_registry",
        "graph_checkpointer",
    )

    async def failed_shutdown():
        raise OSError("manager cleanup failed")

    async with app.router.lifespan_context(app):
        for name in names:
            setattr(app.state, name, Resource(name, closed, fail=True))
        app.state.research_manager = SimpleNamespace(
            shutdown=failed_shutdown, pending_shutdown=set()
        )
    assert closed == list(names)


async def test_lifespan_partial_initialization_closes_already_created_client(tmp_path, monkeypatch):
    monkeypatch.setenv("BINANCE_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BINANCE_AGENT_RUNTIME_CONFIG", str(tmp_path / "absent.json"))
    closed = []

    def client(market, **kwargs):
        if market == "api":
            raise OSError("second client initialization failed")
        return Resource("fapi", closed)

    monkeypatch.setattr(server, "BinanceRestClient", client)
    app = server.create_app(Settings())
    with pytest.raises(StartupError):
        async with app.router.lifespan_context(app):
            pytest.fail("initialization must fail")
    assert closed == ["fapi"]
    assert isinstance(app.state.startup_error, StartupError)


async def test_lifespan_bad_configuration_preserves_original_and_names_field(tmp_path, monkeypatch):
    path = tmp_path / "runtime_config.json"
    original = b'{"llm":{"active":123}}'
    path.write_bytes(original)
    monkeypatch.setenv("BINANCE_AGENT_RUNTIME_CONFIG", str(path))
    monkeypatch.setenv("BINANCE_AGENT_DATA_DIR", str(tmp_path))
    settings = Settings()
    baseline = settings.model_dump()
    app = server.create_app(settings)
    with pytest.raises(StartupError) as error:
        async with app.router.lifespan_context(app):
            pytest.fail("configuration must fail before startup")
    assert path.read_bytes() == original
    assert "llm.active" in str(error.value)
    assert str(path) in str(error.value)
    assert settings.model_dump() == baseline


async def test_deferred_cleanup_cancellation_keeps_worker_ownership_and_eventually_closes(
    tmp_path, monkeypatch
):
    app = demo_app(tmp_path, monkeypatch)
    closed = []
    release = asyncio.Event()

    async def worker():
        await release.wait()

    async def shutdown():
        pass

    task = asyncio.create_task(worker())
    cleanup = None
    try:
        async with app.router.lifespan_context(app):
            app.state.rest_client = Resource("rest", closed)
            app.state.research_manager = SimpleNamespace(shutdown=shutdown, pending_shutdown={task})
        cleanup = app.state.pending_resource_cleanup
        await asyncio.sleep(0)
        assert not closed
        cleanup.cancel()
        await asyncio.sleep(0)
        assert not task.cancelled(), "cleanup cancellation must not cancel resource owners again"
        assert not closed, "active worker still owns the resource"
        release.set()
        await task
        await asyncio.gather(cleanup, return_exceptions=True)
        # The cancelled waiter does not own cleanup any more; the last worker's
        # callback retains and schedules the actual close task.
        await app.state.pending_resource_close
        assert closed == ["rest"]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if cleanup is not None:
            await asyncio.gather(cleanup, return_exceptions=True)
