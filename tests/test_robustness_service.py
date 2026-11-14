from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi.routing import APIRoute, APIWebSocketRoute

from binance_agent.app.run_store import FileRunStore
from binance_agent.app.security import AnalysisLifecycleMiddleware, AnalysisLimiter
from binance_agent.app.server import create_app
from binance_agent.infra.settings import Settings


class SlowGraph:
    async def astream(self, state: Any, *, stream_mode: str) -> Any:
        await asyncio.Event().wait()
        yield {}


def legacy_app(tmp_path: Path) -> Any:
    from test_server_runs import _Aggregator, _TickerClient

    app = create_app(Settings(collaboration={"enabled": False}))
    app.state.run_store = FileRunStore(tmp_path / "runs.jsonl")
    app.state.rest_client = _TickerClient()
    app.state.aggregator = _Aggregator()
    app.state.graph = SlowGraph()
    return app


@asynccontextmanager
async def live_server(app: Any):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(200):
            if server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        assert server.started
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        sock.close()


async def idle(app: Any) -> None:
    for _ in range(100):
        if app.state.analysis_limiter._active == 0:
            break
        await asyncio.sleep(0.01)
    assert app.state.analysis_limiter._active == 0
    assert not app.state.active_research_runs


async def test_real_uvicorn_creation_failure_is_503_and_next_request_admitted(
    tmp_path: Path,
) -> None:
    app = legacy_app(tmp_path)

    def fail(**kwargs: Any) -> str:
        raise PermissionError("read-only ledger")

    app.state.run_store.create = fail
    async with live_server(app) as url, httpx.AsyncClient(base_url=url, trust_env=False) as client:
        for _ in range(3):
            response = await client.get("/analyze", params={"symbol": "BTCUSDT"})
            assert response.status_code == 503
            await idle(app)
    assert app.state.run_store.list_all() == []


async def test_real_uvicorn_first_event_disconnect_releases_slot(tmp_path: Path) -> None:
    app = legacy_app(tmp_path)
    async with live_server(app) as url, httpx.AsyncClient(base_url=url, trust_env=False) as client:
        for _ in range(3):
            async with client.stream("GET", "/analyze?symbol=BTCUSDT") as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        break
            await idle(app)
    assert all(r["status"] == "failed" for r in app.state.run_store.list_all())


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_asgi_send_failure_does_not_leak(tmp_path: Path, cleanup_fails: bool) -> None:
    app = legacy_app(tmp_path)
    if cleanup_fails:

        def fail(*a: Any, **kw: Any) -> None:
            raise OSError("disk full during cleanup")

        app.state.run_store.fail = fail
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/analyze",
        "raw_path": b"/analyze",
        "query_string": b"symbol=BTCUSDT",
        "headers": [(b"host", b"127.0.0.1")],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 80),
    }
    sent_request = False

    async def receive() -> Any:
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.Event().wait()

    async def send(message: Any) -> None:
        if message["type"] == "http.response.body":
            raise OSError("send failed")

    with pytest.raises(Exception, match=r"send failed|unhandled errors"):
        await asyncio.wait_for(app(scope, receive, send), 2)
    await idle(app)


async def test_cancelled_concurrent_release_finishes_before_return() -> None:
    limiter = AnalysisLimiter(1)
    reservation = await limiter.try_acquire()
    assert reservation is not None
    await limiter._lock.acquire()
    first = asyncio.create_task(reservation.release())
    second = asyncio.create_task(reservation.release())
    await asyncio.sleep(0)
    first.cancel()
    limiter._lock.release()
    result = await asyncio.gather(first, second, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError)
    assert limiter._active == 0
    assert reservation._released
    await reservation.release()
    assert limiter._active == 0


async def test_request_cancel_before_response_releases_owned_reservation() -> None:
    limiter = AnalysisLimiter(1)
    started = asyncio.Event()

    async def app(scope: Any, receive: Any, send: Any) -> None:
        await limiter.try_acquire()
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(AnalysisLifecycleMiddleware(app)({"type": "http"}, None, None))
    await started.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert limiter._active == 0


async def test_config_save_failure_keeps_live_config_and_registry(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from test_config_reset import _build_app

    app = _build_app(monkeypatch)
    before = app.state.settings.model_dump()
    registry = app.state.llm_registry
    graph = object()
    app.state.graph = graph

    def fail(data: Any) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("binance_agent.app.server.save_overrides", fail)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://127.0.0.1",
    ) as client:
        response = await client.post("/config/llm-endpoint/switch", json={"endpoint": "b"})
    assert response.status_code == 500
    assert app.state.settings.model_dump() == before
    assert app.state.llm_registry is registry
    assert registry.active == "a"
    assert app.state.graph is graph


async def test_configuration_and_admission_share_gate(monkeypatch: Any) -> None:
    from test_config_reset import _build_app

    app = _build_app(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        reservation = await app.state.analysis_limiter.try_acquire()
        assert reservation is not None
        response = await client.post("/config/llm-endpoint/switch", json={"endpoint": "b"})
        assert response.status_code == 409
        await reservation.release()
        async with app.state.analysis_limiter.reconfigure():
            assert await app.state.analysis_limiter.try_acquire() is None
            response = await client.post("/config/llm-endpoint/switch", json={"endpoint": "b"})
            assert response.status_code == 409


async def test_saved_model_key_cannot_leave_configured_origin(monkeypatch: Any) -> None:
    from test_config_reset import _build_app

    app = _build_app(monkeypatch)
    keys: list[str] = []
    monkeypatch.setattr(
        "binance_agent.app.server.resolve_secret", lambda ref: keys.append(ref) or "secret"
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            "/config/llm/models", json={"endpoint": "a", "base_url": "https://evil.example/v1"}
        )
    assert response.status_code == 422
    assert keys == []


@pytest.fixture
def endpoint_security(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """所有凭证、配置落盘与出站请求均留在内存中,验证真实配置路由。"""
    from test_config_reset import _build_app

    app = _build_app(monkeypatch)
    secrets = {"A": "fake-old-key", "B": "fake-other-key"}
    effects: list[str] = []
    saved: list[dict[str, Any]] = []
    requests: list[httpx.Request] = []

    def resolve(ref: str) -> str | None:
        effects.append("read-secret")
        return secrets.get(ref)

    def store(ref: str, value: str) -> bool:
        effects.append("write-secret")
        secrets[ref] = value
        return True

    def save(data: dict[str, Any]) -> None:
        effects.append("save-config")
        saved.append(data)

    def respond(request: httpx.Request) -> httpx.Response:
        effects.append("http-request")
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": "fixture-model"}]})

    monkeypatch.setattr("binance_agent.app.server.resolve_secret", resolve)
    monkeypatch.setattr("binance_agent.llm.registry.resolve_secret", resolve)
    monkeypatch.setattr("binance_agent.app.server.store_secret", store)
    monkeypatch.setattr("binance_agent.app.server.save_overrides", save)
    return {
        "app": app,
        "secrets": secrets,
        "effects": effects,
        "saved": saved,
        "requests": requests,
        "transport": httpx.MockTransport(respond),
    }


@pytest.mark.parametrize(
    "base_url,key_fields",
    [
        ("https://other.example/v1", {}),
        ("http://a/v1", {}),
        ("https://a:8443/v1", {}),
        ("https://a:0/v1", {}),
        ("https://a:invalid/v1", {}),
        ("https://other.example/v1", {"api_key": None}),
        ("https://other.example/v1", {"api_key": ""}),
        ("https://other.example/v1", {"api_key": "   "}),
    ],
)
async def test_endpoint_origin_change_requires_new_key_without_side_effects(
    endpoint_security: dict[str, Any], base_url: str, key_fields: dict[str, Any]
) -> None:
    app = endpoint_security["app"]
    before = app.state.settings.model_dump()
    registry = app.state.llm_registry
    graph = object()
    app.state.graph = graph
    secrets_before = dict(endpoint_security["secrets"])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1", trust_env=False
    ) as client:
        response = await client.post(
            "/config/llm/endpoint",
            json={"name": "a", "base_url": base_url, "model": "new-model", **key_fields},
        )
    assert response.status_code == 422
    assert "重新提供 API key" in response.json()["detail"]
    assert "新建端点" in response.json()["detail"]
    assert app.state.settings.model_dump() == before
    assert app.state.llm_registry is registry
    assert app.state.graph is graph
    assert endpoint_security["secrets"] == secrets_before
    assert endpoint_security["effects"] == []
    assert endpoint_security["saved"] == []


async def test_endpoint_edit_cannot_retarget_saved_key_before_model_lookup(
    endpoint_security: dict[str, Any],
) -> None:
    app = endpoint_security["app"]
    before = app.state.settings.model_dump()
    target = "https://other.example/v1"
    async with httpx.AsyncClient(
        transport=endpoint_security["transport"], trust_env=False
    ) as outbound:
        app.state.intel_http = outbound
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1", trust_env=False
        ) as client:
            changed = await client.post(
                "/config/llm/endpoint",
                json={"name": "a", "base_url": target, "model": "ma"},
            )
            models = await client.post(
                "/config/llm/models", json={"endpoint": "a", "base_url": target}
            )
    assert changed.status_code == models.status_code == 422
    assert app.state.settings.model_dump() == before
    assert endpoint_security["requests"] == []
    assert endpoint_security["effects"] == []


@pytest.mark.parametrize(
    "current_url,edited_url",
    [
        ("https://a/v1", "https://a/v2"),
        ("https://a/v1", "HTTPS://A:443/v2"),
        ("https://a:443/v1", "https://a/v2"),
        ("http://a/v1", "http://a:80/v2"),
        ("https://a:8443/v1", "https://a:8443/v2"),
    ],
)
async def test_same_origin_endpoint_edit_preserves_saved_key(
    endpoint_security: dict[str, Any], current_url: str, edited_url: str
) -> None:
    app = endpoint_security["app"]
    app.state.settings.llm.endpoints[0].base_url = current_url
    secrets_before = dict(endpoint_security["secrets"])
    async with httpx.AsyncClient(
        transport=endpoint_security["transport"], trust_env=False
    ) as outbound:
        app.state.intel_http = outbound
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1", trust_env=False
        ) as client:
            changed = await client.post(
                "/config/llm/endpoint",
                json={"name": "a", "base_url": edited_url, "model": "new-model", "api_key": ""},
            )
            models = await client.post(
                "/config/llm/models", json={"endpoint": "a", "base_url": current_url}
            )
    assert changed.status_code == models.status_code == 200
    endpoint = app.state.settings.llm.endpoints[0]
    assert endpoint.base_url == edited_url
    assert endpoint.model == "new-model"
    assert endpoint.api_key_ref == "A"
    assert endpoint_security["secrets"] == secrets_before
    assert "write-secret" not in endpoint_security["effects"]
    assert len(endpoint_security["saved"]) == 1
    assert endpoint_security["requests"][0].headers["authorization"] == "Bearer fake-old-key"


async def test_endpoint_origin_change_with_new_key_keeps_old_secret(
    endpoint_security: dict[str, Any],
) -> None:
    app = endpoint_security["app"]
    target = "https://other.example/v1"
    async with httpx.AsyncClient(
        transport=endpoint_security["transport"], trust_env=False
    ) as outbound:
        app.state.intel_http = outbound
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1", trust_env=False
        ) as client:
            changed = await client.post(
                "/config/llm/endpoint",
                json={"name": "a", "base_url": target, "model": "ma", "api_key": "fake-new-key"},
            )
            models = await client.post(
                "/config/llm/models", json={"endpoint": "a", "base_url": target}
            )
    assert changed.status_code == models.status_code == 200
    endpoint = app.state.settings.llm.endpoints[0]
    assert endpoint.base_url == target
    assert endpoint.api_key_ref != "A"
    assert endpoint_security["secrets"]["A"] == "fake-old-key"
    assert endpoint_security["secrets"][endpoint.api_key_ref] == "fake-new-key"
    assert endpoint_security["requests"][0].headers["authorization"] == "Bearer fake-new-key"
    assert len(endpoint_security["saved"]) == 1
    assert "fake-old-key" not in changed.text
    assert "fake-new-key" not in changed.text


async def test_new_keyless_endpoint_does_not_reuse_existing_secret(
    endpoint_security: dict[str, Any],
) -> None:
    app = endpoint_security["app"]
    target = "http://127.0.0.1:12345/v1"
    async with httpx.AsyncClient(
        transport=endpoint_security["transport"], trust_env=False
    ) as outbound:
        app.state.intel_http = outbound
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1", trust_env=False
        ) as client:
            changed = await client.post(
                "/config/llm/endpoint", json={"name": "local", "base_url": target, "model": "ma"}
            )
            models = await client.post(
                "/config/llm/models", json={"endpoint": "local", "base_url": target}
            )
    assert changed.status_code == models.status_code == 200
    assert "write-secret" not in endpoint_security["effects"]
    assert endpoint_security["secrets"] == {"A": "fake-old-key", "B": "fake-other-key"}
    assert "authorization" not in endpoint_security["requests"][0].headers


@pytest.mark.parametrize("delete_first", [False, True])
async def test_keyless_creation_cannot_reconnect_legacy_named_secret(
    endpoint_security: dict[str, Any], delete_first: bool
) -> None:
    app = endpoint_security["app"]
    name = "a" if delete_first else "retired"
    legacy_ref = f"llm_{name}_key"
    endpoint_security["secrets"][legacy_ref] = "fake-legacy-key"
    if delete_first:
        app.state.settings.llm.endpoints[0].api_key_ref = legacy_ref
        app.state.llm_registry.reload(app.state.settings.llm)
    secrets_before = dict(endpoint_security["secrets"])
    target = "https://other.example/v1"
    async with httpx.AsyncClient(
        transport=endpoint_security["transport"], trust_env=False
    ) as outbound:
        app.state.intel_http = outbound
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1", trust_env=False
        ) as client:
            if delete_first:
                deleted = await client.delete(f"/config/llm/endpoint/{name}")
                assert deleted.status_code == 200
            created = await client.post(
                "/config/llm/endpoint", json={"name": name, "base_url": target, "model": "ma"}
            )
            models = await client.post(
                "/config/llm/models", json={"endpoint": name, "base_url": target}
            )
    assert created.status_code == models.status_code == 200
    endpoint = next(e for e in app.state.settings.llm.endpoints if e.name == name)
    assert endpoint.api_key_ref != legacy_ref
    assert "authorization" not in endpoint_security["requests"][0].headers
    assert endpoint_security["secrets"] == secrets_before
    assert "write-secret" not in endpoint_security["effects"]


async def test_desktop_api_disables_generated_docs_but_keeps_ui_and_business_routes() -> None:
    app = create_app(Settings())
    routes = {
        (method, route.path)
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }
    routes.update(
        ("WEBSOCKET", route.path) for route in app.routes if isinstance(route, APIWebSocketRoute)
    )
    assert routes == {
        ("GET", "/"),
        ("GET", "/health"),
        ("GET", "/health/deps"),
        ("GET", "/snapshot"),
        ("GET", "/intel"),
        ("GET", "/runs/{run_id}/evidence/{evidence_id}"),
        ("GET", "/runs/{run_id}"),
        ("GET", "/runs"),
        ("GET", "/runs/{run_id}/resume"),
        ("GET", "/analyze"),
        ("POST", "/config/llm-endpoint/switch"),
        ("GET", "/config/llm-endpoint"),
        ("GET", "/config"),
        ("POST", "/config/reset"),
        ("POST", "/config/llm/endpoint"),
        ("POST", "/config/llm/models"),
        ("POST", "/config/llm/agent"),
        ("DELETE", "/config/llm/endpoint/{name}"),
        ("POST", "/config/intel/source"),
        ("POST", "/config/intel/key"),
        ("POST", "/config/params"),
        ("POST", "/config/network"),
        ("WEBSOCKET", "/ws/price"),
        ("POST", "/research/sessions"),
        ("GET", "/research/sessions"),
        ("GET", "/research/sessions/{session_id}"),
        ("POST", "/research/sessions/{session_id}/turns"),
        ("POST", "/research/sessions/{session_id}/messages"),
        ("GET", "/research/sessions/{session_id}/events"),
        ("POST", "/runs/{run_id}/resume"),
        ("POST", "/runs/{run_id}/stop"),
        ("POST", "/research/sessions/{session_id}/close"),
        ("GET", "/symbols"),
        ("GET", "/reports/example"),
        ("GET", "/runs/{run_id}/report"),
        ("GET", "/runs/{run_id}/report.md"),
        ("POST", "/config/desktop"),
        ("POST", "/config/research"),
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1", trust_env=False
    ) as client:
        for path in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"):
            assert (await client.get(path)).status_code == 404
        for path in ("/", "/ui/app.js", "/ui/styles.css", "/health"):
            assert (await client.get(path)).status_code == 200
        assert (
            await client.get("/config", headers={"Origin": "https://other.example"})
        ).status_code == 403


async def test_history_cold_decode_runs_in_worker_and_hot_poll_parses_nothing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import json
    import threading

    from binance_agent.app import run_store

    app = legacy_app(tmp_path)
    path = app.state.run_store._path
    with path.open("w") as handle:
        for index in range(10000):
            handle.write(
                json.dumps(
                    {"type": "created", "run_id": str(index), "symbol": "BTCUSDT", "market": "fapi"}
                )
                + "\n"
            )
    thread_ids: list[int] = []
    original = run_store.decode_jsonl

    def decode(*args: Any) -> Any:
        thread_ids.append(threading.get_ident())
        return original(*args)

    monkeypatch.setattr(run_store, "decode_jsonl", decode)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get("/runs")
        assert response.status_code == 200
        assert len(thread_ids) == 10000
        assert threading.get_ident() not in thread_ids
        thread_ids.clear()
        for _ in range(3):
            response = await client.get("/runs")
            assert response.status_code == 200
        assert thread_ids == []


@pytest.mark.parametrize("save_fails", [False, True])
async def test_intel_key_activation_follows_metadata_commit(
    monkeypatch: Any, save_fails: bool
) -> None:
    from binance_agent.infra.runtime_config import build_candidate
    from test_config_reset import _build_app

    app = _build_app(monkeypatch)
    settings = app.state.settings
    settings.intel.keys["cryptopanic"] = "old_key_ref"
    before = settings.model_copy(deep=True)
    secrets = {"old_key_ref": "old-value"}
    saved: dict[str, Any] = {}

    def store(ref: str, value: str) -> bool:
        secrets[ref] = value
        return True

    def save(data: Any) -> None:
        if save_fails:
            raise OSError("metadata replace failed")
        saved.update(data)

    monkeypatch.setattr("binance_agent.app.server.store_secret", store)
    monkeypatch.setattr("binance_agent.app.server.resolve_secret", secrets.get)
    monkeypatch.setattr("binance_agent.app.server.save_overrides", save)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://127.0.0.1",
    ) as client:
        response = await client.post(
            "/config/intel/key", json={"source": "cryptopanic", "api_key": "new-value"}
        )
    assert secrets["old_key_ref"] == "old-value"
    if save_fails:
        assert response.status_code == 500
        assert settings.model_dump() == before.model_dump()
    else:
        assert response.status_code == 200
        ref = settings.intel.keys["cryptopanic"]
        assert ref != "old_key_ref" and secrets[ref] == "new-value"
        assert build_candidate(before, saved).intel.keys["cryptopanic"] == ref


async def test_reset_synchronizes_registry_proxy(monkeypatch: Any) -> None:
    from test_config_reset import _build_app

    app = _build_app(monkeypatch)
    settings = app.state.settings
    settings.network.proxy = "http://127.0.0.1:8080"
    app.state.llm_registry.set_proxy(settings.network.proxy)

    class Resource:
        async def aclose(self) -> None:
            pass

    for name in ("rest_client", "spot_client", "intel_http"):
        setattr(app.state, name, Resource())
    monkeypatch.setattr("binance_agent.app.server.BinanceRestClient", lambda *a, **kw: Resource())
    monkeypatch.setattr("binance_agent.app.server.BoundedAsyncClient", lambda *a, **kw: Resource())
    monkeypatch.setattr("binance_agent.app.server.load_settings", lambda: Settings())
    monkeypatch.setattr("binance_agent.app.server.clear_overrides", lambda: None)
    monkeypatch.setattr("binance_agent.app.server.overrides_present", lambda: False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post("/config/reset")
    assert response.status_code == 200
    assert settings.network.proxy is None
    assert app.state.llm_registry._proxy is None


@pytest.mark.parametrize(
    "path",
    [
        "/snapshot?symbol=BTCUSDT",
        "/intel?symbol=BTCUSDT",
        "/watchlist",
        "/alerts/recent",
        "/outcomes/recent",
    ],
)
async def test_cross_site_read_endpoints_reject_before_dependencies(path: str) -> None:
    app = create_app(Settings())  # No dependencies: reaching upstream code would fail.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get(path, headers={"Origin": "https://evil.example"})
    assert response.status_code == 403


async def test_cross_site_websocket_rejected_before_upstream() -> None:
    app = create_app(Settings())
    sent: list[dict[str, Any]] = []
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "scheme": "ws",
        "path": "/ws/price",
        "raw_path": b"/ws/price",
        "query_string": b"symbol=BTCUSDT",
        "headers": [(b"host", b"127.0.0.1"), (b"origin", b"https://evil.example")],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 80),
        "subprotocols": [],
    }

    async def receive() -> Any:
        return {"type": "websocket.connect"}

    async def send(message: Any) -> None:
        sent.append(message)

    await app(scope, receive, send)
    assert sent == [
        {"type": "websocket.close", "code": 1008, "reason": "cross-site request rejected"}
    ]
