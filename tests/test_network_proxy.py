"""统一出站代理测试:overrides 贯通 + registry 缓存失效 + /config/network 端点 + reset 还原。"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from binance_agent.app.server import _current_overrides, create_app
from binance_agent.infra.runtime_config import apply_overrides
from binance_agent.infra.settings import LlmConfig, LlmEndpoint, NetworkConfig, Settings
from binance_agent.llm.registry import LlmRegistry
from binance_agent.market.rest_client import BinanceRestClient

# ---------- runtime_config 贯通 ----------


def test_apply_overrides_sets_network_proxy() -> None:
    settings = Settings()
    apply_overrides(settings, {"network": {"proxy": "http://127.0.0.1:7897"}})
    assert settings.network.proxy == "http://127.0.0.1:7897"


def test_apply_overrides_empty_proxy_clears() -> None:
    settings = Settings(network=NetworkConfig(proxy="http://old:1"))
    apply_overrides(settings, {"network": {"proxy": ""}})
    assert settings.network.proxy is None


def test_apply_overrides_without_network_keeps_default() -> None:
    settings = Settings(network=NetworkConfig(proxy="http://keep:1"))
    apply_overrides(settings, {"params": {"time_window_hours": 12}})
    assert settings.network.proxy == "http://keep:1"


def test_apply_overrides_rejects_proxy_credentials() -> None:
    settings = Settings()

    with pytest.raises(ValueError, match="credentials"):
        apply_overrides(
            settings,
            {"network": {"proxy": "http://alice:secret@127.0.0.1:7897"}},
        )


def test_current_overrides_persists_network_proxy() -> None:
    settings = Settings(network=NetworkConfig(proxy="http://127.0.0.1:7897"))
    reg = LlmRegistry(settings.llm, model_factory=lambda e, m: FakeListChatModel(responses=["{}"]))
    data = _current_overrides(settings, reg)
    assert data["network"] == {"proxy": "http://127.0.0.1:7897"}


# ---------- registry 缓存失效 ----------


def _counting_registry() -> tuple[LlmRegistry, dict[str, int]]:
    calls = {"n": 0}

    def factory(endpoint: LlmEndpoint, model: str) -> FakeListChatModel:
        calls["n"] += 1
        return FakeListChatModel(responses=["{}"])

    config = LlmConfig(
        active_endpoint="a",
        endpoints=[LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A", model="ma")],
    )
    return LlmRegistry(config, model_factory=factory), calls


def test_set_proxy_change_invalidates_model_cache() -> None:
    reg, calls = _counting_registry()
    reg.for_agent("technical")
    reg.for_agent("technical")
    assert calls["n"] == 1  # 命中缓存

    reg.set_proxy("http://127.0.0.1:7897")
    reg.for_agent("technical")
    assert calls["n"] == 2  # 代理变更 → 缓存清空,重建


def test_set_proxy_same_value_keeps_cache() -> None:
    reg, calls = _counting_registry()
    reg.set_proxy("http://p:1")
    reg.for_agent("technical")
    reg.set_proxy("http://p:1")  # 同值,no-op
    reg.for_agent("technical")
    assert calls["n"] == 1


# ---------- rest_client 透传 ----------


async def test_rest_client_accepts_explicit_proxy() -> None:
    client = BinanceRestClient("fapi", proxy="http://127.0.0.1:7897")
    try:
        assert client.market == "fapi"
    finally:
        await client.aclose()


# ---------- /config/network 端点 + /config/reset 还原 ----------


def _stub_aggregator() -> object:
    return type("_Agg", (), {"_disabled": {}})()


def _build_app(monkeypatch: Any) -> Any:
    settings = Settings(
        llm=LlmConfig(
            active_endpoint="a",
            endpoints=[
                LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A", model="ma"),
            ],
        )
    )
    app = create_app(settings)
    app.state.llm_registry = LlmRegistry(
        settings.llm, model_factory=lambda endpoint, model: FakeListChatModel(responses=["{}"])
    )
    app.state.aggregator = _stub_aggregator()
    app.state.intel_http = object()
    app.state.rest_client = object()
    app.state.spot_client = object()
    monkeypatch.setattr("binance_agent.app.server.build_graph", lambda *a, **k: object())
    monkeypatch.setattr("binance_agent.app.server._build_aggregator", lambda *a: _stub_aggregator())
    monkeypatch.setattr("binance_agent.app.server.resolve_secret", lambda ref: None)
    return app


async def test_set_network_persists_and_config_exposes(monkeypatch: Any) -> None:
    app = _build_app(monkeypatch)
    saved: list[dict[str, Any]] = []
    monkeypatch.setattr("binance_agent.app.server.save_overrides", lambda d: saved.append(d))
    monkeypatch.setattr("binance_agent.app.server.overrides_present", lambda: True)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        r = await client.post("/config/network", json={"proxy": " http://127.0.0.1:7897 "})
        assert r.status_code == 200
        assert r.json() == {"proxy": "http://127.0.0.1:7897"}  # 去除首尾空白

        cfg = (await client.get("/config")).json()
        assert cfg["network"] == {"proxy": "http://127.0.0.1:7897"}

    assert saved and saved[-1]["network"] == {"proxy": "http://127.0.0.1:7897"}
    # 出站客户端已按新代理重建(不再是占位 object)
    assert isinstance(app.state.rest_client, BinanceRestClient)
    await app.state.rest_client.aclose()
    await app.state.spot_client.aclose()
    await app.state.intel_http.aclose()


async def test_set_network_empty_clears_proxy(monkeypatch: Any) -> None:
    app = _build_app(monkeypatch)
    app.state.settings.network.proxy = "http://old:1"
    monkeypatch.setattr("binance_agent.app.server.save_overrides", lambda d: None)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        r = await client.post("/config/network", json={"proxy": "  "})
    assert r.status_code == 200
    assert r.json() == {"proxy": ""}
    assert app.state.settings.network.proxy is None
    await app.state.rest_client.aclose()
    await app.state.spot_client.aclose()
    await app.state.intel_http.aclose()


async def test_set_network_rejects_proxy_credentials(monkeypatch: Any) -> None:
    app = _build_app(monkeypatch)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        response = await client.post(
            "/config/network",
            json={"proxy": "http://alice:secret@127.0.0.1:7897"},
        )

    assert response.status_code == 422
    assert "credentials" in response.json()["detail"]


async def test_config_reset_restores_base_proxy(monkeypatch: Any) -> None:
    app = _build_app(monkeypatch)
    app.state.settings.network.proxy = "http://runtime-override:7897"

    base = Settings(
        llm=LlmConfig(
            active_endpoint="a",
            endpoints=[
                LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A", model="ma"),
            ],
        )
    )  # 基底未配代理
    monkeypatch.setattr("binance_agent.app.server.load_settings", lambda: base)
    monkeypatch.setattr("binance_agent.app.server.clear_overrides", lambda: None)
    monkeypatch.setattr("binance_agent.app.server.overrides_present", lambda: False)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        r = await client.post("/config/reset")
    assert r.status_code == 200
    assert app.state.settings.network.proxy is None  # 回到基底(无代理)
    # 代理变更触发出站客户端重建
    assert isinstance(app.state.rest_client, BinanceRestClient)
    await app.state.rest_client.aclose()
    await app.state.spot_client.aclose()
    await app.state.intel_http.aclose()
