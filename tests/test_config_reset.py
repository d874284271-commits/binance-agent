"""/config 配置来源透明化 + /config/reset 重置端点测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from binance_agent.app.server import create_app
from binance_agent.infra.settings import LlmConfig, LlmEndpoint, Settings
from binance_agent.llm.registry import LlmRegistry


def _stub_aggregator() -> object:
    return type("_Agg", (), {"_disabled": {}})()


def _build_app(monkeypatch: Any) -> Any:
    settings = Settings(
        llm=LlmConfig(
            active_endpoint="a",
            endpoints=[
                LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A", model="ma"),
                LlmEndpoint(name="b", base_url="https://b/v1", api_key_ref="B", model="mb"),
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
    monkeypatch.setattr("binance_agent.app.server.build_graph", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr("binance_agent.app.server._build_aggregator", lambda *a: _stub_aggregator())
    monkeypatch.setattr("binance_agent.app.server.resolve_secret", lambda ref: None)
    return app


async def test_config_exposes_has_overrides_flag(monkeypatch: Any) -> None:
    app = _build_app(monkeypatch)
    monkeypatch.setattr("binance_agent.app.server.overrides_present", lambda: True)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        r = await client.get("/config")
    assert r.status_code == 200
    assert r.json()["has_overrides"] is True


async def test_config_exposes_source_catalog(monkeypatch: Any) -> None:
    app = _build_app(monkeypatch)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        response = await client.get("/config")

    assert response.status_code == 200
    catalog = {row["source"]: row for row in response.json()["intel"]["catalog"]}
    assert catalog["sec_rss"]["source_kind"] == "official"
    assert catalog["guardian_api"]["requires_key"] is True


async def test_config_reset_restores_base_active_and_clears(monkeypatch: Any) -> None:
    app = _build_app(monkeypatch)
    # 模拟用户已切到 b(运行时覆盖)
    app.state.llm_registry.switch_active("b")
    assert app.state.llm_registry.active == "b"

    base = Settings(
        llm=LlmConfig(
            active_endpoint="a",
            endpoints=[
                LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A", model="ma"),
                LlmEndpoint(name="b", base_url="https://b/v1", api_key_ref="B", model="mb"),
            ],
        )
    )
    monkeypatch.setattr("binance_agent.app.server.load_settings", lambda: base)
    cleared = {"n": 0}
    monkeypatch.setattr(
        "binance_agent.app.server.clear_overrides",
        lambda: cleared.update(n=cleared["n"] + 1),
    )
    monkeypatch.setattr("binance_agent.app.server.overrides_present", lambda: False)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        r = await client.post("/config/reset")

    assert r.status_code == 200
    body = r.json()
    assert body["reset"] is True
    assert body["active"] == "a"  # 恢复基底 active
    assert body["has_overrides"] is False
    assert cleared["n"] == 1  # 调用了 clear_overrides
    assert app.state.llm_registry.active == "a"  # registry 已重载到基底


async def test_endpoint_structured_output_can_toggle_and_omission_preserves(
    monkeypatch: Any,
) -> None:
    app = _build_app(monkeypatch)
    monkeypatch.setattr("binance_agent.app.server.save_overrides", lambda _data: None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        enabled = await client.post(
            "/config/llm/endpoint",
            json={
                "name": "a",
                "base_url": "https://a/v1",
                "model": "ma",
                "structured_output": True,
            },
        )
        preserved = await client.post(
            "/config/llm/endpoint",
            json={"name": "a", "base_url": "https://a/v2", "model": "ma2"},
        )

    assert enabled.status_code == 200
    assert enabled.json()["endpoints"][0]["structured_output"] is True
    assert preserved.status_code == 200
    endpoint = next(item for item in preserved.json()["endpoints"] if item["name"] == "a")
    assert endpoint["structured_output"] is True
