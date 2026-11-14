from __future__ import annotations

import httpx

from binance_agent.app.server import create_app
from binance_agent.infra.settings import LlmConfig, LlmEndpoint, Settings


async def test_health_deps_returns_ui_expected_shape(monkeypatch) -> None:
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
    app.state.llm_registry = object()
    app.state.aggregator = type("_Aggregator", (), {"_disabled": {"rss": "未配置 RSS feeds"}})()

    monkeypatch.setattr(
        "binance_agent.app.server.resolve_secret",
        lambda ref: "key" if ref == "A" else None,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        response = await client.get("/health/deps")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "warn"}
    assert body["binance"]["status"] in {"ok", "warn"}
    assert body["llm"]["status"] == "warn"
    assert body["llm"]["endpoints"][0]["name"] == "a"
    assert body["llm"]["endpoints"][0]["status"] == "ok"
    assert body["llm"]["endpoints"][1]["status"] == "warn"
    assert body["intel"]["sources"][0]["source"] == "rss"
