from __future__ import annotations

import httpx
import pytest

from binance_agent.app.server import create_app
from binance_agent.infra.settings import Settings


@pytest.mark.parametrize("collaboration", [True, False])
@pytest.mark.parametrize("use_for_all_roles", [True, False, None])
async def test_first_use_without_models_can_open_config_and_example(
    tmp_path, monkeypatch, collaboration, use_for_all_roles
):
    monkeypatch.setenv("BINANCE_AGENT_DATA_DIR", str(tmp_path))
    settings = Settings.model_validate(
        {
            "llm": {"endpoints": [], "active": ""},
            "research": {"enabled": True, "max_requests": 100, "deadline_s": 600},
            "collaboration": {"enabled": collaboration},
            "runtime": {"checkpoint_backend": "sqlite"},
        }
    )
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client,
    ):
        for path in ("/health", "/config", "/reports/example", "/runs"):
            assert (await client.get(path)).status_code == 200
        response = await client.get("/analyze", params={"symbol": "BTCUSDT"})
        assert response.status_code == 409
        assert "模型" in response.text
        assert app.state.run_store.list_all() == []
        response = await client.post(
            "/config/llm/endpoint",
            json={
                "name": "local",
                "base_url": "http://127.0.0.1:12345/v1",
                "model": "test-model",
                **(
                    {"use_for_all_roles": use_for_all_roles}
                    if use_for_all_roles is not None
                    else {}
                ),
            },
        )
        assert response.status_code == 200, response.text
        assert app.state.graph is not None
