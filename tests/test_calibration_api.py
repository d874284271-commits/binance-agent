"""Developer evaluation and replay are not part of the released HTTP product."""

import httpx
import pytest

from binance_agent.app.server import create_app
from binance_agent.infra.settings import Settings


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/outcomes/stats"),
        ("get", "/outcomes/recent"),
        ("get", "/watchlist"),
        ("get", "/alerts/recent"),
        ("post", "/config/calibration/apply"),
        ("post", "/watchlist/example/check"),
        ("post", "/runs/replay"),
    ],
)
async def test_removed_engineering_routes(method, path):
    app = create_app(Settings())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await getattr(client, method)(path)
    assert response.status_code in {404, 405}
