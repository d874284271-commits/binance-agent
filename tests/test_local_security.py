from __future__ import annotations

import httpx

from binance_agent.app.server import create_app
from binance_agent.infra.settings import Settings


async def test_cross_site_origin_cannot_trigger_analyze() -> None:
    app = create_app(Settings())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:8765",
        trust_env=False,
    ) as client:
        response = await client.get(
            "/analyze",
            params={"symbol": "BTCUSDT"},
            headers={"Origin": "https://evil.example"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "cross-site request rejected"


async def test_cross_site_fetch_metadata_cannot_trigger_analyze_without_origin() -> None:
    app = create_app(Settings(app={"max_concurrent_runs": 1}))
    reservation = await app.state.analysis_limiter.try_acquire()
    assert reservation is not None
    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1:8765",
            trust_env=False,
        ) as client:
            response = await client.get(
                "/analyze",
                params={"symbol": "BTCUSDT"},
                headers={"Sec-Fetch-Site": "cross-site"},
            )
    finally:
        await reservation.release()

    assert response.status_code == 403
    assert response.json()["detail"] == "cross-site request rejected"


async def test_cross_site_fetch_metadata_cannot_trigger_dependency_check() -> None:
    app = create_app(Settings())
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:8765",
        trust_env=False,
    ) as client:
        response = await client.get(
            "/health/deps",
            headers={"Sec-Fetch-Site": "cross-site"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "cross-site request rejected"


async def test_non_loopback_host_header_is_rejected() -> None:
    app = create_app(Settings())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://evil.example",
        trust_env=False,
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 400


async def test_ipv6_loopback_host_is_allowed() -> None:
    app = create_app(Settings())
    transport = httpx.ASGITransport(app=app, client=("::1", 1234))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://[::1]",
        trust_env=False,
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200


async def test_analyze_returns_429_when_local_concurrency_is_exhausted() -> None:
    app = create_app(Settings(app={"max_concurrent_runs": 1}))
    reservation = await app.state.analysis_limiter.try_acquire()
    assert reservation is not None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:8765",
        trust_env=False,
    ) as client:
        response = await client.get("/analyze", params={"symbol": "BTCUSDT"})

    await reservation.release()
    assert response.status_code == 429
    assert response.json()["detail"] == "too many concurrent analyses"


async def test_cli_style_request_without_origin_requires_loopback_client() -> None:
    app = create_app(Settings())
    transport = httpx.ASGITransport(app=app, client=("10.0.0.8", 4567))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:8765",
        trust_env=False,
    ) as client:
        response = await client.get("/config")

    assert response.status_code == 403
    assert response.json()["detail"] == "non-loopback client rejected"


async def test_loopback_cli_without_browser_metadata_can_run_dependency_check() -> None:
    app = create_app(Settings(profile="demo"))

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 4567))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1:8765",
            trust_env=False,
        ) as client:
            response = await client.get("/health/deps")

    assert response.status_code == 200
    assert response.json()["deep"] is False
