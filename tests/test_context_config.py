"""Context configuration round trips, endpoint validation and graph injection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from binance_agent.agents.context import ContextPolicy
from binance_agent.app import server
from binance_agent.infra.runtime_config import apply_overrides, load_overrides
from binance_agent.infra.settings import LlmConfig, LlmEndpoint, Settings, load_settings
from binance_agent.llm.registry import LlmRegistry


def configured_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, list[Any]]:
    settings = Settings(
        llm=LlmConfig(
            endpoints=[
                LlmEndpoint(
                    name="default",
                    base_url="https://unused.invalid/v1",
                    api_key_ref="TEST_UNUSED",
                    model="fake",
                )
            ]
        )
    )
    app = server.create_app(settings)
    registry = LlmRegistry(
        settings.llm, model_factory=lambda endpoint, model: FakeListChatModel(responses=["{}"])
    )
    app.state.llm_registry = registry
    app.state.aggregator = type("Aggregator", (), {"_disabled": {}})()
    app.state.intel_http = object()
    app.state.rest_client = object()
    captured: list[Any] = []

    def fake_graph(*args: Any, **kwargs: Any) -> object:
        captured.append(kwargs)
        return object()

    monkeypatch.setattr(server, "build_graph", fake_graph)
    monkeypatch.setattr(server, "_build_aggregator", lambda *args: app.state.aggregator)
    monkeypatch.setattr(server, "resolve_secret", lambda ref: None)
    monkeypatch.setenv("BINANCE_AGENT_RUNTIME_CONFIG", str(tmp_path / "runtime.json"))
    return app, captured


def test_old_settings_defaults_and_toml_context_overrides(tmp_path: Path) -> None:
    old = tmp_path / "old.toml"
    old.write_text("[runtime]\ncheckpoint_retention = 20\n")
    assert load_settings(old).runtime.context == ContextPolicy()
    custom = tmp_path / "custom.toml"
    custom.write_text("[runtime.context]\nmax_output_tokens = 6000\nsafety_margin_tokens = 5000\n")
    settings = load_settings(custom)
    assert settings.runtime.context.max_output_tokens == 6000
    assert settings.runtime.context.window_tokens == 200000
    assert settings.runtime.context.safety_margin_tokens == 5000


def test_context_overrides_merge_existing_policy_and_round_trip() -> None:
    settings = Settings()
    settings.runtime.context = ContextPolicy(max_output_tokens=6000)
    apply_overrides(settings, {"params": {"context": {"trigger_tokens": 150000}}})
    assert settings.runtime.context.max_output_tokens == 6000
    assert settings.runtime.context.trigger_tokens == 150000
    saved = server._current_overrides(settings, LlmRegistry(settings.llm))
    restored = Settings()
    apply_overrides(restored, saved)
    assert restored.runtime.context == settings.runtime.context
    with pytest.raises(ValidationError):
        apply_overrides(settings, {"params": {"context": {"target_tokens": 160000}}})
    assert settings.runtime.context.target_tokens == 120000


async def test_context_api_partial_update_persists_injects_and_resets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app, graphs = configured_app(monkeypatch, tmp_path)
    transport = httpx.ASGITransport(app=app)
    base = app.state.settings.model_copy(deep=True)
    monkeypatch.setattr(server, "load_settings", lambda: base)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        first = await client.get("/config")
        assert first.json()["params"]["context"] == ContextPolicy().model_dump()
        updated = await client.post("/config/params", json={"context": {"max_output_tokens": 6000}})
        assert updated.status_code == 200
        assert updated.json()["context"]["max_output_tokens"] == 6000
        assert updated.json()["context"]["window_tokens"] == 200000
        again = await client.post("/config/params", json={"context": {"trigger_tokens": 155000}})
        assert again.status_code == 200
        assert again.json()["context"]["max_output_tokens"] == 6000
        persisted = load_overrides(tmp_path / "runtime.json")
        assert persisted["params"]["context"]["max_output_tokens"] == 6000
        runtime_cfg = graphs[-1]["runtime_config_for"]("technical")
        assert runtime_cfg["context_policy"] == app.state.settings.runtime.context
        assert runtime_cfg["context_policy"].trigger_tokens == 155000
        assert runtime_cfg["candidates"]
        reset = await client.post("/config/reset")
        assert reset.status_code == 200
        after = await client.get("/config")
        assert after.json()["params"]["context"] == ContextPolicy().model_dump()
        assert not (tmp_path / "runtime.json").exists()
        assert graphs[-1]["runtime_config_for"]("technical")["context_policy"] == ContextPolicy()


@pytest.mark.parametrize(
    "policy",
    [
        {"target_tokens": 180000},
        {"window_tokens": 120000},
        {"max_output_tokens": 0},
        {"unknown_option": 1},
        {"safety_margin_tokens": -1},
    ],
)
async def test_invalid_context_api_does_not_mutate_other_parameters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy: dict[str, int],
) -> None:
    app, graphs = configured_app(monkeypatch, tmp_path)
    before = app.state.settings.model_dump()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1", trust_env=False
    ) as client:
        response = await client.post(
            "/config/params", json={"context": policy, "max_news_items": 5}
        )
    assert response.status_code == 422
    assert app.state.settings.model_dump() == before
    assert graphs == []
    assert not (tmp_path / "runtime.json").exists()


def test_research_fingerprint_includes_policy_and_compaction_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    original = server._research_config_fingerprint(settings)
    changed = settings.model_copy(deep=True)
    changed.runtime.context = ContextPolicy(target_tokens=119000)
    assert server._research_config_fingerprint(changed) != original
    monkeypatch.setattr(server, "COMPACTION_PROMPT_VERSION", "future-prompt")
    assert server._research_config_fingerprint(settings) != original
