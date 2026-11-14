"""LlmRegistry 离线测试:per-agent 路由 / upsert / remove / config_view 不漏 key(model_factory 注入)。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import binance_agent.llm.registry as reg_mod
from binance_agent.infra.settings import LlmConfig, LlmEndpoint
from binance_agent.llm.registry import LlmRegistry


@pytest.fixture(autouse=True)
def _stub_secret(monkeypatch: Any) -> None:
    # 避免真实 keyring 副作用;a 端点视为已设 key,其余未设
    monkeypatch.setattr(reg_mod, "resolve_secret", lambda ref: "x" if ref == "A_KEY" else None)


def _registry() -> LlmRegistry:
    config = LlmConfig(
        active_endpoint="a",
        endpoints=[
            LlmEndpoint(
                name="a",
                base_url="https://a/v1",
                api_key_ref="A_KEY",
                model="ma",
                models=["ma", "ma2"],
            ),
            LlmEndpoint(name="b", base_url="https://b/v1", api_key_ref="B_KEY", model="mb"),
        ],
        agent_models={"technical": ("a", "ma")},
    )
    return LlmRegistry(config, model_factory=lambda ep, m: FakeListChatModel(responses=["{}"]))


def test_set_agent_model_routes() -> None:
    reg = _registry()
    reg.set_agent_model("funding", "b", "mb")
    agents = {a["agent"]: (a["endpoint"], a["model"]) for a in reg.config_view()["agents"]}
    assert agents["funding"] == ("b", "mb")
    assert agents["technical"] == ("a", "ma")
    assert agents["strategy_synth"] == ("a", "ma")  # 未配置回退 active


def test_set_agent_model_unknown_endpoint() -> None:
    reg = _registry()
    with pytest.raises(ValueError):
        reg.set_agent_model("funding", "nope", "x")


def test_set_proxy_log_does_not_expose_userinfo(caplog: Any) -> None:
    reg = _registry()
    caplog.set_level("INFO", logger=reg_mod.__name__)

    reg.set_proxy("http://user:secret@proxy.example:8080")

    assert "proxy.example:8080" in caplog.text
    assert "user" not in caplog.text
    assert "secret" not in caplog.text


def test_upsert_endpoint_adds_and_routes() -> None:
    reg = _registry()
    reg.upsert_endpoint("c", "https://c/v1", "mc", models=["mc"])
    assert "c" in reg.endpoint_names()
    endpoints = {item["name"]: item for item in reg.config_view()["endpoints"]}
    assert endpoints["c"]["structured_output"] is False
    reg.set_agent_model("technical", "c", "mc")
    agents = {a["agent"]: a["endpoint"] for a in reg.config_view()["agents"]}
    assert agents["technical"] == "c"


def test_upsert_endpoint_preserves_existing_tuning_when_not_supplied() -> None:
    config = LlmConfig(
        active_endpoint="a",
        endpoints=[
            LlmEndpoint(
                name="a",
                base_url="https://a/v1",
                api_key_ref="A_KEY",
                model="ma",
                models=["ma"],
                temperature=0.1,
                timeout=120,
                max_retries=5,
                structured_output=True,
                fallback_verified=True,
                max_concurrency=4,
            )
        ],
    )
    reg = LlmRegistry(config, model_factory=lambda ep, m: FakeListChatModel(responses=["{}"]))

    reg.upsert_endpoint("a", "https://new/v1", "new-model", models=["new-model"])

    endpoint = config.endpoints[0]
    assert endpoint.base_url == "https://new/v1"
    assert endpoint.model == "new-model"
    assert endpoint.models == ["new-model"]
    assert endpoint.api_key_ref == "A_KEY"
    assert endpoint.temperature == 0.1
    assert endpoint.timeout == 120
    assert endpoint.max_retries == 5
    assert endpoint.structured_output is True
    assert endpoint.fallback_verified is True
    assert endpoint.max_concurrency == 4


def test_upsert_endpoint_can_explicitly_change_structured_output() -> None:
    reg = _registry()

    reg.upsert_endpoint("a", "https://a/v1", "ma", structured_output=True)

    endpoints = {item["name"]: item for item in reg.config_view()["endpoints"]}
    assert endpoints["a"]["structured_output"] is True


def test_remove_endpoint_falls_back() -> None:
    reg = _registry()
    reg.set_agent_model("technical", "b", "mb")  # technical 指向 b
    reg.remove_endpoint("b")
    assert "b" not in reg.endpoint_names()
    agents = {a["agent"]: a["endpoint"] for a in reg.config_view()["agents"]}
    assert agents["technical"] == "a"  # 指向被删端点 → 回退 active


def test_remove_active_switches_active() -> None:
    reg = _registry()  # active=a
    reg.remove_endpoint("a")
    assert reg.active == "b"


def test_switch_active_updates_backing_config() -> None:
    config = LlmConfig(
        active_endpoint="a",
        endpoints=[
            LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A_KEY", model="ma"),
            LlmEndpoint(name="b", base_url="https://b/v1", api_key_ref="B_KEY", model="mb"),
        ],
    )
    reg = LlmRegistry(config, model_factory=lambda ep, m: FakeListChatModel(responses=["{}"]))

    reg.switch_active("b")

    assert reg.active == "b"
    assert config.active_endpoint == "b"


def test_remove_endpoint_keeps_at_least_one() -> None:
    reg = _registry()
    reg.remove_endpoint("b")
    with pytest.raises(ValueError):
        reg.remove_endpoint("a")  # 不能删到一个不剩


def test_config_view_no_key_plaintext() -> None:
    reg = _registry()
    view = reg.config_view()
    eps = {e["name"]: e for e in view["endpoints"]}
    assert eps["a"]["has_key"] is True
    assert eps["b"]["has_key"] is False
    for e in view["endpoints"]:
        assert set(e.keys()) == {
            "name",
            "base_url",
            "model",
            "models",
            "structured_output",
            "fallback_verified",
            "max_concurrency",
            "has_key",
        }
        assert e["structured_output"] is False
        assert "api_key" not in e and "api_key_ref" not in e


def test_runtime_config_for_uses_endpoint_timeout() -> None:
    # Bug1 回归:runtime_config_for 须按 agent→端点路由返回该端点 timeout(供节点外层 wait_for),
    # 未显式映射的 agent 回退 active 端点 timeout。
    config = LlmConfig(
        active_endpoint="a",
        endpoints=[
            LlmEndpoint(
                name="a", base_url="https://a/v1", api_key_ref="A_KEY", model="ma", timeout=45
            ),
            LlmEndpoint(
                name="b", base_url="https://b/v1", api_key_ref="B_KEY", model="mb", timeout=90
            ),
        ],
        agent_models={"technical": ("a", "ma"), "strategy_synth": ("b", "mb")},
    )
    reg = LlmRegistry(config, model_factory=lambda ep, m: FakeListChatModel(responses=["{}"]))
    technical = reg.runtime_config_for("technical")
    synth = reg.runtime_config_for("strategy_synth")
    funding = reg.runtime_config_for("funding")
    legacy_keys = {"timeout_s", "max_retries", "structured_output"}
    assert {key: technical[key] for key in legacy_keys} == {
        "timeout_s": 45.0,
        "max_retries": 2,
        "structured_output": False,
    }
    assert {key: synth[key] for key in legacy_keys} == {
        "timeout_s": 90.0,
        "max_retries": 2,
        "structured_output": False,
    }
    assert {key: funding[key] for key in legacy_keys} == {
        "timeout_s": 45.0,
        "max_retries": 2,
        "structured_output": False,
    }


def test_runtime_config_for_carries_structured_output_flag() -> None:
    config = LlmConfig(
        active_endpoint="a",
        endpoints=[
            LlmEndpoint(
                name="a",
                base_url="https://a/v1",
                api_key_ref="A_KEY",
                model="ma",
                timeout=45,
                structured_output=True,
            ),
            LlmEndpoint(
                name="b", base_url="https://b/v1", api_key_ref="B_KEY", model="mb", timeout=90
            ),
        ],
        agent_models={"technical": ("a", "ma"), "strategy_synth": ("b", "mb")},
    )
    reg = LlmRegistry(config, model_factory=lambda ep, m: FakeListChatModel(responses=["{}"]))
    assert reg.runtime_config_for("technical")["structured_output"] is True
    assert reg.runtime_config_for("strategy_synth")["structured_output"] is False


def test_runtime_config_for_uses_endpoint_timeout_and_retries() -> None:
    config = LlmConfig(
        active_endpoint="a",
        endpoints=[
            LlmEndpoint(
                name="a",
                base_url="https://a/v1",
                api_key_ref="A_KEY",
                model="ma",
                timeout=45,
                max_retries=3,
            ),
            LlmEndpoint(
                name="b",
                base_url="https://b/v1",
                api_key_ref="B_KEY",
                model="mb",
                timeout=90,
                max_retries=0,
            ),
        ],
        agent_models={"strategy_synth": ("b", "mb")},
    )
    reg = LlmRegistry(config, model_factory=lambda ep, m: FakeListChatModel(responses=["{}"]))

    technical = reg.runtime_config_for("technical")
    synth = reg.runtime_config_for("strategy_synth")
    legacy_keys = {"timeout_s", "max_retries", "structured_output"}
    assert {key: technical[key] for key in legacy_keys} == {
        "timeout_s": 45.0,
        "max_retries": 3,
        "structured_output": False,
    }
    assert {key: synth[key] for key in legacy_keys} == {
        "timeout_s": 90.0,
        "max_retries": 0,
        "structured_output": False,
    }


def test_fallbacks_for_agent_returns_other_endpoint_defaults_without_duplicate() -> None:
    calls: list[tuple[str, str]] = []

    def factory(endpoint: LlmEndpoint, model: str) -> FakeListChatModel:
        calls.append((endpoint.name, model))
        return FakeListChatModel(responses=["{}"])

    config = LlmConfig(
        active_endpoint="a",
        endpoints=[
            LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A_KEY", model="ma"),
            LlmEndpoint(name="b", base_url="https://b/v1", api_key_ref="B_KEY", model="mb"),
            LlmEndpoint(name="c", base_url="https://c/v1", api_key_ref="C_KEY", model="mc"),
        ],
        agent_models={"technical": ("b", "custom-b")},
    )
    reg = LlmRegistry(config, model_factory=factory)

    primary = reg.for_agent("technical")
    fallbacks = reg.fallbacks_for_agent("technical")

    assert primary is not None
    assert len(fallbacks) == 2
    assert calls == [("b", "custom-b"), ("a", "ma"), ("c", "mc")]


def test_runtime_candidates_keep_each_endpoint_execution_policy() -> None:
    config = LlmConfig(
        active_endpoint="a",
        endpoints=[
            LlmEndpoint(
                name="a",
                base_url="https://a/v1",
                api_key_ref="A_KEY",
                model="ma",
                timeout=45,
                max_retries=3,
                structured_output=True,
            ),
            LlmEndpoint(
                name="b",
                base_url="https://b/v1",
                api_key_ref="B_KEY",
                model="mb",
                timeout=90,
                max_retries=0,
                structured_output=False,
            ),
        ],
        agent_models={"technical": ("a", "custom-a")},
    )
    reg = LlmRegistry(config, model_factory=lambda ep, m: FakeListChatModel(responses=["{}"]))

    candidates = reg.runtime_candidates_for("technical")

    assert [candidate.endpoint for candidate in candidates] == ["a", "b"]
    assert [candidate.timeout_s for candidate in candidates] == [45.0, 90.0]
    assert [candidate.max_retries for candidate in candidates] == [3, 0]
    assert [candidate.structured_output for candidate in candidates] == [True, False]


def test_runtime_config_exposes_candidates_without_new_graph_wiring() -> None:
    reg = _registry()

    config = reg.runtime_config_for("technical")

    assert [candidate.endpoint for candidate in config["candidates"]] == ["a", "b"]


def test_fallbacks_for_unmapped_agent_skip_active_duplicate() -> None:
    calls: list[tuple[str, str]] = []

    def factory(endpoint: LlmEndpoint, model: str) -> FakeListChatModel:
        calls.append((endpoint.name, model))
        return FakeListChatModel(responses=["{}"])

    config = LlmConfig(
        active_endpoint="a",
        endpoints=[
            LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A_KEY", model="ma"),
            LlmEndpoint(name="b", base_url="https://b/v1", api_key_ref="B_KEY", model="mb"),
        ],
    )
    reg = LlmRegistry(config, model_factory=factory)

    reg.for_agent("funding")
    fallbacks = reg.fallbacks_for_agent("funding")

    assert len(fallbacks) == 1
    assert calls == [("a", "ma"), ("b", "mb")]


def test_active_model_uses_active_endpoint_default_not_mapping() -> None:
    calls: list[tuple[str, str]] = []

    def factory(endpoint: LlmEndpoint, model: str) -> FakeListChatModel:
        calls.append((endpoint.name, model))
        return FakeListChatModel(responses=["{}"])

    config = LlmConfig(
        active_endpoint="a",
        endpoints=[
            LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A_KEY", model="ma"),
            LlmEndpoint(name="b", base_url="https://b/v1", api_key_ref="B_KEY", model="mb"),
        ],
        agent_models={"technical": ("b", "custom-b")},  # 映射不应影响 active_model
    )
    reg = LlmRegistry(config, model_factory=factory)

    assert reg.active_model() is not None
    assert calls == [("a", "ma")]


def test_active_model_follows_switch_active() -> None:
    calls: list[tuple[str, str]] = []

    def factory(endpoint: LlmEndpoint, model: str) -> FakeListChatModel:
        calls.append((endpoint.name, model))
        return FakeListChatModel(responses=["{}"])

    config = LlmConfig(
        active_endpoint="a",
        endpoints=[
            LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A_KEY", model="ma"),
            LlmEndpoint(name="b", base_url="https://b/v1", api_key_ref="B_KEY", model="mb"),
        ],
    )
    reg = LlmRegistry(config, model_factory=factory)

    reg.active_model()
    reg.switch_active("b")
    reg.active_model()

    assert calls == [("a", "ma"), ("b", "mb")]


def test_reload_replaces_endpoints_active_and_keeps_factory() -> None:
    calls: list[tuple[str, str]] = []

    def factory(endpoint: LlmEndpoint, model: str) -> FakeListChatModel:
        calls.append((endpoint.name, model))
        return FakeListChatModel(responses=["{}"])

    reg = LlmRegistry(
        LlmConfig(
            active_endpoint="a",
            endpoints=[
                LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A_KEY", model="ma")
            ],
        ),
        model_factory=factory,
    )
    reg.active_model()  # 经 factory 构造 (a, ma)

    reg.reload(
        LlmConfig(
            active_endpoint="z",
            endpoints=[
                LlmEndpoint(name="z", base_url="https://z/v1", api_key_ref="Z_KEY", model="mz")
            ],
        )
    )

    assert reg.active == "z"
    assert reg.endpoint_names() == ["z"]
    reg.active_model()  # 缓存已清,用同一 factory 重建 (z, mz)
    assert calls == [("a", "ma"), ("z", "mz")]


async def test_explicit_proxy_clients_are_shared_and_closeable(monkeypatch: Any) -> None:
    sync_clients: list[Any] = []
    async_clients: list[Any] = []

    class _SyncClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.closed = False
            sync_clients.append(self)

        def close(self) -> None:
            self.closed = True

    class _AsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.closed = False
            async_clients.append(self)

        async def aclose(self) -> None:
            self.closed = True

    built_with: list[tuple[Any, Any]] = []

    def _fake_build(
        endpoint: LlmEndpoint,
        model: str,
        *,
        http_client: Any = None,
        http_async_client: Any = None,
    ) -> BaseChatModel:
        built_with.append((http_client, http_async_client))
        return FakeListChatModel(responses=["{}"])

    monkeypatch.setattr(reg_mod.httpx, "Client", _SyncClient)
    monkeypatch.setattr(reg_mod.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(reg_mod, "_build_chat_openai", _fake_build)
    reg = LlmRegistry(
        LlmConfig(
            active_endpoint="a",
            endpoints=[
                LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A_KEY", model="ma"),
                LlmEndpoint(name="b", base_url="https://b/v1", api_key_ref="B_KEY", model="mb"),
            ],
            agent_models={"technical": ("a", "ma"), "funding": ("b", "mb")},
        ),
        proxy="http://127.0.0.1:7897",
    )

    reg.for_agent("technical")
    reg.for_agent("funding")

    assert len(sync_clients) == 1
    assert len(async_clients) == 1
    assert built_with == [(sync_clients[0], async_clients[0])] * 2

    await reg.aclose()

    assert sync_clients[0].closed
    assert async_clients[0].closed


async def test_proxy_switch_retires_old_shared_clients(monkeypatch: Any) -> None:
    sync_clients: list[Any] = []
    async_clients: list[Any] = []

    class _SyncClient:
        def __init__(self, **kwargs: Any) -> None:
            self.closed = False
            sync_clients.append(self)

        def close(self) -> None:
            self.closed = True

    class _AsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            self.closed = False
            async_clients.append(self)

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(reg_mod.httpx, "Client", _SyncClient)
    monkeypatch.setattr(reg_mod.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(
        reg_mod,
        "_build_chat_openai",
        lambda *args, **kwargs: FakeListChatModel(responses=["{}"]),
    )
    reg = LlmRegistry(
        LlmConfig(
            active_endpoint="a",
            endpoints=[
                LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A_KEY", model="ma")
            ],
        ),
        proxy="http://127.0.0.1:7897",
    )
    reg.for_agent("technical")

    reg.set_proxy("http://127.0.0.1:7898")
    await asyncio.sleep(0)

    assert sync_clients[0].closed
    assert async_clients[0].closed

    reg.for_agent("technical")
    assert len(sync_clients) == 2
    assert len(async_clients) == 2
    await reg.aclose()


def test_proxy_log_label_masks_host() -> None:
    domain = reg_mod._proxy_log_label("http://proxy.internal.example:7897")
    local = reg_mod._proxy_log_label("http://127.0.0.1:7897")

    assert domain == "http://*.internal.example:7897"
    assert local == "http://127.*.*.1:7897"
    assert "proxy.internal.example" not in domain
    assert "127.0.0.1" not in local
