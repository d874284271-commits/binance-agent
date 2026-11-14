from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def isolate_proxy_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep offline httpx/respx tests independent of host proxy settings."""
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture(autouse=True)
def isolate_runtime_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[None]:
    """把运行时配置重定向到 tmp,防止测试写入/读取用户真实 ~/.binance_agent/runtime_config.json。

    背景:server 的 /config/* 写操作会调用 save_overrides() 落盘到 _default_path()
    (默认 ~/.binance_agent/runtime_config.json)。若某测试未 monkeypatch save_overrides,
    便会污染用户真实配置(并在下次真实启动时静默覆盖 config.toml)。此处全局兜底。
    """
    target = tmp_path_factory.mktemp("runtime_cfg") / "runtime_config.json"
    monkeypatch.setenv("BINANCE_AGENT_RUNTIME_CONFIG", str(target))
    yield
