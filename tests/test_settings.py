from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from binance_agent.infra import settings as settings_mod


def test_default_config_path_uses_pyinstaller_bundle_config(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("BINANCE_AGENT_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    bundle_dir = tmp_path / "bundle"
    config_dir = bundle_dir / "config"
    config_dir.mkdir(parents=True)
    bundled_config = config_dir / "config.toml"
    bundled_config.write_text("[app]\nport = 8765\n", encoding="utf-8")

    monkeypatch.setattr(settings_mod.sys, "frozen", True, raising=False)
    monkeypatch.setattr(settings_mod.sys, "_MEIPASS", str(bundle_dir), raising=False)
    monkeypatch.setattr(settings_mod.sys, "executable", str(tmp_path / "dist" / "app.exe"))

    assert settings_mod.default_config_path() == bundled_config


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10"])
def test_app_host_rejects_non_loopback_bindings(host: str) -> None:
    with pytest.raises(ValidationError, match="loopback"):
        settings_mod.AppConfig(host=host)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_app_host_accepts_loopback_bindings(host: str) -> None:
    assert settings_mod.AppConfig(host=host).host == host


def test_network_proxy_rejects_embedded_credentials() -> None:
    with pytest.raises(ValidationError, match="credentials"):
        settings_mod.NetworkConfig(proxy="http://alice:secret@127.0.0.1:7897")


@pytest.mark.parametrize(
    "factory",
    [
        lambda proxy: settings_mod.NetworkConfig(proxy=proxy),
        lambda proxy: settings_mod.Settings(network={"proxy": proxy}),
    ],
)
def test_network_proxy_validation_error_hides_embedded_credentials(factory) -> None:
    proxy = "http://alice:secret@127.0.0.1:7897"

    with pytest.raises(ValidationError) as captured:
        factory(proxy)

    error = str(captured.value)
    assert "credentials" in error
    assert proxy not in error
    assert "alice" not in error
    assert "secret" not in error


def test_startup_config_proxy_error_hides_embedded_credentials(tmp_path: Path) -> None:
    proxy = "http://alice:secret@127.0.0.1:7897"
    config = tmp_path / "config.toml"
    config.write_text(f'[network]\nproxy = "{proxy}"\n', encoding="utf-8")

    with pytest.raises(ValidationError) as captured:
        settings_mod.load_settings(config)

    error = str(captured.value)
    assert "credentials" in error
    assert proxy not in error
    assert "alice" not in error
    assert "secret" not in error


def test_outcome_configuration_rejects_invalid_window_and_interval() -> None:
    with pytest.raises(ValidationError):
        settings_mod.RuntimeConfig(outcome_window_hours=0)
    with pytest.raises(ValidationError):
        settings_mod.RuntimeConfig(outcome_kline_interval="7m")


def test_confidence_bucket_caps_are_validated() -> None:
    cfg = settings_mod.ConfidenceConfig(bucket_caps={"[0.6,0.8)": 0.65})
    assert cfg.bucket_caps == {"[0.6,0.8)": 0.65}

    with pytest.raises(ValidationError):
        settings_mod.ConfidenceConfig(bucket_caps={"unknown": 0.5})
    with pytest.raises(ValidationError):
        settings_mod.ConfidenceConfig(bucket_caps={"[0.6,0.8)": 1.1})


def test_legacy_refetch_setting_warns_and_migrates() -> None:
    with pytest.warns(DeprecationWarning, match="preflight_refetch_enabled"):
        settings = settings_mod.Settings(runtime={"strategy_planner_refetch_enabled": False})

    assert settings.runtime.preflight_refetch_enabled is False
