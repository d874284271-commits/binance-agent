"""Desktop browser state survives application restarts in the app data directory."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from binance_agent.app.run_store import FileRunStore
from binance_agent.desktop import shell
from binance_agent.infra.settings import Settings


def test_desktop_uses_persistent_profile_and_waits_for_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BINANCE_AGENT_DATA_DIR", str(tmp_path))
    captured = {}
    server = SimpleNamespace(started=True, should_exit=False, run=lambda: None)
    thread = SimpleNamespace(
        start=lambda: None,
        join=lambda **kwargs: captured.update(join=kwargs),
        is_alive=lambda: False,
    )
    monkeypatch.setattr(shell, "threading", SimpleNamespace(Thread=lambda **_: thread))
    monkeypatch.setattr(
        shell, "uvicorn", SimpleNamespace(Config=shell.uvicorn.Config, Server=lambda _: server)
    )
    monkeypatch.setitem(
        sys.modules,
        "webview",
        SimpleNamespace(
            create_window=lambda *args, **kwargs: SimpleNamespace(),
            start=lambda **kwargs: captured.update(kwargs),
        ),
    )
    settings = Settings.model_validate({"profile": "demo", "hotkey": {"enabled": False}})
    shell.run_desktop(settings)
    assert captured["private_mode"] is False
    assert Path(captured["storage_path"]) == tmp_path / ".demo/.desktop"
    assert Path(captured["storage_path"]).is_dir()
    assert server.should_exit and captured["join"]["timeout"] == 15


def test_close_defers_destroy_until_bridge_can_return(monkeypatch) -> None:
    calls = []
    captured = {}

    class Timer:
        daemon = False

        def __init__(self, interval, callback):
            captured.update(interval=interval, callback=callback, timer=self)

        def start(self):
            calls.append("scheduled")

    monkeypatch.setattr(shell.threading, "Timer", Timer)
    api = shell._WindowApi()
    api.bind(SimpleNamespace(destroy=lambda: calls.append("destroyed")))
    api.close()
    assert calls == ["scheduled"]
    assert captured["timer"].daemon is True
    captured["callback"]()
    assert calls == ["scheduled", "destroyed"]


def test_native_save_uses_selected_saved_report_and_cancel_is_noop(tmp_path, monkeypatch):
    store = FileRunStore(tmp_path / "runs.jsonl")
    first = store.create(symbol="BTCUSDT", market="fapi")
    second = store.create(symbol="ETHUSDT", market="api")
    store.complete(
        first, {"symbol": "BTCUSDT", "decision_type": "no_trade", "rationale": "selected version"}
    )
    store.complete(
        second, {"symbol": "ETHUSDT", "decision_type": "no_trade", "rationale": "different version"}
    )
    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace(FileDialog=SimpleNamespace(SAVE=1)))
    api = shell._WindowApi(lambda: store)
    api.bind(SimpleNamespace(create_file_dialog=lambda *a, **k: None))
    assert api.save_report(first) == {"cancelled": True}
    assert not (tmp_path / "export.md").exists()
    api.bind(SimpleNamespace(create_file_dialog=lambda *a, **k: (str(tmp_path / "export.md"),)))
    assert api.save_report(first)["saved"]
    content = (tmp_path / "export.md").read_text()
    assert "selected version" in content and "different version" not in content
    assert "runtime_events" not in content
