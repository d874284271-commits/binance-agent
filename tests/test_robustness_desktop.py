"""Startup failures preserve data and clean up desktop resources."""

from __future__ import annotations

import builtins
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from binance_agent import main
from binance_agent.app import checkpoints
from binance_agent.desktop import shell
from binance_agent.infra.settings import Settings
from binance_agent.infra.startup_errors import StartupError


def test_corrupt_checkpoint_keeps_original_and_reports_path(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.sqlite"
    original = b"not a SQLite database"
    path.write_bytes(original)
    sidecars = [Path(str(path) + suffix) for suffix in ("-wal", "-shm")]
    for sidecar in sidecars:
        sidecar.write_bytes(b"original sidecar")
    with pytest.raises(StartupError) as error:
        checkpoints.SQLiteCheckpointSaver(path)
    assert path.read_bytes() == original
    assert all(sidecar.read_bytes() == b"original sidecar" for sidecar in sidecars)
    assert str(path) in str(error.value)
    assert "-wal/-shm" in str(error.value)
    assert isinstance(error.value.__cause__, sqlite3.DatabaseError)


def test_checkpoint_initialization_failure_closes_connection(tmp_path, monkeypatch) -> None:
    closed = []

    class Connection:
        def execute(self, sql):
            raise sqlite3.DatabaseError("broken")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(checkpoints.sqlite3, "connect", lambda *a, **k: Connection())
    with pytest.raises(StartupError):
        checkpoints.SQLiteCheckpointSaver(tmp_path / "db")
    assert closed == [True]


def desktop_stub(monkeypatch, tmp_path, *, ready=True, alive=False):
    calls = []
    server = SimpleNamespace(started=ready, should_exit=False, run=lambda: None)
    thread = SimpleNamespace(
        start=lambda: calls.append("thread-start"),
        join=lambda **kw: calls.append("join"),
        is_alive=lambda: alive,
    )
    webview = SimpleNamespace(
        create_window=lambda *a, **kw: calls.append("window") or SimpleNamespace(),
        start=lambda **kw: calls.append("webview-start"),
    )
    monkeypatch.setenv("BINANCE_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        shell, "create_app", lambda settings: SimpleNamespace(state=SimpleNamespace())
    )
    monkeypatch.setattr(shell.threading, "Thread", lambda **kw: thread)
    monkeypatch.setattr(shell.uvicorn, "Server", lambda config: server)
    monkeypatch.setattr(shell, "_show_desktop_error", lambda message: calls.append("error"))
    monkeypatch.setitem(sys.modules, "webview", webview)
    return calls, server, webview


def test_hotkey_failure_keeps_window_and_stops_listener(tmp_path, monkeypatch) -> None:
    calls, server, _ = desktop_stub(monkeypatch, tmp_path)

    class Listener:
        def __init__(self, *args):
            pass

        def start(self):
            raise ImportError("pynput unavailable")

        def stop(self):
            calls.append("hotkey-stop")

    monkeypatch.setattr(shell, "HotkeyListener", Listener)
    monkeypatch.setattr(shell, "_warn_hotkey", lambda *args: calls.append("warning"))
    shell.run_desktop(Settings.model_validate({"hotkey": {"enabled": True}}))
    assert "webview-start" in calls and "warning" in calls and "hotkey-stop" in calls
    assert calls.index("hotkey-stop") < calls.index("join")
    assert server.should_exit


@pytest.mark.parametrize("stage", ["create", "start"])
def test_webview_failure_stops_backend(tmp_path, monkeypatch, stage) -> None:
    calls, server, webview = desktop_stub(monkeypatch, tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("native UI unavailable")

    setattr(webview, "create_window" if stage == "create" else "start", fail)
    with pytest.raises(RuntimeError, match="native UI unavailable"):
        shell.run_desktop(Settings.model_validate({"hotkey": {"enabled": False}}))
    assert server.should_exit and "join" in calls and "error" in calls


def test_backend_early_exit_does_not_open_window(tmp_path, monkeypatch) -> None:
    calls, server, _ = desktop_stub(monkeypatch, tmp_path, ready=False)
    with pytest.raises(StartupError, match="未打开分析窗口"):
        shell.run_desktop(Settings())
    assert "window" not in calls and "webview-start" not in calls
    assert server.should_exit and "join" in calls


def test_backend_timeout_does_not_open_window(tmp_path, monkeypatch) -> None:
    calls, server, _ = desktop_stub(monkeypatch, tmp_path, ready=False, alive=True)
    times = iter([0, 11])
    monkeypatch.setattr(shell.time, "monotonic", lambda: next(times))
    with pytest.raises(StartupError):
        shell.run_desktop(Settings())
    assert "window" not in calls
    assert server.should_exit


def test_webview_import_failure_reports_error_without_backend(tmp_path, monkeypatch) -> None:
    calls, _, _ = desktop_stub(monkeypatch, tmp_path)
    original_import = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "webview":
            raise ImportError("webview missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ImportError, match="webview missing"):
        shell.run_desktop(Settings())
    assert calls == ["error"]


def test_headless_failed_lifespan_is_nonzero(monkeypatch) -> None:
    app = SimpleNamespace(state=SimpleNamespace(startup_error="checkpoint unavailable"))
    monkeypatch.setattr(main, "create_app", lambda settings: app)
    monkeypatch.setattr(
        main.uvicorn, "Server", lambda config: SimpleNamespace(started=False, run=lambda: None)
    )
    with pytest.raises(SystemExit, match="checkpoint unavailable"):
        main._run_server(Settings())


def test_reconnection_does_not_start_a_new_turn() -> None:
    js = Path("src/binance_agent/desktop/ui/app.js").read_text()
    handler = js[js.index("source.onerror =") : js.index("source.onopen =")]
    assert "startResearch" not in handler and "continueTurn" not in handler
    assert "连接暂时断开" in handler


def test_listener_stop_failure_still_stops_backend(tmp_path, monkeypatch) -> None:
    calls, server, _ = desktop_stub(monkeypatch, tmp_path)

    class Listener:
        def __init__(self, *args):
            pass

        def start(self):
            pass

        def stop(self):
            raise RuntimeError("stop failed")

    monkeypatch.setattr(shell, "HotkeyListener", Listener)
    shell.run_desktop(Settings())
    assert "webview-start" in calls and "join" in calls and server.should_exit
