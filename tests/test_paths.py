"""数据目录解析测试:开发态挂 CWD,冻结态挂用户目录,绝对路径直通。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from binance_agent.infra.paths import data_dir, resolve_data_path


def test_data_dir_dev_mode_uses_cwd() -> None:
    assert data_dir() == Path.cwd()


def test_data_dir_frozen_uses_home(monkeypatch: Any) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert data_dir() == Path.home() / ".binance_agent"


def test_resolve_data_path_relative_and_absolute(monkeypatch: Any) -> None:
    assert resolve_data_path(".runs/runs.jsonl") == Path.cwd() / ".runs" / "runs.jsonl"

    absolute = Path.cwd().resolve() / "elsewhere" / "x.db"
    assert resolve_data_path(absolute) == absolute

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert (
        resolve_data_path(".runs/checkpoints.sqlite")
        == Path.home() / ".binance_agent" / ".runs" / "checkpoints.sqlite"
    )


def test_data_dir_environment_override(monkeypatch: Any, tmp_path: Path) -> None:
    target = tmp_path / "runtime-data"
    monkeypatch.setenv("BINANCE_AGENT_DATA_DIR", str(target))

    assert resolve_data_path(".runs/runs.jsonl") == target / ".runs" / "runs.jsonl"
    assert resolve_data_path(".runs/runs.jsonl", profile="demo") == (
        target / ".demo" / ".runs" / "runs.jsonl"
    )
