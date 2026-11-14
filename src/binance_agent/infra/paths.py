"""运行时数据目录解析:开发态写 CWD,冻结包写用户目录。

PyInstaller 打包后 CWD 不可控(快捷方式启动可能落在 system32),
所有相对数据路径(.runs/、checkpoints)统一挂到 ``data_dir()`` 下。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ENV_DATA_DIR = "BINANCE_AGENT_DATA_DIR"


def data_dir(*, profile: str = "default") -> Path:
    """数据根目录;离线 demo 使用独立子目录,避免污染正常运行记录。"""

    override = os.environ.get(_ENV_DATA_DIR)
    if override:
        base = Path(override).expanduser()
    else:
        base = Path.home() / ".binance_agent" if getattr(sys, "frozen", False) else Path.cwd()
    return base / ".demo" if profile == "demo" else base


def resolve_data_path(path: str | Path, *, profile: str = "default") -> Path:
    """相对路径挂到 data_dir 下;绝对路径原样返回。"""
    p = Path(path)
    return p if p.is_absolute() else data_dir(profile=profile) / p
