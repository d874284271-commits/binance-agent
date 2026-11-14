"""PyInstaller 打包入口:等价 ``python -m binance_agent.main``。

单独的入口文件(而非 -m 模块)是 PyInstaller Analysis 的要求;
freeze_support 防止 Windows 上潜在的 multiprocessing 子进程递归启动。
"""

from __future__ import annotations

import io
import multiprocessing
import os
import sys

from binance_agent.main import main


def _ensure_std_streams() -> None:
    """windowed(console=False)应用无重定向启动时 sys.stdout/stderr 为 None。

    uvicorn 启动会调 ``sys.stdout.isatty()`` 判断配色 → AttributeError 秒崩
    (双击 exe / CI Start-Process 均触发;经管道启动则有真实句柄,故本机终端测不出)。
    兜底指向 devnull;应用日志仍走 setup_logging 的滚动文件。
    """
    if sys.stdout is None:
        sys.stdout = io.TextIOWrapper(open(os.devnull, "wb"), encoding="utf-8")  # noqa: SIM115 进程级长生命周期句柄
    if sys.stderr is None:
        sys.stderr = io.TextIOWrapper(open(os.devnull, "wb"), encoding="utf-8")  # noqa: SIM115 进程级长生命周期句柄


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _ensure_std_streams()
    main()
