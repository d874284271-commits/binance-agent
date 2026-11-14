"""启动入口:加载配置 → 脱敏日志 → 启动桌面悬浮窗(或 headless server-only)。

默认打开 pywebview 悬浮窗(后台线程跑 FastAPI)。
headless 模式仅起后端服务,用于开发/CI/无显示环境:
  set BINANCE_AGENT_HEADLESS=1  或  python -m binance_agent.main --headless
运行: uv run python -m binance_agent.main
"""

from __future__ import annotations

import logging
import os
import sys

import uvicorn

from .app.server import create_app
from .infra.logging import setup_logging
from .infra.profiles import apply_profile, resolve_profile
from .infra.settings import Settings, load_settings
from .infra.startup_errors import StartupError

logger = logging.getLogger("binance_agent")


def _run_server(settings: Settings) -> None:
    """运行本机服务,启动失败以非零退出;不把未就绪状态当作正常关闭。"""
    app = create_app(settings)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=settings.app.host,
            port=settings.app.port,
            log_level=settings.app.log_level.lower(),
        )
    )
    server.run()
    if not server.started:
        error = getattr(app.state, "startup_error", None)
        raise SystemExit(str(error or "后端启动失败,请检查端口、配置及启动日志。"))


def main() -> None:
    """按配置和 profile 装配启动入口;显式 headless 或缺少桌面依赖时运行纯后端。"""
    settings = load_settings()
    try:
        profile = resolve_profile()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    apply_profile(settings, profile)
    setup_logging(settings.app.log_level)
    headless = bool(os.environ.get("BINANCE_AGENT_HEADLESS")) or "--headless" in sys.argv
    logger.info(
        "启动 Binance Agent | profile=%s market=%s host=%s port=%s headless=%s",
        settings.profile,
        settings.market.default_market,
        settings.app.host,
        settings.app.port,
        headless,
    )

    if headless:
        try:
            _run_server(settings)
        except StartupError as exc:
            raise SystemExit(str(exc)) from exc
        return

    try:
        from .desktop.shell import run_desktop
    except ImportError:
        logger.warning("pywebview 未安装,回退 server-only(uv sync --extra desktop 可启用悬浮窗)")
        _run_server(settings)
        return

    run_desktop(settings)


if __name__ == "__main__":
    main()
