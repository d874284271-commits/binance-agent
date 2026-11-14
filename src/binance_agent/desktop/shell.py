"""pywebview 桌面壳:后台线程跑 uvicorn,主线程打开研究窗口。

窗口加载同源 http://127.0.0.1:port/(由 FastAPI 提供单页 UI),SSE 同源无跨域问题。
暴露最小 JS API(close/minimize)以便 frameless 窗口可关闭/最小化。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import uvicorn

from ..app.hotkey import HotkeyListener
from ..app.server import create_app
from ..infra.paths import resolve_data_path
from ..infra.settings import Settings
from ..infra.startup_errors import StartupError

logger = logging.getLogger(__name__)


class _WindowApi:
    """暴露给前端的最小桥接(frameless 无原生标题栏)。"""

    def __init__(self, store: Callable[[], Any] | None = None) -> None:
        self._window: Any = None
        self._store = store

    def bind(self, window: Any) -> None:
        self._window = window

    def close(self) -> None:
        """延迟销毁原生窗口,让 JS 桥接响应先返回,避免 Cocoa 退出后回调线程阻塞。"""
        if self._window is not None:
            # Cocoa waits for the JS bridge result on a non-daemon thread.
            # Destroying its WebView inside that call strands the result callback
            # after the native event loop exits and prevents process shutdown.
            timer = threading.Timer(0.1, self._window.destroy)
            timer.daemon = True
            timer.start()

    def minimize(self) -> None:
        if self._window is not None:
            self._window.minimize()

    def save_report(self, run_id: str) -> dict[str, Any]:
        """Only saved reports may leave the app; cancellation is a normal result."""
        if (
            self._window is None
            or self._store is None
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id)
        ):
            return {"error": "无法读取这份研报"}
        from webview import FileDialog

        from ..agents.reports import report_markdown
        from ..app.reports import saved_report
        from ..infra.file_io import atomic_write

        try:
            content = report_markdown(saved_report(self._store(), run_id))
            choice = self._window.create_file_dialog(
                FileDialog.SAVE,
                save_filename=f"研报-{run_id[:8]}.md",
                file_types=("Markdown (*.md)",),
            )
            if not choice:
                return {"cancelled": True}
            path = Path(choice if isinstance(choice, str) else choice[0])
            atomic_write(path, content.encode("utf-8"))
            return {"saved": True, "name": path.name}
        except Exception:
            logger.exception("保存研报失败")
            return {"error": "保存失败,请检查目标文件夹权限后重试"}


def _make_trigger(window: Any) -> Callable[[str], None]:
    """热键回调:把交易对注入悬浮窗并前置窗口。"""

    def trigger(symbol: str) -> None:
        try:
            window.evaluate_js(
                f"window.__externalAnalyze && window.__externalAnalyze({json.dumps(symbol)})"
            )
            window.show()
        except Exception as exc:
            logger.warning("注入分析失败: %s", exc)

    return trigger


def _shutdown_backend(app: Any, server: Any, thread: Any) -> None:
    """先请求后台研究保存退出,再关闭 SSE/服务线程;分别有界等待并记录关闭超时。"""
    manager = getattr(app.state, "research_manager", None)
    loop = getattr(app.state, "backend_loop", None)
    if manager is not None and loop is not None and loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(manager.shutdown(), loop).result(timeout=10)
        except Exception:
            logger.exception("保存并停止后台研究时发生异常")
    server.should_exit = True
    thread.join(timeout=15)
    if thread.is_alive():
        logger.error("后台关闭超时,请检查研究状态和日志")


def _show_desktop_error(message: str) -> None:
    """HTTP 界面无法启动时尝试系统错误框;显示不可用仍保留日志诊断。"""
    logger.error("%s", message)
    try:
        from tkinter import Tk, messagebox

        root = Tk()
        try:
            root.withdraw()
            messagebox.showerror("Binance Agent 启动失败", message, parent=root)
        finally:
            root.destroy()
    except Exception:
        logger.warning("无法显示系统错误窗口,请查看上述启动诊断")


def _warn_hotkey(window: Any, message: str) -> None:
    logger.warning("%s", message)

    def loaded() -> None:
        try:
            window.evaluate_js(f"window.alert({json.dumps(message)})")
        except Exception:
            logger.warning("无法显示热键提示: %s", message)

    try:
        window.events.loaded += loaded
    except AttributeError:
        logger.warning("窗口不支持加载提示,请查看热键日志")


def run_desktop(settings: Settings) -> None:
    """后台线程运行本机服务,就绪后在主线程打开原生窗口。

    后端启动失败不打开 URL;热键失败可降级。退出时停止热键和后台研究,
    WebView 持久化目录按 profile 隔离以支持重开历史。
    """
    app: Any = None
    server: Any = None
    thread: Any = None
    listener: HotkeyListener | None = None
    backend_errors: list[BaseException] = []
    try:
        import webview

        app = create_app(settings)
        config = uvicorn.Config(
            app,
            host=settings.app.host,
            port=settings.app.port,
            log_level=settings.app.log_level.lower(),
        )
        server = uvicorn.Server(config)

        def run_backend() -> None:
            try:
                server.run()
            except BaseException as exc:
                # uvicorn uses SystemExit for bind failures on some platforms.
                backend_errors.append(exc)

        thread = threading.Thread(target=run_backend, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                detail = getattr(app.state, "startup_error", None)
                if detail is None and backend_errors:
                    detail = type(backend_errors[0]).__name__
                raise StartupError(
                    "桌面后端",
                    f"{settings.app.host}:{settings.app.port}",
                    str(detail or "等待就绪超过 10 秒或后端提前退出"),
                    "未打开分析窗口。请检查端口占用、配置和启动日志,修复后重试。",
                )
            time.sleep(0.05)

        url = f"http://{settings.app.host}:{settings.app.port}/"
        logger.info("打开研究窗口 %s", url)
        api = _WindowApi(lambda: app.state.run_store)
        window = webview.create_window(
            "Binance Agent",
            url=url,
            width=1040,
            height=760,
            min_size=(440, 680),
            frameless=True,
            easy_drag=True,
            on_top=settings.desktop.on_top,
            background_color="#0b0e11",
            js_api=api,
        )
        api.bind(window)
        if settings.hotkey.enabled:
            try:
                listener = HotkeyListener(settings.hotkey.combo, _make_trigger(window))
                listener.start()
            except Exception:
                logger.exception("全局热键启动失败")
                _warn_hotkey(
                    window, "全局热键不可用,可继续在窗口内发起分析。请检查桌面依赖和系统权限。"
                )

        storage = resolve_data_path(".desktop", profile=settings.profile)
        storage.mkdir(parents=True, exist_ok=True)
        webview.start(private_mode=False, storage_path=str(storage))
    except Exception as exc:
        _show_desktop_error(str(exc))
        raise
    finally:
        try:
            if listener is not None:
                listener.stop()
        except Exception:
            logger.exception("停止全局热键失败")
        finally:
            if thread is not None and server is not None:
                _shutdown_backend(app, server, thread)
