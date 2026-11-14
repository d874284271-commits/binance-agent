"""本机 HTTP/WS 请求边界与分析准入,在昂贵调用和配置修改前拒绝不合规请求。

Host/Origin 检查服务于 loopback 桌面应用,不提供公网身份认证。
分析槽位由请求或已接管执行的后台任务持有,取消和传输失败时仍须回收。
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from ipaddress import ip_address
from urllib.parse import urlsplit

import anyio
from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from ..infra.async_cleanup import wait_for_cleanup

_PROTECTED_EXACT = {
    "/analyze",
    "/health/deps",
    "/runs/replay",
    "/snapshot",
    "/intel",
    "/watchlist",
    "/runs",
}
_PROTECTED_PREFIXES = ("/config", "/watchlist/", "/research/", "/runs/", "/alerts", "/outcomes")
_ALLOWED_FETCH_SITES = frozenset({"same-origin", "same-site", "none"})


class AnalysisReservation:
    """单次分析的槽位所有权;release 可重复调用,后台接管后须自行负责释放。"""

    def __init__(self, limiter: AnalysisLimiter) -> None:
        self._limiter = limiter
        self._released = False
        self._release_task: asyncio.Task[None] | None = None
        self.detached = False
        self.on_close: Callable[[], None] | None = None
        self.close_stream: Callable[[], Awaitable[None]] | None = None

    def detach(self) -> None:
        """仅在后台任务建立后转交所有权,让请求结束时跳过该槽位的清理。"""
        self.detached = True

    async def release(self) -> None:
        """复用同一回收任务以免重复减计数,调用方取消时仍等待必要清理完成。"""
        if self._release_task is None or self._release_task.cancelled():

            async def finish() -> None:
                await self._limiter._release()
                self._released = True

            self._release_task = asyncio.create_task(finish())
        await wait_for_cleanup(self._release_task)


class AnalysisLimiter:
    """有界分析准入与配置写入的共享互斥边界,不排队等待昂贵任务名额。"""

    def __init__(self, limit: int) -> None:
        self._limit = max(1, int(limit))
        self._active = 0
        self._lock = asyncio.Lock()
        self._configuring = False

    async def try_acquire(self) -> AnalysisReservation | None:
        """繁忙或配置切换时返回 None;成功后由当前请求登记并负责槽位收尾。"""
        async with self._lock:
            if self._configuring or self._active >= self._limit:
                return None
            self._active += 1
        reservation = AnalysisReservation(self)
        owner = _request_owner.get()
        if owner is not None:
            owner.append(reservation)
        return reservation

    @asynccontextmanager
    async def reconfigure(self):
        """无活动分析时独占配置写入,取消时也撤销配置占用标记以恢复准入。"""
        async with self._lock:
            if self._active or self._configuring:
                raise ConfigurationBusy("analysis is active; change settings after it ends")
            self._configuring = True
        try:
            yield
        finally:
            with anyio.CancelScope(shield=True):
                async with self._lock:
                    self._configuring = False

    async def _release(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)


class ConfigurationBusy(RuntimeError):
    pass


_request_owner: contextvars.ContextVar[list[AnalysisReservation] | None] = contextvars.ContextVar(
    "analysis_request_owner", default=None
)


class AnalysisLifecycleMiddleware:
    """将请求槽位持有到完整 ASGI 响应结束,包含 SSE 与发送失败场景。

    先关闭流和记录运行结局,再释放未转交后台的槽位。清理屏蔽请求取消,
    记录收尾失败仍继续释放;最后恢复 ContextVar,避免污染后续请求。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        owned: list[AnalysisReservation] = []
        token = _request_owner.set(owned)
        try:
            await self.app(scope, receive, send)
        finally:
            try:
                with anyio.CancelScope(shield=True):
                    for reservation in owned:
                        if reservation.detached:
                            continue
                        try:
                            if reservation.close_stream is not None:
                                await reservation.close_stream()
                            if reservation.on_close is not None:
                                reservation.on_close()
                        except Exception:
                            logging.getLogger(__name__).exception(
                                "analysis final state could not be persisted"
                            )
                        finally:
                            with suppress(asyncio.CancelledError):
                                await reservation.release()
            finally:
                _request_owner.reset(token)


def _loopback_name(value: str | None) -> bool:
    """仅识别 localhost 与 loopback IP 字面量,不通过 DNS 信任任意域名。"""
    if not value:
        return False
    host = value.strip().strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _protected_path(path: str) -> bool:
    """圈定读取本机研究数据、昂贵调用和配置写操作的 HTTP 来源检查范围。"""
    return (
        path in _PROTECTED_EXACT
        or any(path.startswith(prefix) for prefix in _PROTECTED_PREFIXES)
        or (path.startswith("/runs/") and path.endswith("/resume"))
    )


class LoopbackHostMiddleware:
    """HTTP/WS 仅接受 loopback Host,包含带端口的 IPv6;监听绑定由配置另行限制。"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        raw_host = Headers(scope=scope).get("host", "")
        try:
            host = urlsplit(f"//{raw_host}").hostname
        except ValueError:
            host = None
        if not _loopback_name(host):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008, "reason": "Invalid host"})
            else:
                response = PlainTextResponse("Invalid host header", status_code=400)
                await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class LocalOriginGuardMiddleware(BaseHTTPMiddleware):
    """在受保护路由执行前拒绝跨站浏览器请求,不替代调用者身份认证。

    有 Origin 时要求本地来源,缺少 Origin 时要求客户端地址为 loopback;
    Sec-Fetch-Site 明示跨站则直接拒绝。同源页面脚本仍须由前端保证安全。
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not _protected_path(request.url.path):
            return await call_next(request)

        fetch_site = request.headers.get("sec-fetch-site")
        if fetch_site is not None and fetch_site.strip().lower() not in _ALLOWED_FETCH_SITES:
            return JSONResponse(
                {"detail": "cross-site request rejected"},
                status_code=403,
            )

        origin = request.headers.get("origin")
        if origin:
            try:
                origin_host = urlsplit(origin).hostname
            except ValueError:
                origin_host = None
            if not _loopback_name(origin_host):
                return JSONResponse(
                    {"detail": "cross-site request rejected"},
                    status_code=403,
                )
        elif not _loopback_name(request.client.host if request.client else None):
            return JSONResponse(
                {"detail": "non-loopback client rejected"},
                status_code=403,
            )

        return await call_next(request)


def valid_local_origin(scope: Scope) -> bool:
    """供 WS 接受连接前检查本地来源;Host 限制仍由 LoopbackHostMiddleware 负责。"""
    headers = Headers(scope=scope)
    fetch_site = headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site.strip().lower() not in _ALLOWED_FETCH_SITES:
        return False
    origin = headers.get("origin")
    if origin:
        try:
            return _loopback_name(urlsplit(origin).hostname)
        except ValueError:
            return False
    client = scope.get("client")
    return bool(client and _loopback_name(client[0]))
