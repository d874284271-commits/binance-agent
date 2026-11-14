"""SSE 序列化辅助:把图节点输出(可能含 pydantic 模型)转为 JSON 友好结构。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel

from ..infra.async_cleanup import wait_for_cleanup


class ContextIterator:
    """用同一驱动任务拥有生成器的迭代、关闭和 ContextVar 生命周期。

    响应索取下一事件时才驱动迭代;关闭时取消并收齐驱动任务,之后请求才可释放执行槽。
    """

    def __init__(
        self,
        iterator: AsyncIterator[Any],
        variable: ContextVar[Any] | None = None,
        value: Any = None,
    ):
        self.iterator = iterator
        self.variable = variable
        self.value = value
        self._commands: asyncio.Queue[asyncio.Future[Any]] = asyncio.Queue()
        self._driver: asyncio.Task[None] | None = None
        self._closing: asyncio.Task[None] | None = None

    def __aiter__(self):
        return self

    async def _drive(self) -> None:
        try:
            while True:
                reply = await self._commands.get()
                token = self.variable.set(self.value) if self.variable is not None else None
                try:
                    value = await self.iterator.__anext__()
                except BaseException as exc:
                    if not reply.done():
                        reply.set_exception(exc)
                    return
                else:
                    if not reply.done():
                        reply.set_result(value)
                finally:
                    if token is not None and self.variable is not None:
                        self.variable.reset(token)
        finally:
            close = getattr(self.iterator, "aclose", None)
            if close is not None:
                await close()

    async def __anext__(self):
        if self._closing is not None or (self._driver is not None and self._driver.done()):
            raise StopAsyncIteration
        if self._driver is None:
            self._driver = asyncio.create_task(self._drive())
        reply = asyncio.get_running_loop().create_future()
        await self._commands.put(reply)
        return await reply

    async def aclose(self):
        """共享一次关闭操作,确保重复关闭或等待者取消不会跳过生成器清理。"""
        if self._closing is None:

            async def finish() -> None:
                if self._driver is None:
                    close = getattr(self.iterator, "aclose", None)
                    if close is not None:
                        await close()
                else:
                    self._driver.cancel()
                    await asyncio.gather(self._driver, return_exceptions=True)

            self._closing = asyncio.create_task(finish())
        await wait_for_cleanup(self._closing)


def jsonable(obj: Any) -> Any:
    """递归转换 Pydantic 模型、字典及序列为 JSON 友好结构;其他值保持原样,不做脱敏。"""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {key: jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
    return obj
