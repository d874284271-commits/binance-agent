"""区分等待者取消与清理任务取消,避免资源回收被提前中断或陷入空转。"""

from __future__ import annotations

import asyncio

import anyio


async def wait_for_cleanup(task: asyncio.Task[None]) -> None:
    """等待已拥有的清理任务结束,再向调用者传递等待期间收到的取消。

    子任务自身被取消或失败时直接传播结果;不能重复等待已取消的子任务,
    否则关闭过程会空转。此函数不设置超时,清理任务须自行保证有界完成。
    """
    cancelled = False
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        task.result()
    if cancelled:
        raise asyncio.CancelledError
