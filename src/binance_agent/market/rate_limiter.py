"""基于币安 X-MBX-USED-WEIGHT-1m 响应头的反馈式软限流。

达到可配置软上限后等到下一分钟,再配合 REST 客户端的 429/418 退避。
这只是当前进程观察到的权重,不替代币安按 IP 维护的实际限额。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Mapping

logger = logging.getLogger(__name__)


def _seconds_to_next_minute() -> float:
    return 60.0 - (time.time() % 60.0)


class RateLimiter:
    def __init__(self, soft_limit: int = 2000) -> None:
        self._soft_limit = soft_limit
        self._used_weight = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """达到软上限时串行等待到下一分钟并清零本地权重;不预估本次请求权重。"""
        async with self._lock:
            if self._used_weight >= self._soft_limit:
                wait = _seconds_to_next_minute()
                logger.warning(
                    "权重接近上限(%s/%s),节流 %.1fs",
                    self._used_weight,
                    self._soft_limit,
                    wait,
                )
                await asyncio.sleep(wait)
                self._used_weight = 0

    def reset(self) -> None:
        """清零已用权重(429/418 退避后调用,旧权重已过期)。"""
        self._used_weight = 0

    def observe(self, headers: Mapping[str, str]) -> None:
        """读取响应头中的分钟权重反馈;大小写不敏感,无法解析的值保持原计数。"""
        for key, value in headers.items():
            if key.lower().startswith("x-mbx-used-weight-1m"):
                with contextlib.suppress(ValueError):
                    self._used_weight = int(value)

    @property
    def used_weight(self) -> int:
        return self._used_weight
