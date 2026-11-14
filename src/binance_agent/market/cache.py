"""极简 TTL 缓存:同一交易对短时间内重复请求直接命中,降低权重消耗。"""

from __future__ import annotations

import time
from typing import Any


class TTLCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        """返回尚未过期的原对象;缺失或已过期返回 None,过期项在读取时移除。"""
        item = self._store.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at < time.monotonic():
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any, ttl: float) -> None:
        """用单调时钟设置 ttl 秒有效期;保存对象引用,不复制载荷。"""
        self._store[key] = (time.monotonic() + ttl, value)
