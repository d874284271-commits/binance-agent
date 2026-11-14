"""公共 HTTP 与模型客户端共用的有限 Retry-After 时间解析。"""

from __future__ import annotations

import email.utils
import math
import time
from datetime import timezone
from typing import Any


def parse_retry_after(raw: Any, *, now: float | None = None) -> float | None:
    """把秒数或 HTTP 日期转换为非负等待秒数;无效/负数/非有限输入返回 None。

    now 是用于日期换算的 Unix 秒时间,默认取当前时刻;过期日期返回 0。
    这里仅解析,不等待也不截断时长,重试次数和总期限由调用者约束。
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(str(raw).strip())
    except (ValueError, TypeError):
        try:
            date = email.utils.parsedate_to_datetime(str(raw))
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            value = max(0.0, date.timestamp() - (time.time() if now is None else now))
        except (ValueError, TypeError, IndexError, OverflowError):
            return None
    return value if math.isfinite(value) and value >= 0 else None
