"""akshare_cn 纯函数测试:中文源时间(北京时)→ UTC 正确转换(Bug2 回归)。

模块顶层不 import akshare(仅 fetch 内按需导入),故 _parse_dt 可独立测试,无需安装 intel-cn extra。
"""

from __future__ import annotations

from datetime import datetime, timezone

from binance_agent.intel.providers.akshare_cn import _parse_dt


def test_parse_dt_beijing_to_utc() -> None:
    # 北京时 20:00 → UTC 12:00(-8h);此前 bug 会当作 UTC 20:00 产生 +8h 偏差
    dt = _parse_dt("2026-06-03 20:00")
    assert dt.tzinfo == timezone.utc
    assert dt == datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)


def test_parse_dt_full_datetime_with_seconds() -> None:
    dt = _parse_dt("2026-06-03 20:30:45")
    assert dt == datetime(2026, 6, 3, 12, 30, 45, tzinfo=timezone.utc)


def test_parse_dt_date_only_is_beijing_midnight() -> None:
    # 仅日期 → 北京时 00:00 → UTC 前一日 16:00
    dt = _parse_dt("2026-06-03")
    assert dt == datetime(2026, 6, 2, 16, 0, tzinfo=timezone.utc)


def test_parse_dt_invalid_returns_utc_aware() -> None:
    dt = _parse_dt("不是时间")
    assert dt.tzinfo == timezone.utc
