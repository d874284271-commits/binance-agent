"""FRED —— 美联储官方免费 key,发布日历 → 宏观事件(无预期值/星级)。

仅保留中高重要性发布(NFP/CPI/GDP/FOMC 等)降噪,喂风险 agent 做临近事件告警。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from ..base import (
    FetchContext,
    Provider,
    ProviderResult,
    parse_json_response,
    row_result,
    text_field,
)
from ..models import MacroEvent

_URL = "https://api.stlouisfed.org/fred/releases/dates"

# 发布名关键字 → 重要性(1-3);FRED 无星级,这里据经验映射
_IMPORTANCE: dict[str, int] = {
    "employment situation": 3,  # 非农 NFP
    "consumer price index": 3,  # CPI
    "gross domestic product": 3,  # GDP
    "fomc": 3,
    "federal open market": 3,
    "personal income": 2,  # PCE
    "producer price index": 2,  # PPI
    "retail sales": 2,
    "unemployment": 2,
}


def _importance(name: str) -> int:
    low = name.lower()
    for keyword, level in _IMPORTANCE.items():
        if keyword in low:
            return level
    return 1


class FredProvider(Provider):
    name = "fred"
    source_kind = "official"

    def __init__(self, client: httpx.AsyncClient, key: str, ahead_days: int = 7) -> None:
        self._client = client
        self._key = key
        self._ahead_days = ahead_days

    async def fetch(self, ctx: FetchContext) -> ProviderResult:
        """采集未来配置天数内的 FRED 发布日历,保留本地规则认定的中高重要性事件。

        同名同日去重;上游只有日期时以 UTC 零点表示,不代表实际精确发布时间。
        """
        today = datetime.now(timezone.utc).date()
        end = today + timedelta(days=self._ahead_days)
        resp = await self._client.get(
            _URL,
            params={
                "api_key": self._key,
                "file_type": "json",
                "realtime_start": today.isoformat(),
                "realtime_end": end.isoformat(),
                "include_release_dates_with_no_data": "false",
                "sort_order": "asc",
                "limit": 1000,
            },
        )
        if resp.status_code != 200:
            return ProviderResult(source=self.name, ok=False, error=f"HTTP {resp.status_code}")
        payload = parse_json_response(resp)
        if not isinstance(payload, dict):
            return ProviderResult(source=self.name, ok=False, error="json_parse_error")
        rows = payload.get("release_dates") or []

        if not isinstance(rows, list):
            return ProviderResult(source=self.name, ok=False, error="invalid_payload")
        events: list[MacroEvent] = []
        seen: set[tuple[str, str]] = set()
        skipped = 0
        valid = 0
        for row in rows:
            try:
                name = text_field(row.get("release_name", ""))
                date_str = text_field(row.get("date", ""))
                if not name or (name, date_str) in seen:
                    continue
                event_time = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                valid += 1
                seen.add((name, date_str))
                importance = _importance(name)
                if importance < 2:
                    continue
                events.append(
                    MacroEvent(name=name, time=event_time, importance=importance, source="FRED")
                )
            except (AttributeError, TypeError, ValueError):
                skipped += 1
        return row_result(self.name, skipped, valid, macro_events=events)
