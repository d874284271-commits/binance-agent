"""Fear & Greed 指数(alternative.me)—— 无需 key,0-100 现成情绪分。"""

from __future__ import annotations

import httpx

from ..base import (
    FetchContext,
    Provider,
    ProviderResult,
    finite_number,
    parse_json_response,
    row_result,
)

_URL = "https://api.alternative.me/fng/"


class FearGreedProvider(Provider):
    name = "fear_greed"
    source_kind = "community"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def fetch(self, ctx: FetchContext) -> ProviderResult:
        """获取首个有效的 0-100 整数恐惧贪婪指数;无有效记录时返回失败及坏行警告。"""
        resp = await self._client.get(_URL, params={"limit": 1})
        if resp.status_code != 200:
            return ProviderResult(source=self.name, ok=False, error=f"HTTP {resp.status_code}")
        payload = parse_json_response(resp)
        if not isinstance(payload, dict):
            return ProviderResult(source=self.name, ok=False, error="json_parse_error")
        data = payload.get("data") or []
        if not data:
            return ProviderResult(source=self.name, ok=False, error="empty response")
        if not isinstance(data, list):
            return ProviderResult(source=self.name, ok=False, error="invalid_payload")
        skipped = 0
        for item in data:
            try:
                value = finite_number(item["value"])
                if not value.is_integer() or not 0 <= value <= 100:
                    raise ValueError("invalid_index")
                return row_result(
                    self.name,
                    skipped,
                    1,
                    sentiment_features={
                        "fear_greed": {
                            "value": int(value),
                            "classification": item.get("value_classification"),
                        }
                    },
                )
            except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
                skipped += 1
        return row_result(self.name, skipped, 0)
