"""币安衍生品情绪 —— 复用 market 层 fapi(免费替代 Coinglass)。

多空账户比 + 资金费率 + 标记价,作为数值情绪特征直接纳入(无需 LLM)。
"""

from __future__ import annotations

from typing import Any

from ...market.rest_client import BinanceRestClient
from ..base import FetchContext, Provider, ProviderResult, finite_number, row_result


def _to_float(value: Any) -> float | None:
    try:
        return finite_number(value)
    except (TypeError, ValueError, OverflowError):
        return None


class BinanceDerivProvider(Provider):
    name = "binance_deriv"
    source_kind = "official"

    def __init__(self, rest: BinanceRestClient) -> None:
        self._rest = rest

    async def fetch(self, ctx: FetchContext) -> ProviderResult:
        """复用公共合约 REST 客户端采集多空账户比、资金费率和标记价。

        坏字段保留警告,至少一个有效值时保留部分结果;不调用 LLM,也不关闭共享 client。
        """
        features: dict[str, Any] = {}

        skipped = 0
        valid = 0
        if ctx.symbol is None:
            return ProviderResult(source=self.name, ok=False, error="asset_required")
        lsr = await self._rest.global_long_short_ratio(ctx.symbol, limit=1)
        if isinstance(lsr, list) and lsr:
            for row in reversed(lsr):
                value = _to_float(row.get("longShortRatio")) if isinstance(row, dict) else None
                if value is None:
                    skipped += 1
                    continue
                features["long_short_ratio"] = value
                valid += 1
                break
        elif not isinstance(lsr, list):
            skipped += 1

        premium = await self._rest.premium_index(ctx.symbol)
        if isinstance(premium, dict):
            for output, key in (("funding_rate", "lastFundingRate"), ("mark_price", "markPrice")):
                value = _to_float(premium.get(key))
                if value is None:
                    skipped += 1
                else:
                    features[output] = value
                    valid += 1
        else:
            skipped += 1

        return row_result(self.name, skipped, valid, sentiment_features=features)
