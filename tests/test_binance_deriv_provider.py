"""币安衍生品情绪 provider 离线测试:特征解析 + 畸形负载容错(stub rest client)。"""

from __future__ import annotations

from typing import Any

from binance_agent.intel.base import FetchContext
from binance_agent.intel.providers.binance_deriv import BinanceDerivProvider


class _StubRest:
    def __init__(self, lsr: Any, premium: Any) -> None:
        self._lsr = lsr
        self._premium = premium

    async def global_long_short_ratio(self, symbol: str, limit: int = 1) -> Any:
        return self._lsr

    async def premium_index(self, symbol: str) -> Any:
        return self._premium


def _ctx() -> FetchContext:
    return FetchContext(symbol="BTCUSDT", base_asset="BTC", window_hours=24)


async def test_binance_deriv_parses_features() -> None:
    rest = _StubRest(
        lsr=[{"longShortRatio": "2.35"}],
        premium={"lastFundingRate": "0.0001", "markPrice": "65000.5"},
    )
    result = await BinanceDerivProvider(rest).fetch(_ctx())  # type: ignore[arg-type]

    assert result.ok is True
    assert result.sentiment_features == {
        "long_short_ratio": 2.35,
        "funding_rate": 0.0001,
        "mark_price": 65000.5,
    }


async def test_binance_deriv_tolerates_malformed_payloads() -> None:
    # Both invalid containers are failures, not a successful empty source.
    rest = _StubRest(lsr={"unexpected": "shape"}, premium="not a dict")
    result = await BinanceDerivProvider(rest).fetch(_ctx())  # type: ignore[arg-type]

    assert result.ok is False
    assert result.error == "invalid_rows"
    assert result.warnings == ["skipped_invalid_rows: 2"]
    assert result.sentiment_features == {}


async def test_binance_deriv_non_numeric_values_report_invalid_rows() -> None:
    rest = _StubRest(
        lsr=[{"longShortRatio": "n/a"}],
        premium={"lastFundingRate": None, "markPrice": "abc"},
    )
    result = await BinanceDerivProvider(rest).fetch(_ctx())  # type: ignore[arg-type]

    assert result.ok is False
    assert result.error == "invalid_rows"
    assert result.warnings == ["skipped_invalid_rows: 3"]
    assert result.sentiment_features == {}
