"""M3 指标纯函数测试:固定输入校验数值正确性。"""

from __future__ import annotations

import numpy as np

from binance_agent.indicators import ta


def test_sma() -> None:
    out = ta.sma([1, 2, 3, 4, 5], 3)
    assert np.isnan(out[1])
    assert out[2] == 2.0  # (1+2+3)/3
    assert out[-1] == 4.0  # (3+4+5)/3


def test_ema_recursion() -> None:
    out = ta.ema([1, 2, 3], 2)  # alpha = 2/3
    assert out[0] == 1.0
    assert abs(out[1] - (2 / 3 * 2 + 1 / 3 * 1)) < 1e-9
    assert abs(out[2] - (2 / 3 * 3 + 1 / 3 * out[1])) < 1e-9


def test_rsi_extremes() -> None:
    increasing = list(range(1, 30))
    assert ta.rsi(increasing, 14)[-1] == 100.0  # 无下跌 → 100
    decreasing = list(range(30, 1, -1))
    assert ta.rsi(decreasing, 14)[-1] == 0.0  # 无上涨 → 0


def test_macd_hist_identity() -> None:
    macd_line, signal_line, hist = ta.macd(list(range(1, 60)))
    assert len(macd_line) == len(signal_line) == len(hist) == 59
    assert abs(hist[-1] - (macd_line[-1] - signal_line[-1])) < 1e-9


def test_bollinger_constant_series() -> None:
    upper, middle, lower = ta.bollinger([5.0] * 25, 20)
    assert abs(upper[-1] - 5.0) < 1e-9
    assert abs(middle[-1] - 5.0) < 1e-9
    assert abs(lower[-1] - 5.0) < 1e-9  # std=0 → 三线重合


def test_atr_constant_true_range() -> None:
    lows = [float(i) for i in range(16)]
    highs = [x + 1 for x in lows]
    closes = [x + 0.5 for x in lows]
    out = ta.atr(highs, lows, closes, 14)
    assert abs(out[-1] - 1.5) < 1e-9  # 每根 TR 恒为 1.5


def test_summarize_keys_and_values() -> None:
    closes = [float(i) for i in range(1, 40)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    summary = ta.summarize(highs, lows, closes)

    assert summary["close"] == 39.0
    assert summary["rsi14"] == 100.0
    assert summary["atr14"] is not None
    for key in ("sma20", "ema12", "macd", "boll_upper", "boll_position"):
        assert key in summary
