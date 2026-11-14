"""技术指标:用 numpy 计算,与行情输入保持同一价格单位。

数组型指标返回与输入等长的 float 数组,预热位置填 NaN;EMA 自首值起有值。
调用方提供按时间升序的一维数据、正整数周期,以及等长的 high/low/close。
latest 与 summarize 将末值转换为可序列化标量,不在这里补齐缺失行情。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _as_array(values: Sequence[float] | Array) -> Array:
    return np.asarray(values, dtype=np.float64)


def sma(values: Sequence[float] | Array, period: int) -> Array:
    """计算简单移动平均;前 period - 1 个位置或不足一个窗口时返回 NaN。"""
    a = _as_array(values)
    out = np.full(a.shape, np.nan)
    if len(a) >= period:
        windows = np.lib.stride_tricks.sliding_window_view(a, period)
        out[period - 1 :] = windows.mean(axis=1)
    return out


def ema(values: Sequence[float] | Array, period: int) -> Array:
    """按 adjust=False 递推指数移动平均,平滑系数为 2 / (period + 1)。

    首个输出直接取首个输入,不额外设置预热期;空输入返回空数组。
    """
    a = _as_array(values)
    out = np.full(a.shape, np.nan)
    if len(a) == 0:
        return out
    alpha = 2.0 / (period + 1)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(values: Sequence[float] | Array, period: int = 14) -> Array:
    """按 Wilder 平滑计算 0-100 的 RSI;至少需要 period + 1 个价格。

    前 period 个位置为 NaN;平均跌幅为 0 时输出 100(也包括全程持平)。
    """
    a = _as_array(values)
    n = len(a)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta = np.diff(a)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    def _rsi(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    avg_gain = float(gain[:period].mean())
    avg_loss = float(loss[:period].mean())
    out[period] = _rsi(avg_gain, avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i - 1]) / period
        out[i] = _rsi(avg_gain, avg_loss)
    return out


def macd(
    values: Sequence[float] | Array,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[Array, Array, Array]:
    """返回 MACD 线、信号线和柱值,顺序固定,数值单位与输入价格相同。

    三者沿用 ema 的首值初始化规则,不另行清空慢周期之前的输出。
    """
    a = _as_array(values)
    macd_line = ema(a, fast) - ema(a, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(
    values: Sequence[float] | Array, period: int = 20, num_std: float = 2.0
) -> tuple[Array, Array, Array]:
    """返回上轨、中轨、下轨;标准差使用窗口总体标准差(ddof=0)。

    前 period - 1 个位置为 NaN,轨道宽度为 num_std 倍标准差。
    """
    a = _as_array(values)
    n = len(a)
    middle = sma(a, period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    if n >= period:
        windows = np.lib.stride_tricks.sliding_window_view(a, period)
        std = windows.std(axis=1)
        upper[period - 1 :] = middle[period - 1 :] + num_std * std
        lower[period - 1 :] = middle[period - 1 :] - num_std * std
    return upper, middle, lower


def atr(
    high: Sequence[float] | Array,
    low: Sequence[float] | Array,
    close: Sequence[float] | Array,
    period: int = 14,
) -> Array:
    """按 Wilder 平滑计算真实波幅均值,返回与价格同单位的数组。

    high、low、close 应等长且按时间升序;前 period 个位置为 NaN,
    首个有效值使用 period 段相邻收盘价形成的真实波幅均值。
    """
    h = _as_array(high)
    low_arr = _as_array(low)
    c = _as_array(close)
    n = len(c)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    prev_close = c[:-1]
    true_range = np.maximum(
        h[1:] - low_arr[1:],
        np.maximum(np.abs(h[1:] - prev_close), np.abs(low_arr[1:] - prev_close)),
    )
    current = float(true_range[:period].mean())
    out[period] = current
    for i in range(period + 1, n):
        current = (current * (period - 1) + true_range[i - 1]) / period
        out[i] = current
    return out


def latest(arr: Array) -> float | None:
    """返回末值并四舍五入到 6 位小数;空数组或末值 NaN 返回 None。"""
    if len(arr) == 0 or np.isnan(arr[-1]):
        return None
    return round(float(arr[-1]), 6)


def summarize(
    highs: Sequence[float] | Array,
    lows: Sequence[float] | Array,
    closes: Sequence[float] | Array,
) -> dict[str, float | None]:
    """汇总末值价格、趋势、动量及波动指标,供分析师读取。

    预热不足的指标保留 None;boll_position 表示在上下轨之间的位置,
    0 为下轨、1 为上轨,价格突破时可超出此范围,轨道宽度为 0 时为 None。
    """
    closes_arr = _as_array(closes)
    macd_line, signal_line, hist = macd(closes_arr)
    upper, middle, lower = bollinger(closes_arr)
    last_close = latest(closes_arr)
    boll_upper = latest(upper)
    boll_lower = latest(lower)
    boll_position: float | None = None
    if last_close is not None and boll_upper is not None and boll_lower is not None:
        span = boll_upper - boll_lower
        if span > 0:
            boll_position = round((last_close - boll_lower) / span, 4)
    return {
        "close": last_close,
        "sma20": latest(sma(closes_arr, 20)),
        "ema12": latest(ema(closes_arr, 12)),
        "ema26": latest(ema(closes_arr, 26)),
        "rsi14": latest(rsi(closes_arr, 14)),
        "macd": latest(macd_line),
        "macd_signal": latest(signal_line),
        "macd_hist": latest(hist),
        "boll_upper": boll_upper,
        "boll_middle": latest(middle),
        "boll_lower": boll_lower,
        "boll_position": boll_position,  # 0=下轨 1=上轨
        "atr14": latest(atr(highs, lows, closes_arr, 14)),
    }
