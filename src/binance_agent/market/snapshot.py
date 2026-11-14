"""行情快照:并发拉取多端点,容错组装为结构化 MarketSnapshot。

单端点失败不影响整体(记录到 errors),供编排层注入分析师使用。
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .rest_client import BinanceRestClient

logger = logging.getLogger(__name__)

DEFAULT_INTERVALS: tuple[str, ...] = ("5m", "1h", "4h", "1d")


class Kline(BaseModel):
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


class FundingInfo(BaseModel):
    mark_price: float | None = None
    last_funding_rate: float | None = None
    next_funding_time: int | None = None


class LongShortRatio(BaseModel):
    timestamp: int
    long_short_ratio: float
    long_account: float | None = None
    short_account: float | None = None


class MarketSnapshot(BaseModel):
    symbol: str
    market: str = "fapi"
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    klines: dict[str, list[Kline]] = Field(default_factory=dict)
    ticker_24h: dict[str, Any] = Field(default_factory=dict)
    funding: FundingInfo = Field(default_factory=FundingInfo)
    open_interest: float | None = None
    open_interest_hist: list[dict[str, Any]] = Field(default_factory=list)
    long_short_ratio: list[LongShortRatio] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _to_float(value: Any) -> float | None:
    """把上游标量转为有限浮点数;空值、布尔值、非法文本及无穷值返回 None。"""
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None and number.is_integer() else None


def _parse_klines(raw: Any) -> list[Kline]:
    return _klines_with_errors(raw)[0]


def _klines_with_errors(raw: Any) -> tuple[list[Kline], int]:
    """逐行解析币安 K 线数组,返回有效行及跳过数,保留上游时间单位(毫秒)。

    坏行不会使整个端点失败,但跳过数交由调用方记录到快照 errors。
    """
    if not isinstance(raw, list):
        return [], 1
    klines: list[Kline] = []
    skipped = 0
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 7:
            skipped += 1
            continue
        times = [_to_int(row[0]), _to_int(row[6])]
        values = [_to_float(value) for value in row[1:6]]
        if any(value is None for value in times + values):
            skipped += 1
            continue
        klines.append(
            Kline.model_validate(
                dict(
                    zip(
                        ("open_time", "open", "high", "low", "close", "volume", "close_time"),
                        [times[0], *values, times[1]],
                        strict=True,
                    )
                )
            )
        )
    return klines, skipped


async def build_market_snapshot(
    client: BinanceRestClient,
    symbol: str,
    intervals: Sequence[str] = DEFAULT_INTERVALS,
) -> MarketSnapshot:
    """并发获取多周期 K 线及公共行情,返回容错组装的 MarketSnapshot。

    现货跳过衍生品端点;单端点失败或坏行写入 errors,其他有效数据仍保留。
    缺失字段不以零伪装,是否足以继续分析由下游数据预检决定;此函数不关闭 client。
    """
    symbol = symbol.upper()
    intervals = tuple(intervals)
    snapshot = MarketSnapshot(symbol=symbol, market=client.market)

    kline_coros = [client.klines(symbol, iv) for iv in intervals]
    deriv_coros = []
    if client.has_derivatives:  # 现货无资金费率/OI/多空比,跳过
        deriv_coros = [
            client.premium_index(symbol),
            client.funding_rate_history(symbol, limit=1),
            client.open_interest(symbol),
            client.open_interest_hist(symbol),
            client.global_long_short_ratio(symbol),
        ]
    results = await asyncio.gather(
        *kline_coros, *deriv_coros, client.ticker_24h(symbol), return_exceptions=True
    )

    n = len(intervals)
    for iv, res in zip(intervals, results[:n], strict=True):
        if isinstance(res, BaseException):
            snapshot.errors.append(f"klines[{iv}]: {res}")
        else:
            snapshot.klines[iv], skipped = _klines_with_errors(res)
            if skipped:
                snapshot.errors.append(f"klines[{iv}]: skipped_invalid_rows: {skipped}")

    ticker = results[-1]  # ticker 始终是最后一个

    if client.has_derivatives:
        premium = results[n]
        funding_hist = results[n + 1]
        oi = results[n + 2]
        oi_hist = results[n + 3]
        lsr = results[n + 4]

        if isinstance(premium, BaseException):
            snapshot.errors.append(f"premiumIndex: {premium}")
        elif isinstance(premium, dict):
            snapshot.funding.mark_price = _to_float(premium.get("markPrice"))
            snapshot.funding.last_funding_rate = _to_float(premium.get("lastFundingRate"))
            snapshot.funding.next_funding_time = _to_int(premium.get("nextFundingTime"))
            if any(value is None for value in snapshot.funding.model_dump().values()):
                snapshot.errors.append("premiumIndex: invalid_or_missing_value")
        else:
            snapshot.errors.append("premiumIndex: invalid_payload")

        # premiumIndex 已含资金费率;funding_hist 仅作补充回退
        if (
            snapshot.funding.last_funding_rate is None
            and not isinstance(funding_hist, BaseException)
            and isinstance(funding_hist, list)
            and funding_hist
        ):
            for row in reversed(funding_hist):
                rate = _to_float(row.get("fundingRate")) if isinstance(row, dict) else None
                if rate is None:
                    snapshot.errors.append("fundingRate: skipped_invalid_row")
                    continue
                snapshot.funding.last_funding_rate = rate
                break

        if isinstance(oi, BaseException):
            snapshot.errors.append(f"openInterest: {oi}")
        elif isinstance(oi, dict):
            snapshot.open_interest = _to_float(oi.get("openInterest"))
            if snapshot.open_interest is None:
                snapshot.errors.append("openInterest: invalid_value")
        else:
            snapshot.errors.append("openInterest: invalid_payload")

        if isinstance(oi_hist, BaseException):
            snapshot.errors.append(f"openInterestHist: {oi_hist}")
        elif isinstance(oi_hist, list):
            for row in oi_hist[-12:]:
                if (
                    not isinstance(row, dict)
                    or _to_int(row.get("timestamp")) is None
                    or _to_float(row.get("sumOpenInterest")) is None
                    or (
                        "sumOpenInterestValue" in row
                        and _to_float(row["sumOpenInterestValue"]) is None
                    )
                ):
                    snapshot.errors.append("openInterestHist: skipped_invalid_row")
                    continue
                snapshot.open_interest_hist.append(row)
        else:
            snapshot.errors.append("openInterestHist: invalid_payload")

        if isinstance(lsr, BaseException):
            snapshot.errors.append(f"longShortRatio: {lsr}")
        elif isinstance(lsr, list):
            for item in lsr[-12:]:
                timestamp = _to_int(item.get("timestamp")) if isinstance(item, dict) else None
                ratio = _to_float(item.get("longShortRatio")) if isinstance(item, dict) else None
                if timestamp is None or ratio is None:
                    snapshot.errors.append("longShortRatio: skipped_invalid_row")
                    continue
                snapshot.long_short_ratio.append(
                    LongShortRatio(
                        timestamp=timestamp,
                        long_short_ratio=ratio,
                        long_account=_to_float(item.get("longAccount")),
                        short_account=_to_float(item.get("shortAccount")),
                    )
                )
        else:
            snapshot.errors.append("longShortRatio: invalid_payload")

    if isinstance(ticker, BaseException):
        snapshot.errors.append(f"ticker24h: {ticker}")
    elif isinstance(ticker, dict):
        snapshot.ticker_24h = {
            key: ticker[key]
            for key in (
                "lastPrice",
                "priceChangePercent",
                "highPrice",
                "lowPrice",
                "volume",
                "quoteVolume",
            )
            if _to_float(ticker.get(key)) is not None
        }
        if len(snapshot.ticker_24h) != 6:
            snapshot.errors.append("ticker24h: invalid_or_missing_value")
    else:
        snapshot.errors.append("ticker24h: invalid_payload")

    return snapshot
