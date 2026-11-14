"""币安合约(fapi)及现货(api)的只读 REST 客户端。

仅访问公共行情端点,不使用签名或交易密钥;根据响应权重头节流,
网络错误和 429/418 在有限重试及总时限内退避。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlencode

import httpx

from ..infra.retry import parse_retry_after
from .cache import TTLCache
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

FAPI_BASE = "https://fapi.binance.com"
SPOT_BASE = "https://api.binance.com"
_MAX_BACKOFF = 8.0
BINANCE_KLINE_INTERVALS = frozenset(
    {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
)

# 市场规格:基址 + K线/ticker 路径 + 是否有衍生品数据(资金费率/OI/多空比)
_MARKET_SPECS: dict[str, dict[str, Any]] = {
    "fapi": {
        "base": FAPI_BASE,
        "klines": "/fapi/v1/klines",
        "ticker": "/fapi/v1/ticker/24hr",
        "derivatives": True,
    },
    "api": {
        "base": SPOT_BASE,
        "klines": "/api/v3/klines",
        "ticker": "/api/v3/ticker/24hr",
        "derivatives": False,
    },
}


class BinanceRestClient:
    def __init__(
        self,
        market: str = "fapi",
        *,
        timeout: float = 10.0,
        rate_limiter: RateLimiter | None = None,
        cache: TTLCache | None = None,
        base_url: str | None = None,
        proxy: str | None = None,
    ) -> None:
        if market not in _MARKET_SPECS:
            raise ValueError(f"未知市场: {market}")
        spec = _MARKET_SPECS[market]
        self.market = market
        self.has_derivatives: bool = spec["derivatives"]
        self._spec = spec
        # proxy 显式给定时透传 httpx(覆盖 env);留空则 trust_env 自动读环境变量代理。
        client_kwargs: dict[str, Any] = {
            "base_url": base_url or spec["base"],
            "timeout": timeout,
            "headers": {"Accept": "application/json"},
        }
        if proxy:
            client_kwargs["proxy"] = proxy
        self._client = httpx.AsyncClient(**client_kwargs)
        self._rate_limiter = rate_limiter or RateLimiter()
        self._cache = cache or TTLCache()

    async def symbol_catalog(self) -> list[dict[str, str]]:
        """Actual currently trading pairs, shared cache and rate limiter."""
        path = "/api/v3/exchangeInfo" if self.market == "api" else "/fapi/v1/exchangeInfo"
        payload = await self._get(path, {}, ttl=3600)
        names = {
            "BTC": "Bitcoin / 比特币",
            "ETH": "Ethereum / 以太坊",
            "SOL": "Solana",
            "BNB": "BNB",
            "DOGE": "Dogecoin / 狗狗币",
            "XRP": "XRP",
            "ADA": "Cardano",
        }
        return [
            {
                "symbol": row["symbol"],
                "base_asset": row["baseAsset"],
                "quote_asset": row["quoteAsset"],
                "name": names.get(row["baseAsset"], row["baseAsset"]),
                "market": self.market,
            }
            for row in payload.get("symbols", [])
            if row.get("status") == "TRADING"
            and all(isinstance(row.get(k), str) for k in ("symbol", "baseAsset", "quoteAsset"))
        ]

    async def aclose(self) -> None:
        """关闭本实例持有的 HTTP client,调用方应在应用资源清理时等待完成。"""
        await self._client.aclose()

    async def _get(
        self,
        path: str,
        params: dict[str, Any],
        *,
        ttl: float = 0.0,
        max_retries: int = 3,
    ) -> Any:
        """在 45 秒总时限内获取公共 JSON;该时限包含节流、网络等待及重试退避。"""
        return await asyncio.wait_for(
            self._get_with_retries(path, params, ttl=ttl, max_retries=max_retries), 45.0
        )

    async def _get_with_retries(
        self,
        path: str,
        params: dict[str, Any],
        *,
        ttl: float = 0.0,
        max_retries: int = 3,
    ) -> Any:
        """按市场、路径及参数缓存成功响应,并在每次发送前遵守权重反馈。

        ttl 为秒;429/418 的 Retry-After 超过 30 秒时直接失败,其余退避有限。
        不重试普通业务 HTTP 错误;最终失败上抛,由快照层记录缺失及错误。
        """
        # 含市场前缀:防止显式共享 TTLCache 时 fapi/api 同路径同参数串数据
        cache_key = f"{self.market}#{path}?{urlencode(sorted(params.items()))}"
        if ttl:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            await self._rate_limiter.acquire()
            try:
                resp = await self._client.get(path, params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "网络异常[%s] path=%s attempt=%d: %s", type(exc).__name__, path, attempt, exc
                )
                await asyncio.sleep(min(2.0**attempt, _MAX_BACKOFF))
                continue

            self._rate_limiter.observe(resp.headers)
            if resp.status_code == 200:
                data = resp.json()
                if ttl:
                    self._cache.set(cache_key, data, ttl)
                return data
            if resp.status_code in (429, 418):
                retry_after = parse_retry_after(resp.headers.get("Retry-After"))
                if retry_after is None:
                    retry_after = min(2.0**attempt, _MAX_BACKOFF)
                if retry_after > 30.0:
                    resp.raise_for_status()
                logger.warning(
                    "币安限速 %s,退避 %.1fs (path=%s)", resp.status_code, retry_after, path
                )
                await asyncio.sleep(retry_after)
                # 退避后旧权重已过期(可能跨分钟窗),重置以免 acquire 二次节流
                self._rate_limiter.reset()
                continue
            resp.raise_for_status()

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"请求重试耗尽: {path}")

    # ── 公共行情端点(无需 Key)──
    async def klines(self, symbol: str, interval: str, limit: int = 200) -> Any:
        """返回指定市场的原始 K 线数组,短缓存 10 秒;时间字段仍为 Unix 毫秒。"""
        return await self._get(
            self._spec["klines"],
            {"symbol": symbol, "interval": interval, "limit": limit},
            ttl=10,
        )

    async def klines_range(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> Any:
        """按 Unix 毫秒的 start_time/end_time 回拉原始 K 线,不使用缓存。

        同时提供起止时间才自动分页,并按 open_time 去重排序;页大小为 1..1000,
        时间戳不前进时显式失败,避免截断数据被用于 Outcome 评估。
        """
        if interval not in BINANCE_KLINE_INTERVALS:
            raise ValueError(f"不支持的 K 线 interval: {interval}")
        if not 1 <= limit <= 1000:
            raise ValueError("K 线单页 limit 必须在 1..1000 之间")
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time is None or end_time is None:
            if start_time is not None:
                params["startTime"] = int(start_time)
            if end_time is not None:
                params["endTime"] = int(end_time)
            return await self._get(self._spec["klines"], params, ttl=0)

        cursor = int(start_time)
        end_ms = int(end_time)
        if cursor > end_ms:
            raise ValueError("K 线范围 start_time 不得晚于 end_time")

        rows_by_open_time: dict[int, Any] = {}
        while cursor <= end_ms:
            page_params = {
                **params,
                "startTime": cursor,
                "endTime": end_ms,
            }
            page = await self._get(self._spec["klines"], page_params, ttl=0)
            if not isinstance(page, list) or not page:
                break

            page_open_times: list[int] = []
            for row in page:
                try:
                    open_time = int(row[0])
                except (TypeError, ValueError, IndexError):
                    continue
                if open_time > end_ms:
                    continue
                rows_by_open_time[open_time] = row
                page_open_times.append(open_time)

            if len(page) < limit:
                break
            if not page_open_times:
                raise RuntimeError("K 线分页无可用时间戳, 无法继续")
            next_cursor = max(page_open_times) + 1
            if next_cursor <= cursor:
                raise RuntimeError("K 线分页无进展, 已停止以避免无限循环")
            cursor = next_cursor

        return [rows_by_open_time[key] for key in sorted(rows_by_open_time)]

    async def premium_index(self, symbol: str) -> Any:
        """标记价 + 资金费率 + 下次结算时间。"""
        return await self._get("/fapi/v1/premiumIndex", {"symbol": symbol}, ttl=10)

    async def funding_rate_history(self, symbol: str, limit: int = 10) -> Any:
        """读取合约资金费率历史,缓存 60 秒;返回原始比率而非百分数。"""
        return await self._get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit}, ttl=60)

    async def open_interest(self, symbol: str) -> Any:
        """读取合约当前未平仓量,缓存 30 秒;返回上游原始对象。"""
        return await self._get("/fapi/v1/openInterest", {"symbol": symbol}, ttl=30)

    async def open_interest_hist(self, symbol: str, period: str = "1h", limit: int = 24) -> Any:
        """读取合约未平仓量历史,period 使用币安周期标识,缓存 60 秒。"""
        return await self._get(
            "/futures/data/openInterestHist",
            {"symbol": symbol, "period": period, "limit": limit},
            ttl=60,
        )

    async def global_long_short_ratio(
        self, symbol: str, period: str = "1h", limit: int = 24
    ) -> Any:
        """读取合约全市场账户多空比历史,缓存 60 秒,不代表名义仓位金额比。"""
        return await self._get(
            "/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": period, "limit": limit},
            ttl=60,
        )

    async def ticker_24h(self, symbol: str) -> Any:
        """返回绑定市场的 24 小时 ticker 原始对象,缓存 5 秒。"""
        return await self._get(self._spec["ticker"], {"symbol": symbol}, ttl=5)
