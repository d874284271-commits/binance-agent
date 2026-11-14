"""币安实时价 WebSocket 组合流 + 自动重连。

按 market 选流:合约(fapi)走 wss://fstream.binance.com 的 <sym>@markPrice@1s
(标记价/资金费率);现货(api)走 wss://stream.binance.com:9443 的 <sym>@ticker(最新价 c)。
用组合流端点 /stream?streams=...(应对 2026-04-23 旧 URL 下线,未路由连接只收 /public 的问题);
断线指数退避重连。仅订阅公共行情(无需 Key)。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

# 用新版 asyncio 客户端(支持代理;legacy websockets.connect 不支持)
from websockets.asyncio.client import connect as ws_connect

logger = logging.getLogger(__name__)

FSTREAM_BASE = "wss://fstream.binance.com"
SPOT_STREAM_BASE = "wss://stream.binance.com:9443"
_MAX_BACKOFF = 30.0

# 市场规格:WS 基址 + 价格流名(@markPrice 合约 / @ticker 现货)+ 价格字段 + 是否有衍生品
# (资金费率/标记价)。对称于 rest_client._MARKET_SPECS。
_WS_MARKET_SPECS: dict[str, dict[str, Any]] = {
    "fapi": {
        "base": FSTREAM_BASE,
        "stream": "{sym}@markPrice@1s",
        "price_field": "p",
        "derivatives": True,
    },
    "api": {
        "base": SPOT_STREAM_BASE,
        "stream": "{sym}@ticker",
        "price_field": "c",
        "derivatives": False,
    },
}


def _env_proxy() -> str | None:
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        value = os.environ.get(name)
        if value:
            return value
    return None


class BinanceWsManager:
    def __init__(
        self, market: str = "fapi", *, base_url: str | None = None, proxy: str | None = None
    ) -> None:
        if market not in _WS_MARKET_SPECS:
            raise ValueError(f"未知市场: {market}")
        self.market = market
        self._spec = _WS_MARKET_SPECS[market]
        self._base = base_url or self._spec["base"]
        # 经系统代理路由(websockets 不会自动读 env);可达环境即生效
        self._proxy = proxy if proxy is not None else _env_proxy()

    async def stream(
        self, streams: list[str], *, stop: asyncio.Event | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """持续产出公共组合流的 data 对象;坏 JSON 或空载荷丢弃。

        正常和异常断开都按 1-30 秒退避,收到有效数据后重置退避。stop 可打断
        静默接收和重连等待;退出前取消并收齐接收任务,取消异常继续向调用方传播。
        """
        url = f"{self._base}/stream?streams=" + "/".join(streams)
        backoff = 1.0
        while stop is None or not stop.is_set():
            try:
                async with ws_connect(
                    url, proxy=self._proxy, ping_interval=20, ping_timeout=20, close_timeout=5
                ) as ws:
                    logger.info("WS 连接成功: %s", streams)
                    iterator = ws.__aiter__()
                    while stop is None or not stop.is_set():
                        receive = asyncio.ensure_future(iterator.__anext__())
                        stopped = asyncio.create_task(stop.wait()) if stop is not None else None
                        pending = [receive] + ([stopped] if stopped is not None else [])
                        try:
                            done, _ = await asyncio.wait(
                                pending, return_when=asyncio.FIRST_COMPLETED
                            )
                            if stopped is not None and stopped in done:
                                return
                            try:
                                raw = receive.result()
                            except StopAsyncIteration:
                                break
                        finally:
                            for task in pending:
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(*pending, return_exceptions=True)
                        try:
                            message = json.loads(raw)
                        except (ValueError, TypeError):
                            logger.debug("WS JSON 解析失败,丢弃")
                            continue
                        if not isinstance(message, dict):
                            continue
                        data = message.get("data", message)
                        if not isinstance(data, dict) or not data:
                            continue
                        backoff = 1.0
                        yield data
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("WS 断开,%.1fs 后重连: %s", backoff, exc)
            if stop is not None and stop.is_set():
                return
            # Normal closes need the same backoff as errors. Stop also interrupts idle delay.
            if stop is None:
                await asyncio.sleep(backoff)
            else:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=backoff)
                    return
                except asyncio.TimeoutError:
                    pass
            backoff = min(backoff * 2, _MAX_BACKOFF)

    async def price_stream(
        self, symbol: str, *, stop: asyncio.Event | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """按绑定市场把价格流映射为统一事件,保留上游价格字符串及毫秒时间戳。

        现货使用最新成交价,合约使用标记价;现货没有的资金费率等衍生品字段为 None。
        """
        spec = self._spec
        stream_name = spec["stream"].format(sym=symbol.lower())
        derivatives = spec["derivatives"]
        async for data in self.stream([stream_name], stop=stop):
            price = data.get(spec["price_field"])
            yield {
                "symbol": data.get("s"),
                "market": self.market,
                "price": price,
                "mark_price": price if derivatives else None,
                "funding_rate": data.get("r") if derivatives else None,
                "next_funding_time": data.get("T") if derivatives else None,
            }
