"""Provider 可插拔接口 —— 增删资讯源的关键抽象。

每个源实现统一 ``async fetch(ctx) -> ProviderResult``;aggregator 并发调度、
超时容错、合并去重。单源失败被隔离,不影响整体快照。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx

from .models import MacroEvent, NewsItem, SourceKind

# base_asset → 新闻检索常用名;未命中回退 base_asset 本身
_ASSET_NAMES: dict[str, str] = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "BNB": "BNB",
    "XRP": "XRP",
    "DOGE": "Dogecoin",
    "ADA": "Cardano",
}
_QUOTE_SUFFIXES = ("USDT", "USDC", "FDUSD", "BUSD", "TUSD", "USD", "DAI")


def split_base_asset(symbol: str) -> str:
    """BTCUSDT -> BTC;无法识别则原样返回。"""
    s = symbol.upper()
    for quote in _QUOTE_SUFFIXES:
        if s.endswith(quote) and len(s) > len(quote):
            return s[: -len(quote)]
    return s


def asset_search_name(base_asset: str) -> str:
    """将已知基础资产代码映射为常用新闻检索名;未配置别名时保留输入。"""
    return _ASSET_NAMES.get(base_asset.upper(), base_asset)


def parse_json_response(resp: httpx.Response) -> Any | None:
    """容错解析响应 JSON;畸形/非 JSON 返回 None(调用方应返回 ok=False)。

    重要:provider 内严禁 ``raise_for_status()`` —— 其异常消息内嵌完整请求 URL
    (可能含 auth_token/api_key 查询参数),会经 aggregator 的 error 串泄漏密钥。
    非 200 应显式 ``return ProviderResult(ok=False, error=f"HTTP {code}")``。
    """
    try:
        return resp.json()
    except (ValueError, TypeError):
        return None


@dataclass
class FetchContext:
    symbol: str | None  # 原始交易对,如 BTCUSDT
    base_asset: str | None  # 基础资产,如 BTC
    window_hours: int  # 时间窗(新闻/事件过滤)
    question: str = ""
    start_date: date | None = None
    end_date: date | None = None

    @property
    def search_name(self) -> str:
        return self.question or (asset_search_name(self.base_asset) if self.base_asset else "")


@dataclass
class ProviderResult:
    source: str
    ok: bool = True
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    sentiment_features: dict[str, Any] = field(default_factory=dict)
    news_items: list[NewsItem] = field(default_factory=list)
    macro_events: list[MacroEvent] = field(default_factory=list)


class Provider(ABC):
    """资讯源抽象基类。子类设置 ``name`` 并实现 ``fetch``。"""

    supports_independent: bool = False
    name: str = "provider"
    source_kind: SourceKind = "unknown"

    @abstractmethod
    async def fetch(self, ctx: FetchContext) -> ProviderResult: ...


def row_warnings(skipped: int) -> list[str]:
    """把坏行计数转换为结构化警告文本;没有坏行时返回空列表。"""
    return [f"skipped_invalid_rows: {skipped}"] if skipped else []


def row_result(source: str, skipped: int, valid: int, **kwargs: Any) -> ProviderResult:
    """组装来源结果;有坏行且没有有效行才失败,部分有效时保留结果和警告。"""
    return ProviderResult(
        source=source,
        ok=not (skipped and not valid),
        error="invalid_rows" if skipped and not valid else None,
        warnings=row_warnings(skipped),
        **kwargs,
    )


def finite_number(value: Any) -> float:
    """将上游数值转成有限 float;拒绝布尔值、NaN 和无穷值,转换错误继续抛出。"""
    if isinstance(value, bool):
        raise ValueError("invalid_number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("invalid_number")
    return number


def text_field(value: Any) -> str:
    """要求上游字段确为字符串,不隐式转换对象或数值;错误时抛 ValueError。"""
    if not isinstance(value, str):
        raise ValueError("invalid_text")
    return value
