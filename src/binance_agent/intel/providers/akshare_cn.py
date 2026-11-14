"""AKShare 中文源 —— 免费库间接获取:百度宏观事件(带预期/前值/公布值) + 财联社快讯。

akshare 为同步库且体量大(可选依赖 intel-cn),放线程池执行;字段名按中文列
防御式解析(库版本变动时尽量不崩,best-effort)。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from ..base import FetchContext, Provider, ProviderResult
from ..models import MacroEvent, NewsItem

logger = logging.getLogger(__name__)

# 中国大陆无夏令时,固定 UTC+8;财联社/百度宏观时间均为北京时间。
# 用固定偏移而非 ZoneInfo,避免 Windows 缺 IANA tzdata 时报错。
_BEIJING = timezone(timedelta(hours=8))


def _first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, "", "--"):
            return record[key]
    return None


def _as_str(value: Any) -> str | None:
    if value in (None, "", "--"):
        return None
    return str(value).strip()


def _importance(value: Any) -> int:
    text = str(value or "")
    if "3" in text or "三" in text or text.count("★") == 3:
        return 3
    if "2" in text or "二" in text or text.count("★") == 2:
        return 2
    return 1


def _parse_dt(value: Any) -> datetime:
    """解析中文源时间(北京时间)→ 统一转 UTC。

    直接打 UTC tzinfo 会引入 +8h 偏差,污染 aggregator 时间窗过滤与临近宏观事件告警。
    故先按北京时(UTC+8)本地化,再转 UTC。
    """
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%H:%M"):
        try:
            parsed = datetime.strptime(text, fmt)
        except (ValueError, TypeError):
            continue
        if fmt == "%H:%M":  # 仅时间 → 补北京时"今天"
            now_bj = datetime.now(_BEIJING)
            parsed = parsed.replace(year=now_bj.year, month=now_bj.month, day=now_bj.day)
        return parsed.replace(tzinfo=_BEIJING).astimezone(timezone.utc)
    return datetime.now(timezone.utc)


class AkshareCnProvider(Provider):
    name = "akshare_cn"
    source_kind = "media"

    def __init__(self, max_news: int = 20) -> None:
        self._max_news = max_news

    async def fetch(self, ctx: FetchContext) -> ProviderResult:
        """在线程中调用同步 AKShare 宏观和快讯入口,返回本源的事件及新闻。

        可选依赖缺失由工厂禁用;两个同步取数函数遇到接口异常会各自返回空列表,
        因此空结果不证明上游成功。未处理异常由聚合器隔离,本函数不阻塞事件循环。
        """
        import akshare as ak  # 缺库时由工厂禁用,不会到这里

        events = await asyncio.to_thread(self._fetch_macro, ak)
        news = await asyncio.to_thread(self._fetch_news, ak)
        return ProviderResult(source=self.name, macro_events=events, news_items=news)

    def _fetch_macro(self, ak: Any) -> list[MacroEvent]:
        events: list[MacroEvent] = []
        try:
            df = ak.news_economic_baidu()
        except Exception as exc:  # 接口偶失效,降级
            logger.debug("akshare 宏观事件失败: %s", exc)
            return events
        for record in df.to_dict("records"):
            name = _first(record, "事件", "指标", "经济指标")
            if not name:
                continue
            events.append(
                MacroEvent(
                    name=str(name),
                    time=_parse_dt(_first(record, "日期", "时间", "公布时间")),
                    importance=_importance(_first(record, "重要性", "重要程度")),
                    forecast=_as_str(_first(record, "预测值", "预期")),
                    previous=_as_str(_first(record, "前值")),
                    actual=_as_str(_first(record, "公布值", "今值", "现值")),
                    source="百度-AKShare",
                )
            )
        return events

    def _fetch_news(self, ak: Any) -> list[NewsItem]:
        items: list[NewsItem] = []
        try:
            df = ak.stock_info_global_cls()
        except Exception as exc:  # 接口偶失效,降级
            logger.debug("akshare 快讯失败: %s", exc)
            return items
        for record in df.to_dict("records")[: self._max_news]:
            title = _first(record, "标题", "title")
            content = _first(record, "内容", "content")
            if not title and not content:
                continue
            published_raw = (
                f"{_first(record, '发布日期', '日期') or ''} "
                f"{_first(record, '发布时间', '时间') or ''}".strip()
            )
            try:
                published = (
                    datetime.fromisoformat(published_raw.strip())
                    .replace(tzinfo=_BEIJING)
                    .astimezone(timezone.utc)
                )
            except ValueError:
                published = None
            items.append(
                NewsItem(
                    title=str(title or content)[:120],
                    summary=str(content or "")[:500],
                    source="财联社-AKShare",
                    url="",
                    published_at=published,
                )
            )
        return items
