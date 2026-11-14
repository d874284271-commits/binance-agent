"""效果闭环 CLI:``python -m binance_agent.eval``。

扫描 ``.runs/runs.jsonl`` 中「已完成、建议完成满 N 小时、尚未打标」的 run,逐条回拉建议生成
后 N 小时的真实行情、回溯裁决 outcome、写入 ``.runs/outcomes.jsonl``,并打印质量统计。

设计:核心 :func:`tag_pending` 不做 IO 之外的副作用(不打印、client 由外部注入),便于测试;
``run_cli`` 负责构造真实 store/client 与打印。回拉异常的 run 标记为 fetch_failed 且**不写库**
(下次重试),不中断整批。触发方式为显式 CLI(可挂系统计划任务),不常驻应用进程。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..app.outcome_store import FileOutcomeStore
from ..app.run_store import FileRunStore
from ..infra.paths import resolve_data_path
from ..infra.runtime_config import apply_overrides, load_overrides
from ..infra.settings import Settings, load_settings
from ..market.rest_client import BINANCE_KLINE_INTERVALS, BinanceRestClient
from . import COMPLIANCE_NOTICE
from .calibration import build_export_payload, format_reliability, reliability, suggest_caps
from .stats import aggregate, format_summary
from .tagging import UNEVALUABLE, evaluation_window_ms, normalize_klines, tag_run

logger = logging.getLogger(__name__)

ClientFactory = Callable[[str], BinanceRestClient]
MAX_OUTCOME_WINDOW_HOURS = 24 * 90


@dataclass
class TagSummary:
    scanned: int = 0
    due: int = 0
    tagged: int = 0
    fetch_failed: int = 0
    by_outcome: dict[str, int] = field(default_factory=dict)


async def tag_pending(
    *,
    settings: Settings,
    run_store: FileRunStore,
    outcome_store: FileOutcomeStore,
    client_factory: ClientFactory,
    now: datetime | None = None,
    window_hours: int | None = None,
) -> TagSummary:
    """对到期未打标的 run 回拉打标。``window_hours`` 同时是「最小成熟时长」与「回拉窗口长度」。"""
    now = now or datetime.now(timezone.utc)
    window = window_hours if window_hours is not None else settings.runtime.outcome_window_hours
    interval = settings.runtime.outcome_kline_interval
    if not 1 <= window <= MAX_OUTCOME_WINDOW_HOURS:
        raise ValueError(f"window_hours 必须在 1..{MAX_OUTCOME_WINDOW_HOURS} 之间")
    if interval not in BINANCE_KLINE_INTERVALS:
        raise ValueError(f"不支持的 outcome_kline_interval: {interval}")
    tagged = outcome_store.tagged_run_ids()
    cutoff = now - timedelta(hours=window)

    summary = TagSummary()
    outcomes: Counter[str] = Counter()

    for run in run_store.list_all():
        summary.scanned += 1
        run_id = str(run.get("run_id") or "")
        if not run_id or run.get("status") != "completed" or run_id in tagged:
            continue
        completed = _parse_iso(run.get("completed_at"))
        if completed is None:
            summary.due += 1
            record = tag_run(run, [], window_hours=window, kline_interval=interval)
            record.outcome = UNEVALUABLE
            record.reason = "缺失或无法解析 completed_at, 无法确定评估窗口"
            outcome_store.append(**record.append_kwargs())
            summary.tagged += 1
            outcomes[record.outcome] += 1
            continue
        if completed > cutoff:
            continue  # 尚未成熟(不足 window_hours)
        summary.due += 1

        final = run.get("final_advice")
        needs_market = (
            isinstance(final, dict)
            and final.get("decision_type") == "trade"
            and final.get("direction") in ("long", "short")
        )
        if not needs_market:
            record = tag_run(run, [], window_hours=window, kline_interval=interval)
        else:
            try:
                raw = await _fetch_window(client_factory, run, completed, window, interval)
            except Exception as exc:  # 回拉失败 → 不写库,下次重试;不中断整批
                summary.fetch_failed += 1
                logger.warning("run %s 回拉行情失败,跳过(下次重试): %s", run_id, exc)
                continue
            record = tag_run(
                run, normalize_klines(raw), window_hours=window, kline_interval=interval
            )

        outcome_store.append(**record.append_kwargs())
        summary.tagged += 1
        outcomes[record.outcome] += 1

    summary.by_outcome = dict(outcomes)
    return summary


async def _fetch_window(
    client_factory: ClientFactory,
    run: dict[str, Any],
    completed: datetime,
    window_hours: int,
    interval: str,
) -> Any:
    market = str(run.get("market") or "fapi")
    symbol = str(run.get("symbol") or "")
    start_ms, end_ms = evaluation_window_ms(completed, window_hours)
    client = client_factory(market)
    return await client.klines_range(
        symbol, interval, start_time=start_ms, end_time=end_ms, limit=1000
    )


async def run_cli(
    settings: Settings,
    *,
    window_hours: int | None = None,
    proposal_out: Path | None = None,
) -> TagSummary:
    """构造真实 store/client 执行打标并打印统计;供 ``main`` 与手动调用。"""
    run_store = FileRunStore(resolve_data_path(".runs/runs.jsonl"))
    outcome_store = FileOutcomeStore(resolve_data_path(".runs/outcomes.jsonl"))
    proxy = settings.network.proxy or None
    clients: dict[str, BinanceRestClient] = {}

    def factory(market: str) -> BinanceRestClient:
        key = market if market in ("fapi", "api") else "fapi"
        if key not in clients:
            clients[key] = BinanceRestClient(key, timeout=10.0, proxy=proxy)
        return clients[key]

    try:
        summary = await tag_pending(
            settings=settings,
            run_store=run_store,
            outcome_store=outcome_store,
            client_factory=factory,
            window_hours=window_hours,
        )
    finally:
        for client in clients.values():
            await client.aclose()

    records = outcome_store.list_all()
    print(format_summary(aggregate(records)))
    report = reliability(records)
    print(format_reliability(report, suggest_caps(report)))
    if proposal_out is not None:
        export_proposal(proposal_out, records=records, settings=settings)
    print(COMPLIANCE_NOTICE)
    return summary


def export_proposal(
    path: Path,
    *,
    records: list[dict[str, Any]],
    settings: Settings,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """写出可审计 JSON;仅导出建议,不提供自动应用路径。"""

    payload = build_export_payload(
        records,
        current_policy=settings.confidence.model_dump(mode="json"),
        generated_at=generated_at or datetime.now(timezone.utc),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m binance_agent.eval",
        description="效果闭环:回溯性给历史分析建议打 outcome 标签并统计(仅观测,不下单、非绩效)。",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=None,
        help="覆盖打标窗口/最小成熟时长(小时);默认取 config 的 runtime.outcome_window_hours",
    )
    parser.add_argument(
        "--proposal-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="导出校准报告与人工 proposal JSON;不会自动应用配置",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    apply_overrides(settings, load_overrides())  # 与 server 一致:config.toml 为基,叠加运行时配置
    summary = asyncio.run(
        run_cli(
            settings,
            window_hours=args.window_hours,
            proposal_out=args.proposal_out,
        )
    )
    print(
        f"扫描 {summary.scanned} / 到期 {summary.due} / 新打标 {summary.tagged} / "
        f"回拉失败跳过 {summary.fetch_failed}"
    )


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


if __name__ == "__main__":
    main()
