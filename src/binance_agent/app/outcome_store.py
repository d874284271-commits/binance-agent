"""本地 OutcomeStore:效果闭环打标结果的 JSONL 台账(append-only,每行一条)。

沿用 :class:`~binance_agent.app.alert_store.FileAlertStore` 的「每行一条独立记录」范式:
一条 outcome = 对某次 ``/analyze`` 建议(run)在 T+N 小时后回拉真实行情的回溯裁决
(命中止盈 / 止损 / 未触发 …)。用于观测分析质量,**非交易绩效、不构成收益承诺**。

``tagged_run_ids()`` 供 CLI 幂等去重:已打标的 run 不再重复回拉/写入。
落盘前统一脱敏(reason 可能夹带错误文本)。
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..infra.file_io import append_jsonl, read_jsonl
from ..infra.logging import redact_text


class FileOutcomeStore:
    """Append-only JSONL outcome ledger(每行一条 outcome)。"""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)

    def append(
        self,
        *,
        run_id: str,
        symbol: str,
        market: str | None,
        outcome: str,
        direction: str | None = None,
        decision_type: str | None = None,
        confidence: float | None = None,
        pre_policy_confidence: float | None = None,
        model_confidence: float | None = None,
        entry: float | None = None,
        stop: float | None = None,
        targets: list[float] | None = None,
        r_realized: float | None = None,
        entered_at: str | None = None,
        evaluation_started_at: str | None = None,
        evaluation_ended_at: str | None = None,
        entry_fill: float | None = None,
        target_evaluated: float | None = None,
        policy_version: str | None = None,
        resolved_at: str | None = None,
        kline_interval: str | None = None,
        window_hours: int | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """追加一条 outcome,返回脱敏后的记录(与落盘内容一致)。"""
        record = {
            "outcome_id": uuid.uuid4().hex,
            "run_id": run_id,
            "symbol": symbol,
            "market": market,
            "direction": direction,
            "decision_type": decision_type,
            "confidence": confidence,
            "pre_policy_confidence": pre_policy_confidence,
            "model_confidence": model_confidence,
            "entry": entry,
            "stop": stop,
            "targets": targets or [],
            "outcome": outcome,
            "r_realized": r_realized,
            "entered_at": entered_at,
            "evaluation_started_at": evaluation_started_at,
            "evaluation_ended_at": evaluation_ended_at,
            "entry_fill": entry_fill,
            "target_evaluated": target_evaluated,
            "policy_version": policy_version,
            "resolved_at": resolved_at or _now_iso(),
            "kline_interval": kline_interval,
            "window_hours": window_hours,
            "reason": reason,
            "created_at": _now_iso(),
        }
        line = redact_text(json.dumps(record, ensure_ascii=False, default=str))
        with self._lock:
            append_jsonl(self._path, line)
        redacted: dict[str, Any] = json.loads(line)
        return redacted

    def list_recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """返回最新在前的若干条 outcome(append 顺序为时间正序,读取时反转)。"""
        records = self._read_all()
        records.reverse()
        return records[: max(0, limit)]

    def list_all(self) -> list[dict[str, Any]]:
        """全部 outcome 记录(append 顺序,最早在前),供统计聚合使用。"""
        return self._read_all()

    def tagged_run_ids(self) -> set[str]:
        """已打标的 run_id 集合,供 CLI 幂等去重:已在库中的 run 不再重复回拉/写入。"""
        return {str(run_id) for record in self._read_all() if (run_id := record.get("run_id"))}

    def _read_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return read_jsonl(self._path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
