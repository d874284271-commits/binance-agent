"""本地 AlertStore:用 JSONL 记录 watchlist 检查产生的提醒事件(append-only)。

与 :class:`~binance_agent.app.run_store.FileRunStore` 不同,这里每一行就是一条独立
alert,读取时无需按 id 聚合——直接解析每行,返回最新在前的若干条,供 /alerts/recent
展示「观察项检查历史」。落盘前同样统一脱敏(reason 可能夹带错误文本)。
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


class FileAlertStore:
    """Append-only JSONL alert ledger(每行一条 alert)。"""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)

    def append(
        self,
        *,
        symbol: str,
        status: str,
        reason: str,
        item_id: str | None = None,
        market: str | None = None,
    ) -> dict[str, Any]:
        """追加一条 alert,返回脱敏后的记录(与落盘内容一致)。"""
        record = {
            "alert_id": uuid.uuid4().hex,
            "item_id": item_id,
            "symbol": symbol,
            "market": market,
            "status": status,
            "reason": reason,
            "created_at": _now_iso(),
        }
        line = redact_text(json.dumps(record, ensure_ascii=False, default=str))
        with self._lock:
            append_jsonl(self._path, line)
        redacted: dict[str, Any] = json.loads(line)
        return redacted

    def list_recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """返回最新在前的若干条 alert(append 顺序为时间正序,读取时反转)。"""
        with self._lock:
            records = read_jsonl(self._path)
        records.reverse()
        return records[: max(0, limit)]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
