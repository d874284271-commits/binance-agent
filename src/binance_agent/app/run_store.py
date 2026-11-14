"""本地 RunStore/EventLog:用 JSONL 记录每次分析的可复盘轨迹。"""

from __future__ import annotations

import copy
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..infra.file_io import decode_jsonl, file_lock
from ..infra.logging import redact_text
from ..intel.evidence import load_evidence_documents, save_evidence_documents
from ..intel.models import EvidenceDocument
from .streaming import jsonable


class FileRunStore:
    """只追加的 JSONL 运行账本,按 run_id 聚合事件为可回看的记录。

    事件保留中间阶段,读取缓存按文件变化增量更新;返回副本避免调用者改写缓存。
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}
        self._summaries: dict[str, dict[str, Any]] = {}
        self._signature: tuple[int, int, int, int, int] | None = None
        self._local_append_signature: tuple[int, int, int, int, int] | None = None
        self._offset = 0
        self._line_number = 0

    def save_evidence(self, run_id: str, documents: list[EvidenceDocument]) -> None:
        with self._lock:
            existing = {
                d.evidence_id: d for d in load_evidence_documents(self._path.parent, run_id)
            }
            existing.update({d.evidence_id: d for d in documents})
            save_evidence_documents(self._path.parent, run_id, list(existing.values()))

    def load_evidence(self, run_id: str) -> list[EvidenceDocument]:
        with self._lock:
            return load_evidence_documents(self._path.parent, run_id)

    def create(self, *, symbol: str, market: str) -> str:
        """生成 run_id 并追加 running 记录;文件写入失败向上传递,不返回未落盘的成功结果。"""
        run_id = uuid.uuid4().hex
        self._append(
            {
                "type": "created",
                "run_id": run_id,
                "symbol": symbol.upper(),
                "market": market,
                "status": "running",
                "created_at": _now_iso(),
            }
        )
        return run_id

    def append_event(self, run_id: str, event_type: str, payload: Any) -> None:
        """序列化并追加一个运行事件,保留追加顺序供派生视图和回放读取。"""
        self._append(
            {
                "type": "event",
                "run_id": run_id,
                "event_type": event_type,
                "payload": jsonable(payload),
                "created_at": _now_iso(),
            }
        )

    def complete(self, run_id: str, final_advice: Any) -> None:
        self._append(
            {
                "type": "completed",
                "run_id": run_id,
                "status": "completed",
                "final_advice": jsonable(final_advice),
                "completed_at": _now_iso(),
            }
        )

    def fail(self, run_id: str, message: str) -> None:
        self._append(
            {
                "type": "failed",
                "run_id": run_id,
                "status": "failed",
                "error": message,
                "completed_at": _now_iso(),
            }
        )

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            return copy.deepcopy(self._records(copy_records=False).get(run_id))

    def list_recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        records = list(self._records().values())
        records.sort(key=lambda record: str(record.get("created_at") or ""), reverse=True)
        return records[: max(0, limit)]

    def list_all(self) -> list[dict[str, Any]]:
        """全部 run 记录,按创建(文件追加)顺序——最早在前。

        ``created_at`` 在低精度时钟(如 Windows ~15ms 分辨率)上可能平手,排序无法
        区分先后;文件追加顺序才是创建先后的可靠真相,供 watchlist「每 symbol 取最新」派生使用。
        """
        return list(self._records().values())

    def list_recent_summaries(
        self, *, limit: int = 20, offset: int = 0, newest_first_on_ties: bool = False
    ) -> list[dict[str, Any]]:
        """从增量摘要缓存分页取最近记录,避免复制正文;同时间排序按调用方要求选择追加顺序。"""
        with self._lock:
            self._records(copy_records=False)
            summaries = sorted(
                reversed(self._summaries.values())
                if newest_first_on_ties
                else self._summaries.values(),
                key=lambda record: str(record.get("created_at") or ""),
                reverse=True,
            )
            start = max(0, offset)
            return copy.deepcopy(summaries[start : start + max(0, limit)])

    def _append(self, event: dict[str, Any]) -> None:
        # 落盘前统一脱敏:错误消息/快照中若夹带密钥不应进 .runs JSONL(logging filter 不覆盖此路径)
        line = redact_text(json.dumps(event, ensure_ascii=False, default=str))
        with self._lock, file_lock(self._path), self._path.open("a+b") as handle:
            before = self._file_signature()
            known = self._signature is not None and before in (
                self._signature,
                self._local_append_signature,
            )
            handle.seek(0, 2)
            if handle.tell():
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write(line.encode("utf-8") + b"\n")
            handle.flush()
            # Only an uninterrupted chain of this instance's locked appends is
            # known to preserve previously cached bytes. External changes rebuild.
            self._local_append_signature = self._file_signature() if known else None

    def _file_signature(self) -> tuple[int, int, int, int, int]:
        stat = self._path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def _records(self, *, copy_records: bool = True) -> dict[str, dict[str, Any]]:
        with self._lock, file_lock(self._path), self._path.open("rb") as handle:
            signature = self._file_signature()
            if signature == self._signature:
                return copy.deepcopy(self._cache) if copy_records else self._cache
            incremental = (
                self._signature is not None
                and signature == self._local_append_signature
                and signature[:2] == self._signature[:2]
                and signature[2] > self._offset
            )
            if not incremental:
                self._cache = {}
                self._summaries = {}
                self._offset = 0
                self._line_number = 0
            handle.seek(self._offset)
            touched: set[str] = set()
            for raw in handle:
                # Do not consume an unfinished append; retry it on the next change.
                if not raw.endswith(b"\n"):
                    break
                self._offset += len(raw)
                self._line_number += 1
                event = decode_jsonl(raw, self._path, self._line_number)
                if event is None or not event.get("run_id"):
                    continue
                run_id = str(event["run_id"])
                record = self._cache.setdefault(run_id, _empty_record(run_id))
                _apply_event(record, event)
                touched.add(run_id)
            for run_id in touched:
                _derive_record_views(self._cache[run_id])
                self._summaries[run_id] = _summary_record(self._cache[run_id])
            self._signature = signature
            self._local_append_signature = None
            return copy.deepcopy(self._cache) if copy_records else self._cache


def _empty_record(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "symbol": None,
        "market": None,
        "status": "unknown",
        "created_at": None,
        "completed_at": None,
        "market_snapshot": None,
        "intel_snapshot": None,
        "node_outputs": [],
        "runtime_events": [],
        "context_compactions": {},
        "tool_events": [],
        "planner_steps": [],
        "planner_decision": None,
        "preflight_report": None,
        "debate_grounding": None,
        "data_quality_report": None,
        "risk_gate_result": None,
        "final_advice": None,
        "error": None,
        "checkpoint_status": "none",
        "checkpoint_backend": None,
        "checkpoint_thread_id": None,
        "resumable": False,
        "run_health": {},
        "evidence_summary": {},
        "evidence_factors": [],
        "events": [],
    }


def _summary_record(record: dict[str, Any]) -> dict[str, Any]:
    final_advice = record.get("final_advice") or {}
    return {
        "run_id": record.get("run_id"),
        "session_id": record.get("session_id"),
        "mode": record.get("mode"),
        "target": record.get("target"),
        "title": record.get("title"),
        "report": record.get("report"),
        "symbol": record.get("symbol"),
        "market": record.get("market"),
        "status": record.get("status"),
        "created_at": record.get("created_at"),
        "completed_at": record.get("completed_at"),
        "checkpoint_status": record.get("checkpoint_status"),
        "checkpoint_backend": record.get("checkpoint_backend"),
        "resumable": record.get("resumable"),
        "error": record.get("error"),
        "final_advice": {
            "decision_type": final_advice.get("decision_type"),
            "direction": final_advice.get("direction"),
            "confidence": final_advice.get("confidence"),
            "no_trade_reason": final_advice.get("no_trade_reason"),
        }
        if isinstance(final_advice, dict)
        else None,
    }


def _derive_record_views(record: dict[str, Any]) -> None:
    runtime_events = record.get("runtime_events") or []
    tool_events = record.get("tool_events") or []
    failed_runtime = [
        event for event in runtime_events if isinstance(event, dict) and not event.get("ok")
    ]
    fallback_agents = [str(event.get("agent")) for event in failed_runtime if event.get("agent")]
    # repaired:LLM 输出非法 JSON 后经二次 repair 才救回——ok=True 但走了降级路径,单独透出可见
    repaired_agents = [
        str(event.get("agent"))
        for event in runtime_events
        if isinstance(event, dict) and event.get("repaired") and event.get("agent")
    ]
    data_quality = record.get("data_quality_report") or {}
    grounding = record.get("debate_grounding") or {}
    risk_gate = record.get("risk_gate_result") or {}
    final_advice = record.get("final_advice") or {}
    llm_status = "ok" if not failed_runtime else "warn"
    user_message = "LLM 正常" if llm_status == "ok" else "部分 LLM 节点失败或使用降级结果"

    record["run_health"] = {
        "market_status": _tool_status(tool_events, "market_snapshot"),
        "intel_status": _tool_status(tool_events, "intel_snapshot"),
        "llm_status": llm_status,
        "user_message": user_message,
        "fallback_agents": _dedup_strings(fallback_agents),
        "repaired_agents": _dedup_strings(repaired_agents),
        "data_quality": data_quality.get("level") if isinstance(data_quality, dict) else None,
        "risk_gate": risk_gate.get("reason") if isinstance(risk_gate, dict) else None,
        "preflight": (record.get("preflight_report") or {}).get("status"),
    }
    record["evidence_summary"] = {
        "decision_type": final_advice.get("decision_type")
        if isinstance(final_advice, dict)
        else None,
        "direction": final_advice.get("direction") if isinstance(final_advice, dict) else None,
        "confidence": final_advice.get("confidence") if isinstance(final_advice, dict) else None,
        "data_quality": data_quality.get("level") if isinstance(data_quality, dict) else None,
        "risk_warnings": final_advice.get("risk_warnings", [])
        if isinstance(final_advice, dict)
        else [],
        "evidence_integrity": grounding.get("verified_ratio", grounding.get("grounded_ratio")),
    }
    record.setdefault("evidence_factors", [])


def _tool_status(tool_events: list[Any], tool_name: str) -> str:
    matches = [
        event for event in tool_events if isinstance(event, dict) and event.get("tool") == tool_name
    ]
    if not matches:
        return "unavailable"
    latest = matches[-1]
    if latest.get("ok"):
        return "ok"
    output = latest.get("output")
    return "warn" if output else "error"


def _dedup_strings(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _apply_event(record: dict[str, Any], event: dict[str, Any]) -> None:
    event_type = event.get("type")
    if event_type == "created":
        record["symbol"] = event.get("symbol")
        record["market"] = event.get("market")
        record["status"] = event.get("status", "running")
        record["created_at"] = event.get("created_at")
        _refresh_resumable(record)
        return

    if event_type == "event":
        name = str(event.get("event_type") or "")
        payload = event.get("payload")
        record["events"].append(
            {
                "event_type": name,
                "payload": payload,
                "created_at": event.get("created_at"),
            }
        )
        _apply_payload(record, name, payload)
        _refresh_resumable(record)
        return

    if event_type == "completed":
        record["status"] = "completed"
        record["completed_at"] = event.get("completed_at")
        record["final_advice"] = event.get("final_advice")
        _refresh_resumable(record)
        return

    if event_type == "failed":
        record["status"] = "failed"
        record["completed_at"] = event.get("completed_at")
        record["error"] = event.get("error")
        _refresh_resumable(record)


def _apply_payload(record: dict[str, Any], event_type: str, payload: Any) -> None:
    if event_type in {"research_budget", "research_input", "news_research"}:
        record[event_type] = payload
        return
    if event_type == "market":
        record["market_snapshot"] = payload
        return
    if event_type == "intel":
        record["intel_snapshot"] = payload
        return
    if event_type == "tool" and isinstance(payload, dict):
        record["tool_events"].append(payload)
        return
    if event_type == "node" and isinstance(payload, dict):
        record["node_outputs"].append(payload)
        update = payload.get("update") or {}
        if isinstance(update, dict):
            if isinstance(update.get("context_compactions"), dict):
                record.setdefault("context_compactions", {}).update(update["context_compactions"])
            if isinstance(update.get("news_research"), dict):
                record["news_research"] = update["news_research"]
            if isinstance(update.get("market_snapshot"), dict):
                record["market_snapshot"] = update["market_snapshot"]
            if isinstance(update.get("intel_snapshot"), dict):
                record["intel_snapshot"] = update["intel_snapshot"]
            preflight_report = update.get("preflight_report")
            if isinstance(preflight_report, dict):
                record["preflight_report"] = preflight_report
                final_quality = preflight_report.get("final_quality")
                if isinstance(final_quality, dict):
                    record["data_quality_report"] = final_quality
            if isinstance(update.get("debate_grounding"), dict):
                record["debate_grounding"] = update["debate_grounding"]
            runtime_events = update.get("runtime_events") or []
            record["runtime_events"].extend(runtime_events)
            tool_events = update.get("tool_events") or []
            record["tool_events"].extend(tool_events)
            planner_steps = update.get("planner_steps") or []
            record["planner_steps"].extend(planner_steps)
            if update.get("planner_decision") is not None:
                record["planner_decision"] = update.get("planner_decision")
            risk_assessment = update.get("risk_assessment")
            if isinstance(risk_assessment, dict) and risk_assessment.get("data_quality"):
                record["data_quality_report"] = risk_assessment.get("data_quality")
            if update.get("risk_gate_result") is not None:
                record["risk_gate_result"] = update.get("risk_gate_result")
            if update.get("final_advice") is not None:
                record["final_advice"] = update.get("final_advice")
        return
    if event_type == "final" and isinstance(payload, dict):
        record["final_advice"] = payload.get("advice")
        return
    if event_type == "data_quality":
        record["data_quality_report"] = payload
        return
    if event_type == "risk_gate":
        record["risk_gate_result"] = payload
        return
    if event_type == "checkpoint" and isinstance(payload, dict):
        record["checkpoint_status"] = str(payload.get("status") or "active")
        record["checkpoint_backend"] = payload.get("backend")
        record["checkpoint_thread_id"] = payload.get("thread_id")


def _refresh_resumable(record: dict[str, Any]) -> None:
    checkpoint_status = str(record.get("checkpoint_status") or "none")
    run_status = str(record.get("status") or "")
    record["resumable"] = run_status == "failed" and checkpoint_status in {
        "active",
        "resumable",
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
