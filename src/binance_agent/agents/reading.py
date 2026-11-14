"""确定性正文阅读视图;已保存文本不代表模型已完成阅读或核验。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..intel.content import split_text_chunks
from ..intel.models import EvidenceDocument

READING_VERSION = "reading-v2"
ANALYSIS_VERSION = "question-analysis-v2"


def merge_intervals(intervals: list[list[int]], length: int) -> list[list[int]]:
    """将字符半开区间裁剪到 [0, length),合并重叠或相邻区间,丢弃空区间。"""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        start, end = max(0, start), min(length, end)
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def coverage_for(
    doc: EvidenceDocument,
    record: dict[str, Any] | None = None,
    *,
    analysis_key: str | None = None,
) -> dict[str, Any]:
    """按正文内容版本、阅读计划版本及可选问题版本重建覆盖记录。

    内容完整性或版本不匹配时丢弃旧覆盖;complete 仅表示非空正文的 processed
    区间无缺口,不代表所有事实已 verified。返回新记录,不直接修改传入 record。
    """
    old = record or {}
    actual_hash = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
    integrity_valid = doc.status == "available" and actual_hash == doc.content_sha256
    valid = (
        integrity_valid
        and old.get("content_sha256") == doc.content_sha256
        and old.get("plan_version") == READING_VERSION
        and (analysis_key is None or old.get("analysis_key") == analysis_key)
    )
    record = dict(old) if valid else {}
    if analysis_key is not None:
        record["analysis_key"] = analysis_key
    length = len(doc.text)
    for stage in ("submitted", "processed", "verified"):
        record[f"{stage}_intervals"] = merge_intervals(record.get(f"{stage}_intervals", []), length)
    missing: list[list[int]] = []
    cursor = 0
    for start, end in record["processed_intervals"]:
        if start > cursor:
            missing.append([cursor, start])
        cursor = end
    if cursor < length:
        missing.append([cursor, length])
    complete = integrity_valid and bool(length) and not missing
    record.update(
        evidence_id=doc.evidence_id,
        content_sha256=doc.content_sha256,
        actual_sha256=actual_hash,
        integrity_valid=integrity_valid,
        plan_version=READING_VERSION,
        total_chars=length,
        saved=True,
        covered_chars=sum(end - start for start, end in record["processed_intervals"]),
        unprocessed_intervals=missing,
        complete=complete,
        status="unknown" if not integrity_valid else "complete" if complete else "incomplete",
        reason="unavailable_saved_document"
        if doc.status != "available"
        else "content_hash_mismatch"
        if not integrity_valid
        else None
        if complete
        else "unread_saved_text",
    )
    return record


def coverage_gaps(records: list[dict[str, Any]]) -> list[str]:
    """将未完成的正文覆盖记录转换为可展示的缺口原因,不补做阅读。"""
    return [
        f"正文阅读覆盖缺口:{r.get('evidence_id', 'unknown')}:{r.get('reason') or 'unread_saved_text'}"
        for r in records
        if not r.get("complete")
    ]


def document_view(documents: list[EvidenceDocument]) -> list[dict[str, Any]]:
    """按证据层级及实际文本内容去重,并保留各来源的独立元数据。

    同 URL 的正文已包含标题或摘要时不再重复提交短文本;不同来源的标题、
    发布时间与内容完整性仍分别保留,内容去重不等于来源独立性核验。
    """
    bodies = [d for d in documents if d.status == "available" and d.level == "body"]
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for doc in sorted(documents, key=lambda d: (d.level != "body", d.source_kind != "official")):
        if doc.status != "available" or not doc.text:
            continue
        if doc.level != "body" and any(
            doc.url == body.url and doc.text in body.text for body in bodies
        ):
            continue
        actual_hash = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
        key = (doc.level, actual_hash)
        source = {
            # Deduplicate text only. Conflicting titles/publication dates belong
            # to their individual sources and must reach both research and QA.
            **doc.model_dump(mode="json", exclude={"text"}),
            "integrity_valid": actual_hash == doc.content_sha256,
        }
        if key in groups:
            groups[key]["sources"].append(source)
            continue
        groups[key] = {
            **doc.model_dump(mode="json"),
            "sources": [source],
            "integrity_valid": actual_hash == doc.content_sha256,
        }
    return list(groups.values())


def pending_chunks(
    views: list[dict[str, Any]], records: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """返回尚未被 processed 区间完整覆盖的字符块;每篇优先首尾,再读中间。

    块保留原文偏移和所有来源;此函数不决定预算批次大小,也不更新覆盖记录。
    """
    pending: list[dict[str, Any]] = []
    for view in views:
        text = view["text"]
        # No fixed article cap: the request budget determines the batch size.
        chunks = split_text_chunks(text, max_chunks=max(1, len(text)))
        if len(chunks) > 2:
            chunks = [chunks[0], chunks[-1], *chunks[1:-1]]
        covered = records.get(view["evidence_id"], {}).get("processed_intervals", [])
        for chunk in chunks:
            start, end = int(chunk["start"]), int(chunk["end"])
            if any(a <= start and b >= end for a, b in covered):
                continue
            pending.append(
                {
                    **view,
                    "text": chunk["text"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_start": start,
                    "chunk_end": end,
                }
            )
    return pending


def mark_batch(records: dict[str, dict[str, Any]], batch: list[dict[str, Any]], stage: str) -> None:
    """将批次的字符区间追加到各原始来源的指定阶段记录,原地更新 records。

    stage 由调用方传入 submitted、processed 或 verified;缺少的来源记录跳过,
    区间裁剪与合并由 coverage_for 统一完成。
    """
    for chunk in batch:
        for source in chunk.get("sources", [{"evidence_id": chunk["evidence_id"]}]):
            record = records.get(source["evidence_id"])
            if record is not None:
                record.setdefault(f"{stage}_intervals", []).append(
                    [chunk["chunk_start"], chunk["chunk_end"]]
                )


def merge_facts(previous: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并事实并保留首次记录;身份比较忽略 fact_id、support、citations、source_kind。

    此处仅去重,不采用后一次重复事实的核验结论覆盖旧记录。
    """
    facts: dict[str, dict[str, Any]] = {}
    for fact in [*previous, *new]:
        identity = {
            k: v
            for k, v in fact.items()
            if k not in {"fact_id", "support", "citations", "source_kind"}
        }
        key = json.dumps(identity, ensure_ascii=False, sort_keys=True)
        facts.setdefault(key, fact)
    return list(facts.values())


def verification_documents(
    documents: list[EvidenceDocument], records: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """返回累计成功处理过的全部原文区间,包括此前读到的反证。

    局部引文窗口不能证明全文限定条件;返回内容超出模型窗口时由 runtime 拒绝,
    任务保持未解决,不在这里截断原文来制造已核验结果。
    """
    result: list[dict[str, Any]] = []
    for view in document_view(documents):
        intervals = records.get(view["evidence_id"], {}).get("processed_intervals", [])
        for start, end in merge_intervals(intervals, len(view["text"])):
            result.append(
                {
                    **view,
                    "text": view["text"][start:end],
                    "chunk_start": start,
                    "chunk_end": end,
                    "purpose": "cumulative_original_context",
                }
            )
    return result
