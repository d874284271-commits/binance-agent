"""校验新生成模型输出的结构与字段契约,独立于历史模型的宽松默认值。

函数通过时返回 None,失败抛 ValueError 或模型校验错误,不修改输入。
校验只判断结构及明确的字段关系,不证明文本安全、引用真实或结论正确。
"""

from __future__ import annotations

import math
from typing import Any

from ..intel.models import FactCard

OUTPUT_SCHEMA_VERSION = "structured-output-v2"


def _enum(data: dict[str, Any], key: str, values: set[str]) -> None:
    value = data.get(key)
    if not isinstance(value, str) or value not in values:
        raise ValueError(key)


def confidence_value(value: Any) -> None:
    """要求非布尔、有限且位于 [0, 1] 的数值;先比较范围,避免巨大整数转 float 溢出。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence")
    if not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError("confidence")


def _text(data: dict[str, Any], key: str, *, required: bool = False) -> None:
    if key not in data and not required:
        return
    value = data.get(key)
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError(key)


def _strings(data: dict[str, Any], key: str) -> None:
    if key in data and (
        not isinstance(data[key], list) or any(not isinstance(x, str) for x in data[key])
    ):
        raise ValueError(key)


def analyst_output(data: dict[str, Any]) -> None:
    """要求有效立场、置信度和非空摘要,并校验可选 key_points 字符串列表。"""
    _enum(data, "stance", {"bullish", "bearish", "neutral"})
    confidence_value(data.get("confidence"))
    _text(data, "summary", required=True)
    _strings(data, "key_points")


def strategy_output(data: dict[str, Any]) -> None:
    """校验策略结构及方向/价位一致性:no_trade 无方向和价位,trade 必须给足价位。"""
    _enum(data, "direction", {"long", "short", "neutral"})
    _enum(data, "decision_type", {"trade", "no_trade", "watchlist"})
    confidence_value(data.get("confidence"))
    for key in ("rationale", "no_trade_reason"):
        _text(data, key)
    for key in ("entry_zone", "stop_loss"):
        if data.get(key) is not None:
            _text(data, key)
    for key in ("targets", "reentry_conditions"):
        _strings(data, key)
    if data["decision_type"] == "no_trade":
        if data["direction"] != "neutral" or any(
            data.get(k) for k in ("entry_zone", "stop_loss", "targets")
        ):
            raise ValueError("no_trade must have no direction or trade levels")
    elif data["decision_type"] == "trade":
        if data["direction"] == "neutral":
            raise ValueError("trade requires direction")
        for key in ("entry_zone", "stop_loss"):
            _text(data, key, required=True)
        if not data.get("targets") or any(not target.strip() for target in data["targets"]):
            raise ValueError("trade requires targets")


def researcher_output(data: dict[str, Any]) -> None:
    """校验事实卡必需的声明、证据 ID 和引文,以及显式 gaps/conflicts 列表。"""
    if "stance" in data:
        _enum(data, "stance", {"bullish", "bearish", "neutral"})
    if "confidence" in data:
        confidence_value(data["confidence"])
    if not isinstance(data.get("facts"), list):
        raise ValueError("facts")
    for key in ("gaps", "conflicts"):
        if key not in data:
            raise ValueError(key)
        _strings(data, key)
    _text(data, "search_query")
    if "critical" in data and not isinstance(data["critical"], bool):
        raise ValueError("critical")
    for fact in data["facts"]:
        if not isinstance(fact, dict):
            raise ValueError("facts row")
        for key in ("statement", "evidence_id", "quote"):
            _text(fact, key, required=True)
        FactCard.model_validate({**fact, "fact_id": "pending-schema-check"})


def debate_output(data: dict[str, Any]) -> None:
    """校验非空论据及 claim 的引用字段类型,不验证引用值是否来自真实快照。"""
    _text(data, "argument", required=True)
    for claim in _objects(data, "claims"):
        for key in ("statement", "evidence"):
            _text(claim, key, required=True)
        confidence_value(claim.get("confidence"))
        for key in ("source_path", "source_ref"):
            if claim.get(key) is not None:
                _text(claim, key)
        value = claim.get("source_value")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError("source_value")
        if isinstance(value, (int, float)):
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("source_value")


def _objects(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = data.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(key)
    return rows


def planner_output(data: dict[str, Any]) -> None:
    """仅接受规划动作白名单,并校验可选重评条件和置信上限。"""
    _enum(data, "action", {"no_trade", "watchlist", "continue_to_synth", "rerun_quality_check"})
    _text(data, "reason")
    _strings(data, "reentry_conditions")
    if data.get("confidence_cap") is not None:
        confidence_value(data["confidence_cap"])


def supervisor_output(data: dict[str, Any]) -> None:
    """校验主管分配任务的候选 ID 列表和非空问题,不在此决定任务预算。"""
    for task in _objects(data, "tasks"):
        if "candidate_ids" not in task:
            raise ValueError("candidate_ids")
        _strings(task, "candidate_ids")
        _text(task, "question", required=True)
        _text(task, "completion")
        if "critical" in task and not isinstance(task["critical"], bool):
            raise ValueError("critical")


def supervisor_review_output(data: dict[str, Any]) -> None:
    """要求明确的布尔 follow_up,并校验缺口与后续查询的类型。"""
    _boolean(data, "follow_up")
    _strings(data, "missing")
    _text(data, "query")


def _boolean(data: dict[str, Any], key: str) -> None:
    if not isinstance(data.get(key), bool):
        raise ValueError(key)


def evidence_verifier_output(data: dict[str, Any]) -> None:
    """要求 verdict 使用唯一非负整数索引和布尔 supported,拒绝重复索引。"""
    seen: set[int] = set()
    for verdict in _objects(data, "verdicts"):
        index = verdict.get("index")
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("verdict index")
        seen.add(index)
        _boolean(verdict, "supported")
        _text(verdict, "reason")


def final_verifier_output(data: dict[str, Any]) -> None:
    """要求最终核验返回真正的布尔 supported,不把字符串 true 当作通过。"""
    _boolean(data, "supported")
    _text(data, "reason")
