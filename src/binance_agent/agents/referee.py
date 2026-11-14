"""辩论证据完整性 referee:核验结构化 claim 引用的具体标量和值。

本模块只验证「引用路径存在且引用值与 canonical namespace 一致」,不判断论点语义真伪。
无 LLM、无网络,调用方和测试都通过 :func:`ground_debate` 这一条 seam 使用它。

证据强弱按「独立数据点」计量:``(side, source_path)`` 为一个证据单元,同一路径重复引用
只算一次;逐轮统计首次被核验引用的新路径,供辩论收敛判定。
仅原始行情路径计入独立证据;分析师 stance/confidence 的字面匹配不增加独立支持。
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any

from .state import DebateTurn

VERIFIED = "verified"
MISSING = "missing"
MISMATCH = "mismatch"
INVALID = "invalid"

SIDES = ("bull", "bear")

# 读取旧调用方的符号兼容;新报告只产生上面的证据完整性术语。
GROUNDED = VERIFIED
UNGROUNDED = MISSING
CONTRADICTED = MISMATCH


def ground_debate(turns: list[DebateTurn], namespace: dict[str, Any]) -> dict[str, Any]:
    """核验所有 claim 的 source_path/source_value,返回证据完整性报告。

    ``namespace`` 必须是辩手实际看到的 canonical ``snapshot_data`` 与按 analyst 名键控的
    ``analyst_reports``。容器、空值、缺失值和白名单外路径均不能通过。
    """
    claims_report: list[dict[str, Any]] = []
    counts = {VERIFIED: 0, MISSING: 0, MISMATCH: 0, INVALID: 0}

    for turn in turns:
        for claim_index, claim in enumerate(turn.structured_claims):
            path = (claim.source_path or "").strip()
            verdict, observed = _verify_source(path, claim.source_value, namespace)
            counts[verdict] += 1
            claims_report.append(
                {
                    "claim_id": f"{turn.side}:{turn.round}:{claim_index}",
                    "claim_index": claim_index,
                    "side": turn.side,
                    "round": turn.round,
                    "statement": claim.statement or "",
                    "source_path": path or None,
                    "source_value": claim.source_value,
                    "verdict": verdict,
                    "observed": observed,
                }
            )

    total = len(claims_report)
    verified_ratio = round(counts[VERIFIED] / total, 4) if total else 0.0
    units = _evidence_units(claims_report)
    return {
        "verified_ratio": verified_ratio,
        "grounded_ratio": verified_ratio,
        "verified_count": counts[VERIFIED],
        "missing_count": counts[MISSING],
        "mismatch_count": counts[MISMATCH],
        "invalid_count": counts[INVALID],
        "total_claims": total,
        "by_side": {side: _side_summary(side, claims_report, units) for side in SIDES},
        "duplicate_claims": sum(1 for c in claims_report if c["source_path"])
        - len({(c["side"], c["source_path"]) for c in claims_report if c["source_path"]}),
        "shared_sources": sorted(
            path
            for (side, path), ok in units.items()
            if side == "bull" and ok and units.get(("bear", path))
        ),
        "rounds": _round_novelty(claims_report, sorted({turn.round for turn in turns})),
        "claims": claims_report,
    }


def new_verified_sources_in_round(
    turns: list[DebateTurn], namespace: dict[str, Any], round_no: int
) -> int:
    """第 ``round_no`` 轮首次被核验引用的路径数;该轮无 turn 时为 0。"""
    for entry in ground_debate(turns, namespace)["rounds"]:
        if entry["round"] == round_no:
            return int(entry["new_verified_sources"])
    return 0


def _evidence_units(claims: list[dict[str, Any]]) -> dict[tuple[str, str], bool]:
    """``(side, source_path)`` → 是否 verified;同一单元下任一 claim 未通过即不算(同侧自相矛盾)。"""
    units: dict[tuple[str, str], bool] = {}
    for claim in claims:
        path = claim["source_path"]
        if not _independent_path(path):
            continue
        key = (claim["side"], path)
        units[key] = units.get(key, True) and claim["verdict"] == VERIFIED
    return units


def _side_summary(
    side: str, claims: list[dict[str, Any]], units: dict[tuple[str, str], bool]
) -> dict[str, Any]:
    side_claims = [c for c in claims if c["side"] == side]
    side_units = [ok for (unit_side, _), ok in units.items() if unit_side == side]
    verified_sources = sum(side_units)
    return {
        "total_claims": len(side_claims),
        "verified_count": sum(1 for c in side_claims if c["verdict"] == VERIFIED),
        "sources": len(side_units),
        "verified_sources": verified_sources,
        "source_verified_ratio": (
            round(verified_sources / len(side_units), 4) if side_units else 0.0
        ),
    }


def _round_novelty(claims: list[dict[str, Any]], round_numbers: list[int]) -> list[dict[str, Any]]:
    """逐轮统计核验通过的路径,以及其中此前任何一方都未核验引用过的新路径。"""
    seen: set[str] = set()
    rounds: list[dict[str, Any]] = []
    for round_no in round_numbers:
        verified = {
            c["source_path"]
            for c in claims
            if c["round"] == round_no
            and c["verdict"] == VERIFIED
            and _independent_path(c["source_path"])
        }
        rounds.append(
            {
                "round": round_no,
                "verified_sources": len(verified),
                "new_verified_sources": len(verified - seen),
            }
        )
        seen |= verified
    return rounds


def _verify_source(path: str, claimed: Any, namespace: dict[str, Any]) -> tuple[str, Any]:
    if not _allowed_path(path) or not _is_nonempty_scalar(claimed):
        return INVALID, None
    ok, value = _navigate(path, namespace)
    if not ok:
        return MISSING, None
    if not _is_nonempty_scalar(value):
        return INVALID, None
    return (_compare(value, claimed), value)


def _resolve_ref(ref: str, namespace: dict[str, Any]) -> tuple[str, Any]:
    """v0.1 测试/调用兼容:把 ``path=value`` 读取为新的结构化证据契约。"""
    path, separator, raw_value = ref.partition("=")
    claimed: Any = raw_value.strip() if separator else None
    return _verify_source(path.strip(), claimed, namespace)


def _allowed_path(path: str) -> bool:
    parts = path.split(".")
    if len(parts) < 2 or any(not part for part in parts):
        return False
    if parts[0] == "snapshot_data":
        return True
    return (
        parts[0] == "analyst_reports" and len(parts) == 3 and parts[2] in {"stance", "confidence"}
    )


def _independent_path(path: str | None) -> bool:
    """Only canonical market data, never generated narrative or snapshot metadata."""
    if not path:
        return False
    parts = path.split(".")
    if len(parts) < 2 or parts[0] != "snapshot_data":
        return False
    if parts[1] == "open_interest":
        return len(parts) == 2
    return len(parts) >= 3 and parts[1] in {
        "ticker_24h",
        "funding",
        "open_interest_hist_tail",
        "long_short_ratio_tail",
        "technicals",
    }


def _navigate(path: str, namespace: dict[str, Any]) -> tuple[bool, Any]:
    """按点路径下钻 dict/list;任一层缺失即失败。"""
    cur: Any = namespace
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False, None
    return True, cur


def _compare(observed: Any, claimed: Any) -> str:
    if isinstance(observed, bool) or isinstance(claimed, bool):
        return (
            VERIFIED
            if type(observed) is bool and type(claimed) is bool and observed == claimed
            else MISMATCH
        )
    obs = _as_decimal(observed)
    clm = _as_decimal(claimed)
    if obs is not None and clm is not None:
        return VERIFIED if obs == clm else MISMATCH
    return VERIFIED if type(observed) is type(claimed) and observed == claimed else MISMATCH


def _as_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _is_nonempty_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (dict, list)):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, float):
        return isfinite(value)
    return isinstance(value, (int, bool))
