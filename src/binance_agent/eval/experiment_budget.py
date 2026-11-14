"""Cross-batch experiment budget ledger.

The ledger is separate from a per-run request ledger so changing an output
directory or restarting a process cannot silently create another allowance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

PHASE_ALLOCATIONS: dict[str, float] = {
    "compatibility_probe": 5.0,
    "dev": 20.0,
    "holdout": 15.0,
    "fault_retest": 10.0,
}
TOTAL_BUDGET_USD = 50.0
DEFAULT_EXPERIMENT_LEDGER = (
    Path(__file__).resolve().parents[3] / "output" / "research-v1" / "experiment-ledger.json"
)


def reserve_phase(path: Path | str, phase: str) -> float:
    """Reserve one fixed phase allowance and return its remaining ceiling.

    A phase is global to the ledger path.  Returning the remaining amount makes
    starting a second process or changing its output directory consume only
    what the first process did not reserve.
    """
    if phase not in PHASE_ALLOCATIONS:
        raise ValueError(f"unknown experiment budget phase: {phase}")
    target = Path(path)
    ledger = _load(target)
    phases = ledger["phases"]
    if not phases[phase]["reserved"]:
        reserved = sum(
            float(item["allocation_usd"]) for item in phases.values() if item.get("reserved")
        )
        allocation = PHASE_ALLOCATIONS[phase]
        if reserved + allocation > TOTAL_BUDGET_USD:
            raise RuntimeError("experiment budget total ceiling exhausted")
        phases[phase]["reserved"] = True
        _save(target, ledger)
    remaining = float(phases[phase]["allocation_usd"]) - float(
        phases[phase].get("reserved_upper_usd", 0.0)
    )
    if remaining <= 0:
        raise RuntimeError(f"experiment budget phase {phase} exhausted")
    return round(remaining, 8)


def record_phase_snapshot(path: Path | str, phase: str, snapshot: dict[str, Any]) -> None:
    """Persist conservative upper reservations and unknown billing counts."""
    if phase not in PHASE_ALLOCATIONS:
        raise ValueError(f"unknown experiment budget phase: {phase}")
    target = Path(path)
    ledger = _load(target)
    item = ledger["phases"][phase]
    attempts = [
        attempt
        for attempt in snapshot.get("attempts", [])
        if isinstance(attempt, dict) and attempt.get("id")
    ]
    if attempts:
        known_ids = {str(value) for value in item.get("attempt_ids", [])}
        unknown_ids = {str(value) for value in item.get("unknown_attempt_ids", [])}
        if "legacy_unknown_cost_requests" not in item:
            # Old ledgers have only a count. Account for overlapping IDs in the
            # supplied snapshot, and retain the unidentifiable remainder.
            overlapping_unknown = {
                str(attempt["id"])
                for attempt in attempts
                if str(attempt["id"]) in known_ids and attempt.get("status") == "unknown"
            }
            unknown_ids.update(overlapping_unknown)
            item["legacy_unknown_cost_requests"] = max(
                0, int(item.get("unknown_cost_requests", 0)) - len(unknown_ids)
            )
        accounted = dict(item.get("attempt_accounted_usd") or {})
        incremental = 0.0
        for attempt in attempts:
            identity = str(attempt["id"])
            reserved = attempt.get("reserved_cost_usd")
            reserved_amount = (
                float(reserved)
                if isinstance(reserved, (int, float))
                and not isinstance(reserved, bool)
                and math.isfinite(reserved)
                and reserved >= 0
                else 0.0
            )
            # Legacy IDs were already charged their original reservation.
            # New IDs start at zero; persisted per-ID amounts make later
            # pending->settled updates and repeated snapshots idempotent.
            previous = float(
                accounted.get(identity, reserved_amount if identity in known_ids else 0.0)
            )
            committed = max(previous, reserved_amount)
            for field in ("cost_usd", "committed_cost_usd"):
                value = attempt.get(field)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                ):
                    committed = max(committed, float(value))
            incremental += committed - previous
            accounted[identity] = committed
        item["attempt_accounted_usd"] = accounted
        item["attempt_ids"] = sorted(known_ids | {str(attempt["id"]) for attempt in attempts})
        item["reserved_upper_usd"] = round(
            float(item.get("reserved_upper_usd", 0.0)) + incremental, 8
        )
        for attempt in attempts:
            identity = str(attempt["id"])
            if attempt.get("status") == "unknown":
                unknown_ids.add(identity)
            elif attempt.get("status") in {"ok", "error", "not_sent"}:
                unknown_ids.discard(identity)
        item["unknown_attempt_ids"] = sorted(unknown_ids)
        item["unknown_cost_requests"] = int(item["legacy_unknown_cost_requests"]) + len(unknown_ids)
        item["requests_reserved"] = max(
            int(item.get("requests_reserved", 0)), len(item["attempt_ids"])
        )
    else:
        # Keep compatibility with older snapshots which did not expose the
        # attempt list; their cumulative reservation is still a safe upper
        # bound, but cannot be deduplicated across independent processes.
        item["reserved_upper_usd"] = max(
            float(item.get("reserved_upper_usd", 0.0)),
            float(snapshot.get("cost_committed_usd", 0.0)),
        )
        item["unknown_cost_requests"] = max(
            int(item.get("unknown_cost_requests", 0)),
            int(snapshot.get("unknown_cost_requests", 0)),
        )
        item["requests_reserved"] = max(
            int(item.get("requests_reserved", 0)),
            int(snapshot.get("requests_reserved", 0)),
        )
    _save(target, ledger)


def _load(path: Path) -> dict[str, Any]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("version") == 1:
            return payload
    return {
        "version": 1,
        "total_allocation_usd": TOTAL_BUDGET_USD,
        "phases": {
            name: {
                "allocation_usd": allocation,
                "reserved": False,
                "reserved_upper_usd": 0.0,
                "unknown_cost_requests": 0,
                "requests_reserved": 0,
                "attempt_ids": [],
            }
            for name, allocation in PHASE_ALLOCATIONS.items()
        },
    }


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
