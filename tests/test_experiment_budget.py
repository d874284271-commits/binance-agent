from __future__ import annotations

import json

import pytest

from binance_agent.eval.experiment_budget import record_phase_snapshot, reserve_phase


def test_experiment_ledger_reserves_fixed_phases_and_records_upper_cost(tmp_path) -> None:
    path = tmp_path / "experiment-ledger.json"
    assert reserve_phase(path, "compatibility_probe") == 5.0
    assert reserve_phase(path, "dev") == 20.0
    record_phase_snapshot(
        path,
        "dev",
        {"cost_committed_usd": 3.2, "unknown_cost_requests": 1, "requests_reserved": 4},
    )
    payload = json.loads(path.read_text())
    assert payload["total_allocation_usd"] == 50.0
    assert payload["phases"]["dev"]["reserved_upper_usd"] == 3.2
    assert payload["phases"]["dev"]["unknown_cost_requests"] == 1


def test_experiment_ledger_rejects_unknown_phase(tmp_path) -> None:
    with pytest.raises(ValueError):
        reserve_phase(tmp_path / "ledger.json", "unknown")


def test_experiment_ledger_deduplicates_attempts_and_returns_remaining_budget(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    assert reserve_phase(path, "dev") == 20.0
    snapshot = {
        "attempts": [
            {"id": "a", "reserved_cost_usd": 3.0, "status": "unknown"},
            {"id": "b", "reserved_cost_usd": 2.0, "status": "ok"},
        ],
        "cost_committed_usd": 5.0,
        "unknown_cost_requests": 1,
        "requests_reserved": 2,
    }
    record_phase_snapshot(path, "dev", snapshot)
    assert reserve_phase(path, "dev") == 15.0
    record_phase_snapshot(path, "dev", snapshot)
    assert reserve_phase(path, "dev") == 15.0


def test_new_batch_preserves_historical_unknown_billing(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    reserve_phase(path, "holdout")
    record_phase_snapshot(
        path,
        "holdout",
        {
            "attempts": [{"id": "old", "reserved_cost_usd": 2.0, "status": "unknown"}],
        },
    )
    record_phase_snapshot(
        path,
        "holdout",
        {
            "attempts": [{"id": "new", "reserved_cost_usd": 1.0, "status": "ok"}],
        },
    )
    item = json.loads(path.read_text())["phases"]["holdout"]
    assert item["unknown_cost_requests"] == 1
    assert item["reserved_upper_usd"] == 3.0
    assert reserve_phase(path, "holdout") == 12.0


def test_legacy_unknown_reservations_survive_new_attempt_ledger(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    reserve_phase(path, "dev")
    record_phase_snapshot(
        path,
        "dev",
        {
            "cost_committed_usd": 2.0,
            "unknown_cost_requests": 3,
            "requests_reserved": 4,
        },
    )
    record_phase_snapshot(
        path,
        "dev",
        {
            "attempts": [{"id": "new", "reserved_cost_usd": 1.0, "status": "unknown"}],
        },
    )
    item = json.loads(path.read_text())["phases"]["dev"]
    assert item["unknown_cost_requests"] == 4
    assert item["reserved_upper_usd"] == 3.0


async def test_probe_rejects_new_experiment_ledger_before_spending(tmp_path, monkeypatch) -> None:
    import argparse

    from binance_agent.eval import research_live

    canonical = tmp_path / "approved" / "experiment-ledger.json"
    monkeypatch.setattr(research_live, "DEFAULT_EXPERIMENT_LEDGER", canonical)
    args = argparse.Namespace(
        output=str(tmp_path / "new-output"), experiment_ledger=str(tmp_path / "new-ledger.json")
    )
    with pytest.raises(ValueError, match="canonical"):
        await research_live.probe(args, "unused")
    assert not (tmp_path / "new-output").exists()
    assert not (tmp_path / "new-ledger.json").exists()
    assert not canonical.exists()


def test_pending_to_actual_overrun_is_added_once_across_snapshots(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    reserve_phase(path, "dev")
    pending = {
        "id": "same-attempt",
        "reserved_cost_usd": 3.0,
        "cost_usd": None,
        "status": "pending",
    }
    record_phase_snapshot(path, "dev", {"attempts": [pending]})
    settled = {**pending, "status": "ok", "cost_usd": 5.0}
    record_phase_snapshot(path, "dev", {"attempts": [settled]})
    assert reserve_phase(path, "dev") == 15.0
    record_phase_snapshot(path, "dev", {"attempts": [settled, settled]})
    # Another output directory returning the same ID cannot charge again or refund.
    record_phase_snapshot(path, "dev", {"attempts": [{**settled, "cost_usd": 4.0}]})
    record_phase_snapshot(path, "dev", {"attempts": [pending]})
    assert reserve_phase(path, "dev") == 15.0


def test_legacy_known_attempt_accounts_only_actual_above_original_reservation(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    reserve_phase(path, "dev")
    record_phase_snapshot(
        path, "dev", {"attempts": [{"id": "old", "reserved_cost_usd": 3.0, "status": "unknown"}]}
    )
    data = json.loads(path.read_text())
    data["phases"]["dev"].pop("attempt_accounted_usd", None)
    path.write_text(json.dumps(data))
    record_phase_snapshot(
        path,
        "dev",
        {"attempts": [{"id": "old", "reserved_cost_usd": 3.0, "cost_usd": 5.0, "status": "ok"}]},
    )
    assert reserve_phase(path, "dev") == 15.0
    item = json.loads(path.read_text())["phases"]["dev"]
    assert item["unknown_cost_requests"] == 0
    assert item["attempt_accounted_usd"]["old"] == 5.0


def test_new_settled_attempt_uses_actual_overrun_but_never_refunds_reservation(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    reserve_phase(path, "dev")
    snapshot = {
        "attempts": [
            {"id": "over", "reserved_cost_usd": 1.0, "cost_usd": 3.0, "status": "ok"},
            {"id": "under", "reserved_cost_usd": 2.0, "cost_usd": 0.5, "status": "ok"},
            {"id": "unknown", "reserved_cost_usd": 4.0, "cost_usd": None, "status": "unknown"},
        ]
    }
    record_phase_snapshot(path, "dev", snapshot)
    assert reserve_phase(path, "dev") == 11.0
    record_phase_snapshot(path, "dev", snapshot)
    assert reserve_phase(path, "dev") == 11.0
    assert json.loads(path.read_text())["phases"]["dev"]["unknown_cost_requests"] == 1
