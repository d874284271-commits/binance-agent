"""FileOutcomeStore:效果闭环 append-only JSONL 台账测试(与 alert/run store 同源范式)。"""

from __future__ import annotations

from pathlib import Path

from binance_agent.app.outcome_store import FileOutcomeStore


def _append(store: FileOutcomeStore, *, run_id: str, outcome: str = "target_hit", **kw) -> dict:
    return store.append(
        run_id=run_id,
        symbol=kw.pop("symbol", "BTCUSDT"),
        market=kw.pop("market", "fapi"),
        outcome=outcome,
        **kw,
    )


def test_append_then_list_roundtrip(tmp_path: Path) -> None:
    store = FileOutcomeStore(tmp_path / "outcomes.jsonl")
    rec = _append(store, run_id="run1", confidence=0.7, r_realized=1.5)
    assert rec["run_id"] == "run1"
    assert rec["outcome"] == "target_hit"
    assert rec["confidence"] == 0.7
    assert rec["r_realized"] == 1.5
    assert rec["outcome_id"]
    assert rec["created_at"]
    assert rec["resolved_at"]  # store 未传时补当前时刻

    items = store.list_recent(limit=10)
    assert len(items) == 1
    assert items[0]["run_id"] == "run1"


def test_append_roundtrips_evaluation_policy_fields(tmp_path: Path) -> None:
    store = FileOutcomeStore(tmp_path / "outcomes.jsonl")

    rec = _append(
        store,
        run_id="run1",
        evaluation_started_at="2026-01-01T00:00:00+00:00",
        evaluation_ended_at="2026-01-02T00:00:00+00:00",
        pre_policy_confidence=0.9,
        entry_fill=102.0,
        target_evaluated=110.0,
        policy_version="ohlc-conservative-v1",
    )

    assert rec["evaluation_started_at"] == "2026-01-01T00:00:00+00:00"
    assert rec["evaluation_ended_at"] == "2026-01-02T00:00:00+00:00"
    assert rec["pre_policy_confidence"] == 0.9
    assert rec["entry_fill"] == 102.0
    assert rec["target_evaluated"] == 110.0
    assert rec["policy_version"] == "ohlc-conservative-v1"


def test_list_recent_newest_first_and_limit(tmp_path: Path) -> None:
    store = FileOutcomeStore(tmp_path / "outcomes.jsonl")
    for i in range(5):
        _append(store, run_id=f"run{i}")
    items = store.list_recent(limit=3)
    assert [it["run_id"] for it in items] == ["run4", "run3", "run2"]


def test_tagged_run_ids_dedup(tmp_path: Path) -> None:
    store = FileOutcomeStore(tmp_path / "outcomes.jsonl")
    _append(store, run_id="run1")
    _append(store, run_id="run2")
    _append(store, run_id="run1")  # 重复 run_id 仍只算一个
    assert store.tagged_run_ids() == {"run1", "run2"}


def test_empty_store(tmp_path: Path) -> None:
    store = FileOutcomeStore(tmp_path / "outcomes.jsonl")
    assert store.list_recent() == []
    assert store.list_all() == []
    assert store.tagged_run_ids() == set()


def test_append_redacts_secret_in_reason(tmp_path: Path) -> None:
    store = FileOutcomeStore(tmp_path / "outcomes.jsonl")
    secret = "sk-ABCDEF1234567890ABCDEF"
    rec = _append(store, run_id="run1", outcome="fetch_failed", reason=f"boom {secret} tail")
    assert secret not in rec["reason"]
    # 落盘文件也不含明文密钥
    assert secret not in (tmp_path / "outcomes.jsonl").read_text(encoding="utf-8")


def test_read_skips_blank_and_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.jsonl"
    store = FileOutcomeStore(path)
    _append(store, run_id="run1")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n")  # 空行
        f.write("{not valid json}\n")  # 坏行
    _append(store, run_id="run2")
    ids = {r["run_id"] for r in store.list_all()}
    assert ids == {"run1", "run2"}
