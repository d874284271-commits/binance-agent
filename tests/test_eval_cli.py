"""效果闭环 CLI 编排 tag_pending 的确定性单测。

注入 fake run_store / client_factory / now,验证成熟度与去重门控、非 trade 不回拉、
回拉失败不写库、非 completed 跳过。核心不做 IO 之外副作用,不触真实网络。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from binance_agent.app.outcome_store import FileOutcomeStore
from binance_agent.eval.__main__ import export_proposal, tag_pending
from binance_agent.eval.tagging import normalize_klines  # noqa: F401 (确保模块可导入)
from binance_agent.infra.settings import Settings

NOW = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)  # cutoff = 2026-01-01T00:00Z(window=24h)
MATURE = "2025-12-31T00:00:00+00:00"  # < cutoff → 成熟
IMMATURE = "2026-01-01T12:00:00+00:00"  # > cutoff → 未成熟


def _settings() -> Any:
    return SimpleNamespace(
        runtime=SimpleNamespace(outcome_window_hours=24, outcome_kline_interval="5m")
    )


class _FakeRunStore:
    def __init__(self, runs: list[dict[str, Any]]) -> None:
        self._runs = runs

    def list_all(self) -> list[dict[str, Any]]:
        return self._runs


class _FakeClient:
    """回拉 K 线的假客户端;raise=True 时模拟回拉失败。"""

    def __init__(self, rows: Any, *, raise_exc: bool = False) -> None:
        self.rows = rows
        self.raise_exc = raise_exc
        self.calls = 0
        self.last_kwargs: dict[str, Any] = {}

    async def klines_range(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.last_kwargs = kwargs
        if self.raise_exc:
            raise RuntimeError("upstream 500")
        return self.rows


def _trade_run(
    run_id: str,
    created_at: str,
    *,
    completed_at: str | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "symbol": "BTCUSDT",
        "market": "fapi",
        "status": status,
        "created_at": created_at,
        "completed_at": completed_at or created_at,
        "final_advice": {
            "direction": "long",
            "decision_type": "trade",
            "confidence": 0.7,
            "pre_policy_confidence": 0.9,
            "model_confidence": 0.8,
            "entry_zone": "100-102",
            "stop_loss": "95",
            "targets": ["110"],
        },
    }


def _no_trade_run(run_id: str, created_at: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "symbol": "BTCUSDT",
        "market": "fapi",
        "status": "completed",
        "created_at": created_at,
        "completed_at": created_at,
        "final_advice": {"direction": "neutral", "decision_type": "no_trade"},
    }


# 币安 /klines 原始数组:[open_time, open, high, low, close, volume, close_time, ...]
_TARGET_HIT_ROWS = [
    [1767139200000, "100", "103", "99", "101", "1", 1767139499999],
    [
        1767139500000,
        "105",
        "112",
        "104",
        "108",
        "1",
        1767139799999,
    ],  # high 112 ≥ 目标 110
]


@pytest.mark.asyncio
async def test_mature_trade_run_gets_tagged(tmp_path: Path) -> None:
    outcome_store = FileOutcomeStore(tmp_path / "outcomes.jsonl")
    client = _FakeClient(_TARGET_HIT_ROWS)
    summary = await tag_pending(
        settings=_settings(),
        run_store=_FakeRunStore([_trade_run("run1", MATURE)]),
        outcome_store=outcome_store,
        client_factory=lambda market: client,
        now=NOW,
    )
    assert summary.due == 1
    assert summary.tagged == 1
    assert client.calls == 1
    assert client.last_kwargs["start_time"] == 1767139200000
    assert summary.by_outcome == {"target_hit": 1}
    assert outcome_store.tagged_run_ids() == {"run1"}
    [stored] = outcome_store.list_all()
    assert stored["evaluation_started_at"] == MATURE
    assert stored["evaluation_ended_at"] == "2026-01-01T00:00:00+00:00"
    assert stored["pre_policy_confidence"] == 0.9
    assert stored["entry_fill"] == 102.0
    assert stored["target_evaluated"] == 110.0
    assert stored["policy_version"] == "ohlc-conservative-v1"


@pytest.mark.asyncio
async def test_immature_run_skipped(tmp_path: Path) -> None:
    outcome_store = FileOutcomeStore(tmp_path / "outcomes.jsonl")
    client = _FakeClient(_TARGET_HIT_ROWS)
    summary = await tag_pending(
        settings=_settings(),
        run_store=_FakeRunStore([_trade_run("run1", IMMATURE)]),
        outcome_store=outcome_store,
        client_factory=lambda market: client,
        now=NOW,
    )
    assert summary.scanned == 1
    assert summary.due == 0
    assert summary.tagged == 0
    assert client.calls == 0


@pytest.mark.asyncio
async def test_maturity_is_measured_from_completed_at_not_created_at(tmp_path: Path) -> None:
    outcome_store = FileOutcomeStore(tmp_path / "outcomes.jsonl")
    client = _FakeClient(_TARGET_HIT_ROWS)
    summary = await tag_pending(
        settings=_settings(),
        run_store=_FakeRunStore([_trade_run("run1", MATURE, completed_at=IMMATURE)]),
        outcome_store=outcome_store,
        client_factory=lambda market: client,
        now=NOW,
    )
    assert summary.due == 0
    assert summary.tagged == 0
    assert client.calls == 0


@pytest.mark.asyncio
async def test_completed_run_without_completed_at_is_persisted_as_unevaluable(
    tmp_path: Path,
) -> None:
    run = _trade_run("run1", MATURE)
    run.pop("completed_at")
    outcome_store = FileOutcomeStore(tmp_path / "outcomes.jsonl")
    client = _FakeClient(_TARGET_HIT_ROWS)

    summary = await tag_pending(
        settings=_settings(),
        run_store=_FakeRunStore([run]),
        outcome_store=outcome_store,
        client_factory=lambda market: client,
        now=NOW,
    )

    assert summary.due == 1
    assert summary.tagged == 1
    assert summary.by_outcome == {"unevaluable": 1}
    assert client.calls == 0
    [record] = outcome_store.list_all()
    assert record["outcome"] == "unevaluable"
    assert "completed_at" in record["reason"]


@pytest.mark.asyncio
async def test_already_tagged_run_is_idempotent(tmp_path: Path) -> None:
    outcome_store = FileOutcomeStore(tmp_path / "outcomes.jsonl")
    outcome_store.append(run_id="run1", symbol="BTCUSDT", market="fapi", outcome="target_hit")
    client = _FakeClient(_TARGET_HIT_ROWS)
    summary = await tag_pending(
        settings=_settings(),
        run_store=_FakeRunStore([_trade_run("run1", MATURE)]),
        outcome_store=outcome_store,
        client_factory=lambda market: client,
        now=NOW,
    )
    assert summary.tagged == 0
    assert client.calls == 0  # 已打标 → 不回拉


@pytest.mark.asyncio
async def test_no_trade_run_tagged_without_market_fetch(tmp_path: Path) -> None:
    outcome_store = FileOutcomeStore(tmp_path / "outcomes.jsonl")
    client = _FakeClient(_TARGET_HIT_ROWS)
    summary = await tag_pending(
        settings=_settings(),
        run_store=_FakeRunStore([_no_trade_run("run1", MATURE)]),
        outcome_store=outcome_store,
        client_factory=lambda market: client,
        now=NOW,
    )
    assert summary.tagged == 1
    assert client.calls == 0  # 非 trade → 不回拉行情
    assert summary.by_outcome == {"not_applicable": 1}


@pytest.mark.asyncio
async def test_fetch_failure_not_persisted_for_retry(tmp_path: Path) -> None:
    outcome_store = FileOutcomeStore(tmp_path / "outcomes.jsonl")
    client = _FakeClient(None, raise_exc=True)
    summary = await tag_pending(
        settings=_settings(),
        run_store=_FakeRunStore([_trade_run("run1", MATURE)]),
        outcome_store=outcome_store,
        client_factory=lambda market: client,
        now=NOW,
    )
    assert summary.fetch_failed == 1
    assert summary.tagged == 0
    # 未写库 → 下次可重试
    assert outcome_store.tagged_run_ids() == set()


@pytest.mark.asyncio
async def test_non_completed_run_skipped(tmp_path: Path) -> None:
    outcome_store = FileOutcomeStore(tmp_path / "outcomes.jsonl")
    client = _FakeClient(_TARGET_HIT_ROWS)
    summary = await tag_pending(
        settings=_settings(),
        run_store=_FakeRunStore([_trade_run("run1", MATURE, status="running")]),
        outcome_store=outcome_store,
        client_factory=lambda market: client,
        now=NOW,
    )
    assert summary.due == 0
    assert summary.tagged == 0
    assert client.calls == 0


@pytest.mark.asyncio
async def test_outcome_window_rejects_unbounded_history(tmp_path: Path) -> None:
    outcome_store = FileOutcomeStore(tmp_path / "outcomes.jsonl")
    client = _FakeClient(_TARGET_HIT_ROWS)

    with pytest.raises(ValueError, match="window_hours"):
        await tag_pending(
            settings=_settings(),
            run_store=_FakeRunStore([]),
            outcome_store=outcome_store,
            client_factory=lambda market: client,
            now=NOW,
            window_hours=24 * 90 + 1,
        )


def test_proposal_export_writes_auditable_json_without_applying(tmp_path: Path) -> None:
    target = tmp_path / "audit" / "proposal.json"
    settings = Settings()
    records = [
        {
            "outcome_id": f"o-{index}",
            "run_id": f"r-{index}",
            "outcome": "stop_hit",
            "confidence": 0.85,
            "decision_type": "trade",
            "policy_version": "ohlc-conservative-v1",
        }
        for index in range(20)
    ]

    payload = export_proposal(target, records=records, settings=settings, generated_at=NOW)

    assert target.exists()
    assert payload["schema_version"] == 2
    assert payload["diagnostic_proposal"]["dataset_fingerprint"]
    assert payload["proposal"] is None
    assert settings.confidence.bucket_caps == {}
