"""Regression contracts for corrupt input, atomic writes and incremental history."""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet

from binance_agent.app.alert_store import FileAlertStore
from binance_agent.app.outcome_store import FileOutcomeStore
from binance_agent.app.run_store import FileRunStore
from binance_agent.eval.stats import aggregate
from binance_agent.infra import file_io
from binance_agent.infra.runtime_config import (
    RuntimeConfigError,
    apply_overrides,
    build_candidate,
    load_overrides,
    save_overrides,
)
from binance_agent.infra.secrets import SecretStore, SecretStoreError
from binance_agent.infra.settings import Settings


def _set_secret(path: str, key: bytes, number: int) -> None:
    SecretStore(Fernet(key), Path(path)).set(str(number), str(number))


def test_jsonl_corrupt_utf8_isolated_and_tail_preserved(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    for cls in (FileRunStore, FileOutcomeStore, FileAlertStore):
        path = tmp_path / cls.__name__
        path.write_bytes(b'{"type":"created","run_id":"old"}\n\xff\n{"unfinished":')
        store = cls(path)
        if isinstance(store, FileRunStore):
            run_id = store.create(symbol="BTCUSDT", market="api")
            assert store.get("old") is not None
            assert store.get(run_id) is not None
        elif isinstance(store, FileOutcomeStore):
            store.append(run_id="new", symbol="BTC", market="api", outcome="open")
            assert len(store.list_all()) == 2
        else:
            store.append(symbol="BTC", status="ok", reason="")
            assert len(store.list_recent()) == 2
        assert b'{"unfinished":\n' in path.read_bytes()
        assert "跳过损坏 JSONL" in caplog.text


def test_run_cache_incremental_and_rebuilds(tmp_path: Path) -> None:
    import binance_agent.app.run_store as module

    path = tmp_path / "runs.jsonl"
    store = FileRunStore(path)
    first = store.create(symbol="BTC", market="api")
    assert store.get(first)["symbol"] == "BTC"  # type: ignore[index]
    with patch.object(module, "decode_jsonl", wraps=module.decode_jsonl) as decode:
        store.list_recent_summaries()
        assert decode.call_count == 0
        external = FileRunStore(path)
        second = external.create(symbol="ETH", market="api")
        assert store.get(second)["symbol"] == "ETH"  # type: ignore[index]
        assert decode.call_count == 2
        decode.reset_mock()
        third = store.create(symbol="BNB", market="api")
        fourth = store.create(symbol="XRP", market="api")
        assert store.get(third) is not None and store.get(fourth) is not None
        assert decode.call_count == 2  # two local appends, no old records decoded
    original = path.read_bytes()
    path.write_bytes(original.replace(b'"BTC"', b'"SOL"'))
    assert store.get(first)["symbol"] == "SOL"  # type: ignore[index]
    # A changed old byte plus external append is not distinguishable by size alone.
    rewritten = path.read_bytes().replace(b'"SOL"', b'"ADA"')
    path.write_bytes(rewritten + b'{"type":"created","run_id":"external","symbol":"EXT"}\n')
    assert store.get(first)["symbol"] == "ADA"  # type: ignore[index]
    assert store.get("external") is not None
    # Replacement and truncation invalidate stale records.
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b'{"type":"created","run_id":"replacement","symbol":"X"}\n')
    os.replace(replacement, path)
    assert store.get(first) is None
    assert store.get("replacement") is not None
    path.write_bytes(b"")
    assert store.list_all() == []


def test_atomic_replace_failure_keeps_config_and_candidate_is_independent(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    save_overrides({"params": {"max_news_items": 10}}, path)
    original = path.read_bytes()
    settings = Settings()
    before = settings.model_dump()
    with pytest.raises(ValueError):
        apply_overrides(
            settings,
            {
                "params": {"max_news_items": 20, "context": {"max_tokens": "bad"}},
                "confidence": {"max_confidence": "bad"},
            },
        )
    assert settings.model_dump() == before
    candidate = build_candidate(settings, {"params": {"max_news_items": 20}})
    assert candidate.intel.max_news_items == 20
    assert settings.model_dump() == before
    with (
        patch.object(file_io.os, "replace", side_effect=OSError("disk error")),
        pytest.raises(OSError),
    ):
        save_overrides({"params": {}}, path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".config.json.*"))


@pytest.mark.parametrize("content", [b"[]", b"{", b"\xff"])
def test_invalid_saved_configuration_is_not_empty(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "runtime.json"
    path.write_bytes(content)
    with pytest.raises(RuntimeConfigError, match=r"runtime\.json"):
        load_overrides(path)
    assert path.read_bytes() == content


def test_secret_rmw_process_lock_and_private_mode(tmp_path: Path) -> None:
    path = tmp_path / "secrets.enc"
    key = Fernet.generate_key()
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(_set_secret, str(path), key, number) for number in range(8)]
        for future in futures:
            future.result(timeout=30)
    store = SecretStore(Fernet(key), path)
    assert [store.get(str(number)) for number in range(8)] == [str(n) for n in range(8)]
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    original = path.read_bytes()
    with (
        patch.object(file_io.os, "replace", side_effect=OSError("disk error")),
        pytest.raises(OSError),
    ):
        store.set("extra", "value")
    assert path.read_bytes() == original
    path.write_bytes(b"")
    with pytest.raises(SecretStoreError):
        store.set("extra", "value")
    assert path.read_bytes() == b""


def test_stats_reject_boolean_and_nonfinite_numbers() -> None:
    records = [
        {"outcome": "target_hit", "confidence": value, "r_realized": value}
        for value in (True, False, float("inf"), float("nan"), 0.5)
    ]
    stats = aggregate(records)
    assert stats["overall"]["avg_r"] == 0.5
    assert sum(bucket["count"] for bucket in stats["by_confidence_bucket"].values()) == 1
    json.dumps(stats, allow_nan=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"params": {"per_source_timeout": float("inf")}},
        {"params": {"max_news_items": 1.5}},
        {"params": {"max_news_items": True}},
        {"params": {"strategy_planner_mode": "unknown"}},
        {"llm": {"endpoints": [{"name": "broken"}]}},
        {"intel_sources": {"rss": "false"}},
    ],
)
def test_invalid_override_fields_do_not_partially_activate(overrides: dict[str, object]) -> None:
    settings = Settings()
    original = settings.model_dump()
    with pytest.raises(ValueError):
        apply_overrides(settings, overrides)
    assert settings.model_dump() == original


@pytest.mark.parametrize("name", ["max_news_items", "max_debate_rounds"])
def test_zero_count_override_survives_restart(tmp_path: Path, name: str) -> None:
    path = tmp_path / "runtime.json"
    override = {"params": {name: 0}}
    candidate = build_candidate(Settings(), override)
    save_overrides(override, path)
    restored = build_candidate(Settings(), load_overrides(path))
    assert candidate.model_dump() == restored.model_dump()
    assert (
        restored.intel.max_news_items if name == "max_news_items" else restored.debate.max_rounds
    ) == 0


@pytest.mark.parametrize(
    "name",
    [
        "time_window_hours",
        "per_source_timeout",
        "max_news_items",
        "max_debate_rounds",
        "strategy_planner_max_steps",
    ],
)
def test_overlarge_numeric_override_reports_field(name: str) -> None:
    with pytest.raises(RuntimeConfigError, match=name):
        build_candidate(Settings(), {"params": {name: 10**400}})


def _create_initial_key_and_secret(path: str, key_path: str, number: int) -> None:
    import binance_agent.infra.secrets as secrets

    key_file = Path(key_path)

    def get_password(service: str, name: str) -> str | None:
        if key_file.exists():
            return key_file.read_text()
        # Expose an unprotected concurrent initial creation deterministically.
        time.sleep(0.05)
        return None

    def set_password(service: str, name: str, value: str) -> None:
        with key_file.with_suffix(".creations").open("a") as handle:
            handle.write("created\n")
        key_file.write_text(value)

    sys.modules["keyring"] = SimpleNamespace(get_password=get_password, set_password=set_password)
    secrets._STORE_PATH = Path(path)
    store = secrets._default_store()
    assert store is not None
    store.set(str(number), str(number))


def test_initial_fernet_key_is_created_once_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "secrets.enc"
    key_file = tmp_path / "fake-keyring-key"
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as executor:
        futures = [
            executor.submit(_create_initial_key_and_secret, str(path), str(key_file), n)
            for n in range(8)
        ]
        for future in futures:
            future.result(timeout=30)
    assert key_file.with_suffix(".creations").read_text().splitlines() == ["created"]
    store = SecretStore(Fernet(key_file.read_bytes()), path)
    assert [store.get(str(n)) for n in range(8)] == [str(n) for n in range(8)]


def test_external_change_between_local_appends_forces_rebuild(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    store = FileRunStore(path)
    first = store.create(symbol="BTC", market="api")
    assert store.get(first) is not None
    store.create(symbol="ETH", market="api")
    path.write_bytes(path.read_bytes().replace(b'"BTC"', b'"SOL"'))
    store.create(symbol="BNB", market="api")
    assert store.get(first)["symbol"] == "SOL"  # type: ignore[index]
