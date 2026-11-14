from __future__ import annotations

from pathlib import Path

from binance_agent.app.alert_store import FileAlertStore


def test_append_then_list_returns_record(tmp_path: Path) -> None:
    store = FileAlertStore(tmp_path / "alerts.jsonl")
    rec = store.append(
        symbol="BTCUSDT",
        status="ok",
        reason="行情可达 · LLM 可达",
        item_id="run1",
        market="fapi",
    )
    assert rec["symbol"] == "BTCUSDT"
    assert rec["status"] == "ok"
    assert rec["item_id"] == "run1"
    assert rec["market"] == "fapi"
    assert rec["alert_id"]
    assert rec["created_at"]

    items = store.list_recent(limit=10)
    assert len(items) == 1
    assert items[0]["symbol"] == "BTCUSDT"
    assert items[0]["reason"] == "行情可达 · LLM 可达"


def test_list_recent_newest_first_and_limit(tmp_path: Path) -> None:
    store = FileAlertStore(tmp_path / "alerts.jsonl")
    for i in range(5):
        store.append(symbol=f"SYM{i}", status="ok", reason=str(i))

    items = store.list_recent(limit=3)
    assert [it["symbol"] for it in items] == ["SYM4", "SYM3", "SYM2"]


def test_empty_store_returns_empty(tmp_path: Path) -> None:
    store = FileAlertStore(tmp_path / "alerts.jsonl")
    assert store.list_recent() == []


def test_append_redacts_secret_in_reason(tmp_path: Path) -> None:
    store = FileAlertStore(tmp_path / "alerts.jsonl")
    secret = "sk-ABCDEF1234567890ABCDEF"
    rec = store.append(symbol="BTCUSDT", status="error", reason=f"boom {secret} tail")
    assert secret not in rec["reason"]

    items = store.list_recent()
    assert secret not in items[0]["reason"]
    # 落盘文件也不应包含明文密钥
    assert secret not in (tmp_path / "alerts.jsonl").read_text(encoding="utf-8")
