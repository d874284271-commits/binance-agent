"""RunStore/EventLog:本地 JSONL 执行轨迹持久化测试。"""

from __future__ import annotations

from pathlib import Path

from binance_agent.app.run_store import FileRunStore


def test_file_run_store_roundtrip(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path / "runs.jsonl")
    run_id = store.create(symbol="BTCUSDT", market="fapi")

    store.append_event(run_id, "market", {"symbol": "BTCUSDT"})
    store.append_event(
        run_id,
        "node",
        {
            "node": "technical",
            "update": {
                "analyst_reports": [{"analyst": "technical", "stance": "neutral"}],
                "runtime_events": [{"agent": "technical", "ok": True, "latency_ms": 12.3}],
            },
        },
    )
    store.append_event(
        run_id,
        "node",
        {
            "node": "risk_manager",
            "update": {
                "risk_assessment": {
                    "level": "low",
                    "warnings": [],
                    "data_quality": {"level": "ok", "score": 1.0},
                }
            },
        },
    )
    store.append_event(
        run_id,
        "node",
        {
            "node": "strategy_synth",
            "update": {"risk_gate_result": {"veto": False, "warnings": []}},
        },
    )
    store.complete(run_id, {"decision_type": "no_trade"})

    record = store.get(run_id)
    assert record is not None
    assert record["run_id"] == run_id
    assert record["symbol"] == "BTCUSDT"
    assert record["market"] == "fapi"
    assert record["status"] == "completed"
    assert record["market_snapshot"]["symbol"] == "BTCUSDT"
    assert record["node_outputs"][0]["node"] == "technical"
    assert record["runtime_events"][0]["agent"] == "technical"
    assert record["data_quality_report"]["level"] == "ok"
    assert record["risk_gate_result"]["veto"] is False
    assert record["final_advice"]["decision_type"] == "no_trade"

    recent = store.list_recent(limit=5)
    assert recent[0]["run_id"] == run_id


def test_file_run_store_missing_run(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path / "runs.jsonl")
    assert store.get("missing") is None


def test_list_recent_summaries_omits_heavy_snapshots(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path / "runs.jsonl")
    run_id = store.create(symbol="BTCUSDT", market="fapi")
    store.append_event(
        run_id,
        "market",
        {
            "ticker_24h": {"lastPrice": "100"},
            "klines": {"1h": [{"close": i} for i in range(200)]},
            "errors": [],
        },
    )
    store.append_event(
        run_id,
        "intel",
        {"source_status": {"rss": "ok"}, "news_items": [{"title": "x"}]},
    )
    store.complete(run_id, {"decision_type": "no_trade", "confidence": 0.2})

    summary = store.list_recent_summaries(limit=1)[0]

    assert summary["run_id"] == run_id
    assert summary["symbol"] == "BTCUSDT"
    assert summary["final_advice"]["decision_type"] == "no_trade"
    assert "market_snapshot" not in summary
    assert "intel_snapshot" not in summary
    assert "events" not in summary
    assert "node_outputs" not in summary


def test_file_run_store_redacts_registered_secret(tmp_path: Path) -> None:
    # #5 回归:落盘前脱敏。错误消息若夹带已注册密钥,不应明文进 .runs JSONL
    # (logging RedactingFilter 不覆盖此写盘路径)。用无 sk- 前缀的串,确保走"注册密钥"替换而非正则兜底。
    from binance_agent.infra.logging import register_secret

    secret = "BLAZEKEY-deadbeefcafe-secret-9911"
    register_secret(secret)
    store = FileRunStore(tmp_path / "runs.jsonl")
    run_id = store.create(symbol="BTCUSDT", market="fapi")
    store.fail(run_id, f"upstream 401: auth failed with key={secret}")

    raw = (tmp_path / "runs.jsonl").read_text(encoding="utf-8")
    assert secret not in raw
    assert "***REDACTED***" in raw

    record = store.get(run_id)
    assert record is not None
    assert secret not in str(record["error"])


def test_file_run_store_tracks_checkpoint_resumable_status(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path / "runs.jsonl")
    run_id = store.create(symbol="ETHUSDT", market="fapi")

    store.append_event(
        run_id,
        "checkpoint",
        {"backend": "sqlite", "thread_id": run_id, "status": "resumable"},
    )
    store.fail(run_id, "strategy_synth interrupted")

    failed = store.get(run_id)
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["checkpoint_status"] == "resumable"
    assert failed["checkpoint_backend"] == "sqlite"
    assert failed["resumable"] is True

    store.append_event(run_id, "resume", {"source": "checkpoint"})
    store.append_event(run_id, "checkpoint", {"thread_id": run_id, "status": "completed"})
    store.complete(run_id, {"decision_type": "no_trade"})

    completed = store.get(run_id)
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["checkpoint_status"] == "completed"
    assert completed["resumable"] is False


def test_file_run_store_tracks_tool_events(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path / "runs.jsonl")
    run_id = store.create(symbol="BTCUSDT", market="fapi")

    store.append_event(
        run_id,
        "tool",
        {
            "tool": "market_snapshot",
            "ok": True,
            "latency_ms": 3.5,
            "input": {"symbol": "BTCUSDT", "market": "fapi"},
            "output": {"market_snapshot": {"symbol": "BTCUSDT"}},
            "errors": [],
        },
    )

    record = store.get(run_id)
    assert record is not None
    assert record["tool_events"][0]["tool"] == "market_snapshot"
    assert record["tool_events"][0]["ok"] is True
    assert record["tool_events"][0]["output"]["market_snapshot"]["symbol"] == "BTCUSDT"


def test_file_run_store_tracks_planner_steps_from_node_output(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path / "runs.jsonl")
    run_id = store.create(symbol="BTCUSDT", market="fapi")

    store.append_event(
        run_id,
        "node",
        {
            "node": "strategy_planner",
            "update": {
                "planner_steps": [
                    {
                        "step": 1,
                        "thought": "检查数据质量",
                        "action": "data_quality_check",
                        "observation": {"ok": True},
                        "decision": "continue",
                    }
                ],
                "planner_decision": {"decision_type": "watchlist", "reason": "max_steps"},
                "tool_events": [{"tool": "data_quality_check", "ok": True}],
            },
        },
    )

    record = store.get(run_id)
    assert record is not None
    assert record["planner_steps"][0]["action"] == "data_quality_check"
    assert record["planner_decision"]["decision_type"] == "watchlist"
    assert record["tool_events"][0]["tool"] == "data_quality_check"


def test_run_record_derives_run_health_from_events(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path / "runs.jsonl")
    run_id = store.create(symbol="BTCUSDT", market="fapi")
    store.append_event(
        run_id,
        "tool",
        {
            "tool": "market_snapshot",
            "ok": True,
            "latency_ms": 3.5,
            "input": {"symbol": "BTCUSDT", "market": "fapi"},
            "output": {"market_snapshot": {"symbol": "BTCUSDT"}},
            "errors": [],
        },
    )
    store.append_event(
        run_id,
        "tool",
        {
            "tool": "intel_snapshot",
            "ok": False,
            "latency_ms": 5.0,
            "input": {"symbol": "BTCUSDT"},
            "output": {"intel_snapshot": {"symbol": "BTCUSDT"}},
            "errors": [{"category": "timeout", "message": "rss timeout"}],
        },
    )
    store.append_event(
        run_id,
        "node",
        {
            "node": "technical",
            "update": {
                "runtime_events": [
                    {
                        "agent": "technical",
                        "ok": False,
                        "error_category": "llm_error",
                        "model_index": 1,
                        "attempts": 3,
                    }
                ]
            },
        },
    )
    store.append_event(
        run_id,
        "node",
        {
            "node": "risk_manager",
            "update": {
                "risk_assessment": {
                    "data_quality": {
                        "level": "degraded",
                        "score": 0.75,
                        "trade_blocked": False,
                        "reasons": ["资讯源成功率过低:40%"],
                    }
                }
            },
        },
    )
    store.complete(run_id, {"decision_type": "no_trade", "confidence": 0.2})

    record = store.get(run_id)

    assert record is not None
    assert record["run_health"]["market_status"] == "ok"
    assert record["run_health"]["intel_status"] == "warn"
    assert record["run_health"]["llm_status"] == "warn"
    assert record["run_health"]["fallback_agents"] == ["technical"]
    assert record["run_health"]["data_quality"] == "degraded"
    assert record["evidence_summary"]["decision_type"] == "no_trade"
    assert record["evidence_summary"]["confidence"] == 0.2
    assert record["evidence_factors"] == []


def test_run_health_surfaces_repaired_agents(tmp_path: Path) -> None:
    # repaired=True(JSON 经二次修复才救回)的 agent 应出现在 run_health.repaired_agents,
    # 且不算 fallback(ok=True)。
    store = FileRunStore(tmp_path / "runs.jsonl")
    run_id = store.create(symbol="BTCUSDT", market="fapi")
    store.append_event(
        run_id,
        "node",
        {
            "node": "strategy_synth",
            "update": {
                "runtime_events": [
                    {"agent": "strategy_synth", "ok": True, "repaired": True},
                    {"agent": "technical", "ok": True, "repaired": False},
                ]
            },
        },
    )
    store.complete(run_id, {"decision_type": "no_trade"})

    record = store.get(run_id)
    assert record is not None
    assert record["run_health"]["repaired_agents"] == ["strategy_synth"]


def test_preflight_node_promotes_effective_snapshots_and_report(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path / "runs.jsonl")
    run_id = store.create(symbol="BTCUSDT", market="fapi")
    store.append_event(run_id, "market", {"ticker_24h": {"lastPrice": "90"}})
    store.append_event(run_id, "intel", {"source_status": {"rss": "error"}})
    store.append_event(
        run_id,
        "node",
        {
            "node": "data_preflight",
            "update": {
                "market_snapshot": {"ticker_24h": {"lastPrice": "101"}},
                "intel_snapshot": {"source_status": {"rss": "ok"}},
                "preflight_report": {
                    "status": "ready",
                    "effective_snapshot_version": "sha256:abc",
                    "final_quality": {"level": "ok", "score": 1.0},
                },
            },
        },
    )

    record = store.get(run_id)

    assert record is not None
    assert record["market_snapshot"]["ticker_24h"]["lastPrice"] == "101"
    assert record["intel_snapshot"]["source_status"]["rss"] == "ok"
    assert record["preflight_report"]["effective_snapshot_version"] == "sha256:abc"
    assert record["data_quality_report"]["level"] == "ok"
    assert record["run_health"]["fallback_agents"] == []
    assert record["run_health"]["llm_status"] == "ok"
