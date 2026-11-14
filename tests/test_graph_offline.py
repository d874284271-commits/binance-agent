"""M4 全链路离线测试:注入固定快照 + 假 LLM,跑完整图,断言关键不变式。

校验:三份分析报告齐全 / 辩论轮次 ≤ 上限 / 临近宏观事件触发高风险告警 /
final_advice 字段完整含 disclaimer。无需网络与真实 LLM(CI 友好)。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from binance_agent.agents.graph import build_graph


def _fakes() -> dict[str, BaseChatModel]:
    return {
        "technical": FakeListChatModel(
            responses=['{"stance":"bullish","confidence":0.6,"summary":"技术面偏多"}']
        ),
        "funding": FakeListChatModel(
            responses=['{"stance":"bearish","confidence":0.5,"summary":"资金面承压"}']
        ),
        "news_sentiment": FakeListChatModel(
            responses=['{"stance":"neutral","confidence":0.5,"summary":"情绪中性"}']
        ),
        "bull_debater": FakeListChatModel(
            responses=[
                '{"argument":"多头:动量与趋势向上",'
                '"claims":[{"statement":"趋势向上","evidence":"analyst_reports.technical.stance=bullish",'
                '"source_path":"analyst_reports.technical.stance","source_value":"bullish",'
                '"confidence":0.6}]}',
                '{"argument":"多头:短周期仍有支撑",'
                '"claims":[{"statement":"仍有支撑","evidence":"snapshot_data.ticker_24h.lastPrice=100",'
                '"source_path":"snapshot_data.ticker_24h.lastPrice","source_value":"100",'
                '"confidence":0.55}]}',
            ]
        ),
        "bear_debater": FakeListChatModel(
            responses=[
                '{"argument":"空头:临近事件且资金过热",'
                '"claims":[{"statement":"事件风险偏高","evidence":"risk macro event in intel snapshot",'
                '"source_path":"intel_snapshot.macro_events","source_value":1,"confidence":0.6}]}',
                '{"argument":"空头:资金费率仍需警惕",'
                '"claims":[{"statement":"资金费率偏高","evidence":"snapshot_data.funding.last_funding_rate=0.0006",'
                '"source_path":"snapshot_data.funding.last_funding_rate","source_value":0.0006,'
                '"confidence":0.55}]}',
            ]
        ),
        "research_manager": FakeListChatModel(responses=["分歧明显,倾向区间震荡"]),
        "strategy_synth": FakeListChatModel(
            responses=[
                '{"direction":"neutral","entry_zone":"100-102","stop_loss":"98",'
                '"targets":["105","108"],"confidence":0.55,"rationale":"综合中性偏谨慎"}'
            ]
        ),
    }


class _FailingChatModel(FakeListChatModel):
    async def ainvoke(  # type: ignore[override]
        self, messages: Any, *args: Any, **kwargs: Any
    ) -> AIMessage:
        raise TimeoutError("llm timeout")


def _complete_market_snapshot(*, funding_rate: float) -> dict[str, Any]:
    rows = [
        {
            "open": i,
            "high": i + 2,
            "low": i - 2,
            "close": i + 0.5,
            "volume": 10,
        }
        for i in range(1, 50)
    ]
    return {
        "klines": {"5m": rows, "1h": rows, "4h": rows, "1d": rows},
        "funding": {"last_funding_rate": funding_rate},
        "ticker_24h": {
            "lastPrice": "100",
            "priceChangePercent": "1.2",
            "highPrice": "105",
            "lowPrice": "95",
            "volume": "1000",
            "quoteVolume": "100000",
        },
        "errors": [],
    }


def _initial_state(*, with_event: bool = True, complete_market: bool = True) -> dict[str, Any]:
    soon = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    macro = [{"name": "美国CPI", "time": soon, "importance": 3}] if with_event else []
    funding_rate = 0.0006 if with_event else 0.0001
    long_short = 1.2
    market_snapshot = (
        _complete_market_snapshot(funding_rate=funding_rate)
        if complete_market
        else {
            "klines": {},
            "funding": {"last_funding_rate": funding_rate},
            "ticker_24h": {},
            "errors": ["klines[1h]: boom"],
        }
    )
    return {
        "symbol": "BTCUSDT",
        "market": "fapi",
        "market_snapshot": market_snapshot,
        "intel_snapshot": {
            "sentiment_features": {"long_short_ratio": long_short},
            "news_items": [],
            "macro_events": macro,
            "source_status": {"fear_greed": "ok", "reddit": "ok", "rss": "ok"},
        },
        "analyst_reports": [],
        "debate_turns": [],
        "debate_round": 0,
        "max_debate_rounds": 2,
        "research_summary": None,
        "risk_assessment": None,
        "final_advice": None,
    }


async def test_full_graph_offline_with_event() -> None:
    fakes = _fakes()
    app = build_graph(lambda agent: fakes[agent])
    result = await app.ainvoke(_initial_state(with_event=True))

    # 三份分析报告齐全(并行 reducer 正确)
    assert len(result["analyst_reports"]) == 3
    assert {r.analyst for r in result["analyst_reports"]} == {
        "technical",
        "funding",
        "news_sentiment",
    }

    # 辩论轮次受硬上限约束
    assert result["debate_round"] == 2
    assert len(result["debate_turns"]) == 4
    assert max(t.round for t in result["debate_turns"]) < result["max_debate_rounds"]

    # 事件感知风控:临近高重要性事件 → 高风险告警
    risk = result["risk_assessment"]
    assert risk["level"] == "high"
    assert any("宏观事件" in w for w in risk["warnings"])

    # 最终建议字段完整 + 强制免责声明 + 风险告警透传
    advice = result["final_advice"]
    assert advice.symbol == "BTCUSDT"
    assert advice.direction in {"long", "short", "neutral"}
    assert advice.disclaimer
    assert any("CPI" in w or "宏观事件" in w for w in advice.risk_warnings)

    # referee 核验证据完整性:产出独立报告,且不重复追加 debate_turns(len 仍为 4)
    grounding = result["debate_grounding"]
    assert grounding["total_claims"] == 4
    # 现有 fake claims:3 条可在快照/报告定位、1 条(intel_snapshot.macro_events)命名空间外 → 0.75
    assert grounding["verified_ratio"] == 0.75
    assert len(result["debate_turns"]) == 4


def _bogus_ref_fakes() -> dict[str, BaseChatModel]:
    """辩手 claims 的结构化引用全部无法通过证据完整性核验。"""
    fakes = _fakes()
    for name in ("technical", "funding", "news_sentiment"):
        fakes[name] = FakeListChatModel(
            responses=['{"stance":"bullish","confidence":0.8,"summary":"一致看多"}']
        )
    fakes["bull_debater"] = FakeListChatModel(
        responses=[
            '{"argument":"多头","claims":[{"statement":"编造1","evidence":"x",'
            '"source_path":"snapshot_data.nope.foo","source_value":1,"confidence":0.6}]}',
            '{"argument":"多头2","claims":[{"statement":"编造2","evidence":"x",'
            '"source_path":"analyst_reports.ghost.confidence","source_value":0.9,'
            '"confidence":0.6}]}',
        ]
    )
    fakes["bear_debater"] = FakeListChatModel(
        responses=[
            '{"argument":"空头","claims":[{"statement":"编造3","evidence":"x",'
            '"source_path":"totally.made.up","source_value":1,"confidence":0.6}]}',
            '{"argument":"空头2","claims":[{"statement":"编造4","evidence":"x",'
            '"source_path":"intel_snapshot.macro_events","source_value":5,"confidence":0.6}]}',
        ]
    )
    fakes["research_manager"] = FakeListChatModel(responses=["倾向看多"])
    fakes["strategy_synth"] = FakeListChatModel(
        responses=[
            '{"direction":"long","decision_type":"trade","entry_zone":"100-102","stop_loss":"96",'
            '"targets":["108","112"],"confidence":0.9,"rationale":"看多"}'
        ]
    )
    return fakes


async def test_ungrounded_debate_caps_final_confidence() -> None:
    fakes = _bogus_ref_fakes()
    app = build_graph(lambda agent: fakes[agent])
    result = await app.ainvoke(_initial_state(with_event=False))

    grounding = result["debate_grounding"]
    assert grounding["verified_ratio"] == 0.0

    advice = result["final_advice"]
    assert advice.confidence <= 0.6  # 表演性辩论 → 置信度被压低
    assert any("辩论" in reason for reason in advice.confidence_reasons)


async def test_graph_threads_llm_fallbacks_into_nodes() -> None:
    fakes = _fakes()
    primary = dict(fakes)
    primary["technical"] = _FailingChatModel(responses=[])
    fallback = FakeListChatModel(
        responses=['{"stance":"bullish","confidence":0.7,"summary":"fallback technical"}']
    )
    app = build_graph(
        lambda agent: primary[agent],
        fallbacks_for=lambda agent: [fallback] if agent == "technical" else [],
    )

    result = await app.ainvoke(_initial_state(with_event=False))

    technical = next(
        report for report in result["analyst_reports"] if report.analyst == "technical"
    )
    event = next(event for event in result["runtime_events"] if event["agent"] == "technical")
    assert technical.summary == "fallback technical"
    assert event["ok"] is True
    assert event["model_index"] == 1


async def test_full_graph_offline_no_event_low_risk() -> None:
    fakes = _fakes()
    app = build_graph(lambda agent: fakes[agent])
    result = await app.ainvoke(_initial_state(with_event=False))

    assert len(result["analyst_reports"]) == 3
    assert result["risk_assessment"]["level"] == "low"
    assert result["final_advice"].risk_warnings == []
    assert result["final_advice"].decision_type == "no_trade"
    assert result["final_advice"].no_trade_reason


async def test_full_graph_missing_critical_market_data_forces_no_trade() -> None:
    fakes = _fakes()
    app = build_graph(lambda agent: fakes[agent])
    result = await app.ainvoke(_initial_state(with_event=False, complete_market=False))

    advice = result["final_advice"]
    assert advice.decision_type == "no_trade"
    assert advice.direction == "neutral"
    assert advice.entry_zone is None
    assert advice.stop_loss is None
    assert advice.targets == []
    assert advice.no_trade_reason


async def test_full_graph_single_analyst_failure_still_completes() -> None:
    fakes = _fakes()
    fakes["technical"] = _FailingChatModel(responses=[])
    app = build_graph(lambda agent: fakes[agent])
    result = await app.ainvoke(_initial_state(with_event=False))

    reports = {report.analyst: report for report in result["analyst_reports"]}
    assert reports["technical"].stance == "neutral"
    assert reports["technical"].confidence <= 0.2
    assert "LLM" in reports["technical"].summary
    assert result["final_advice"].decision_type == "no_trade"


async def test_full_graph_strategy_synth_failure_returns_no_trade() -> None:
    fakes = _fakes()
    fakes["strategy_synth"] = _FailingChatModel(responses=[])
    app = build_graph(lambda agent: fakes[agent])
    result = await app.ainvoke(_initial_state(with_event=False))

    advice = result["final_advice"]
    assert advice.decision_type == "no_trade"
    assert advice.direction == "neutral"
    assert advice.entry_zone is None
    assert advice.stop_loss is None
    assert advice.targets == []
    assert advice.no_trade_reason


async def test_full_graph_strategy_repair_caps_final_confidence() -> None:
    fakes = _fakes()
    fakes["technical"] = FakeListChatModel(
        responses=['{"stance":"bullish","confidence":0.9,"summary":"技术面强"}']
    )
    fakes["funding"] = FakeListChatModel(
        responses=['{"stance":"bullish","confidence":0.85,"summary":"资金面支持"}']
    )
    fakes["news_sentiment"] = FakeListChatModel(
        responses=['{"stance":"bullish","confidence":0.8,"summary":"情绪支持"}']
    )
    fakes["strategy_synth"] = FakeListChatModel(
        responses=[
            "not json",
            '{"direction":"long","decision_type":"trade","entry_zone":"100",'
            '"stop_loss":"95","targets":["115"],"confidence":0.95,'
            '"rationale":"一致看多"}',
        ]
    )
    app = build_graph(lambda agent: fakes[agent])
    result = await app.ainvoke(_initial_state(with_event=False))

    advice = result["final_advice"]
    assert advice.decision_type == "trade"
    assert advice.model_confidence == 0.95
    assert advice.confidence <= 0.65
    assert any("JSON repair" in reason for reason in advice.confidence_reasons)


async def test_strategy_synth_evaluates_risk_gate_once(monkeypatch: Any) -> None:
    # Bug4 回归:strategy_synth 内 evaluate_risk_gate 仅应执行一次
    # (此前 nodes 直接算一次 + apply_quality_and_risk_gates 内再算一次 = 2 次,
    #  且持久化的 risk_gate_result 与实际改 advice 用的 gate 是两个对象)。
    import binance_agent.agents.nodes as nodes_mod
    import binance_agent.agents.validation as val_mod

    real = val_mod.evaluate_risk_gate
    calls = {"n": 0}

    def _counting(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real(*args, **kwargs)

    # 两处绑定都要替换:nodes:606 用 nodes 命名空间,apply 内部用 validation 命名空间
    monkeypatch.setattr(val_mod, "evaluate_risk_gate", _counting)
    monkeypatch.setattr(nodes_mod, "evaluate_risk_gate", _counting)

    fakes = _fakes()
    fakes["strategy_synth"] = FakeListChatModel(
        responses=[
            '{"direction":"long","decision_type":"trade","entry_zone":"100-102",'
            '"stop_loss":"95","targets":["110","120"],"confidence":0.7,"rationale":"看多"}'
        ]
    )
    app = build_graph(lambda agent: fakes[agent])
    result = await app.ainvoke(_initial_state(with_event=False))

    assert calls["n"] == 1  # 去重前为 2
    assert result["final_advice"] is not None
    assert result["risk_gate_result"] is not None
