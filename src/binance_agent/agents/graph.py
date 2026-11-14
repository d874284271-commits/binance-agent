"""LangGraph 编排核心:数据预检 → 并行分析师 → 有限辩论 → 研究主管 → 风控 → 策略合成。

裸 StateGraph(不依赖已移除的 prebuilt)。节点经 ``model_for(agent)`` 绑定各自 LLM,
便于 per-agent 选型与离线注入假模型;risk_manager 为规则式无 LLM。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from functools import partial
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from ..infra.settings import (
    ResearchConfig,
    StrategyPlannerMode,
    planner_flags_for_mode,
    planner_mode_from_flags,
)
from . import nodes, preflight
from .state import AgentState, StrategyAdvice

ModelFor = Callable[[str], BaseChatModel]
FallbacksFor = Callable[[str], list[BaseChatModel]]


def _should_continue_debate(state: AgentState) -> str | list[str]:
    # debate_controller 在达上限或上一轮无新增已核验证据时写入 stop_reason
    if state.get("debate_stop_reason") or state["debate_round"] >= state["max_debate_rounds"]:
        return "debate_referee"
    return ["bull_debater", "bear_debater"]


def _after_data_preflight(state: AgentState) -> str | list[str]:
    if preflight.preflight_allows_analysis(state):
        return ["technical", "funding", "news_sentiment"]
    return END


def build_graph(
    model_for: ModelFor,
    *,
    checkpointer: Any | None = None,
    interrupt_after: list[str] | None = None,
    strategy_planner_mode: StrategyPlannerMode | None = None,
    enable_strategy_planner: bool = False,
    planner_max_steps: int = 3,
    planner_tool_registry: Any | None = None,
    enable_strategy_planner_llm_actions: bool = False,
    enforce_strategy_planner_decision: bool | None = None,
    runtime_config_for: Callable[[str], dict[str, Any]] | None = None,
    fallbacks_for: FallbacksFor | None = None,
    preflight_refetch_enabled: bool = True,
    analysis_clock: Callable[[], datetime] | None = None,
    preflight_tool_registry: Any | None = None,
    preflight_tool_context: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    planner_tool_context: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    confidence_bucket_caps: dict[str, float] | None = None,
    confidence_policy_version: str = "confidence-policy/v2",
    research_config: ResearchConfig | None = None,
    research_fetch: Any = None,
    research_search: Any = None,
    research_persist: Any = None,
    research_load: Any = None,
    topic_graph: Any = None,
) -> Any:
    """装配并编译分析图,返回可运行的 LangGraph;构建本身不发起模型或数据请求。

    model_for 与 runtime_config_for 决定各角色模型及端点能力,checkpointer
    由调用方管理。预检阻断时直接结束;研究工作流配置决定单角色、无辩论或
    有限辩论路径。风控和最终证据闸门仍保留,replay 只消费保存产物。
    """
    graph: StateGraph = StateGraph(AgentState)

    def _rc(agent: str) -> dict[str, Any]:
        cfg = dict(runtime_config_for(agent) if runtime_config_for else {})
        if fallbacks_for:
            cfg["fallbacks"] = fallbacks_for(agent)
        return cfg

    evidence_mode = research_config is not None and research_config.enabled
    single_mode = (
        evidence_mode and research_config is not None and research_config.workflow == "single"
    )

    effective_mode = strategy_planner_mode or planner_mode_from_flags(
        enable_strategy_planner, enable_strategy_planner_llm_actions
    )
    (
        effective_planner_enabled,
        effective_llm_actions,
        default_enforce_planner_decision,
    ) = planner_flags_for_mode(effective_mode)
    enforce_planner_decision = (
        default_enforce_planner_decision
        if enforce_strategy_planner_decision is None
        else enforce_strategy_planner_decision
    )

    # planner_* 两个依赖参数保留给旧 server 一个版本的调用兼容;补抓能力已提升到 preflight。
    effective_preflight_registry = (
        preflight_tool_registry if preflight_tool_registry is not None else planner_tool_registry
    )
    effective_preflight_context = (
        preflight_tool_context if preflight_tool_context is not None else planner_tool_context
    )
    graph.add_node(
        "data_preflight",
        partial(
            preflight.run_data_preflight,
            tool_registry=effective_preflight_registry,
            tool_context=effective_preflight_context,
            refetch_enabled=preflight_refetch_enabled,
            analysis_clock=analysis_clock,
        ),
    )

    graph.add_node(
        "technical",
        partial(nodes.technical_analyst, llm=model_for("technical"), runtime_cfg=_rc("technical")),
    )
    graph.add_node(
        "funding",
        partial(nodes.funding_analyst, llm=model_for("funding"), runtime_cfg=_rc("funding")),
    )
    if evidence_mode:
        from .research import build_news_research_graph

        assert research_config is not None
        graph.add_node(
            "news_sentiment",
            build_news_research_graph(
                model_for,
                config=research_config,
                runtime_config_for=_rc,
                fetch=research_fetch,
                search=research_search,
                persist=research_persist,
                load=research_load,
            ),
        )
    else:
        graph.add_node(
            "news_sentiment",
            partial(
                nodes.news_sentiment_analyst,
                llm=model_for("news_sentiment"),
                runtime_cfg=_rc("news_sentiment"),
            ),
        )

    graph.add_node(
        "bull_debater",
        partial(nodes.bull_debater, llm=model_for("bull_debater"), runtime_cfg=_rc("bull_debater")),
    )
    graph.add_node(
        "bear_debater",
        partial(nodes.bear_debater, llm=model_for("bear_debater"), runtime_cfg=_rc("bear_debater")),
    )
    graph.add_node("debate_controller", nodes.debate_controller)
    graph.add_node("debate_referee", nodes.debate_referee)  # 规则式,无 LLM;核验证据完整性
    graph.add_node(
        "research_manager",
        partial(
            nodes.research_manager,
            llm=model_for("research_manager"),
            runtime_cfg=_rc("research_manager"),
        ),
    )
    if effective_planner_enabled:
        planner_llm = model_for("strategy_planner") if effective_llm_actions else None
        graph.add_node(
            "strategy_planner",
            partial(
                nodes.strategy_planner,
                tool_registry=planner_tool_registry,
                max_steps=planner_max_steps,
                llm=planner_llm,
                enable_llm_actions=effective_llm_actions,
                planner_mode=effective_mode,
                planner_enforced=enforce_planner_decision,
                runtime_cfg=_rc("strategy_planner"),
            ),
        )
    graph.add_node("risk_manager", nodes.risk_manager)  # 规则式,无 LLM
    graph.add_node(
        "strategy_synth",
        partial(
            nodes.strategy_synth,
            llm=model_for("strategy_synth"),
            runtime_cfg=_rc("strategy_synth"),
            apply_planner_decision=enforce_planner_decision,
            confidence_bucket_caps=dict(confidence_bucket_caps or {}),
            confidence_policy_version=confidence_policy_version,
            evidence_mode=evidence_mode,
        ),
    )

    # 确定性预检建立 canonical snapshots;blocked 时直接 END,绝不消耗任何 LLM。
    async def replay_saved(state: AgentState) -> dict[str, Any]:
        """仅根据已保存建议与证据回放,不发起模型、抓取或压缩请求。

        缺失建议返回 no_trade;研究记录交最终证据闸门校验已有核验版本,
        不能把缺少原文或不匹配的历史核验升级为当前已核验结论。
        """
        if state.get("mode") == "topic":
            from .reports import incomplete_report

            return {
                "final_report": state.get("final_report")
                or incomplete_report(dict(state), "旧记录未保存专题结果,请重新研究").model_dump(
                    mode="json"
                )
            }
        advice = state.get("candidate_advice") or state.get("final_advice")
        if advice is None:
            advice = StrategyAdvice(
                symbol=state["symbol"],
                decision_type="no_trade",
                confidence=0.0,
                rationale="旧记录没有保存最终建议,无法离线复原。",
                no_trade_reason="missing_saved_advice",
            )
        elif isinstance(advice, dict):
            advice = StrategyAdvice.model_validate(advice)
        if evidence_mode or state.get("news_research"):
            from .research import finalize_evidence_advice

            return await finalize_evidence_advice(
                {**state, "candidate_advice": advice},
                llm=None,  # type: ignore[arg-type]  # replay branch never constructs a runtime
            )
        return {"final_advice": advice.model_copy(deep=True)}

    graph.add_node("replay_saved", replay_saved)
    if topic_graph is not None:
        graph.add_node("topic_research", topic_graph)
        graph.add_edge("topic_research", END)
    graph.add_conditional_edges(
        START,
        lambda state: (
            "replay_saved"
            if state.get("research_replay")
            else "topic_research"
            if state.get("mode") == "topic" and topic_graph is not None
            else "data_preflight"
        ),
    )
    graph.add_edge("replay_saved", END)
    if single_mode:
        graph.add_conditional_edges(
            "data_preflight",
            lambda state: "news_sentiment" if preflight.preflight_allows_analysis(state) else END,
        )
    else:
        graph.add_conditional_edges("data_preflight", _after_data_preflight)
    if single_mode:
        graph.add_edge("news_sentiment", "risk_manager")
    elif evidence_mode:

        def debate_route(state: AgentState) -> list[str] | str:
            stances = {r.stance for r in state["analyst_reports"] if r.confidence >= 0.4}
            research = state.get("news_research") or {}
            if (
                research_config is not None
                and research_config.workflow == "multi_debate"
                and {"bullish", "bearish"} <= stances
                and not research.get("critical_unresolved")
                and not research.get("gaps")
                and not research.get("conflicts")
            ):
                return ["bull_debater", "bear_debater"]
            return "debate_referee"

        graph.add_node(
            "analysis_complete",
            lambda state: {"max_debate_rounds": min(2, max(1, state["max_debate_rounds"]))},
        )
        graph.add_edge(["technical", "funding", "news_sentiment"], "analysis_complete")
        graph.add_conditional_edges("analysis_complete", debate_route)
    else:
        graph.add_edge(["technical", "funding", "news_sentiment"], "bull_debater")
        graph.add_edge(["technical", "funding", "news_sentiment"], "bear_debater")
    graph.add_edge(["bull_debater", "bear_debater"], "debate_controller")
    # 条件边:未达上限且仍有新证据则继续辩论,否则进入 referee 核验证据完整性后转研究主管
    graph.add_conditional_edges("debate_controller", _should_continue_debate)
    graph.add_edge("debate_referee", "research_manager")  # referee 只在收敛后跑一次,不引入重辩循环
    if effective_planner_enabled:
        graph.add_edge("research_manager", "strategy_planner")
        graph.add_edge("strategy_planner", "risk_manager")
    else:
        graph.add_edge("research_manager", "risk_manager")
    graph.add_edge("risk_manager", "strategy_synth")
    if evidence_mode:
        from .research import finalize_evidence_advice

        graph.add_node(
            "final_evidence_gate",
            partial(
                finalize_evidence_advice,
                llm=model_for("evidence_verifier"),
                runtime_cfg=_rc("evidence_verifier"),
            ),
        )
        graph.add_edge("strategy_synth", "final_evidence_gate")
        graph.add_edge("final_evidence_gate", END)
    else:
        graph.add_edge("strategy_synth", END)

    return graph.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)
