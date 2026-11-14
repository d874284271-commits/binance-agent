"""FastAPI 应用工厂。

M0: /health
M1: /snapshot(行情快照)
M2: /intel(资讯情报)
M4: /analyze(SSE 多智能体分阶段)、/config/llm-endpoint/switch(运行时切换端点)
M5: 静态悬浮窗 UI(/ 与 /ui)
依赖经 lifespan 创建/释放,绑定到 app.state。
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import uuid
import warnings
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from .. import __version__
from ..agents.budget import EndpointPool, RunBudget, run_budget_scope
from ..agents.collaboration import AgentEngine
from ..agents.context import COMPACTION_PROMPT_VERSION, CONTEXT_VERSION, ContextPolicy
from ..agents.graph import build_graph
from ..agents.output_validation import OUTPUT_SCHEMA_VERSION
from ..agents.reading import ANALYSIS_VERSION
from ..agents.state import DISCLAIMER
from ..agents.tools import ToolInvocationResult, ToolRegistry, build_default_tool_registry
from ..demo import (
    DEMO_REQUEST_ID,
    DemoHttpClient,
    DemoIntelAggregator,
    DemoMarketData,
    DemoOutcomeStore,
    DemoResearchContent,
    DemoRestClient,
    DemoWsManager,
    build_demo_registry,
)
from ..infra.bounded_http import BoundedAsyncClient
from ..infra.logging import redact_text
from ..infra.paths import resolve_data_path
from ..infra.profiles import apply_profile
from ..infra.runtime_config import (
    RuntimeConfigError,
    activate_candidate,
    apply_overrides,
    clear_overrides,
    load_overrides,
    overrides_present,
    save_overrides,
)
from ..infra.runtime_config import (
    _default_path as runtime_config_path,
)
from ..infra.secrets import resolve_secret, store_secret
from ..infra.settings import (
    ConfidenceConfig,
    NetworkConfig,
    Settings,
    StrategyPlannerMode,
    effective_strategy_planner_mode,
    load_settings,
    planner_flags_for_mode,
    planner_mode_from_flags,
)
from ..infra.startup_errors import StartupError, startup_error
from ..intel.aggregator import IntelAggregator, build_providers
from ..intel.content import ContentFetcher, HttpxPinnedTransport
from ..intel.sources import SOURCE_SPECS, source_catalog, source_spec
from ..llm.registry import AGENT_NAMES, LlmRegistry
from ..market.rest_client import BinanceRestClient
from ..market.snapshot import build_market_snapshot
from ..market.ws_manager import BinanceWsManager
from .alert_store import FileAlertStore
from .checkpoints import build_checkpointer, checkpoint_backend_name, has_checkpoint_thread
from .collaboration_store import CollaborationStore, SessionConflict
from .outcome_store import FileOutcomeStore
from .research_sessions import RunManager, register_session_routes
from .run_store import FileRunStore
from .security import (
    AnalysisLifecycleMiddleware,
    AnalysisLimiter,
    ConfigurationBusy,
    LocalOriginGuardMiddleware,
    LoopbackHostMiddleware,
    valid_local_origin,
)
from .streaming import ContextIterator, jsonable

logger = logging.getLogger(__name__)

_UI_DIR = Path(__file__).resolve().parent.parent / "desktop" / "ui"

# 需 key 的数据源申请页(供 UI 引导用户去哪申请)
_SOURCE_APPLY_URLS: dict[str, str] = {
    "cryptopanic": "https://cryptopanic.com/developers/api",
    "finnhub": "https://finnhub.io/dashboard",
    "fred": "https://fred.stlouisfed.org/docs/api/api_key.html",
    "guardian_api": "https://open-platform.theguardian.com/access/",
    "gnews_api": "https://gnews.io/",
    "newsapi": "https://newsapi.org/pricing",
    "mediastack": "https://mediastack.com/pricing",
}


def _build_aggregator(
    settings: Settings, intel_http: httpx.AsyncClient, rest_client: BinanceRestClient
) -> IntelAggregator:
    providers, disabled = build_providers(settings, intel_http, rest_client)
    return IntelAggregator(
        providers,
        per_source_timeout=settings.intel.per_source_timeout,
        window_hours=settings.intel.time_window_hours,
        max_news_items=settings.intel.max_news_items,
        retain_syndication=settings.research.enabled,
        disabled=disabled,
    )


def _research_config_fingerprint(settings: Settings, *, include_collaboration: bool = True) -> str:
    """Return a secret-free fingerprint for checkpoint compatibility.

    A research checkpoint is only safe to continue when the graph policy and
    endpoint execution contract are unchanged.  Secret values are deliberately
    excluded; the referenced key name remains part of the contract.
    """
    payload = {
        "robustness_contract": "robustness/v2",
        "product_contract": "reports/v1",
        "intel": settings.intel.model_dump(mode="json"),
        "confidence": settings.confidence.model_dump(mode="json"),
        "context_policy": settings.runtime.context.model_dump(mode="json"),
        "context_version": CONTEXT_VERSION,
        "research_analysis_version": ANALYSIS_VERSION,
        "model_output_schema": OUTPUT_SCHEMA_VERSION,
        "compaction_prompt_version": COMPACTION_PROMPT_VERSION,
        "research": settings.research.model_dump(mode="json"),
        "debate": settings.debate.model_dump(mode="json"),
        "runtime": {
            "strategy_planner_mode": settings.runtime.strategy_planner_mode,
            "strategy_planner_enabled": settings.runtime.strategy_planner_enabled,
            "strategy_planner_max_steps": settings.runtime.strategy_planner_max_steps,
            "strategy_planner_llm_actions": settings.runtime.strategy_planner_llm_actions,
            "preflight_refetch_enabled": settings.runtime.preflight_refetch_enabled,
        },
        "llm": {
            "active_endpoint": settings.llm.active_endpoint,
            "agent_models": settings.llm.agent_models,
            "endpoints": [
                {
                    key: getattr(endpoint, key)
                    for key in (
                        "name",
                        "base_url",
                        "api_key_ref",
                        "model",
                        "models",
                        "temperature",
                        "timeout",
                        "max_retries",
                        "structured_output",
                        "fallback_verified",
                        "max_concurrency",
                    )
                }
                for endpoint in settings.llm.endpoints
            ],
        },
    }
    if include_collaboration and settings.collaboration.enabled:
        payload["collaboration"] = settings.collaboration.model_dump(mode="json")
        payload["network"] = settings.network.model_dump(mode="json")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _current_overrides(settings: Settings, registry: LlmRegistry) -> dict[str, Any]:
    """从当前 settings + registry 构造可持久化 overrides(无 key 明文)。"""
    return {
        "llm": {
            "active": registry.active,
            "endpoints": [e.model_dump() for e in settings.llm.endpoints],
            "agent_models": {a: list(p) for a, p in settings.llm.agent_models.items()},
        },
        "desktop": settings.desktop.model_dump(),
        "hotkey": settings.hotkey.model_dump(),
        "research": {
            "max_requests": settings.research.max_requests,
            "deadline_s": settings.research.deadline_s,
        },
        "intel_sources": dict(settings.intel.sources),
        "intel_keys": dict(settings.intel.keys),
        "confidence": settings.confidence.model_dump(mode="json"),
        "network": {"proxy": settings.network.proxy},
        "params": {
            "context": settings.runtime.context.model_dump(mode="json"),
            "time_window_hours": settings.intel.time_window_hours,
            "per_source_timeout": settings.intel.per_source_timeout,
            "max_news_items": settings.intel.max_news_items,
            "max_debate_rounds": settings.debate.max_rounds,
            "strategy_planner_mode": effective_strategy_planner_mode(settings.runtime),
            "strategy_planner_enabled": settings.runtime.strategy_planner_enabled,
            "strategy_planner_max_steps": settings.runtime.strategy_planner_max_steps,
            "strategy_planner_llm_actions": settings.runtime.strategy_planner_llm_actions,
            "preflight_refetch_enabled": settings.runtime.preflight_refetch_enabled,
        },
    }


def _dep_item(
    *,
    status: str,
    severity: str,
    reason: str = "",
    action_label: str = "建议",
    action_hint: str = "查看配置与网络状态。",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "severity": severity,
        "reason": reason,
        "action_label": action_label,
        "action_hint": action_hint,
        **extra,
    }


def _graph_config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def _derive_watchlist(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从按创建(文件追加)顺序的 run 记录派生观察清单。

    每个 symbol 取最后出现(最新)的一条 run:若其 ``final_advice.decision_type ==
    "watchlist"`` 则收录(被后续 trade/no_trade 覆盖则自动出清单)。以文件顺序而非
    ``created_at`` 判定先后,避免低精度时钟(Windows ~15ms)平手导致选错。``item_id``
    为该 run 的 id,供 /watchlist/{item_id}/check 反查。
    """
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        symbol = record.get("symbol")
        if symbol:
            latest[symbol] = record  # 后出现者覆盖 → 最终为该 symbol 最新一条
    items: list[dict[str, Any]] = []
    for record in latest.values():
        advice = record.get("final_advice")
        if not isinstance(advice, dict) or advice.get("decision_type") != "watchlist":
            continue
        items.append(
            {
                "item_id": record.get("run_id"),
                "symbol": record.get("symbol"),
                "market": record.get("market"),
                "decision_type": "watchlist",
                "status": record.get("status"),
                "direction": advice.get("direction"),
                "confidence": advice.get("confidence"),
                "reentry_conditions": advice.get("reentry_conditions") or [],
                "created_at": record.get("created_at"),
            }
        )
    items.sort(key=lambda it: str(it.get("created_at") or ""), reverse=True)
    return items


def _build_runtime_graph(settings: Settings, registry: LlmRegistry, app: FastAPI) -> Any:
    planner_mode = effective_strategy_planner_mode(settings.runtime)
    planner_enabled, planner_llm_actions, enforce_planner_decision = planner_flags_for_mode(
        planner_mode
    )

    def _preflight_tool_context() -> dict[str, Any]:
        # 延迟求值:preflight 节点每次执行时从 app.state 现取依赖,
        # 规避 aggregator/rest_client 运行时重建(改源/改 key/改代理)导致图内引用过期。
        return {
            "aggregator": getattr(app.state, "aggregator", None),
            "rest_clients": {
                "fapi": getattr(app.state, "rest_client", None),
                "api": getattr(app.state, "spot_client", None),
            },
            "build_market_snapshot": getattr(
                app.state, "market_snapshot_builder", build_market_snapshot
            ),
        }

    def research_runtime_config(agent: str) -> dict[str, Any]:
        cfg = registry.runtime_config_for(agent)
        cfg["context_policy"] = settings.runtime.context
        if settings.research.enabled:
            allowed = {e.name for e in settings.llm.endpoints if e.fallback_verified}
            candidates = cfg.get("candidates", [])
            cfg["candidates"] = [
                c for i, c in enumerate(candidates) if i == 0 or c.endpoint in allowed
            ]
        return cfg

    if settings.research.enabled and not hasattr(app.state, "content_fetcher"):
        app.state.content_fetcher = ContentFetcher(
            transport=HttpxPinnedTransport(),
            trusted_official_domains={
                "sec.gov",
                "federalreserve.gov",
                "cftc.gov",
                "ecb.europa.eu",
                "ethereum.org",
                "github.com",
            },
        )
    from ..agents.topic import build_topic_graph
    from ..intel.search import TavilySearch

    search = getattr(app.state, "research_search", None)
    key = (
        resolve_secret(settings.intel.keys.get("tavily") or "TAVILY_API_KEY")
        if settings.profile != "demo"
        else None
    )
    if key and hasattr(app.state, "intel_http"):

        async def configured_search(query: str) -> list[dict[str, Any]]:
            assert key is not None
            return await TavilySearch(app.state.intel_http, key)(query)

        search = configured_search
    topic_graph = build_topic_graph(
        registry.for_agent,
        config=settings.research,
        runtime_config_for=research_runtime_config,
        gather=lambda symbol, **kw: app.state.aggregator.gather(symbol, **kw),
        fetch=app.state.content_fetcher.fetch if settings.research.enabled else None,
        search=search,
        persist=lambda run_id, docs: app.state.run_store.save_evidence(run_id, docs),
        load=lambda run_id: app.state.run_store.load_evidence(run_id),
    )
    return build_graph(
        registry.for_agent,
        checkpointer=getattr(app.state, "graph_checkpointer", None),
        strategy_planner_mode=planner_mode,
        enable_strategy_planner=planner_enabled,
        planner_max_steps=settings.runtime.strategy_planner_max_steps,
        planner_tool_registry=getattr(app.state, "tool_registry", None),
        enable_strategy_planner_llm_actions=planner_llm_actions,
        enforce_strategy_planner_decision=enforce_planner_decision,
        runtime_config_for=research_runtime_config,
        fallbacks_for=registry.fallbacks_for_agent,
        preflight_refetch_enabled=settings.runtime.preflight_refetch_enabled,
        preflight_tool_registry=getattr(app.state, "tool_registry", None),
        preflight_tool_context=(
            _preflight_tool_context if settings.runtime.preflight_refetch_enabled else None
        ),
        confidence_bucket_caps=settings.confidence.bucket_caps,
        confidence_policy_version=settings.confidence.policy_version,
        research_config=settings.research,
        research_fetch=(app.state.content_fetcher.fetch if settings.research.enabled else None),
        research_search=search,
        research_persist=lambda run_id, docs: app.state.run_store.save_evidence(run_id, docs),
        research_load=lambda run_id: app.state.run_store.load_evidence(run_id),
        topic_graph=topic_graph,
    )


def _research_budget(
    app: FastAPI, settings: Settings, run_id: str, *, resume: bool = False
) -> RunBudget | None:
    """为研究建立有限预算,恢复时沿用已记录的用量而不重置额度。

    研究未启用时返回 None;缺少明确上限时拒绝执行。预算变化写入该 run 的事件账本,
    调用方必须在发起研究请求前完成本步骤,存储失败继续向上传播。
    """
    if not settings.research.enabled:
        return None
    cfg = settings.research
    if cfg.max_requests is None or cfg.deadline_s is None:
        raise HTTPException(
            status_code=409,
            detail="research requires explicit max_requests and deadline_s after provider probing",
        )
    if not hasattr(app.state, "research_endpoint_pool"):
        app.state.research_endpoint_pool = EndpointPool(
            per_endpoint_concurrency={e.name: e.max_concurrency for e in settings.llm.endpoints}
        )
    store = app.state.run_store

    def callback(snapshot: dict[str, Any]) -> None:
        store.append_event(run_id, "research_budget", snapshot)

    previous = (store.get(run_id) or {}).get("research_budget") if resume else None
    if previous:
        restored = RunBudget.from_snapshot(
            previous, endpoint_pool=app.state.research_endpoint_pool, on_change=callback
        )
        callback(restored.snapshot())
        return restored
    return RunBudget(
        max_requests=cfg.max_requests,
        deadline_s=cfg.deadline_s,
        endpoint_pool=app.state.research_endpoint_pool,
        on_change=callback,
    )


def _append_checkpoint_event(
    run_store: FileRunStore,
    run_id: str,
    checkpointer: Any,
    status: str,
) -> None:
    if checkpointer is None:
        return
    run_store.append_event(
        run_id,
        "checkpoint",
        {
            "thread_id": run_id,
            "backend": checkpoint_backend_name(checkpointer),
            "status": status,
        },
    )


def _enrich_final_payload(
    run_store: FileRunStore, run_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    record = run_store.get(run_id)
    if record is None:
        return payload
    enriched = dict(payload)
    for key in (
        "run_health",
        "evidence_summary",
        "evidence_factors",
        "news_research",
        "research_budget",
    ):
        enriched[key] = record.get(key) or ([] if key == "evidence_factors" else {})
    return enriched


def _sse_event(event: str, payload: Any) -> dict[str, str]:
    """SSE 事件封装:JSON 序列化 + 落盘前同款脱敏。"""
    return {
        "event": event,
        "data": redact_text(json.dumps(payload, ensure_ascii=False, default=str)),
    }


async def _stream_graph_events(
    *,
    graph: Any,
    state: dict[str, Any] | None,
    run_id: str,
    run_store: FileRunStore,
    checkpointer: Any,
    budget: RunBudget | None = None,
) -> AsyncGenerator[dict[str, str], None]:
    """消费 graph 更新流 → node/interrupt/final SSE 事件(analyze 与 resume 共用)。

    state=None 表示从 checkpoint 续跑;先保存事件再发送,以便断线后读取已有记录。
    关闭或取消消费时一并关闭上游流;异常交给调用方处理,两端点的失败语义不同。
    """
    final_advice: Any = None
    with run_budget_scope(budget):
        async with aclosing(_graph_astream(graph, state, _graph_config(run_id))) as stream_5:
            async for update in stream_5:
                for node, node_output in update.items():
                    payload = jsonable(node_output)
                    if node == "__research_progress__":
                        run_store.append_event(run_id, "research_progress", payload)
                        yield _sse_event("research_progress", payload)
                        continue
                    if isinstance(payload, dict) and payload.get("evidence_documents"):
                        from ..intel.models import EvidenceDocument

                        run_store.save_evidence(
                            run_id,
                            [
                                EvidenceDocument.model_validate(d)
                                for d in payload["evidence_documents"]
                            ],
                        )
                        payload.pop("evidence_documents", None)
                    if node == "__interrupt__":
                        _append_checkpoint_event(run_store, run_id, checkpointer, "resumable")
                        run_store.fail(run_id, "checkpoint interrupted")
                        yield _sse_event("interrupt", {"run_id": run_id, "payload": payload})
                        return
                    run_store.append_event(run_id, "node", {"node": node, "update": payload})
                    yield _sse_event("node", {"node": node, "update": payload})
                    if isinstance(payload, dict) and payload.get("final_advice"):
                        final_advice = payload.get("final_advice")
    final_payload = {"advice": final_advice, "disclaimer": DISCLAIMER, "run_id": run_id}
    run_store.append_event(run_id, "final", final_payload)
    _append_checkpoint_event(run_store, run_id, checkpointer, "completed")
    run_store.complete(run_id, final_advice)
    yield _sse_event("final", _enrich_final_payload(run_store, run_id, final_payload))


async def _invoke_tool(
    *,
    registry: ToolRegistry,
    run_store: FileRunStore,
    run_id: str,
    name: str,
    payload: dict[str, Any],
    context: dict[str, Any],
) -> ToolInvocationResult:
    result = await registry.invoke(name, payload, context=context)
    run_store.append_event(run_id, "tool", result.model_dump(mode="json"))
    return result


async def _graph_astream(
    graph: Any,
    state: dict[str, Any] | None,
    config: dict[str, Any],
) -> AsyncGenerator[dict[str, Any], None]:
    """适配正式图与离线替身的流接口,将研究进度合并到统一事件流。

    有 checkpoint 的正式图同步提交状态;退出任一分支时关闭上游迭代器,
    避免 SSE 消费结束后图仍继续运行。兼容分支只适配签名,不重试执行。
    """
    try:
        params = inspect.signature(graph.astream).parameters
    except (TypeError, ValueError):
        accepts_config = True
    else:
        accepts_config = "config" in params or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )
    if accepts_config and hasattr(graph, "get_graph"):
        async with aclosing(
            graph.astream(
                state,
                config=config,
                stream_mode=["updates", "custom"],
                subgraphs=True,
                **({"durability": "sync"} if getattr(graph, "checkpointer", None) else {}),
            )
        ) as stream_4:
            async for namespace, mode, content in stream_4:
                if mode == "custom":
                    yield {"__research_progress__": content}
                elif not namespace:
                    yield content
    elif accepts_config:
        async with aclosing(graph.astream(state, config=config, stream_mode="updates")) as stream_3:
            async for update in stream_3:
                yield update
    else:
        async with aclosing(graph.astream(state, stream_mode="updates")) as stream_2:
            async for update in stream_2:
                yield update


async def _prepare_session_state(
    app: FastAPI,
    settings: Settings,
    run_id: str,
    symbol: str | None,
    market: str | None,
    question: str,
) -> dict[str, Any]:
    """Freeze one turn's inputs independently of event subscribers."""
    store = app.state.run_store
    previous = store.get(run_id) or {}
    registry = app.state.tool_registry
    client = app.state.spot_client if market == "api" else app.state.rest_client
    market_payload = previous.get("market_snapshot")
    topic = previous.get("mode") == "topic"
    if not market_payload and symbol is not None:
        result = await _invoke_tool(
            registry=registry,
            run_store=store,
            run_id=run_id,
            name="market_snapshot",
            payload={"symbol": symbol, "market": market},
            context={
                "rest_client": client,
                "build_market_snapshot": getattr(
                    app.state, "market_snapshot_builder", build_market_snapshot
                ),
            },
        )
        market_payload = (result.output or {}).get("market_snapshot") or {}
        store.append_event(run_id, "market", market_payload)
    intel_payload = previous.get("intel_snapshot")
    if not intel_payload and not topic:
        result = await _invoke_tool(
            registry=registry,
            run_store=store,
            run_id=run_id,
            name="intel_snapshot",
            payload={"symbol": symbol, "research_question": question},
            context={"aggregator": app.state.aggregator},
        )
        intel_payload = (result.output or {}).get("intel_snapshot") or {}
        store.append_event(run_id, "intel", intel_payload)
    latest_price = None
    with contextlib.suppress(httpx.HTTPError, ValueError, TypeError, KeyError):
        if symbol is not None:
            latest_price = float((await client.ticker_24h(symbol)).get("lastPrice"))
    return {
        "mode": previous.get("mode"),
        "target": previous.get("target"),
        "title": previous.get("title"),
        "research_demo": settings.profile == "demo",
        "research_run_id": run_id,
        "research_question": question,
        "research_budget": store.budget(run_id) or {},
        "analysis_at": datetime.now(timezone.utc).isoformat(),
        **({"symbol": symbol, "market": market} if symbol is not None else {}),
        "market_snapshot": market_payload or {},
        "intel_snapshot": intel_payload or {},
        "snapshot_freshness_enabled": settings.profile != "demo",
        "snapshot_freshness_now": datetime.now(timezone.utc).isoformat(),
        "latest_price": latest_price,
        "analyst_reports": [],
        "debate_turns": [],
        "runtime_events": [],
        "tool_events": [],
        "planner_steps": [],
        "planner_decision": None,
        "debate_round": 0,
        "max_debate_rounds": settings.debate.max_rounds,
        "debate_stop_reason": None,
        "research_summary": None,
        "risk_assessment": None,
        "risk_gate_result": None,
        "final_advice": None,
        "context_compactions": {},
    }


def _initialize_collaboration(app: FastAPI, settings: Settings) -> RunManager:
    """Also available to offline integration fixtures with injected dependencies."""
    if settings.runtime.checkpoint_backend != "sqlite":
        raise ValueError("collaboration requires runtime.checkpoint_backend='sqlite'")
    path = resolve_data_path(settings.runtime.checkpoint_path, profile=settings.profile)
    store = CollaborationStore(path, legacy=app.state.run_store)
    app.state.run_store = store
    engine = AgentEngine(
        store,
        app.state.graph_checkpointer,
        idle_s=settings.collaboration.idle_unload_s,
        max_resident=settings.collaboration.max_resident_agents,
        max_workers=settings.research.max_workers,
    )
    manager = RunManager(
        app,
        settings,
        store,
        engine,
        prepare=lambda run_id, symbol, market, question: _prepare_session_state(
            app, settings, run_id, symbol, market, question
        ),
        stream_graph=_graph_astream,
        fingerprint=lambda: _research_config_fingerprint(settings),
    )
    app.state.research_manager = manager
    if hasattr(engine.checkpointer, "pin_thread") and store.legacy:
        for record in store.legacy.list_all():
            if record.get("resumable"):
                engine.checkpointer.pin_thread(record["run_id"])
    manager.start()
    return manager


def _endpoint_origin(url: str) -> tuple[str, str, int]:
    """取得凭证绑定的 HTTP origin;路径无关,省略端口与协议默认端口等价。

    不解析 DNS 或发起请求。缺少有效 HTTP(S) origin、主机名或端口非法时抛 ValueError,
    调用方必须拒绝沿用已有密钥;显式端口 0 不视为省略端口。
    """
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname
    if scheme not in {"http", "https"} or hostname is None:
        raise ValueError("endpoint URL must contain an HTTP(S) origin")
    port = parsed.port
    return (
        scheme,
        hostname.lower(),
        port if port is not None else (443 if scheme == "https" else 80),
    )


class SwitchEndpointRequest(BaseModel):
    endpoint: str


class UpsertEndpointRequest(BaseModel):
    name: str
    base_url: str
    model: str
    models: list[str] | None = None
    api_key: str | None = None  # 仅经 body;加密落盘,绝不回传/记日志
    structured_output: bool | None = None
    fallback_verified: bool | None = None
    max_concurrency: int | None = Field(None, ge=1, le=16)
    use_for_all_roles: bool = False


class FetchModelsRequest(BaseModel):
    base_url: str
    api_key: str | None = None
    endpoint: str | None = None  # 提供则用已存 key(api_key 留空时)


class SetAgentModelRequest(BaseModel):
    agent: str
    endpoint: str
    model: str


class SetIntelSourceRequest(BaseModel):
    source: str
    enabled: bool


class SetIntelKeyRequest(BaseModel):
    source: str
    api_key: str  # 仅经 body;加密落盘,绝不回传/记日志


class SetParamsRequest(BaseModel):
    context: dict[str, int] | None = None
    time_window_hours: int | None = Field(None, ge=1, le=168)
    max_debate_rounds: int | None = Field(None, ge=0, le=20)
    per_source_timeout: float | None = Field(None, gt=0, le=60)
    max_news_items: int | None = Field(None, ge=0, le=200)
    strategy_planner_mode: StrategyPlannerMode | None = None
    strategy_planner_enabled: bool | None = None
    strategy_planner_max_steps: int | None = Field(None, ge=1, le=20)
    strategy_planner_llm_actions: bool | None = None
    preflight_refetch_enabled: bool | None = None
    # v0.2 读取兼容;新客户端应使用 preflight_refetch_enabled。
    strategy_planner_refetch_enabled: bool | None = None


class DesktopRequest(BaseModel):
    on_top: bool = False
    hotkey_enabled: bool = False


class ResearchLimitsRequest(BaseModel):
    max_requests: int = Field(100, ge=1, le=500)
    deadline_s: float = Field(600, gt=0, le=3600)


class SetNetworkRequest(BaseModel):
    # 空串/纯空白 = 清除显式代理,回退环境变量(trust_env)。
    proxy: str = ""


class ReplayRunsRequest(BaseModel):
    run_ids: list[str] | None = Field(None, max_length=100)
    limit: int = Field(20, ge=1, le=100)
    max_debate_rounds: int | None = Field(None, ge=1, le=20)
    completed_only: bool = True


class CalibrationApplyRequest(BaseModel):
    proposal_id: str = Field(min_length=8, max_length=128)
    fingerprint: str = Field(min_length=16, max_length=128)
    bucket_caps: dict[str, float]

    @field_validator("bucket_caps")
    @classmethod
    def _valid_bucket_caps(cls, value: dict[str, float]) -> dict[str, float]:
        return ConfidenceConfig(bucket_caps=value).bucket_caps


def _own_legacy_run(app: FastAPI, reservation: Any, store: FileRunStore, run_id: str) -> None:
    """把旧式 SSE run 的收尾交给请求生命周期,不接管协作后台任务。

    请求提前结束且记录仍为 running 时标记失败;槽位由 reservation 单独释放,
    因而记录收尾失败也不会阻止并发额度回收。
    """

    def finish() -> None:
        app.state.active_research_runs.discard(run_id)
        record = store.get(run_id)
        if record and record.get("status") == "running":
            store.fail(run_id, "analysis response ended before completion")

    reservation.on_close = finish


def create_app(settings: Settings) -> FastAPI:
    """组装本机 UI/API 与请求边界;依赖的创建和回收留给 lifespan。

    settings 是运行中的共享配置,配置写接口通过统一准入锁切换它与依赖,
    调用者仍须按 AppConfig 的 loopback 约束绑定监听地址。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """启动运行依赖;关闭时先停止研究,再回收不再被任务持有的资源。

        尚未结束的任务保留资源所有权,其完成回调负责延后关闭,避免关闭客户端后仍有任务使用。
        """
        stage_resource, stage_path = (
            "应用数据目录",
            resolve_data_path(".runs", profile=settings.profile),
        )
        ready = False
        try:
            app.state.backend_loop = asyncio.get_running_loop()
            # demo 不读取正常 profile 的运行时覆盖;demo-live 则先加载用户端点,再强制其安全能力开关。
            if settings.profile != "demo":
                stage_resource, stage_path = "运行时配置", runtime_config_path()
                try:
                    apply_overrides(settings, load_overrides())
                except ValueError as exc:
                    errors = getattr(exc, "errors", None)
                    fields = (
                        ", ".join(
                            ".".join(map(str, item["loc"]))
                            for item in errors(include_input=False, include_url=False)
                        )
                        if callable(errors)
                        else str(exc)
                        if isinstance(exc, RuntimeConfigError)
                        else "JSON/字段"
                    )
                    raise StartupError(
                        "运行时配置",
                        stage_path,
                        fields,
                        "保留原文件,修复所列字段后重新启动;未启用部分覆盖。",
                    ) from exc
            apply_profile(settings, settings.profile)

            _proxy: str | None = settings.network.proxy or None
            if settings.profile == "demo":
                demo_market = DemoMarketData()
                rest_client: Any = DemoRestClient("fapi")
                spot_client: Any = DemoRestClient("api")
                intel_http: Any = DemoHttpClient()
                aggregator: Any = DemoIntelAggregator()
                registry = build_demo_registry(settings.llm)
                app.state.demo_market_data = demo_market
                app.state.market_snapshot_builder = demo_market.build_snapshot
                if settings.research.enabled:
                    app.state.content_fetcher = DemoResearchContent()
                ws_manager: Any = DemoWsManager("fapi")
                spot_ws_manager: Any = DemoWsManager("api")
            else:
                if _proxy:
                    logger.info("出站代理已配置(地址已脱敏)")
                rest_client = BinanceRestClient("fapi", timeout=10.0, proxy=_proxy)
                app.state.rest_client = rest_client
                spot_client = BinanceRestClient("api", timeout=10.0, proxy=_proxy)
                app.state.spot_client = spot_client
                _intel_http_kwargs: dict[str, Any] = {
                    "timeout": settings.intel.per_source_timeout,
                    "headers": {"User-Agent": "binance-agent/0.2 (+local analysis)"},
                    "follow_redirects": True,
                }
                if _proxy:
                    _intel_http_kwargs["proxy"] = _proxy
                intel_http = BoundedAsyncClient(**_intel_http_kwargs)
                app.state.intel_http = intel_http
                aggregator = _build_aggregator(settings, intel_http, rest_client)
                registry = LlmRegistry(settings.llm, proxy=_proxy)
                app.state.llm_registry = registry
                app.state.market_snapshot_builder = build_market_snapshot
                ws_manager = BinanceWsManager("fapi", proxy=_proxy)
                spot_ws_manager = BinanceWsManager("api", proxy=_proxy)
            app.state.rest_client = rest_client
            app.state.spot_client = spot_client
            app.state.intel_http = intel_http
            app.state.aggregator = aggregator
            app.state.llm_registry = registry
            if not hasattr(app.state, "tool_registry"):
                app.state.tool_registry = build_default_tool_registry()
            stage_resource, stage_path = (
                "checkpoint SQLite",
                resolve_data_path(settings.runtime.checkpoint_path, profile=settings.profile),
            )
            if not hasattr(app.state, "graph_checkpointer"):
                app.state.graph_checkpointer = build_checkpointer(
                    settings.runtime.checkpoint_backend,
                    path=resolve_data_path(
                        settings.runtime.checkpoint_path, profile=settings.profile
                    ),
                    retention=settings.runtime.checkpoint_retention,
                )
            app.state.graph = _build_runtime_graph(settings, registry, app)
            stage_resource, stage_path = (
                "运行历史目录",
                resolve_data_path(".runs/runs.jsonl", profile=settings.profile),
            )
            if not hasattr(app.state, "run_store"):
                app.state.run_store = FileRunStore(
                    resolve_data_path(".runs/runs.jsonl", profile=settings.profile)
                )
            stage_resource, stage_path = (
                "协作 SQLite",
                resolve_data_path(settings.runtime.checkpoint_path, profile=settings.profile),
            )
            if settings.collaboration.enabled:
                _initialize_collaboration(app, settings)
            stage_resource, stage_path = (
                "告警目录",
                resolve_data_path(".runs/alerts.jsonl", profile=settings.profile),
            )
            if not hasattr(app.state, "alert_store"):
                app.state.alert_store = FileAlertStore(
                    resolve_data_path(".runs/alerts.jsonl", profile=settings.profile)
                )
            # 效果闭环:仅在开关开启时挂 outcome_store(关闭则不建文件、/outcomes 端点 404,现有行为零变化)
            stage_resource, stage_path = (
                "效果历史目录",
                resolve_data_path(".runs/outcomes.jsonl", profile=settings.profile),
            )
            if settings.runtime.enable_outcome_tracking and not hasattr(app.state, "outcome_store"):
                if settings.profile == "demo":
                    app.state.outcome_store = DemoOutcomeStore()
                else:
                    app.state.outcome_store = FileOutcomeStore(
                        resolve_data_path(".runs/outcomes.jsonl", profile=settings.profile)
                    )
            app.state.ws_manager = ws_manager
            app.state.spot_ws_manager = spot_ws_manager
            ready = True
            yield
        except Exception as exc:
            if not ready:
                diagnostic = (
                    exc
                    if isinstance(exc, StartupError)
                    else startup_error(
                        stage_resource, stage_path, exc, sqlite="SQLite" in stage_resource
                    )
                )
                app.state.startup_error = diagnostic
                raise diagnostic from exc
            raise
        finally:
            manager = getattr(app.state, "research_manager", None)
            if manager is not None:
                try:
                    await manager.shutdown()
                except Exception:
                    logger.exception("协作任务关闭失败")
            pending: set[asyncio.Task[None]] = (
                getattr(manager, "pending_shutdown", set()) if manager is not None else set()
            )

            # Tasks that ignored cancellation still own their resources. Keep them
            # alive until the tasks exit; never report pending work as completed.
            async def close_resources() -> None:
                seen: set[int] = set()
                for name in (
                    "run_store",
                    "rest_client",
                    "spot_client",
                    "intel_http",
                    "llm_registry",
                    "graph_checkpointer",
                ):
                    resource = getattr(app.state, name, None)
                    if resource is None or id(resource) in seen:
                        continue
                    seen.add(id(resource))
                    close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
                    if not callable(close):
                        continue
                    try:
                        result = close()
                        if inspect.isawaitable(result):
                            await result
                    except Exception:
                        logger.exception("资源关闭失败: %s", name)

            if pending:
                closing: asyncio.Task[None] | None = None

                def owners_done(task: asyncio.Task[Any] | None = None) -> None:
                    nonlocal closing
                    if closing is None and all(owner.done() for owner in pending):
                        closing = asyncio.create_task(close_resources())
                        app.state.pending_resource_close = closing

                for owner in pending:
                    owner.add_done_callback(owners_done)

                async def finish_pending() -> None:
                    try:
                        await asyncio.wait(pending)
                    finally:
                        owners_done()
                        if closing is not None:
                            await asyncio.shield(closing)
                        else:
                            logger.error("关闭时仍有协作任务持有资源;结束后回收")

                app.state.pending_resource_cleanup = asyncio.create_task(finish_pending())
            else:
                await close_resources()

    app = FastAPI(
        title="Binance Agent",
        version=__version__,
        summary="本地多智能体金融策略分析助手(仅分析,不下单)",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(LocalOriginGuardMiddleware)
    app.add_middleware(LoopbackHostMiddleware)
    app.state.settings = settings
    app.state.analysis_limiter = AnalysisLimiter(settings.app.max_concurrent_runs)
    app.state.active_research_runs = set()
    register_session_routes(app)
    from .reports import register_report_routes

    register_report_routes(app)

    @app.middleware("http")
    async def protect_active_config(request: Any, call_next: Any) -> Any:
        """配置写请求与分析准入互斥;繁忙时返回 409,不执行配置处理器。"""
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.url.path.startswith(
            "/config"
        ):
            from fastapi.responses import JSONResponse

            try:
                async with app.state.analysis_limiter.reconfigure():
                    return await call_next(request)
            except ConfigurationBusy as exc:
                return JSONResponse({"detail": str(exc)}, status_code=409)
        return await call_next(request)

    def _select_client(market: str) -> BinanceRestClient:
        if market == "fapi":
            client: BinanceRestClient = app.state.rest_client
            return client
        if market == "api":
            spot: BinanceRestClient = app.state.spot_client
            return spot
        raise HTTPException(status_code=422, detail="未知市场,支持 fapi(合约) / api(现货)")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
            "profile": settings.profile,
            "market": settings.market.default_market,
            "disclaimer": DISCLAIMER,
        }

    @app.get("/health/deps")
    async def health_deps(deep: bool = Query(False)) -> dict[str, Any]:
        aggregator: IntelAggregator = app.state.aggregator
        llm_endpoints = []
        for endpoint in settings.llm.endpoints:
            has_key = resolve_secret(endpoint.api_key_ref) is not None
            llm_endpoints.append(
                _dep_item(
                    name=endpoint.name,
                    model=endpoint.model,
                    status="ok" if has_key else "warn",
                    severity="info" if has_key else "warning",
                    reason="" if has_key else f"缺少 key({endpoint.api_key_ref})",
                    action_hint="在高级配置中保存 API key。",
                )
            )

        disabled = getattr(aggregator, "_disabled", {})
        intel_sources = [
            _dep_item(
                source=source,
                status="warn",
                severity="warning",
                reason=reason,
                action_hint="在快速设置中配置数据源或安装可选依赖。",
            )
            for source, reason in disabled.items()
        ]
        llm_status = "ok" if all(item["status"] == "ok" for item in llm_endpoints) else "warn"
        intel_status = "ok" if not intel_sources else "warn"
        status = "ok" if llm_status == "ok" and intel_status == "ok" else "warn"
        return {
            "status": status,
            "deep": deep,
            "binance": _dep_item(
                status="ok",
                severity="info",
                reason="公共行情端点按需检查",
                action_hint="如分析失败,检查网络、代理或 Binance 可达性。",
            ),
            "llm": {"status": llm_status, "endpoints": llm_endpoints},
            "intel": {"status": intel_status, "sources": intel_sources},
        }

    @app.get("/symbols")
    async def symbols(market: Literal["api", "fapi"] = "fapi") -> dict[str, Any]:
        if settings.profile == "demo":
            return {
                "market": market,
                "synthetic": True,
                "symbols": [
                    {
                        "symbol": x,
                        "base_asset": x.removesuffix("USDT"),
                        "quote_asset": "USDT",
                        "name": n,
                        "market": market,
                    }
                    for x, n in (
                        ("BTCUSDT", "Bitcoin / 比特币"),
                        ("ETHUSDT", "Ethereum / 以太坊"),
                        ("SOLUSDT", "Solana"),
                    )
                ],
            }
        client = app.state.spot_client if market == "api" else app.state.rest_client
        try:
            return {"market": market, "symbols": await client.symbol_catalog()}
        except (httpx.HTTPError, ValueError, KeyError, TimeoutError) as exc:
            raise HTTPException(
                503, "暂时无法读取交易对目录,请检查网络后重试;独立专题仍可使用"
            ) from exc

    @app.get("/snapshot")
    async def snapshot(
        symbol: str = Query(
            ...,
            min_length=2,
            max_length=40,
            pattern="^[A-Za-z0-9]+$",
            description="交易对,如 BTCUSDT",
        ),
        market: str = Query("fapi", description="市场: fapi(合约) / api(现货)"),
    ) -> dict[str, object]:
        client = _select_client(market)
        builder = getattr(app.state, "market_snapshot_builder", build_market_snapshot)
        snap = await builder(client, symbol)
        return snap.model_dump(mode="json")

    @app.get("/intel")
    async def intel(
        symbol: str = Query(
            ...,
            min_length=2,
            max_length=40,
            pattern="^[A-Za-z0-9]+$",
            description="交易对,如 BTCUSDT",
        ),
    ) -> dict[str, object]:
        aggregator: IntelAggregator = app.state.aggregator
        snap = await aggregator.gather(symbol)
        return snap.model_dump(mode="json")

    @app.get("/runs/{run_id}/evidence/{evidence_id}")
    async def get_evidence(run_id: str, evidence_id: str) -> dict[str, Any]:
        if app.state.run_store.get(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        for doc in app.state.run_store.load_evidence(run_id):
            if doc.evidence_id == evidence_id:
                return cast(dict[str, Any], doc.model_dump(mode="json"))
        raise HTTPException(status_code=404, detail="evidence not found")

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        store: FileRunStore = app.state.run_store
        record = await asyncio.to_thread(store.get, run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return record

    @app.get("/runs")
    async def list_runs(
        limit: int = Query(20, ge=1, le=100),
        detail: str = Query("summary", pattern="^(summary|full)$"),
    ) -> list[dict[str, Any]]:
        store: FileRunStore = app.state.run_store
        if detail == "full":
            return await asyncio.to_thread(store.list_recent, limit=limit)
        return await asyncio.to_thread(store.list_recent_summaries, limit=limit)

    @app.get("/runs/{run_id}/resume")
    async def resume_run(run_id: str) -> EventSourceResponse:
        """恢复协作后台轮次或旧 checkpoint,并返回对应 SSE 流。

        旧记录先核对 checkpoint、研究模式和配置指纹,再占用槽位与记录恢复事件。
        取消或异常保留可恢复状态;请求生命周期负责关闭流并释放旧式运行的槽位。
        """
        manager = getattr(app.state, "research_manager", None)
        if manager is not None:
            try:
                run = manager.store.run(run_id)
            except KeyError:
                pass  # Legacy checkpoint runs retain their existing recovery path.
            else:
                with manager.store._lock:
                    cursor = manager.store._conn.execute(
                        "SELECT COALESCE(max(seq),0) FROM research_events WHERE session_id=?",
                        (run["session_id"],),
                    ).fetchone()[0]
                try:
                    await manager.control(run_id, "resume", uuid.uuid4().hex)
                except SessionConflict as exc:
                    raise HTTPException(409, str(exc)) from exc
                return EventSourceResponse(
                    manager.subscribe(run["session_id"], after=cursor, until_run=run_id)
                )
        graph = app.state.graph
        run_store: FileRunStore = app.state.run_store
        record = run_store.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        checkpointer = getattr(app.state, "graph_checkpointer", None)
        if checkpointer is not None and not has_checkpoint_thread(checkpointer, run_id):
            _append_checkpoint_event(run_store, run_id, checkpointer, "missing")
            raise HTTPException(status_code=409, detail=f"checkpoint not found for run {run_id}")

        original_mode = bool((record.get("research_input") or {}).get("evidence_mode", False))
        if original_mode != settings.research.enabled:
            raise HTTPException(
                status_code=409, detail="resume requires the original research mode"
            )
        if settings.research.enabled:
            saved_fingerprint = str(
                (record.get("research_input") or {}).get("config_fingerprint") or ""
            )
            current_fingerprint = _research_config_fingerprint(
                settings, include_collaboration=False
            )
            if not saved_fingerprint:
                raise HTTPException(
                    status_code=409,
                    detail="research checkpoint has no config fingerprint; start a new run",
                )
            if saved_fingerprint != current_fingerprint:
                raise HTTPException(
                    status_code=409,
                    detail="research configuration changed; start a new run",
                )
        if run_id in app.state.active_research_runs:
            raise HTTPException(status_code=409, detail="run already active")
        app.state.active_research_runs.add(run_id)
        reservation = await app.state.analysis_limiter.try_acquire()
        if reservation is None:
            app.state.active_research_runs.discard(run_id)
            raise HTTPException(status_code=429, detail="too many concurrent analyses")

        _own_legacy_run(app, reservation, run_store, run_id)
        resume_payload = {
            "run_id": run_id,
            "from_status": record.get("status"),
            "source": "checkpoint",
        }
        try:
            run_store.append_event(run_id, "resume", resume_payload)
            _append_checkpoint_event(run_store, run_id, checkpointer, "active")
        except BaseException as exc:
            app.state.active_research_runs.discard(run_id)
            await reservation.release()
            if isinstance(exc, OSError):
                raise HTTPException(
                    503, "analysis storage unavailable; resume not started"
                ) from exc
            raise

        async def event_stream() -> AsyncIterator[dict[str, str]]:
            try:
                yield _sse_event("resume", resume_payload)
                async with aclosing(
                    _stream_graph_events(
                        graph=graph,
                        state=None,
                        budget=_research_budget(app, settings, run_id, resume=True),
                        run_id=run_id,
                        run_store=run_store,
                        checkpointer=checkpointer,
                    )
                ) as stream_1:
                    async for item in stream_1:
                        yield item
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    _append_checkpoint_event(run_store, run_id, checkpointer, "resumable")
                with contextlib.suppress(Exception):
                    run_store.fail(run_id, "cancelled by client")
                raise
            except Exception as exc:
                logger.exception("恢复流程异常: %s", exc)
                with contextlib.suppress(Exception):
                    _append_checkpoint_event(run_store, run_id, checkpointer, "resumable")
                with contextlib.suppress(Exception):
                    run_store.fail(run_id, str(exc))
                yield _sse_event("failed", {"message": str(exc), "run_id": run_id})

            finally:
                app.state.active_research_runs.discard(run_id)

        stream = ContextIterator(
            event_stream(), DEMO_REQUEST_ID if settings.profile == "demo" else None, run_id
        )
        reservation.close_stream = stream.aclose
        return EventSourceResponse(stream)

    @app.get("/analyze")
    async def analyze(
        symbol: str = Query(
            ...,
            min_length=2,
            max_length=40,
            pattern="^[A-Za-z0-9]+$",
            description="交易对,如 BTCUSDT",
        ),
        market: str = Query("fapi", description="市场: fapi(合约) / api(现货)"),
        research_question: str = Query("", max_length=2000),
    ) -> EventSourceResponse:
        """接受本机研究请求,在准入成功后创建运行并输出 SSE 进度。

        协作模式将执行权交给后台管理器,断开订阅不终止研究;旧式模式由请求持有运行,
        断开或取消时记录失败并回收流和并发槽位。
        """
        manager = getattr(app.state, "research_manager", None)
        if manager is not None:
            if market not in {"api", "fapi"}:
                raise HTTPException(422, "unknown market")
            try:
                session, turn = await manager.new_session_turn(symbol, market, research_question)
            except SessionConflict as exc:
                raise HTTPException(409, str(exc)) from exc
            return EventSourceResponse(
                manager.subscribe(session["session_id"], until_run=turn["run_id"])
            )
        if research_question and not settings.research.enabled:
            raise HTTPException(
                status_code=409,
                detail="enable evidence research before supplying research_question",
            )
        if settings.research.enabled and (
            settings.research.max_requests is None or settings.research.deadline_s is None
        ):
            raise HTTPException(
                status_code=409,
                detail="set research.max_requests and research.deadline_s before live use",
            )
        reservation = await app.state.analysis_limiter.try_acquire()
        if reservation is None:
            raise HTTPException(status_code=429, detail="too many concurrent analyses")
        try:
            client = _select_client(market)
        except BaseException:
            await reservation.release()
            raise
        symbol = symbol.upper()
        aggregator: IntelAggregator = app.state.aggregator
        graph = app.state.graph
        run_store: FileRunStore = app.state.run_store
        tool_registry: ToolRegistry = getattr(
            app.state, "tool_registry", build_default_tool_registry()
        )

        try:
            run_id = run_store.create(symbol=symbol, market=market)
            _own_legacy_run(app, reservation, run_store, run_id)
            run_store.append_event(
                run_id,
                "research_input",
                {
                    "research_question": research_question,
                    "evidence_mode": settings.research.enabled,
                    "config_fingerprint": (
                        _research_config_fingerprint(settings)
                        if settings.research.enabled
                        else None
                    ),
                },
            )
            app.state.active_research_runs.add(run_id)
            budget = _research_budget(app, settings, run_id)
            checkpointer = getattr(app.state, "graph_checkpointer", None)
        except BaseException as exc:
            await reservation.release()
            if isinstance(exc, OSError):
                raise HTTPException(
                    503, "analysis storage unavailable; no analysis started"
                ) from exc
            raise

        async def event_stream() -> AsyncIterator[dict[str, str]]:
            try:
                yield _sse_event("run", {"run_id": run_id, "symbol": symbol, "market": market})
                _append_checkpoint_event(run_store, run_id, checkpointer, "active")
                market_tool = await _invoke_tool(
                    registry=tool_registry,
                    run_store=run_store,
                    run_id=run_id,
                    name="market_snapshot",
                    payload={"symbol": symbol, "market": market},
                    context={
                        "rest_client": client,
                        "build_market_snapshot": getattr(
                            app.state, "market_snapshot_builder", build_market_snapshot
                        ),
                    },
                )
                market_payload = (market_tool.output or {}).get("market_snapshot") or {}
                run_store.append_event(run_id, "market", market_payload)
                yield _sse_event("market", market_payload)

                latest_price = None
                try:
                    fresh_ticker = await client.ticker_24h(symbol)
                    latest_price = float(fresh_ticker.get("lastPrice"))
                except Exception as exc:
                    logger.info("fresh ticker probe failed symbol=%s: %s", symbol, exc)

                intel_tool = await _invoke_tool(
                    registry=tool_registry,
                    run_store=run_store,
                    run_id=run_id,
                    name="intel_snapshot",
                    payload={"symbol": symbol, "research_question": research_question},
                    context={"aggregator": aggregator},
                )
                intel_payload = (intel_tool.output or {}).get("intel_snapshot") or {}
                run_store.append_event(run_id, "intel", intel_payload)
                yield _sse_event("intel", intel_payload)

                state: dict[str, Any] = {
                    "research_demo": settings.profile == "demo",
                    "research_run_id": run_id,
                    "research_question": research_question,
                    "research_budget": budget.snapshot() if budget is not None else {},
                    "analysis_at": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "market": market,
                    "market_snapshot": market_payload,
                    "snapshot_freshness_enabled": settings.profile != "demo",
                    "snapshot_freshness_now": datetime.now(timezone.utc),
                    "latest_price": latest_price,
                    "intel_snapshot": intel_payload,
                    "analyst_reports": [],
                    "debate_turns": [],
                    "runtime_events": [],
                    "tool_events": [],
                    "planner_steps": [],
                    "planner_decision": None,
                    "debate_round": 0,
                    "max_debate_rounds": settings.debate.max_rounds,
                    "debate_stop_reason": None,
                    "research_summary": None,
                    "risk_assessment": None,
                    "risk_gate_result": None,
                    "final_advice": None,
                }
                async with aclosing(
                    _stream_graph_events(
                        graph=graph,
                        state=state,
                        budget=budget,
                        run_id=run_id,
                        run_store=run_store,
                        checkpointer=checkpointer,
                    )
                ) as stream_0:
                    async for item in stream_0:
                        yield item
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    _append_checkpoint_event(
                        run_store,
                        run_id,
                        checkpointer,
                        "resumable" if has_checkpoint_thread(checkpointer, run_id) else "missing",
                    )
                with contextlib.suppress(Exception):
                    run_store.fail(run_id, "cancelled by client")
                raise
            except Exception as exc:  # 流式中异常 → 以 failed 事件回推,避免连接悬挂
                logger.exception("分析流程异常: %s", exc)
                with contextlib.suppress(Exception):
                    _append_checkpoint_event(
                        run_store,
                        run_id,
                        checkpointer,
                        "resumable" if has_checkpoint_thread(checkpointer, run_id) else "missing",
                    )
                with contextlib.suppress(Exception):
                    run_store.fail(run_id, str(exc))
                yield _sse_event("failed", {"message": str(exc), "run_id": run_id})
            finally:
                app.state.active_research_runs.discard(run_id)

        stream = ContextIterator(
            event_stream(), DEMO_REQUEST_ID if settings.profile == "demo" else None, run_id
        )
        reservation.close_stream = stream.aclose
        return EventSourceResponse(stream)

    @app.post("/config/llm-endpoint/switch")
    async def switch_llm_endpoint(req: SwitchEndpointRequest) -> dict[str, object]:
        candidate, registry = _draft_configuration()
        try:
            registry.switch_active(req.endpoint)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await _commit_configuration(candidate, registry)
        return {"active": registry.active, "endpoints": registry.endpoint_names()}

    @app.get("/config/llm-endpoint")
    async def get_llm_endpoint() -> dict[str, object]:
        registry: LlmRegistry = app.state.llm_registry
        return {"active": registry.active, "endpoints": registry.endpoint_names()}

    @app.get("/config")
    async def get_config() -> dict[str, object]:
        registry: LlmRegistry = app.state.llm_registry
        aggregator: IntelAggregator = app.state.aggregator
        key_refs = {
            source: spec.key_ref
            for source, spec in SOURCE_SPECS.items()
            if spec.key_ref is not None
        }
        key_refs["tavily"] = "TAVILY_API_KEY"
        key_refs.update(settings.intel.keys)
        keyed = [
            {
                "source": source,
                "has_key": settings.profile != "demo" and resolve_secret(ref) is not None,
                "apply_url": _SOURCE_APPLY_URLS.get(source),
            }
            for source, ref in key_refs.items()
            if ref
        ]
        return {
            "profile": settings.profile,
            "version": __version__,
            "desktop": {
                "on_top": settings.desktop.on_top,
                "hotkey_enabled": settings.hotkey.enabled,
            },
            "market": {"default": settings.market.default_market},
            "llm": registry.config_view(include_secret_status=settings.profile != "demo"),
            "has_overrides": False if settings.profile == "demo" else overrides_present(),
            "confidence": settings.confidence.model_dump(mode="json"),
            "network": {"proxy": settings.network.proxy or ""},
            "research": settings.research.model_dump(mode="json"),
            "collaboration": settings.collaboration.model_dump(mode="json"),
            "intel": {
                "sources": dict(settings.intel.sources),
                "catalog": source_catalog(settings.intel.sources, settings.intel.keys),
                "disabled": dict(aggregator._disabled),
                "keyed": keyed,
            },
            "params": {
                "context": settings.runtime.context.model_dump(mode="json"),
                "time_window_hours": settings.intel.time_window_hours,
                "max_debate_rounds": settings.debate.max_rounds,
                "per_source_timeout": settings.intel.per_source_timeout,
                "max_news_items": settings.intel.max_news_items,
                "strategy_planner_mode": effective_strategy_planner_mode(settings.runtime),
                "strategy_planner_enabled": settings.runtime.strategy_planner_enabled,
                "strategy_planner_max_steps": settings.runtime.strategy_planner_max_steps,
                "strategy_planner_llm_actions": settings.runtime.strategy_planner_llm_actions,
                "preflight_refetch_enabled": settings.runtime.preflight_refetch_enabled,
                "strategy_planner_refetch_enabled": settings.runtime.preflight_refetch_enabled,
            },
        }

    def _draft_configuration() -> tuple[Settings, LlmRegistry]:
        """复制待修改配置并建立候选 registry,不替换当前设置或持久化文件。"""
        candidate = settings.model_copy(deep=True)
        old: LlmRegistry = app.state.llm_registry
        registry = LlmRegistry(
            candidate.llm, model_factory=old._factory, proxy=candidate.network.proxy
        )
        return candidate, registry

    async def _commit_configuration(
        candidate: Settings, registry: LlmRegistry, *, reset: bool = False
    ) -> None:
        """准备替换依赖、持久化配置后切换运行状态,最后回收旧依赖。

        调用方必须持有 reconfigure 准入锁,防止新任务读到切换中的状态。
        准备或持久化失败时保留原配置;异常和取消均进入候选资源清理路径。
        """
        created: list[Any] = []
        retired: list[Any] = []
        try:
            candidate = Settings.model_validate(candidate.model_dump())
            registry.reload(candidate.llm)
            registry.set_proxy(candidate.network.proxy)
            rest: Any = getattr(app.state, "rest_client", None)
            intel_client: Any = getattr(app.state, "intel_http", None)
            replacements: dict[str, Any] = {}
            if candidate.network.proxy != settings.network.proxy and settings.profile != "demo":
                proxy = candidate.network.proxy
                rest = BinanceRestClient("fapi", timeout=10.0, proxy=proxy)
                created.append(rest)
                spot = BinanceRestClient("api", timeout=10.0, proxy=proxy)
                created.append(spot)
                intel_client = BoundedAsyncClient(
                    timeout=candidate.intel.per_source_timeout,
                    headers={"User-Agent": "binance-agent/0.2 (+local analysis)"},
                    follow_redirects=True,
                    proxy=proxy,
                )
                created.append(intel_client)
                replacements.update(
                    rest_client=rest,
                    spot_client=spot,
                    intel_http=intel_client,
                    ws_manager=BinanceWsManager("fapi", proxy=proxy),
                    spot_ws_manager=BinanceWsManager("api", proxy=proxy),
                )
                retired.extend([app.state.rest_client, app.state.spot_client, app.state.intel_http])
            aggregator = getattr(app.state, "aggregator", None)
            if candidate.intel != settings.intel or candidate.network != settings.network or reset:
                aggregator = (
                    DemoIntelAggregator()
                    if settings.profile == "demo"
                    else _build_aggregator(candidate, intel_client, rest)
                )
            graph = _build_runtime_graph(candidate, registry, app)
            if settings.profile != "demo":
                if reset:
                    clear_overrides()
                else:
                    save_overrides(_current_overrides(candidate, registry))
            old_registry = app.state.llm_registry
            activate_candidate(settings, candidate)
            for key, value in replacements.items():
                setattr(app.state, key, value)
            app.state.llm_registry = registry
            if aggregator is not None:
                app.state.aggregator = aggregator
            app.state.graph = graph
            manager = getattr(app.state, "research_manager", None)
            if manager is not None:
                manager.pool = EndpointPool(
                    per_endpoint_concurrency={
                        ep.name: ep.max_concurrency for ep in settings.llm.endpoints
                    }
                )
            retired.append(old_registry)
        except BaseException:
            for resource in [registry, *created]:
                with contextlib.suppress(Exception):
                    await resource.aclose()
            raise
        for resource in retired:
            try:
                await resource.aclose()
            except Exception:
                logger.exception("旧配置资源关闭失败")

    @app.post("/config/reset")
    async def reset_config() -> dict[str, object]:
        """清除运行时覆盖,回到 config.toml 基底(实时重建 registry/aggregator/graph)。

        覆盖范围(llm/intel/params/network)恢复为基底;app/market/hotkey/checkpoint 等仅基底项不受影响。
        """
        base = apply_profile(load_settings(), settings.profile)
        candidate, registry = _draft_configuration()
        for name in ("llm", "intel", "debate", "runtime", "confidence", "network"):
            setattr(candidate, name, getattr(base, name))
        registry.reload(candidate.llm)
        await _commit_configuration(candidate, registry, reset=True)
        return {
            "reset": True,
            "has_overrides": False if settings.profile == "demo" else overrides_present(),
            "active": registry.active,
            "endpoints": registry.endpoint_names(),
        }

    @app.post("/config/llm/endpoint")
    async def upsert_llm_endpoint(req: UpsertEndpointRequest) -> dict[str, object]:
        """新增或编辑端点;已有密钥只可随同 origin 的路径和模型修改继续使用。

        跨 origin 必须显式提供新 key,并在任何密钥访问或配置副作用前校验。
        新建无 key 端点也使用独立空引用,避免删除重建后接回同名历史密钥。
        新 key 使用独立引用,不覆盖旧密钥;无需 key 的服务可以单独新建端点。
        """
        existing = next((e for e in settings.llm.endpoints if e.name == req.name), None)
        new_key = req.api_key if req.api_key and req.api_key.strip() else None
        if existing is not None and new_key is None:
            try:
                same_origin = _endpoint_origin(req.base_url) == _endpoint_origin(existing.base_url)
            except ValueError:
                same_origin = False
            if not same_origin:
                raise HTTPException(
                    422,
                    "更改端点的协议、主机或端口时必须重新提供 API key;无需 key 的服务请新建端点。",
                )
        candidate, registry = _draft_configuration()
        if new_key is not None:  # 仅在提供新 key 时加密落盘并切到自管引用
            ref = f"llm_{req.name}_key_{uuid.uuid4().hex}"
            if not store_secret(ref, new_key):
                raise HTTPException(
                    status_code=500, detail="密钥加密存储不可用(keyring/cryptography 缺失)"
                )
        elif existing is not None:
            ref = existing.api_key_ref  # 编辑已有供应商但不改 key → 保留原引用
        else:
            ref = f"llm_{req.name}_key_{uuid.uuid4().hex}"
        registry.upsert_endpoint(
            req.name,
            req.base_url,
            req.model,
            models=req.models,
            api_key_ref=ref,
            structured_output=req.structured_output,
            fallback_verified=req.fallback_verified,
            max_concurrency=req.max_concurrency,
        )
        if req.use_for_all_roles:
            candidate.llm.agent_models = {
                role: (req.name, req.model) for role in AGENT_NAMES if role != "strategy_planner"
            }
            candidate.llm.active_endpoint = req.name
        await _commit_configuration(candidate, registry)
        return registry.config_view()

    @app.post("/config/llm/models")
    async def fetch_llm_models(req: FetchModelsRequest) -> dict[str, object]:
        """查询端点模型;沿用已存 key 前校验 origin,失败时不得读取密钥或出站。

        显式输入的新 key 只用于本次查询、不落盘;查询失败返回空列表供 UI 手填。
        """
        key = req.api_key if req.api_key and req.api_key.strip() else None
        if not key and req.endpoint:
            endpoint = next(
                (item for item in settings.llm.endpoints if item.name == req.endpoint), None
            )

            try:
                matches = endpoint is not None and _endpoint_origin(
                    req.base_url
                ) == _endpoint_origin(endpoint.base_url)
            except ValueError:
                matches = False
            if not matches:
                raise HTTPException(422, "saved key requires the configured endpoint origin")
            assert endpoint is not None
            key = resolve_secret(endpoint.api_key_ref)
        url = req.base_url.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        client: httpx.AsyncClient = app.state.intel_http
        try:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # 拉取失败 → 前端回退手填(url 不含 key,异常可安全返回)
            return {"models": [], "error": str(exc)[:200]}
        items = data.get("data") if isinstance(data, dict) else data
        models = []
        for item in items or []:
            mid = item.get("id") if isinstance(item, dict) else str(item)
            if mid:
                models.append(str(mid))
        return {"models": models}

    @app.post("/config/llm/agent")
    async def set_agent_model(req: SetAgentModelRequest) -> dict[str, object]:
        candidate, registry = _draft_configuration()
        if req.agent not in AGENT_NAMES:
            raise HTTPException(status_code=400, detail=f"未知 agent: {req.agent}")
        try:
            registry.set_agent_model(req.agent, req.endpoint, req.model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await _commit_configuration(candidate, registry)
        return registry.config_view()

    @app.delete("/config/llm/endpoint/{name}")
    async def delete_llm_endpoint(name: str) -> dict[str, object]:
        candidate, registry = _draft_configuration()
        try:
            registry.remove_endpoint(name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await _commit_configuration(candidate, registry)
        return registry.config_view()

    @app.post("/config/intel/source")
    async def set_intel_source(req: SetIntelSourceRequest) -> dict[str, object]:
        candidate, registry = _draft_configuration()
        if source_spec(req.source) is None:
            raise HTTPException(status_code=400, detail=f"未知数据源: {req.source}")
        candidate.intel.sources[req.source] = req.enabled
        await _commit_configuration(candidate, registry)
        aggregator: IntelAggregator = app.state.aggregator
        return {"sources": dict(candidate.intel.sources), "disabled": dict(aggregator._disabled)}

    @app.post("/config/intel/key")
    async def set_intel_key(req: SetIntelKeyRequest) -> dict[str, object]:
        spec = source_spec(req.source)
        old_ref = settings.intel.keys.get(req.source) or (
            "tavily" if req.source == "tavily" else spec.key_ref if spec else None
        )
        if not old_ref:
            raise HTTPException(status_code=400, detail=f"该数据源无需 key: {req.source}")
        candidate, registry = _draft_configuration()
        # A failed metadata commit must not replace the key used by the live
        # configuration. Unreferenced encrypted values are safe to retain.
        ref = f"intel_{req.source}_key_{uuid.uuid4().hex}"
        if not store_secret(ref, req.api_key):
            raise HTTPException(500, "密钥加密存储不可用(keyring/cryptography 缺失)")
        candidate.intel.keys[req.source] = ref
        await _commit_configuration(candidate, registry)
        aggregator: IntelAggregator = app.state.aggregator
        return {
            "source": req.source,
            "has_key": resolve_secret(ref) is not None,
            "disabled": dict(aggregator._disabled),
        }

    @app.post("/config/params")
    async def set_params(req: SetParamsRequest) -> dict[str, object]:
        candidate, registry = _draft_configuration()
        context_policy = None
        if req.context is not None:
            try:
                context_policy = ContextPolicy.model_validate(
                    {**candidate.runtime.context.model_dump(), **req.context}
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        if context_policy is not None:
            candidate.runtime.context = context_policy
        if req.time_window_hours is not None:
            candidate.intel.time_window_hours = req.time_window_hours
        if req.per_source_timeout is not None:
            candidate.intel.per_source_timeout = req.per_source_timeout
        if req.max_news_items is not None:
            candidate.intel.max_news_items = req.max_news_items
        if req.max_debate_rounds is not None:
            candidate.debate.max_rounds = req.max_debate_rounds
        if req.strategy_planner_mode is not None:
            candidate.runtime.strategy_planner_mode = req.strategy_planner_mode
            enabled, llm_actions, _ = planner_flags_for_mode(req.strategy_planner_mode)
            candidate.runtime.strategy_planner_enabled = enabled
            candidate.runtime.strategy_planner_llm_actions = llm_actions
        if req.strategy_planner_enabled is not None:
            candidate.runtime.strategy_planner_enabled = req.strategy_planner_enabled
        if req.strategy_planner_max_steps is not None:
            candidate.runtime.strategy_planner_max_steps = req.strategy_planner_max_steps
        if req.strategy_planner_llm_actions is not None:
            candidate.runtime.strategy_planner_llm_actions = req.strategy_planner_llm_actions
        if req.preflight_refetch_enabled is not None:
            candidate.runtime.preflight_refetch_enabled = req.preflight_refetch_enabled
        if req.strategy_planner_refetch_enabled is not None:
            warnings.warn(
                "strategy_planner_refetch_enabled is deprecated; use preflight_refetch_enabled",
                DeprecationWarning,
                stacklevel=2,
            )
            candidate.runtime.preflight_refetch_enabled = req.strategy_planner_refetch_enabled
            candidate.runtime.strategy_planner_refetch_enabled = (
                req.strategy_planner_refetch_enabled
            )
        if req.strategy_planner_mode is None and (
            req.strategy_planner_enabled is not None or req.strategy_planner_llm_actions is not None
        ):
            candidate.runtime.strategy_planner_mode = planner_mode_from_flags(
                candidate.runtime.strategy_planner_enabled,
                candidate.runtime.strategy_planner_llm_actions,
            )
        await _commit_configuration(candidate, registry)
        return {
            "context": candidate.runtime.context.model_dump(mode="json"),
            "time_window_hours": candidate.intel.time_window_hours,
            "max_debate_rounds": candidate.debate.max_rounds,
            "per_source_timeout": candidate.intel.per_source_timeout,
            "max_news_items": candidate.intel.max_news_items,
            "strategy_planner_mode": effective_strategy_planner_mode(candidate.runtime),
            "strategy_planner_enabled": candidate.runtime.strategy_planner_enabled,
            "strategy_planner_max_steps": candidate.runtime.strategy_planner_max_steps,
            "strategy_planner_llm_actions": candidate.runtime.strategy_planner_llm_actions,
            "preflight_refetch_enabled": candidate.runtime.preflight_refetch_enabled,
            "strategy_planner_refetch_enabled": candidate.runtime.preflight_refetch_enabled,
        }

    @app.post("/config/desktop")
    async def set_desktop(req: DesktopRequest) -> dict[str, Any]:
        candidate, registry = _draft_configuration()
        candidate.desktop.on_top = req.on_top
        candidate.hotkey.enabled = req.hotkey_enabled
        await _commit_configuration(candidate, registry)
        return {"restart_required": True}

    @app.post("/config/research")
    async def set_research_limits(req: ResearchLimitsRequest) -> dict[str, Any]:
        candidate, registry = _draft_configuration()
        candidate.research.max_requests = req.max_requests
        candidate.research.deadline_s = req.deadline_s
        await _commit_configuration(candidate, registry)
        return candidate.research.model_dump()

    @app.post("/config/network")
    async def set_network(req: SetNetworkRequest) -> dict[str, object]:
        # 代理改动需重建受其影响的出站客户端(lifespan 已用旧 proxy 建好)。
        if settings.profile == "demo":
            raise HTTPException(status_code=409, detail="demo profile is offline")
        try:
            network = NetworkConfig(proxy=req.proxy.strip() or None)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="invalid proxy URL; embedded credentials are not allowed",
            ) from exc
        proxy = network.proxy
        candidate, registry = _draft_configuration()
        candidate.network = network
        # The staged registry has no clients yet; set its transport before building.
        registry.set_proxy(proxy)
        await _commit_configuration(candidate, registry)
        logger.info("出站代理已更新: %s", "(已配置,地址已脱敏)" if proxy else "(已清除)")
        return {"proxy": settings.network.proxy or ""}

    @app.websocket("/ws/price")
    async def ws_price(websocket: WebSocket) -> None:
        """本地来源与参数验证通过后订阅行情;断开时停止上游并回收监听任务。"""
        if not valid_local_origin(websocket.scope):
            await websocket.close(code=1008, reason="cross-site request rejected")
            return
        await websocket.accept()
        symbol = websocket.query_params.get("symbol", "").strip().lower()
        market = websocket.query_params.get("market", "fapi").strip().lower()
        if not (2 <= len(symbol) <= 40 and symbol.isascii() and symbol.isalnum()):
            await websocket.close(code=1003)
            return
        if market == "fapi":
            manager: BinanceWsManager = app.state.ws_manager
        elif market == "api":
            manager = app.state.spot_ws_manager
        else:
            await websocket.close(code=1003)
            return
        stop = asyncio.Event()

        async def _watch_disconnect() -> None:
            try:
                while True:
                    await websocket.receive_text()
            except Exception:
                stop.set()

        watcher = asyncio.create_task(_watch_disconnect())
        try:
            async for evt in manager.price_stream(symbol, stop=stop):
                await websocket.send_json(evt)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # 连接异常 → 静默收尾
            logger.debug("ws_price 结束: %s", exc)
        finally:
            stop.set()
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            with contextlib.suppress(Exception):
                await websocket.close()

    # 悬浮窗单页 UI(pywebview 加载同源 http://127.0.0.1:port/)
    if _UI_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(_UI_DIR)), name="ui")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(str(_UI_DIR / "index.html"))

    app.add_middleware(AnalysisLifecycleMiddleware)
    return app
