"""OpenAI 兼容多端点 LLM 注册表 —— per-agent 模型 + 运行时切换。

便宜模型跑分析师、强模型跑辩论/合成;``model_factory`` 可注入用于离线测试
(如 FakeListChatModel),无需真实网络。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx
from langchain_core.language_models import BaseChatModel

from ..agents.runtime import RuntimeModelCandidate
from ..infra.secrets import resolve_secret
from ..infra.settings import LlmConfig, LlmEndpoint

logger = logging.getLogger(__name__)

ModelFactory = Callable[[LlmEndpoint, str], BaseChatModel]

# LLM agent(与 agents/graph.py 节点一致;risk_manager 规则式不在内)
AGENT_NAMES: tuple[str, ...] = (
    "research_supervisor",
    "news_researcher",
    "evidence_verifier",
    "technical",
    "funding",
    "news_sentiment",
    "bull_debater",
    "bear_debater",
    "research_manager",
    "strategy_planner",
    "strategy_synth",
)


def _proxy_log_label(proxy: str | None) -> str:
    if not proxy:
        return "(清除,回退环境变量)"
    try:
        parsed = urlsplit(proxy)
        host = parsed.hostname
        if not host:
            return "(已配置)"
        if ":" in host:
            host = "[****]"
        else:
            labels = host.split(".")
            if len(labels) == 4 and all(label.isdigit() for label in labels):
                host = f"{labels[0]}.*.*.{labels[-1]}"
            elif len(labels) >= 2:
                host = "*." + ".".join(labels[-2:])
            else:
                host = host[:2] + "***"
        port = f":{parsed.port}" if parsed.port is not None else ""
        scheme = f"{parsed.scheme}://" if parsed.scheme else ""
        return f"{scheme}{host}{port}"
    except ValueError:
        return "(已配置)"


def _build_chat_openai(
    endpoint: LlmEndpoint,
    model: str,
    *,
    http_client: httpx.Client | None = None,
    http_async_client: httpx.AsyncClient | None = None,
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    # 用 kwargs 字典传参,兼容不同 langchain-openai 版本并避开第三方构造的类型摩擦
    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": endpoint.base_url,
        "api_key": resolve_secret(endpoint.api_key_ref) or "not-set",
        "temperature": endpoint.temperature,
        "timeout": endpoint.timeout,
        "max_retries": 0,  # AgentRuntime owns all retries and attempt accounting.
    }
    # 显式代理:覆盖环境变量,使 LLM 出站与行情/资讯走同一代理。
    # 留空则不注入 http_client → ChatOpenAI 内部走 trust_env 自动读环境变量。
    if http_client is not None and http_async_client is not None:
        kwargs["http_client"] = http_client
        kwargs["http_async_client"] = http_async_client
    return ChatOpenAI(**kwargs)


class LlmRegistry:
    def __init__(
        self,
        config: LlmConfig,
        *,
        model_factory: ModelFactory | None = None,
        proxy: str | None = None,
    ) -> None:
        self._config = config
        self._active = config.active_endpoint
        self._endpoints: dict[str, LlmEndpoint] = {e.name: e for e in config.endpoints}
        # 注入的测试工厂保持 2 参签名;默认构造器由 Registry 注入共享代理客户端。
        self._factory: ModelFactory | None = model_factory
        self._proxy = proxy
        self._cache: dict[tuple[str, str], BaseChatModel] = {}
        self._http_client: httpx.Client | None = None
        self._http_async_client: httpx.AsyncClient | None = None
        self._closing_tasks: set[asyncio.Task[None]] = set()

    def _proxy_clients(self) -> tuple[httpx.Client | None, httpx.AsyncClient | None]:
        if not self._proxy or self._factory is not None:
            return None, None
        if self._http_client is None or self._http_async_client is None:
            self._http_client = httpx.Client(proxy=self._proxy, timeout=None)
            self._http_async_client = httpx.AsyncClient(proxy=self._proxy, timeout=None)
        return self._http_client, self._http_async_client

    def _model(self, endpoint_name: str, model: str) -> BaseChatModel:
        key = (endpoint_name, model)
        if key not in self._cache:
            endpoint = self._endpoints[endpoint_name]
            if self._factory is not None:
                self._cache[key] = self._factory(endpoint, model)
            else:
                http_client, http_async_client = self._proxy_clients()
                self._cache[key] = _build_chat_openai(
                    endpoint,
                    model,
                    http_client=http_client,
                    http_async_client=http_async_client,
                )
        return self._cache[key]

    async def aclose(self) -> None:
        """关闭 Registry 持有的显式代理客户端并释放模型缓存。"""
        self._cache.clear()
        http_client, self._http_client = self._http_client, None
        http_async_client, self._http_async_client = self._http_async_client, None
        if http_client is not None:
            http_client.close()
        if http_async_client is not None:
            await http_async_client.aclose()
        if self._closing_tasks:
            await asyncio.gather(*tuple(self._closing_tasks), return_exceptions=True)
            self._closing_tasks.clear()

    def _retire_proxy_clients(self) -> None:
        http_client, self._http_client = self._http_client, None
        http_async_client, self._http_async_client = self._http_async_client, None
        if http_client is not None:
            http_client.close()
        if http_async_client is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(http_async_client.aclose())
            return
        task = loop.create_task(http_async_client.aclose())
        self._closing_tasks.add(task)
        task.add_done_callback(self._finish_close_task)

    def _finish_close_task(self, task: asyncio.Task[None]) -> None:
        self._closing_tasks.discard(task)
        try:
            task.result()
        except (Exception, asyncio.CancelledError):
            logger.debug("关闭旧 LLM 代理客户端失败", exc_info=True)

    def set_proxy(self, proxy: str | None) -> None:
        """更新出站代理;清缓存使下次构造的 LLM client 生效。图重建由调用方负责。"""
        if proxy == self._proxy:
            return
        self._retire_proxy_clients()
        self._proxy = proxy
        self._cache.clear()
        logger.info("LLM 出站代理已更新: %s", _proxy_log_label(proxy))

    def for_agent(self, agent: str) -> BaseChatModel:
        """按角色映射选择模型,映射失效时回到 active 端点;复用注册表中的模型缓存。"""
        endpoint_name, model = self._resolve(agent)
        return self._model(endpoint_name, model)

    def active_model(self) -> BaseChatModel:
        """active 端点的默认模型(用于连通性探测)。

        显式取 active,避免 ``for_agent`` 因 agent_models 映射误路由到其他端点。
        无可用 active 端点时抛 ``RuntimeError``(与 ``_resolve`` 一致)。
        """
        if self._active not in self._endpoints:
            raise RuntimeError("未配置任何可用 LLM 端点")
        endpoint = self._endpoints[self._active]
        return self._model(endpoint.name, endpoint.model)

    def fallbacks_for_agent(self, agent: str) -> list[BaseChatModel]:
        """按端点配置顺序返回其他端点的默认模型,排除主端点及重复候选。

        仅构造或复用模型对象,不在这里发起请求;带端点执行参数的路径使用
        runtime_candidates_for,避免把主端点能力套到 fallback。
        """
        primary_endpoint, primary_model = self._resolve(agent)
        fallbacks: list[BaseChatModel] = []
        seen = {(primary_endpoint, primary_model)}
        for endpoint in self._endpoints.values():
            if endpoint.name == primary_endpoint:
                continue
            pair = (endpoint.name, endpoint.model)
            if pair in seen:
                continue
            seen.add(pair)
            fallbacks.append(self._model(endpoint.name, endpoint.model))
        return fallbacks

    def runtime_candidates_for(self, agent: str) -> list[RuntimeModelCandidate]:
        """按 primary → 其他端点顺序返回带端点级执行策略的运行时候选。"""
        primary_endpoint, primary_model = self._resolve(agent)
        resolved: list[tuple[LlmEndpoint, str]] = [
            (self._endpoints[primary_endpoint], primary_model)
        ]
        resolved.extend(
            (endpoint, endpoint.model)
            for endpoint in self._endpoints.values()
            if endpoint.name != primary_endpoint
        )
        return [
            RuntimeModelCandidate(
                model=self._model(endpoint.name, model),
                endpoint=endpoint.name,
                timeout_s=float(endpoint.timeout),
                max_retries=endpoint.max_retries,
                structured_output=endpoint.structured_output,
            )
            for endpoint, model in resolved
        ]

    def runtime_config_for(self, agent: str) -> dict[str, Any]:
        """该 agent 端点的 AgentRuntime 外层配置(供节点构造 AgentRuntime)。

        ``candidates`` 是生效路径,逐端点携带 timeout/retry/结构化输出能力。旧的三个
        primary 字段暂留给既有调用方,避免图构建和离线注入发生不必要破坏。
        """
        endpoint_name, _ = self._resolve(agent)
        endpoint = self._endpoints[endpoint_name]
        return {
            "timeout_s": float(endpoint.timeout),
            "max_retries": endpoint.max_retries,
            "structured_output": endpoint.structured_output,
            "candidates": self.runtime_candidates_for(agent),
        }

    def _resolve(self, agent: str) -> tuple[str, str]:
        """返回角色的端点与模型;无有效角色映射时使用 active,无可用端点则显式失败。"""
        mapping = self._config.agent_models.get(agent)
        if mapping:
            endpoint_name, model = mapping
            if endpoint_name in self._endpoints:
                return endpoint_name, model
        if self._active not in self._endpoints:
            raise RuntimeError("未配置任何可用 LLM 端点")
        endpoint = self._endpoints[self._active]
        return endpoint.name, endpoint.model

    def switch_active(self, endpoint_name: str) -> None:
        """切换已存在的默认端点并清空模型缓存,同时修改当前内存配置。

        未知端点抛 ValueError;持久化和图重建由调用方提交配置时完成。
        """
        if endpoint_name not in self._endpoints:
            raise ValueError(f"未知端点: {endpoint_name}")
        self._active = endpoint_name
        self._config.active_endpoint = endpoint_name
        self._cache.clear()
        logger.info("LLM active 端点切换为 %s", endpoint_name)

    def reload(self, config: LlmConfig) -> None:
        """用新的 LlmConfig 重置端点/active/缓存(保留 model_factory)。供「重置为默认」使用。"""
        self._config = config
        self._active = config.active_endpoint
        self._endpoints = {e.name: e for e in config.endpoints}
        self._cache.clear()
        logger.info("LLM registry 重载:%d 个端点,active=%s", len(self._endpoints), self._active)

    @property
    def active(self) -> str:
        return self._active

    def endpoint_names(self) -> list[str]:
        return list(self._endpoints)

    def set_agent_model(self, agent: str, endpoint: str, model: str) -> None:
        """运行时设置某 agent 的 (endpoint, model);清缓存,图重建由调用方负责。"""
        if endpoint not in self._endpoints:
            raise ValueError(f"未知端点: {endpoint}")
        self._config.agent_models[agent] = (endpoint, model)
        self._cache.clear()

    def upsert_endpoint(
        self,
        name: str,
        base_url: str,
        model: str,
        *,
        models: list[str] | None = None,
        api_key_ref: str | None = None,
        structured_output: bool | None = None,
        fallback_verified: bool | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        """新增或更新一个供应商端点(api_key_ref 默认 llm_<name>_key,与加密库约定一致)。"""
        existing = self._endpoints.get(name)
        endpoint = LlmEndpoint(
            name=name,
            base_url=base_url,
            api_key_ref=api_key_ref or (existing.api_key_ref if existing else f"llm_{name}_key"),
            model=model,
            models=models if models is not None else (existing.models if existing else []),
            temperature=existing.temperature if existing else 0.3,
            timeout=existing.timeout if existing else 60,
            max_retries=existing.max_retries if existing else 2,
            structured_output=(
                structured_output
                if structured_output is not None
                else (existing.structured_output if existing else False)
            ),
            fallback_verified=(
                fallback_verified
                if fallback_verified is not None
                else (existing.fallback_verified if existing else False)
            ),
            max_concurrency=(
                max_concurrency
                if max_concurrency is not None
                else (existing.max_concurrency if existing else 2)
            ),
        )
        self._endpoints[name] = endpoint
        if existing is None:
            self._config.endpoints.append(endpoint)
        else:
            self._config.endpoints = [
                endpoint if e.name == name else e for e in self._config.endpoints
            ]
        self._cache.clear()

    def remove_endpoint(self, name: str) -> None:
        """删除一个供应商端点;至少保留一个;active 与指向它的 agent 映射自动回退。"""
        if name not in self._endpoints:
            raise ValueError(f"未知端点: {name}")
        if len(self._endpoints) <= 1:
            raise ValueError("至少保留一个端点")
        del self._endpoints[name]
        self._config.endpoints = [e for e in self._config.endpoints if e.name != name]
        if self._active == name:
            self._active = next(iter(self._endpoints))
            self._config.active_endpoint = self._active
        for agent in [a for a, (ep, _) in self._config.agent_models.items() if ep == name]:
            del self._config.agent_models[agent]  # 指向被删端点 → 回退 active
        self._cache.clear()

    def config_view(self, *, include_secret_status: bool = True) -> dict[str, Any]:
        """供 GET /config 的只读视图;has_key 仅布尔,绝不回传 key 明文。"""
        endpoints = [
            {
                "name": e.name,
                "base_url": e.base_url,
                "model": e.model,
                "models": e.models,
                "structured_output": e.structured_output,
                "fallback_verified": e.fallback_verified,
                "max_concurrency": e.max_concurrency,
                "has_key": include_secret_status and resolve_secret(e.api_key_ref) is not None,
            }
            for e in self._endpoints.values()
        ]
        agents = []
        for agent in AGENT_NAMES:
            try:
                endpoint_name, model = self._resolve(agent)
            except RuntimeError:
                continue
            agents.append({"agent": agent, "endpoint": endpoint_name, "model": model})
        return {"active": self._active, "endpoints": endpoints, "agents": agents}
