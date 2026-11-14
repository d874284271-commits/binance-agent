"""统一 LLM 调用层:timeout/retry/fallback/JSON repair/调用元数据。

节点负责业务 payload 与结构化 fallback;runtime 只负责把一次 LLM 调用变成
可观测、可降级的结构化结果,避免单个模型异常拖垮整条分析链路。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..infra.retry import parse_retry_after
from .budget import BudgetExceeded, BudgetPersistenceError, current_run_budget
from .context import (
    COMPACTION_PROMPT,
    COUNTING_METHOD,
    ContextEnvelope,
    ContextPolicy,
    ContextUsage,
    compaction_cache_key,
    stable_json,
    validate_compaction,
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True, slots=True)
class RuntimeModelCandidate:
    """一次运行时调用候选及其端点级执行能力。"""

    model: BaseChatModel
    endpoint: str
    timeout_s: float = 60.0
    max_retries: int = 1
    structured_output: bool = False
    max_output_tokens: int | None = None


class RuntimeErrorInfo(BaseModel):
    category: str
    message: str
    model_index: int
    attempt: int


class RuntimeResult(BaseModel):
    data: dict[str, Any]
    text: str = ""
    ok: bool = False
    repaired: bool = False
    model_index: int = 0
    attempts: int = 0
    latency_ms: float = 0.0
    endpoint: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    error_category: str | None = None
    errors: list[RuntimeErrorInfo] = Field(default_factory=list)
    budget: dict[str, Any] | None = None
    context_usage: dict[str, Any] = Field(default_factory=dict)
    context_updates: dict[str, Any] = Field(default_factory=dict)


class AgentRuntime:
    def __init__(
        self,
        llm: BaseChatModel | None = None,
        *,
        candidates: list[RuntimeModelCandidate] | None = None,
        fallbacks: list[BaseChatModel] | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 1,
        structured_output: bool = False,
        context_policy: ContextPolicy | dict[str, Any] | None = None,
        context_cache: dict[str, Any] | None = None,
    ) -> None:
        self.context_policy = (
            context_policy
            if isinstance(context_policy, ContextPolicy)
            else ContextPolicy.model_validate(context_policy or {})
        )
        self._context_cache = dict(context_cache or {})
        self._compaction_locks: dict[str, asyncio.Lock] = {}
        if candidates:
            self._candidates = list(candidates)
        elif llm is not None:
            self._candidates = [
                RuntimeModelCandidate(
                    model=model,
                    endpoint="primary" if index == 0 else f"fallback-{index}",
                    timeout_s=timeout_s,
                    max_retries=max(0, max_retries),
                    structured_output=structured_output,
                )
                for index, model in enumerate([llm, *(fallbacks or [])])
            ]
        else:
            raise ValueError("AgentRuntime 至少需要一个模型候选")

    async def invoke_structured(
        self,
        agent: str,
        system_prompt: str,
        payload: dict[str, Any] | ContextEnvelope,
        *,
        fallback_data: dict[str, Any],
        output_validator: Callable[[dict[str, Any]], None] | None = None,
    ) -> RuntimeResult:
        """请求 JSON 对象并执行可选输出契约校验,返回带 usage、错误及上下文记录的结果。

        常规模型失败使用调用方的 fallback_data,ok 仍为 False;协作输入已过期时
        可直接返回 input_superseded。协作调用先经 AgentEngine 串行化及持久结果
        复用,避免恢复时重做已完成的逻辑调用。
        """
        from .collaboration import current_scope

        scope = current_scope()
        if scope is not None:
            return await scope.engine.invoke(
                scope,
                agent,
                payload,
                lambda prepared: self._invoke(
                    agent,
                    system_prompt,
                    prepared,
                    fallback_data,
                    json_mode=True,
                    output_validator=output_validator,
                ),
            )
        return await self._invoke(
            agent,
            system_prompt,
            payload,
            fallback_data,
            json_mode=True,
            output_validator=output_validator,
        )

    async def invoke_text(
        self,
        agent: str,
        system_prompt: str,
        payload: dict[str, Any] | ContextEnvelope,
        *,
        fallback_text: str,
    ) -> RuntimeResult:
        """请求普通文本并提供 fallback_text 作为失败回退数据,结果保留明确的 ok 状态。

        本入口不强制 JSON;协作、预算与窗口准入沿用结构化入口。调用方必须检查
        ok,失败时不能把 text 当作有效模型结论,也不能假定 text 总是包含回退文本。
        """
        from .collaboration import current_scope

        scope = current_scope()
        if scope is not None:
            return await scope.engine.invoke(
                scope,
                agent,
                payload,
                lambda prepared: self._invoke(
                    agent, system_prompt, prepared, {"text": fallback_text}, json_mode=False
                ),
            )
        return await self._invoke(
            agent, system_prompt, payload, {"text": fallback_text}, json_mode=False
        )

    def _output_limit(self, candidate: RuntimeModelCandidate, *, compact: bool = False) -> int:
        policy_limit = (
            self.context_policy.compact_output_tokens
            if compact
            else self.context_policy.max_output_tokens
        )
        return (
            min(policy_limit, candidate.max_output_tokens)
            if candidate.max_output_tokens
            else policy_limit
        )

    def _count_messages(
        self, candidate: RuntimeModelCandidate, messages: list[Any]
    ) -> tuple[int, str]:
        """优先使用模型显式提供的离线计数器,否则按 UTF-8 字节数保守估计。

        不调用可能联网下载 tokenizer 的通用计数方法;返回计数和计数方式用于 trace。
        """
        # An explicitly offline adapter hook is safe; BaseChatModel.get_num_tokens
        # may download tokenizer assets and is deliberately never called.
        counter = getattr(candidate.model, "count_tokens_offline", None)
        if callable(counter):
            try:
                value = counter(messages)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    return value, "adapter_offline"
            except Exception:
                pass
        return sum(len(str(m.content).encode("utf-8")) + 256 for m in messages), COUNTING_METHOD

    async def _compact_context(
        self,
        target: RuntimeModelCandidate,
        agent: str,
        system_prompt: str,
        envelope: ContextEnvelope,
        usage: ContextUsage,
        updates: dict[str, Any],
        token_usage: dict[str, int],
    ) -> dict[str, Any] | None:
        """按完整输入版本串行复用压缩结果,仅压缩论述块,不改 protected_data。

        每个缓存键最多发起一次压缩请求,失败也缓存;不修复 JSON、不递归压缩。
        来源覆盖或目标长度不合格返回 None,压缩请求的费用及 usage 计入当前预算。
        """
        key = compaction_cache_key(agent, system_prompt, envelope, self.context_policy)
        usage.cache_key = key
        async with self._compaction_locks.setdefault(key, asyncio.Lock()):
            cached = self._context_cache.get(key)
            if cached is None:
                usage.compression_attempts += 1
                try:
                    compact_payload = {
                        "target_node": agent,
                        "token_budget": max(
                            0,
                            self.context_policy.target_tokens
                            - self._count_messages(
                                target, _messages(system_prompt, envelope.protected_data)
                            )[0],
                        ),
                        "blocks": [block.model_dump() for block in envelope.compressible_blocks],
                    }
                    text, tokens = await self._ainvoke_text(
                        self._candidates[0],
                        _messages(COMPACTION_PROMPT, compact_payload),
                        json_mode=True,
                        logical_task=agent,
                        attempt_kind="compact",
                    )
                    _merge_token_usage(token_usage, tokens)
                    # Strict JSON: no repair, fenced extraction, retry, or recursive compaction.
                    try:
                        value = json.loads(text)
                    except (TypeError, ValueError):
                        value = None
                    cached = {"response": value, "failure_reason": None}
                except BudgetPersistenceError:
                    raise
                except Exception as exc:
                    cached = {"response": None, "failure_reason": _classify_error(exc)}
                self._context_cache[key] = cached
                updates[key] = cached
        sections = (
            validate_compaction(cached.get("response"), envelope)
            if isinstance(cached, dict)
            else None
        )
        if sections is not None:
            compacted = envelope.render(sections)
            after, _ = self._count_messages(target, _messages(system_prompt, compacted))
            before, _ = self._count_messages(target, _messages(system_prompt, envelope.render()))
            # Protected fields come only from the original envelope, never model output.
            if after <= self.context_policy.target_tokens and after < before:
                usage.input_tokens_after = after
                usage.compression_status = "compacted" if key in updates else "cache_hit"
                usage.failure_reason = None
                return compacted
        usage.compression_status = "failed"
        usage.failure_reason = (
            cached.get("failure_reason") if isinstance(cached, dict) else None
        ) or "invalid_or_oversized_summary"
        return None

    async def _invoke(
        self,
        agent: str,
        system_prompt: str,
        payload: dict[str, Any] | ContextEnvelope,
        fallback_data: dict[str, Any],
        *,
        json_mode: bool,
        output_validator: Callable[[dict[str, Any]], None] | None = None,
    ) -> RuntimeResult:
        """依次尝试模型候选,在窗口、预算与端点重试上限内返回一次逻辑调用结果。

        无效结构或空输出直接保守回退;上下文拒绝只允许输入确实缩小后再试。
        压缩和 JSON 修复也是受预算约束的请求,持久化失败及取消不会被包装为成功。
        """
        started = time.perf_counter()
        errors: list[RuntimeErrorInfo] = []
        attempts = 0
        token_usage: dict[str, int] = {}
        updates: dict[str, Any] = {}
        envelope = payload if isinstance(payload, ContextEnvelope) else None
        effective_payload = payload.render() if isinstance(payload, ContextEnvelope) else payload
        before, method = self._count_messages(
            self._candidates[0], _messages(system_prompt, effective_payload)
        )
        usage = ContextUsage(
            input_tokens_before=before,
            input_tokens_after=before,
            counting_method=method,
            window_tokens=self.context_policy.window_tokens,
            output_tokens_reserved=self._output_limit(self._candidates[0]),
            safety_margin_tokens=self.context_policy.safety_margin_tokens,
        )
        compaction_checked = False

        def result(**kwargs: Any) -> RuntimeResult:
            return RuntimeResult(
                attempts=attempts,
                latency_ms=_elapsed_ms(started),
                token_usage=token_usage,
                errors=errors,
                budget=_budget_snapshot(),
                context_usage=usage.model_dump(),
                context_updates=updates,
                **kwargs,
            )

        for model_index, candidate in enumerate(self._candidates):
            # Each endpoint may use a different offline counter. Report before
            # and after in this candidate's units and recheck its soft threshold,
            # while allowing at most one compaction attempt for the input version.
            usage.input_tokens_before, usage.counting_method = self._count_messages(
                candidate,
                _messages(
                    system_prompt,
                    payload.render() if isinstance(payload, ContextEnvelope) else payload,
                ),
            )
            current_count, _ = self._count_messages(
                candidate, _messages(system_prompt, effective_payload)
            )
            if not compaction_checked and current_count >= self.context_policy.trigger_tokens:
                if envelope is not None and envelope.compressible_blocks:
                    compaction_checked = True
                    compacted = await self._compact_context(
                        candidate, agent, system_prompt, envelope, usage, updates, token_usage
                    )
                    if compacted is not None:
                        effective_payload = compacted
                else:
                    usage.compression_status = "protected_only"
            attempt = 0
            # A materially reduced request after a provider's smaller-window rejection
            # gets one separate attempt, even when ordinary retries are disabled.
            reduced_retry = False
            while attempt <= max(0, candidate.max_retries) or reduced_retry:
                reduced_retry = False
                attempts += 1
                usage.output_tokens_reserved = self._output_limit(candidate)
                messages = _messages(system_prompt, effective_payload)
                usage.input_tokens_after, usage.counting_method = self._count_messages(
                    candidate, messages
                )
                try:
                    text, tokens = await self._ainvoke_text(
                        candidate,
                        messages,
                        json_mode=json_mode,
                        logical_task=agent,
                        attempt_kind="invoke",
                    )
                    _merge_token_usage(token_usage, tokens)
                except BudgetPersistenceError:
                    raise
                except Exception as exc:
                    category = _classify_error(exc)
                    if category == "input_superseded":
                        return result(
                            data=fallback_data, error_category=category, model_index=model_index
                        )
                    errors.append(
                        RuntimeErrorInfo(
                            category=category,
                            message=_safe_error_message(category),
                            model_index=model_index,
                            attempt=attempt,
                        )
                    )
                    if category == "context_length":
                        if (
                            not compaction_checked
                            and envelope is not None
                            and envelope.compressible_blocks
                        ):
                            compaction_checked = True
                            compacted = await self._compact_context(
                                candidate,
                                agent,
                                system_prompt,
                                envelope,
                                usage,
                                updates,
                                token_usage,
                            )
                            if compacted is not None:
                                effective_payload = compacted
                                reduced_retry = True
                                attempt += 1
                                continue
                        break  # deterministic rejection: never resend identical input
                    if category in {"budget", "auth", "content_policy", "model_not_found"}:
                        break
                    attempt += 1
                    continue
                if not text.strip():
                    return result(
                        data=fallback_data, error_category="empty_response", model_index=model_index
                    )
                if not json_mode:
                    return result(
                        data={"text": text},
                        text=text,
                        ok=True,
                        model_index=model_index,
                        endpoint=candidate.endpoint,
                    )
                try:
                    parsed = _parse_json_object(text)
                except InvalidStructuredOutput:
                    return result(
                        data=fallback_data, error_category="schema_invalid", model_index=model_index
                    )
                if parsed is not None:
                    try:
                        json.dumps(parsed, allow_nan=False)
                        if not parsed:
                            raise ValueError("empty object")
                        if output_validator is not None:
                            output_validator(parsed)
                    except (ValueError, TypeError, KeyError):
                        return result(
                            data=fallback_data,
                            error_category="schema_invalid",
                            model_index=model_index,
                        )
                    return result(
                        data=parsed,
                        text=text,
                        ok=True,
                        model_index=model_index,
                        endpoint=candidate.endpoint,
                    )
                errors.append(
                    RuntimeErrorInfo(
                        category="json_parse",
                        message="LLM output is not valid JSON",
                        model_index=model_index,
                        attempt=attempt,
                    )
                )
                try:
                    repaired_text, repair_usage = await self._repair_json(agent, candidate, text)
                except BudgetPersistenceError:
                    raise
                except Exception as exc:
                    category = _classify_error(exc)
                    if category == "input_superseded":
                        return result(
                            data=fallback_data, error_category=category, model_index=model_index
                        )
                    errors.append(
                        RuntimeErrorInfo(
                            category=category,
                            message=_safe_error_message(category),
                            model_index=model_index,
                            attempt=attempt,
                        )
                    )
                    if category in {
                        "context_length",
                        "budget",
                        "auth",
                        "content_policy",
                        "model_not_found",
                    }:
                        break
                    attempt += 1
                    continue
                _merge_token_usage(token_usage, repair_usage)
                if not repaired_text.strip():
                    return result(
                        data=fallback_data, error_category="empty_response", model_index=model_index
                    )
                try:
                    repaired = _parse_json_object(repaired_text)
                except InvalidStructuredOutput:
                    return result(
                        data=fallback_data, error_category="schema_invalid", model_index=model_index
                    )
                if repaired is not None:
                    try:
                        json.dumps(repaired, allow_nan=False)
                        if not repaired:
                            raise ValueError("empty object")
                        if output_validator is not None:
                            output_validator(repaired)
                    except (ValueError, TypeError, KeyError):
                        return result(
                            data=fallback_data,
                            error_category="schema_invalid",
                            model_index=model_index,
                        )
                    return result(
                        data=repaired,
                        text=repaired_text,
                        ok=True,
                        repaired=True,
                        model_index=model_index,
                        endpoint=candidate.endpoint,
                    )
                attempt += 1
        return result(
            data=fallback_data,
            text="" if json_mode else str(fallback_data["text"]),
            ok=False,
            error_category=errors[-1].category if errors else "llm_error",
        )

    async def _ainvoke_text(
        self,
        candidate: RuntimeModelCandidate,
        messages: list[Any],
        *,
        json_mode: bool = False,
        logical_task: str = "runtime",
        attempt_kind: str = "invoke",
    ) -> tuple[str, dict[str, int]]:
        """执行一个实际模型请求,返回去除 think 块的文本及供应商 token usage。

        发送前检查完整窗口并预留预算;并发等待后再次检查输入版本,过时请求不发送。
        超时或取消若发生在发送后按可能已计费结算;成功响应先结算再返回,结算失败
        不重发成功请求。所有路径都由预算结算负责释放本次端点并发槽。
        """
        from .collaboration import InputSuperseded, current_scope

        scope = current_scope()
        if scope is not None and not scope.is_current():
            raise InputSuperseded("new input is pending; request was not sent")
        # 仅在期望 JSON 的调用(invoke_structured / _repair_json)且端点开启结构化输出时,
        # 绑定 response_format=json_object 从源头约束模型输出;invoke_text(散文)不绑定。
        # 失败仍回落上层的 greedy 正则 + LLM repair 两级兜底。
        output_limit = self._output_limit(candidate, compact=attempt_kind == "compact")
        input_count, _ = self._count_messages(candidate, messages)
        if (
            input_count + output_limit + self.context_policy.safety_margin_tokens
            > self.context_policy.window_tokens
        ):
            raise ContextLengthExceeded("local context window exceeded")
        bindings: dict[str, Any] = {"max_tokens": output_limit}
        if json_mode and candidate.structured_output:
            bindings["response_format"] = {"type": "json_object"}
        runnable: Any = candidate.model.bind(**bindings)
        budget = current_run_budget()
        reservation = None
        if budget is not None:
            reservation = await budget.reserve(
                candidate.endpoint,
                logical_task,
                kind=attempt_kind,
                # Billing reservation remains conservative even with an offline tokenizer.
                input_tokens_upper=sum(len(str(m.content).encode("utf-8")) + 256 for m in messages),
                output_tokens_limit=output_limit,
            )
        sent = False
        try:
            if scope is not None and not scope.is_current():
                raise InputSuperseded("new input arrived during admission; request was not sent")
            timeout = candidate.timeout_s
            if budget is not None:
                remaining = budget.remaining_s()
                if remaining is not None:
                    if remaining <= 0:
                        raise BudgetExceeded("run deadline exceeded")
                    timeout = min(timeout, remaining)
            sent = True
            response = await asyncio.wait_for(runnable.ainvoke(messages), timeout=timeout)
        except asyncio.CancelledError:
            if reservation is not None and budget is not None:
                await budget.settle(reservation, outcome="cancelled", possibly_billed=sent)
            raise
        except Exception as exc:
            if reservation is not None and budget is not None:
                category = _classify_error(exc)
                await budget.settle(
                    reservation,
                    outcome="error" if sent else "not_sent",
                    possibly_billed=sent,
                    error_category=category,
                    retry_after_s=_retry_after_seconds(exc),
                )
            raise
        usage = _extract_token_usage(response)
        if reservation is not None and budget is not None:
            await budget.settle(reservation, usage=usage, outcome="ok")
        return _content_to_text(response.content), usage

    async def _repair_json(
        self, agent: str, candidate: RuntimeModelCandidate, text: str
    ) -> tuple[str, dict[str, int]]:
        """用同一候选模型修复一次非法 JSON;这也是独立登记的预算请求。"""
        messages = [
            SystemMessage(
                content=(
                    "你是 JSON 修复器。只输出一个合法 JSON 对象,不要 markdown,不要解释。"
                    f"目标 agent: {agent}"
                )
            ),
            HumanMessage(content=text),
        ]
        return await self._ainvoke_text(
            candidate, messages, json_mode=True, logical_task=agent, attempt_kind="repair"
        )


class ContextLengthExceeded(ValueError):
    """完整请求连同输出预留和安全余量超出本地上下文窗口。"""


def _messages(system_prompt: str, payload: dict[str, Any]) -> list[Any]:
    return [SystemMessage(content=system_prompt), HumanMessage(content=stable_json(payload))]


class InvalidStructuredOutput(ValueError):
    """JSON 已可解析但根类型不符合对象契约,不应额外消耗 JSON 修复请求。"""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """解析普通、围栏或带说明文字的 JSON 对象;找不到可解析对象时返回 None。

    完整合法 JSON 的根为数组或标量时抛 InvalidStructuredOutput,避免用修复
    掩盖错误的结构化输出。字段级业务约束由调用方的 output_validator 校验。
    """
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = "\n".join(candidate.splitlines()[1:-1]).strip()
    try:
        data = json.loads(candidate)
    except (TypeError, ValueError):
        pass
    else:
        if not isinstance(data, dict):
            raise InvalidStructuredOutput("structured output must be an object")
        return data
    match = _JSON_RE.search(text or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        raw = content
    elif isinstance(content, list):
        parts = [part.get("text") if isinstance(part, dict) else part for part in content]
        raw = " ".join(part for part in parts if isinstance(part, str))
    else:
        raw = ""
    return _strip_think(raw)


def _extract_token_usage(response: Any) -> dict[str, int]:
    raw = getattr(response, "usage_metadata", None)
    if not isinstance(raw, dict):
        metadata = getattr(response, "response_metadata", None)
        raw = metadata.get("token_usage") if isinstance(metadata, dict) else None
    if not isinstance(raw, dict):
        return {}
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }
    usage: dict[str, int] = {}
    for target, candidates in aliases.items():
        for key in candidates:
            value = raw.get(key)
            if isinstance(value, int) and value >= 0:
                usage[target] = value
                break
    if "total_tokens" not in usage and usage:
        usage["total_tokens"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return usage


def _merge_token_usage(total: dict[str, int], addition: dict[str, int]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if key in addition:
            total[key] = total.get(key, 0) + addition[key]


def _strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


def _classify_error(exc: BaseException) -> str:
    from .collaboration import InputSuperseded

    if isinstance(exc, InputSuperseded):
        return "input_superseded"
    if isinstance(exc, ContextLengthExceeded):
        return "context_length"
    raw_error = (str(exc) + " " + json.dumps(getattr(exc, "body", None), default=str)).lower()
    if any(
        marker in raw_error
        for marker in (
            "context_length_exceeded",
            "context_length_error",
            "prompt_too_long",
            "too_many_tokens",
            "max_context_length",
            "maximum context length",
            "context window exceeded",
            "context window is too small",
            "input is too long",
            "prompt is too long",
            "exceeds the context window",
        )
    ):
        return "context_length"
    if isinstance(exc, BudgetPersistenceError):
        return "budget_persistence"
    if isinstance(exc, BudgetExceeded):
        return "budget"
    if isinstance(exc, TimeoutError | asyncio.TimeoutError):
        return "timeout"
    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return "rate_limit"
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return "auth"
        if isinstance(exc, openai.NotFoundError):
            return "model_not_found"
        if isinstance(exc, openai.BadRequestError):
            # content-policy 拒绝以 400 BadRequestError 返回,部分携带 code。
            body = getattr(exc, "body", None)
            raw = json.dumps(body, ensure_ascii=False, default=str) if body is not None else ""
            if "content_policy_violation" in raw or "content_policy" in raw:
                return "content_policy"
            return "llm_error"
        if isinstance(exc, openai.InternalServerError):
            return "http_5xx"
        if isinstance(exc, openai.APITimeoutError):
            return "timeout"
        if isinstance(exc, openai.APIConnectionError):
            return "network"
    except ImportError:
        pass
    return "llm_error"


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Read Retry-After from common SDK exceptions without exposing response data."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("retry-after")
    except (AttributeError, TypeError):
        return None
    if raw is None:
        return None
    return parse_retry_after(raw)


def _safe_error_message(category: str) -> str:
    """Do not persist provider exception bodies, which can include request details."""
    labels = {
        "budget_persistence": "Budget checkpoint failed; execution stopped",
        "model_not_found": "Configured model is unavailable",
        "schema_invalid": "Model output does not match the required schema",
        "empty_response": "Model returned empty output",
        "input_superseded": "New input is pending; no request was sent",
        "context_length": "LLM request exceeds the context window",
        "budget": "LLM request was not sent because the run budget was exhausted",
        "timeout": "LLM request timed out",
        "rate_limit": "LLM provider rate limited the request",
        "auth": "LLM provider authentication failed",
        "network": "LLM network request failed",
        "http_5xx": "LLM provider returned a server error",
        "content_policy": "LLM provider rejected the request",
    }
    return labels.get(category, "LLM request failed")


def _budget_snapshot() -> dict[str, Any] | None:
    budget = current_run_budget()
    return budget.snapshot() if budget is not None else None


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
