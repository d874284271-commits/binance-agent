"""AgentRuntime:统一 LLM 调用层的超时、fallback、JSON repair 与安全降级测试。"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from binance_agent.agents.runtime import AgentRuntime


class _ScriptLLM:
    def __init__(self, script: list[Any], *, delay: float = 0.0) -> None:
        self.script = list(script)
        self.delay = delay
        self.calls: list[Any] = []

    def bind(self, **kwargs: Any) -> _ScriptLLM:
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.calls.append(messages)
        if self.delay:
            await asyncio.sleep(self.delay)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, AIMessage):
            return item
        return AIMessage(content=str(item))


async def test_runtime_timeout_falls_back_to_next_model() -> None:
    primary = _ScriptLLM(['{"stance":"bullish","confidence":0.9}'], delay=0.05)
    fallback = _ScriptLLM(['{"stance":"neutral","confidence":0.4}'])
    runtime = AgentRuntime(
        cast(BaseChatModel, primary),
        fallbacks=[cast(BaseChatModel, fallback)],
        timeout_s=0.01,
        max_retries=0,
    )

    result = await runtime.invoke_structured(
        "technical",
        "system",
        {"symbol": "BTCUSDT"},
        fallback_data={"stance": "neutral", "confidence": 0.2},
    )

    assert result.data == {"stance": "neutral", "confidence": 0.4}
    assert result.ok
    assert result.model_index == 1
    assert any(error.category == "timeout" for error in result.errors)
    assert result.latency_ms >= 0


async def test_runtime_repairs_bad_json_with_second_call() -> None:
    llm = _ScriptLLM(["这不是 JSON", '{"stance":"bearish","confidence":0.6}'])
    runtime = AgentRuntime(cast(BaseChatModel, llm), timeout_s=1.0, max_retries=0)

    result = await runtime.invoke_structured(
        "technical",
        "system",
        {"symbol": "BTCUSDT"},
        fallback_data={"stance": "neutral", "confidence": 0.2},
    )

    assert result.data == {"stance": "bearish", "confidence": 0.6}
    assert result.ok
    assert result.repaired
    assert len(llm.calls) == 2


async def test_runtime_uses_safe_fallback_when_repair_still_fails() -> None:
    llm = _ScriptLLM(["bad json", "still bad"])
    runtime = AgentRuntime(cast(BaseChatModel, llm), timeout_s=1.0, max_retries=0)

    result = await runtime.invoke_structured(
        "technical",
        "system",
        {"symbol": "BTCUSDT"},
        fallback_data={"stance": "neutral", "confidence": 0.2},
    )

    assert result.data == {"stance": "neutral", "confidence": 0.2}
    assert not result.ok
    assert result.error_category == "json_parse"


async def test_runtime_does_not_swallow_cancellation() -> None:
    llm = _ScriptLLM([asyncio.CancelledError()])
    runtime = AgentRuntime(cast(BaseChatModel, llm), timeout_s=1.0, max_retries=0)

    try:
        await runtime.invoke_structured(
            "technical",
            "system",
            {"symbol": "BTCUSDT"},
            fallback_data={"stance": "neutral", "confidence": 0.2},
        )
    except asyncio.CancelledError:
        return
    raise AssertionError("CancelledError should propagate")


async def test_runtime_records_endpoint_and_aggregates_token_usage_across_repair() -> None:
    llm = _ScriptLLM(
        [
            AIMessage(
                content="bad json",
                usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            ),
            AIMessage(
                content='{"stance":"neutral"}',
                usage_metadata={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            ),
        ]
    )
    runtime = AgentRuntime(cast(BaseChatModel, llm), timeout_s=1.0, max_retries=0)

    result = await runtime.invoke_structured(
        "technical",
        "system",
        {"symbol": "BTCUSDT"},
        fallback_data={"stance": "neutral"},
    )

    assert result.endpoint == "primary"
    assert result.token_usage == {"input_tokens": 14, "output_tokens": 5, "total_tokens": 19}


async def _openai_error(cls: type, *, body: Any = None) -> BaseException:
    """构造一个可用的 openai 异常实例(需要真实 httpx response 提供 request)。"""
    import httpx

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=200, request=request, json=body or {})
    if cls.__name__ == "APIConnectionError":
        return cls(request=request)
    return cls("boom", response=response, body=body)


async def _runtime_error_category(exc: BaseException) -> str:
    """构造运行时并注入一个失败脚本,返回首个错误的 category。"""
    llm = _ScriptLLM([exc])
    runtime = AgentRuntime(cast(BaseChatModel, llm), timeout_s=1.0, max_retries=0)
    result = await runtime.invoke_structured(
        "technical",
        "system",
        {"symbol": "BTCUSDT"},
        fallback_data={"stance": "neutral", "confidence": 0.2},
    )
    assert result.errors
    return result.errors[0].category


async def test_runtime_classifies_openai_rate_limit() -> None:
    import openai

    exc = await _openai_error(openai.RateLimitError)
    assert await _runtime_error_category(exc) == "rate_limit"


async def test_runtime_classifies_openai_auth_error() -> None:
    import openai

    exc = await _openai_error(openai.AuthenticationError)
    assert await _runtime_error_category(exc) == "auth"


async def test_runtime_classifies_openai_permission_denied() -> None:
    import openai

    exc = await _openai_error(openai.PermissionDeniedError)
    assert await _runtime_error_category(exc) == "auth"


async def test_runtime_classifies_openai_5xx() -> None:
    import openai

    exc = await _openai_error(openai.InternalServerError)
    assert await _runtime_error_category(exc) == "http_5xx"


async def test_runtime_classifies_openai_content_policy() -> None:
    import openai

    exc = await _openai_error(
        openai.BadRequestError, body={"error": {"code": "content_policy_violation"}}
    )
    assert await _runtime_error_category(exc) == "content_policy"


async def test_runtime_classifies_openai_network_error() -> None:
    import openai

    exc = await _openai_error(openai.APIConnectionError)
    assert await _runtime_error_category(exc) == "network"
