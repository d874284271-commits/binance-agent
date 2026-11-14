"""AgentRuntime 结构化输出(json mode)绑定测试。

验证:端点开启 structured_output 时,仅 invoke_structured/_repair_json 路径绑定
response_format=json_object;invoke_text(散文)永不绑定;关闭时只绑定输出上限。
用带 .bind() 追踪的 fake。
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from binance_agent.agents.runtime import AgentRuntime, RuntimeModelCandidate


class _BindTrackingLLM:
    """记录 .bind() 调用的 fake;bind 返回自身(忽略参数,模拟 FakeListChatModel 行为)。"""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.bind_calls: list[dict[str, Any]] = []
        self.calls = 0

    def bind(self, **kwargs: Any) -> _BindTrackingLLM:
        self.bind_calls.append(kwargs)
        return self

    async def ainvoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return AIMessage(content=str(item))


async def test_structured_output_binds_json_mode_on_invoke_structured() -> None:
    llm = _BindTrackingLLM(['{"stance":"bullish","confidence":0.8}'])
    runtime = AgentRuntime(cast(BaseChatModel, llm), timeout_s=1.0, structured_output=True)

    result = await runtime.invoke_structured(
        "technical", "system", {"symbol": "BTCUSDT"}, fallback_data={"stance": "neutral"}
    )

    assert result.ok
    assert llm.bind_calls == [{"max_tokens": 4096, "response_format": {"type": "json_object"}}]


async def test_structured_output_disabled_binds_only_output_limit() -> None:
    llm = _BindTrackingLLM(['{"stance":"bullish","confidence":0.8}'])
    runtime = AgentRuntime(cast(BaseChatModel, llm), timeout_s=1.0, structured_output=False)

    result = await runtime.invoke_structured(
        "technical", "system", {"symbol": "BTCUSDT"}, fallback_data={"stance": "neutral"}
    )

    assert result.ok
    assert llm.bind_calls == [{"max_tokens": 4096}]


async def test_invoke_text_binds_only_output_limit_even_when_json_enabled() -> None:
    # research_manager 走 invoke_text 返回散文,绝不能被逼成 json_object
    llm = _BindTrackingLLM(["证据不足,倾向观望。"])
    runtime = AgentRuntime(cast(BaseChatModel, llm), timeout_s=1.0, structured_output=True)

    result = await runtime.invoke_text(
        "research_manager", "system", {"symbol": "BTCUSDT"}, fallback_text="降级"
    )

    assert result.ok
    assert result.text == "证据不足,倾向观望。"
    assert llm.bind_calls == [{"max_tokens": 4096}]


async def test_repair_path_also_binds_json_mode() -> None:
    # 首次非法 JSON → repair 二次调用;两次都在 structured 路径,均应绑定
    llm = _BindTrackingLLM(["这不是 JSON", '{"stance":"bearish","confidence":0.6}'])
    runtime = AgentRuntime(cast(BaseChatModel, llm), timeout_s=1.0, structured_output=True)

    result = await runtime.invoke_structured(
        "technical", "system", {"symbol": "BTCUSDT"}, fallback_data={"stance": "neutral"}
    )

    assert result.repaired
    assert llm.bind_calls == [
        {"max_tokens": 4096, "response_format": {"type": "json_object"}},
        {"max_tokens": 4096, "response_format": {"type": "json_object"}},
    ]


async def test_fallback_uses_its_own_structured_output_capability() -> None:
    primary = _BindTrackingLLM(["这不是 JSON", "仍然不是 JSON"])
    fallback = _BindTrackingLLM(['{"stance":"neutral","confidence":0.4}'])
    runtime = AgentRuntime(
        candidates=[
            RuntimeModelCandidate(
                model=cast(BaseChatModel, primary),
                endpoint="json-capable",
                timeout_s=1.0,
                max_retries=0,
                structured_output=True,
            ),
            RuntimeModelCandidate(
                model=cast(BaseChatModel, fallback),
                endpoint="plain-only",
                timeout_s=1.0,
                max_retries=0,
                structured_output=False,
            ),
        ]
    )

    result = await runtime.invoke_structured(
        "technical", "system", {"symbol": "BTCUSDT"}, fallback_data={"stance": "neutral"}
    )

    assert result.ok
    assert result.model_index == 1
    assert primary.bind_calls == [
        {"max_tokens": 4096, "response_format": {"type": "json_object"}},
        {"max_tokens": 4096, "response_format": {"type": "json_object"}},
    ]
    assert fallback.bind_calls == [{"max_tokens": 4096}]


async def test_fallback_uses_its_own_retry_budget() -> None:
    primary = _BindTrackingLLM([RuntimeError("primary unavailable")])
    fallback = _BindTrackingLLM(
        [RuntimeError("temporary fallback failure"), '{"stance":"neutral"}']
    )
    runtime = AgentRuntime(
        candidates=[
            RuntimeModelCandidate(
                model=cast(BaseChatModel, primary),
                endpoint="primary",
                timeout_s=1.0,
                max_retries=0,
            ),
            RuntimeModelCandidate(
                model=cast(BaseChatModel, fallback),
                endpoint="fallback",
                timeout_s=1.0,
                max_retries=1,
            ),
        ]
    )

    result = await runtime.invoke_structured(
        "technical", "system", {"symbol": "BTCUSDT"}, fallback_data={"stance": "neutral"}
    )

    assert result.ok
    assert result.model_index == 1
    assert result.attempts == 3
    assert primary.calls == 1
    assert fallback.calls == 2
