"""agent/llm_client.py 单元测试：覆盖统一重试骨架——网络层 APIError/
RateLimitError 重试、parse_response 触发的 LLMOutputRetry 重试、
成功路径、以及耗尽重试后包装成 LLMError。全部 mock 底层 Anthropic
客户端，不发起真实网络请求。
"""

from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import pytest
from anthropic.types import Message, TextBlock, Usage

from agent.config import AgentConfig
from agent.exceptions import LLMError
from agent.llm_client import LLMClient, LLMOutputRetry


def _make_message(text: str = "ok") -> Message:
    return Message(
        id="msg_1",
        content=[TextBlock(type="text", text=text)],
        model="claude-sonnet-4-6",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _first_text(message: Message) -> str:
    """从 Message.content 里取出第一个文本块的正文。"""
    block = message.content[0]
    assert isinstance(block, TextBlock)
    return block.text


def _rate_limit_error() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def _api_error() -> anthropic.APIError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(message="connection failed", request=request)


@pytest.fixture(autouse=True)
def _reset_shared_client():
    """每个测试前重置模块级单例，避免测试间通过全局状态互相污染。"""
    import agent.llm_client as llm_client_module

    llm_client_module._shared_anthropic_client = None
    yield
    llm_client_module._shared_anthropic_client = None


@pytest.fixture
def mock_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AsyncMock()
    with patch("anthropic.AsyncAnthropic", return_value=client):
        yield client


async def test_call_with_retry_success_first_attempt(mock_client):
    mock_client.messages.create = AsyncMock(return_value=_make_message("hello"))
    config = AgentConfig(llm_retry=3)
    llm = LLMClient(config)

    result = await llm.call_with_retry(
        caller_name="test",
        parse_response=_first_text,
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[],
    )
    assert result == "hello"
    assert mock_client.messages.create.await_count == 1


async def test_call_with_retry_retries_on_api_error_then_succeeds(mock_client):
    mock_client.messages.create = AsyncMock(
        side_effect=[_api_error(), _make_message("recovered")]
    )
    config = AgentConfig(llm_retry=3)
    llm = LLMClient(config)

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await llm.call_with_retry(
            caller_name="test",
            parse_response=_first_text,
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[],
        )
    assert result == "recovered"
    assert mock_client.messages.create.await_count == 2


async def test_call_with_retry_retries_on_rate_limit_with_fixed_delay(mock_client):
    mock_client.messages.create = AsyncMock(
        side_effect=[_rate_limit_error(), _make_message("ok")]
    )
    config = AgentConfig(llm_retry=3, rate_limit_delay=42)
    llm = LLMClient(config)

    with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
        result = await llm.call_with_retry(
            caller_name="test",
            parse_response=_first_text,
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[],
        )
    assert result == "ok"
    mock_sleep.assert_awaited_once_with(42)


async def test_call_with_retry_retries_on_llm_output_retry(mock_client):
    mock_client.messages.create = AsyncMock(
        side_effect=[_make_message("bad"), _make_message("good")]
    )
    config = AgentConfig(llm_retry=3)
    llm = LLMClient(config)

    calls = {"n": 0}

    def _parse(message: Message) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMOutputRetry("输出格式不对")
        return _first_text(message)

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await llm.call_with_retry(
            caller_name="test",
            parse_response=_parse,
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[],
        )
    assert result == "good"


async def test_call_with_retry_exhausts_and_raises_llm_error(mock_client):
    mock_client.messages.create = AsyncMock(side_effect=_api_error())
    config = AgentConfig(llm_retry=2)
    llm = LLMClient(config)

    with patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(LLMError) as exc_info:
            await llm.call_with_retry(
                caller_name="MyCaller",
                parse_response=_first_text,
                model="claude-sonnet-4-6",
                max_tokens=10,
                messages=[],
            )
    assert exc_info.value.stage == "request"
    assert exc_info.value.retry_count == 2
    assert "MyCaller" in exc_info.value.message
    assert mock_client.messages.create.await_count == 2


async def test_call_with_retry_rejects_stream_kwarg(mock_client):
    config = AgentConfig(llm_retry=1)
    llm = LLMClient(config)
    with pytest.raises(AssertionError):
        await llm.call_with_retry(
            caller_name="test",
            parse_response=lambda m: m,
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[],
            stream=True,
        )


def test_get_shared_client_raises_llm_error_without_api_key(monkeypatch):
    import agent.llm_client as llm_client_module

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm_client_module._shared_anthropic_client = None

    with pytest.raises(LLMError):
        llm_client_module._get_shared_anthropic_client()
