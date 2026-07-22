"""agent/action_selector.py 单元测试：覆盖 tool_use block -> LLMAction 的
纯解析逻辑（_build_llm_action / _get_str_field / _extract_tool_use），
不涉及真实 LLM 调用。
"""

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from agent.action_selector import _build_llm_action, _extract_tool_use, _get_str_field, _format_observation
from agent.types import ObserveResult


def _tool_use(name: str, **input_fields) -> ToolUseBlock:
    return ToolUseBlock(type="tool_use", id="tool_1", name=name, input=input_fields)


# ---------- _get_str_field ----------

def test_get_str_field_returns_none_when_missing():
    assert _get_str_field({}, "selector") is None


def test_get_str_field_returns_value_when_str():
    assert _get_str_field({"selector": "#a"}, "selector") == "#a"


def test_get_str_field_raises_on_non_str_type():
    with pytest.raises(ValueError):
        _get_str_field({"selector": 123}, "selector")


# ---------- _extract_tool_use ----------

def _make_message(content) -> Message:
    return Message(
        id="msg_1",
        content=content,
        model="claude-sonnet-4-6",
        role="assistant",
        stop_reason="tool_use",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def test_extract_tool_use_finds_tool_use_block():
    block = _tool_use("click", selector="#a", reason="点击")
    message = _make_message([TextBlock(type="text", text="thinking..."), block])
    assert _extract_tool_use(message) is block


def test_extract_tool_use_returns_none_when_absent():
    message = _make_message([TextBlock(type="text", text="just text")])
    assert _extract_tool_use(message) is None


def test_extract_tool_use_returns_none_for_empty_content():
    message = _make_message([])
    assert _extract_tool_use(message) is None


# ---------- _build_llm_action ----------

def test_build_llm_action_click_with_selector():
    action = _build_llm_action(_tool_use("click", selector="#submit", reason="提交"))
    assert action["action"] == "click"
    assert action["selector"] == "#submit"
    assert action["reason"] == "提交"


def test_build_llm_action_click_with_text_only():
    action = _build_llm_action(_tool_use("click", text="登录", reason="登录"))
    assert action["action"] == "click"
    assert action["text"] == "登录"


def test_build_llm_action_click_without_selector_or_text_raises():
    with pytest.raises(ValueError):
        _build_llm_action(_tool_use("click", reason="点击"))


def test_build_llm_action_type_requires_selector_and_text():
    action = _build_llm_action(_tool_use("type", selector="#input", text="hello", reason="输入"))
    assert action["selector"] == "#input"
    assert action["text"] == "hello"

    with pytest.raises(ValueError):
        _build_llm_action(_tool_use("type", selector="#input", reason="输入"))

    with pytest.raises(ValueError):
        _build_llm_action(_tool_use("type", text="hello", reason="输入"))


def test_build_llm_action_scroll_validates_direction():
    action = _build_llm_action(_tool_use("scroll", direction="down", reason="滚动"))
    assert action["value"] == "down"

    with pytest.raises(ValueError):
        _build_llm_action(_tool_use("scroll", direction="sideways", reason="滚动"))


def test_build_llm_action_extract_requires_instruction():
    action = _build_llm_action(_tool_use("extract", instruction="抓取标题", reason="抽取"))
    assert action["value"] == "抓取标题"

    with pytest.raises(ValueError):
        _build_llm_action(_tool_use("extract", reason="抽取"))


def test_build_llm_action_select_requires_selector_and_value():
    action = _build_llm_action(
        _tool_use("select", selector="#lang", value="English", reason="切换语言")
    )
    assert action["selector"] == "#lang"
    assert action["value"] == "English"

    with pytest.raises(ValueError):
        _build_llm_action(_tool_use("select", selector="#lang", reason="切换语言"))


def test_build_llm_action_done_requires_value():
    action = _build_llm_action(_tool_use("done", value="任务完成", reason="收尾"))
    assert action["value"] == "任务完成"

    with pytest.raises(ValueError):
        _build_llm_action(_tool_use("done", reason="收尾"))


def test_build_llm_action_screenshot_only_needs_reason():
    action = _build_llm_action(_tool_use("screenshot", reason="截图记录"))
    assert action["action"] == "screenshot"
    assert action["selector"] is None
    assert action["text"] is None
    assert action["value"] is None


def test_build_llm_action_unknown_tool_name_raises():
    with pytest.raises(ValueError):
        _build_llm_action(_tool_use("unknown_action", reason="?"))


def test_build_llm_action_missing_reason_raises():
    with pytest.raises(ValueError):
        _build_llm_action(_tool_use("screenshot"))


def test_build_llm_action_blank_reason_raises():
    with pytest.raises(ValueError):
        _build_llm_action(_tool_use("screenshot", reason="   "))


# ---------- _format_observation ----------

def test_format_observation_lists_elements_with_selectors():
    obs: ObserveResult = {
        "url": "https://x",
        "title": "Test",
        "visible_text_summary": "页面说明：请选择提交操作。",
        "text_hash": "h1",
        "interactive_elements": [
            {"role": "button", "name": "提交", "selector": "css=#submit", "href": None},
        ],
        "screenshot_path": "/tmp/step-001.png",
    }
    formatted = _format_observation(obs)
    assert "css=#submit" in formatted
    assert "提交" in formatted
    assert "页面说明：请选择提交操作。" in formatted
    assert '<untrusted_page_content format="json">' in formatted


def test_format_observation_handles_no_elements():
    obs: ObserveResult = {
        "url": "https://x",
        "title": "Test",
        "visible_text_summary": "",
        "text_hash": "h1",
        "interactive_elements": [],
        "screenshot_path": "/tmp/step-001.png",
    }
    formatted = _format_observation(obs)
    assert '<untrusted_page_content format="json">' in formatted
    assert '"interactive_elements": []' in formatted
