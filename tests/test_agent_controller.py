"""agent/agent_controller.py 单元测试：只覆盖模块级纯函数（动作签名、
结果摘要、去重判定、history 拼装等），不实例化 AgentController 本体
（其依赖真实 Playwright + LLM 调用，属于集成测试范畴）。
"""

import json

from agent.agent_controller import (
    _MAX_HISTORY_STEPS,
    AgentController,
    _action_signature,
    _extract_page_key,
    _find_element_role,
    _strip_structural_prefix,
    _summarize_result,
    _unwrap_extract_data,
)
from agent.types import LLMAction, ObserveResult, ToolResult

# _append_history 是 AgentController 的静态方法，不是模块级函数
_append_history = AgentController._append_history


# ---------- _action_signature ----------

def test_action_signature_extracts_four_fields():
    action: LLMAction = {"action": "click", "selector": "#a", "text": None, "value": None, "reason": "x"}
    assert _action_signature(action) == ("click", "#a", None, None)


def test_action_signature_equal_for_identical_actions():
    a1: LLMAction = {"action": "click", "selector": "#a", "text": None, "value": None, "reason": "x"}
    a2: LLMAction = {"action": "click", "selector": "#a", "text": None, "value": None, "reason": "x"}
    assert _action_signature(a1) == _action_signature(a2)


def test_action_signature_differs_when_selector_changes():
    a1: LLMAction = {"action": "click", "selector": "#a", "text": None, "value": None, "reason": "x"}
    a2: LLMAction = {"action": "click", "selector": "#b", "text": None, "value": None, "reason": "x"}
    assert _action_signature(a1) != _action_signature(a2)


# ---------- _summarize_result ----------

def test_summarize_result_failure_includes_error_msg():
    action: LLMAction = {"action": "click", "selector": None, "text": None, "value": None, "reason": "x"}
    result: ToolResult = {"success": False, "page_changed": False, "output": None, "error_msg": "元素未找到"}
    assert _summarize_result(action, result) == "失败（元素未找到）"


def test_summarize_result_failure_without_error_msg():
    action: LLMAction = {"action": "click", "selector": None, "text": None, "value": None, "reason": "x"}
    result: ToolResult = {"success": False, "page_changed": False, "output": None, "error_msg": None}
    assert _summarize_result(action, result) == "失败（未知错误）"


def test_summarize_result_extract_with_output():
    action: LLMAction = {"action": "extract", "selector": None, "text": None, "value": None, "reason": "x"}
    result: ToolResult = {"success": True, "page_changed": False, "output": '{"title": "hi"}', "error_msg": None}
    summary = _summarize_result(action, result)
    assert "extract 已返回数据" in summary
    assert '"title"' in summary


def test_summarize_result_extract_without_output():
    action: LLMAction = {"action": "extract", "selector": None, "text": None, "value": None, "reason": "x"}
    result: ToolResult = {"success": True, "page_changed": False, "output": None, "error_msg": None}
    assert "未返回任何数据" in _summarize_result(action, result)


def test_summarize_result_extract_truncates_long_output():
    action: LLMAction = {"action": "extract", "selector": None, "text": None, "value": None, "reason": "x"}
    long_output = "x" * 5000
    result: ToolResult = {"success": True, "page_changed": False, "output": long_output, "error_msg": None}
    summary = _summarize_result(action, result)
    assert "已截断" in summary


def test_summarize_result_screenshot():
    action: LLMAction = {"action": "screenshot", "selector": None, "text": None, "value": None, "reason": "x"}
    result: ToolResult = {"success": True, "page_changed": False, "output": "/tmp/step-001.png", "error_msg": None}
    assert "/tmp/step-001.png" in _summarize_result(action, result)


def test_summarize_result_click_success():
    action: LLMAction = {"action": "click", "selector": None, "text": None, "value": None, "reason": "x"}
    result: ToolResult = {"success": True, "page_changed": False, "output": None, "error_msg": None}
    assert _summarize_result(action, result) == "成功"


# ---------- _strip_structural_prefix ----------

def test_strip_structural_prefix_removes_heading_marker():
    assert _strip_structural_prefix("# 标题内容") == "标题内容"


def test_strip_structural_prefix_removes_list_marker():
    assert _strip_structural_prefix("- 列表项") == "列表项"


def test_strip_structural_prefix_removes_button_marker():
    assert _strip_structural_prefix("[按钮] 提交") == "提交"


def test_strip_structural_prefix_only_strips_once():
    # observer 每个节点只加一层前缀，不应循环剥离导致误伤正文
    assert _strip_structural_prefix("- - 说明") == "- 说明"


def test_strip_structural_prefix_no_marker_unchanged():
    assert _strip_structural_prefix("普通文本") == "普通文本"


# ---------- _find_element_role ----------

def test_find_element_role_matches_selector():
    obs: ObserveResult = {
        "url": "https://x",
        "title": "Test",
        "visible_text_summary": "",
        "text_hash": "h1",
        "interactive_elements": [
            {"role": "select", "name": "语言", "selector": "css=#lang", "href": None},
            {"role": "button", "name": "提交", "selector": "css=#submit", "href": None},
        ],
        "screenshot_path": "/tmp/step-001.png",
    }
    assert _find_element_role(obs, "css=#lang") == "select"
    assert _find_element_role(obs, "css=#submit") == "button"


def test_find_element_role_returns_none_when_not_found():
    obs: ObserveResult = {
        "url": "https://x",
        "title": "Test",
        "visible_text_summary": "",
        "text_hash": "h1",
        "interactive_elements": [],
        "screenshot_path": "/tmp/step-001.png",
    }
    assert _find_element_role(obs, "css=#missing") is None


def test_find_element_role_returns_none_for_empty_selector():
    obs: ObserveResult = {
        "url": "https://x",
        "title": "Test",
        "visible_text_summary": "",
        "text_hash": "h1",
        "interactive_elements": [{"role": "button", "name": "x", "selector": "", "href": None}],
        "screenshot_path": "/tmp/step-001.png",
    }
    assert _find_element_role(obs, None) is None
    assert _find_element_role(obs, "") is None


# ---------- _extract_page_key ----------

def test_extract_page_key_uses_url_and_text_hash():
    obs: ObserveResult = {
        "url": "https://x",
        "title": "Test",
        "visible_text_summary": "",
        "text_hash": "abc123",
        "interactive_elements": [],
        "screenshot_path": "/tmp/step-001.png",
    }
    assert _extract_page_key(obs) == ("https://x", "abc123")


# ---------- _unwrap_extract_data ----------

def test_unwrap_extract_data_extracts_data_field():
    output = json.dumps({"url": "https://x", "data": {"title": "hi"}})
    assert _unwrap_extract_data(output) == {"title": "hi"}


def test_unwrap_extract_data_none_for_empty_output():
    assert _unwrap_extract_data(None) is None
    assert _unwrap_extract_data("") is None


def test_unwrap_extract_data_none_for_invalid_json():
    assert _unwrap_extract_data("not json") is None


def test_unwrap_extract_data_none_when_data_key_missing():
    assert _unwrap_extract_data(json.dumps({"url": "https://x"})) is None


def test_unwrap_extract_data_none_when_not_a_dict():
    assert _unwrap_extract_data(json.dumps([1, 2, 3])) is None


# ---------- _append_history ----------

def test_append_history_adds_user_and_assistant_pair():
    history: list[dict] = []
    action: LLMAction = {"action": "click", "selector": "#a", "text": None, "value": None, "reason": "点击提交按钮"}
    result: ToolResult = {"success": True, "page_changed": False, "output": None, "error_msg": None}
    _append_history(history, 0, "点击提交按钮完成表单", action, result, stagnant_streak=0, content_changed_from_baseline=False)

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert "点击提交按钮完成表单" in history[1]["content"]


def test_append_history_truncates_to_max_history_steps():
    history: list[dict] = []
    action: LLMAction = {"action": "click", "selector": "#a", "text": None, "value": None, "reason": "点击"}
    result: ToolResult = {"success": True, "page_changed": False, "output": None, "error_msg": None}
    for step in range(_MAX_HISTORY_STEPS + 5):
        _append_history(history, step, "计划", action, result, stagnant_streak=0, content_changed_from_baseline=False)

    assert len(history) == 2 * _MAX_HISTORY_STEPS


def test_append_history_stagnation_nudge_when_no_real_change():
    history: list[dict] = []
    action: LLMAction = {"action": "click", "selector": "#a", "text": None, "value": None, "reason": "点击"}
    result: ToolResult = {"success": True, "page_changed": False, "output": None, "error_msg": None}
    _append_history(history, 3, "计划", action, result, stagnant_streak=3, content_changed_from_baseline=False)
    assert "没有产生任何新进展" in history[0]["content"]


def test_append_history_stagnation_nudge_when_content_actually_changed():
    history: list[dict] = []
    action: LLMAction = {"action": "click", "selector": "#a", "text": None, "value": None, "reason": "点击"}
    result: ToolResult = {"success": True, "page_changed": False, "output": None, "error_msg": None}
    _append_history(history, 3, "计划", action, result, stagnant_streak=3, content_changed_from_baseline=True)
    assert "已经真实生效过" in history[0]["content"]


def test_append_history_maintains_role_alternation():
    history: list[dict] = []
    action: LLMAction = {"action": "click", "selector": "#a", "text": None, "value": None, "reason": "点击"}
    result: ToolResult = {"success": True, "page_changed": False, "output": None, "error_msg": None}
    for step in range(6):
        _append_history(history, step, "计划", action, result, stagnant_streak=0, content_changed_from_baseline=False)

    roles = [entry["role"] for entry in history]
    for i, role in enumerate(roles):
        expected = "user" if i % 2 == 0 else "assistant"
        assert role == expected
