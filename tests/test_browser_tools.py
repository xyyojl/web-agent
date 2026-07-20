"""agent/browser_tools.py 单元测试：
- 纯正则/解析逻辑（敏感字段检测、extract 响应 JSON 解析）用真实输入直接测。
- 依赖 Playwright Page 的函数（click 三级降级、select、scroll、type、
  登录页检测）用轻量 Mock/Fake Page 对象模拟，不启动真实浏览器。
"""

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from anthropic.types import Message, TextBlock, Usage
from playwright.async_api import Error as PlaywrightError

from agent.browser_tools import (
    _check_sensitive,
    _detect_login_page,
    _parse_extract_response,
    browser_click,
    browser_scroll,
    browser_select,
    browser_type,
)
from agent.exceptions import SafetyError
from agent.llm_client import LLMOutputRetry


# ---------- _check_sensitive ----------

@pytest.mark.parametrize(
    "selector",
    [
        "css=#password",
        "css=input[name='passwd']",
        "css=#credit-card",
        "css=#creditcard",
        "css=#cvv",
        "css=#ssn",
        "css=#bank_account",
        "css=#PASSWORD",  # 大小写不敏感
    ],
)
def test_check_sensitive_raises_for_sensitive_selectors(selector):
    with pytest.raises(SafetyError) as exc_info:
        _check_sensitive(selector)
    assert exc_info.value.trigger == "sensitive_field"
    assert exc_info.value.selector == selector


@pytest.mark.parametrize("selector", ["css=#username", "css=#email", "text=提交", "css=#search-box"])
def test_check_sensitive_passes_for_normal_selectors(selector):
    _check_sensitive(selector)  # 不应抛出


def test_check_sensitive_empty_selector_does_not_raise():
    _check_sensitive("")


# ---------- _detect_login_page ----------

def _fake_page(counts: dict[str, int]):
    """构造一个假 Page：page.locator(signal).count() 按 counts 映射返回，
    未在映射中的 signal 抛 PlaywrightError（模拟语法不兼容被跳过）。
    """
    page = MagicMock()

    def _locator(signal):
        locator = MagicMock()
        if signal in counts:
            locator.count = AsyncMock(return_value=counts[signal])
        else:
            locator.count = AsyncMock(side_effect=PlaywrightError("unsupported selector"))
        return locator

    page.locator.side_effect = _locator
    return page


async def test_detect_login_page_returns_first_matching_signal():
    page = _fake_page({"input[type=password]": 1})
    signal = await _detect_login_page(page)
    assert signal == "input[type=password]"


async def test_detect_login_page_returns_none_when_no_signal_matches():
    page = _fake_page({
        "input[type=password]": 0,
        'form[action*="login"]': 0,
        'button:has-text("登录")': 0,
        'button:has-text("Sign in")': 0,
    })
    assert await _detect_login_page(page) is None


async def test_detect_login_page_skips_signals_that_raise_and_still_finds_match():
    # input[type=password] 不在映射内会抛 PlaywrightError，应被跳过继续检查下一个信号
    page = _fake_page({'form[action*="login"]': 1})
    signal = await _detect_login_page(page)
    assert signal == 'form[action*="login"]'


# ---------- browser_type ----------

async def test_browser_type_rejects_sensitive_selector():
    page = MagicMock()
    with pytest.raises(SafetyError):
        await browser_type(page, "css=#password", "secret123")
    page.locator.assert_not_called()


async def test_browser_type_success():
    page = MagicMock()
    locator = MagicMock()
    locator.count = AsyncMock(return_value=1)
    locator.evaluate_all = AsyncMock(return_value=[{
        "tag_name": "input", "type": "text", "id": "username",
        "name": "username", "autocomplete": "username",
        "aria_label": None, "placeholder": None, "label_text": "用户名",
    }])
    locator.fill = AsyncMock(return_value=None)
    page.locator.return_value = locator

    result = await browser_type(page, "css=#username", "alice")
    assert result["success"] is True
    assert result["output"] == "css=#username"
    locator.fill.assert_awaited_once_with("alice")


async def test_browser_type_fill_failure_returns_tool_result():
    page = MagicMock()
    locator = MagicMock()
    locator.count = AsyncMock(return_value=1)
    locator.evaluate_all = AsyncMock(return_value=[{
        "tag_name": "input", "type": "text", "id": "username",
        "name": "username", "autocomplete": None,
        "aria_label": None, "placeholder": None, "label_text": None,
    }])
    locator.fill = AsyncMock(side_effect=RuntimeError("元素不可见"))
    page.locator.return_value = locator

    result = await browser_type(page, "css=#username", "alice")
    assert result["success"] is False
    err = result["error_msg"]
    assert err is not None
    assert "输入失败" in err


# ---------- browser_type attribute-based sensitive detection (DS-R2) ----------


async def test_browser_type_rejects_password_type_attribute():
    """selector 不含敏感词，但元素 type=password → SafetyError，fill() 未被调用。"""
    page = MagicMock()
    locator = MagicMock()
    locator.count = AsyncMock(return_value=1)
    locator.evaluate_all = AsyncMock(return_value=[{
        "tag_name": "input", "type": "password", "id": "credential-input",
        "name": "credential", "autocomplete": None,
        "aria_label": None, "placeholder": None, "label_text": None,
    }])
    locator.fill = AsyncMock(return_value=None)
    page.locator.return_value = locator

    with pytest.raises(SafetyError) as exc_info:
        await browser_type(page, "css=#credential-input", "secret123")
    assert exc_info.value.trigger == "sensitive_field"
    assert exc_info.value.selector == "css=#credential-input"
    assert "type=password" in str(exc_info.value)
    locator.fill.assert_not_called()


async def test_browser_type_rejects_autocomplete_new_password():
    """selector 不含敏感词，但 autocomplete=new-password → SafetyError。"""
    page = MagicMock()
    locator = MagicMock()
    locator.count = AsyncMock(return_value=1)
    locator.evaluate_all = AsyncMock(return_value=[{
        "tag_name": "input", "type": "text", "id": "credential-input",
        "name": "credential", "autocomplete": "new-password",
        "aria_label": None, "placeholder": None, "label_text": None,
    }])
    locator.fill = AsyncMock(return_value=None)
    page.locator.return_value = locator

    with pytest.raises(SafetyError) as exc_info:
        await browser_type(page, "css=#credential-input", "secret123")
    assert "autocomplete=new-password" in str(exc_info.value)
    locator.fill.assert_not_called()


async def test_browser_type_rejects_aria_label_bank_card():
    """aria-label="银行卡号" → SafetyError。"""
    page = MagicMock()
    locator = MagicMock()
    locator.count = AsyncMock(return_value=1)
    locator.evaluate_all = AsyncMock(return_value=[{
        "tag_name": "input", "type": "text", "id": "field-1",
        "name": "bank_card", "autocomplete": None,
        "aria_label": "银行卡号", "placeholder": None, "label_text": None,
    }])
    locator.fill = AsyncMock(return_value=None)
    page.locator.return_value = locator

    with pytest.raises(SafetyError) as exc_info:
        await browser_type(page, "css=#field-1", "1234567890123456")
    assert "银行卡号" in str(exc_info.value)
    locator.fill.assert_not_called()


async def test_browser_type_rejects_sensitive_label_text():
    """关联 label 文本命中敏感正则 → SafetyError。"""
    page = MagicMock()
    locator = MagicMock()
    locator.count = AsyncMock(return_value=1)
    locator.evaluate_all = AsyncMock(return_value=[{
        "tag_name": "input", "type": "text", "id": "field-2",
        "name": "data", "autocomplete": None,
        "aria_label": None, "placeholder": None, "label_text": "CVV 安全码",
    }])
    locator.fill = AsyncMock(return_value=None)
    page.locator.return_value = locator

    with pytest.raises(SafetyError):
        await browser_type(page, "css=#field-2", "123")
    locator.fill.assert_not_called()


async def test_browser_type_attribute_read_failure_rejects():
    """属性读取抛出 PlaywrightError → 返回失败 ToolResult，fill() 未被调用。"""
    page = MagicMock()
    locator = MagicMock()
    locator.count = AsyncMock(return_value=1)
    locator.evaluate_all = AsyncMock(side_effect=PlaywrightError("frame detached"))
    locator.fill = AsyncMock(return_value=None)
    page.locator.return_value = locator

    result = await browser_type(page, "css=#some-field", "value")
    assert result["success"] is False
    err = result["error_msg"]
    assert err is not None
    assert "无法完成敏感字段安全检查" in err
    locator.fill.assert_not_called()


async def test_browser_type_not_blocked_for_password_help_text():
    """[R2-5] aria-label="密码找回说明" 但 type=text 且非密码类字段 → 不被拦截。"""
    page = MagicMock()
    locator = MagicMock()
    locator.count = AsyncMock(return_value=1)
    locator.evaluate_all = AsyncMock(return_value=[{
        "tag_name": "input", "type": "text", "id": "help-text",
        "name": "help", "autocomplete": None,
        "aria_label": "密码找回说明", "placeholder": None, "label_text": None,
    }])
    locator.fill = AsyncMock(return_value=None)
    page.locator.return_value = locator

    result = await browser_type(page, "css=#help-text", "some text")
    assert result["success"] is True
    locator.fill.assert_awaited_once_with("some text")


async def test_browser_type_checks_all_matched_elements():
    """[R2-1] selector 匹配多个元素，任一命中敏感特征即抛出 SafetyError。"""
    page = MagicMock()
    locator = MagicMock()
    locator.count = AsyncMock(return_value=2)
    locator.evaluate_all = AsyncMock(return_value=[
        {
            "tag_name": "input", "type": "text", "id": "normal-field",
            "name": "normal", "autocomplete": None,
            "aria_label": None, "placeholder": None, "label_text": None,
        },
        {
            "tag_name": "input", "type": "password", "id": "hidden-pwd",
            "name": "pwd", "autocomplete": None,
            "aria_label": None, "placeholder": None, "label_text": None,
        },
    ])
    locator.fill = AsyncMock(return_value=None)
    page.locator.return_value = locator

    with pytest.raises(SafetyError) as exc_info:
        await browser_type(page, "css=input", "value")
    assert "element[1]" in str(exc_info.value)
    assert "type=password" in str(exc_info.value)
    locator.fill.assert_not_called()


async def test_browser_type_many_elements_logs_warning(caplog):
    """[R2-1] 元素数量超过阈值时记录 warning，但仍完成检查。"""
    import logging

    page = MagicMock()
    locator = MagicMock()
    count = 51
    attrs_list = [
        {
            "tag_name": "input", "type": "text", "id": f"field-{i}",
            "name": f"name-{i}", "autocomplete": None,
            "aria_label": None, "placeholder": None, "label_text": None,
        }
        for i in range(count)
    ]
    locator.count = AsyncMock(return_value=count)
    locator.evaluate_all = AsyncMock(return_value=attrs_list)
    locator.fill = AsyncMock(return_value=None)
    page.locator.return_value = locator

    with caplog.at_level(logging.WARNING, logger="agent.browser_tools"):
        result = await browser_type(page, "css=input", "value")
    assert result["success"] is True
    assert any("超过阈值" in record.message for record in caplog.records)


# ---------- browser_select ----------

async def test_browser_select_success_by_label():
    page = MagicMock()
    locator = MagicMock()
    locator.select_option = AsyncMock(return_value=None)
    page.locator.return_value = locator

    result = await browser_select(page, "css=#lang", "English")
    assert result["success"] is True
    locator.select_option.assert_awaited_once_with(label="English")


async def test_browser_select_falls_back_to_value_when_label_fails():
    page = MagicMock()
    locator = MagicMock()
    locator.select_option = AsyncMock(
        side_effect=[PlaywrightError("no such label"), None]
    )
    page.locator.return_value = locator

    result = await browser_select(page, "css=#lang", "en")
    assert result["success"] is True
    assert locator.select_option.await_count == 2


async def test_browser_select_fails_when_both_label_and_value_fail():
    # 第一次（label）必须是 PlaywrightError 才会被捕获并进入 value 兜底分支；
    # 第二次（value）用普通 Exception 模拟彻底失败。
    page = MagicMock()
    locator = MagicMock()
    locator.select_option = AsyncMock(
        side_effect=[PlaywrightError("no such label"), RuntimeError("完全失败")]
    )
    page.locator.return_value = locator

    result = await browser_select(page, "css=#lang", "xx")
    assert result["success"] is False
    err = result["error_msg"]
    assert err is not None
    assert "下拉框选择失败" in err


# ---------- browser_scroll ----------

async def test_browser_scroll_detects_page_change():
    page = MagicMock()
    page.evaluate = AsyncMock(side_effect=[0, 500])
    page.keyboard.press = AsyncMock(return_value=None)
    page.wait_for_timeout = AsyncMock(return_value=None)

    result = await browser_scroll(page, "down")
    assert result["success"] is True
    assert result["page_changed"] is True
    assert result["output"] == "500"
    page.keyboard.press.assert_awaited_once_with("PageDown")


async def test_browser_scroll_no_change_detected():
    page = MagicMock()
    page.evaluate = AsyncMock(side_effect=[100, 100])
    page.keyboard.press = AsyncMock(return_value=None)
    page.wait_for_timeout = AsyncMock(return_value=None)

    result = await browser_scroll(page, "up")
    assert result["success"] is True
    assert result["page_changed"] is False
    page.keyboard.press.assert_awaited_once_with("PageUp")


async def test_browser_scroll_failure_returns_tool_result():
    page = MagicMock()
    page.evaluate = AsyncMock(side_effect=RuntimeError("页面已关闭"))

    result = await browser_scroll(page, "down")
    assert result["success"] is False
    err = result["error_msg"]
    assert err is not None
    assert "滚动失败" in err


# ---------- browser_click three-level fallback ----------

async def test_browser_click_css_success():
    page = MagicMock()
    page.url = "https://x"
    page.evaluate = AsyncMock(return_value="Overview")
    page.wait_for_timeout = AsyncMock(return_value=None)
    locator = MagicMock()
    locator.click = AsyncMock(return_value=None)
    page.locator.return_value = locator

    result = await browser_click(page, selector="css=#submit")
    assert result["success"] is True
    out = result["output"]
    assert out is not None
    assert '"selector_level": "css"' in out


async def test_browser_click_falls_back_from_css_to_text():
    page = MagicMock()
    page.url = "https://x"
    page.evaluate = AsyncMock(return_value="Overview")
    page.wait_for_timeout = AsyncMock(return_value=None)

    css_locator = MagicMock()
    css_locator.click = AsyncMock(side_effect=PlaywrightError("找不到元素"))
    page.locator.return_value = css_locator

    text_locator = MagicMock()
    text_locator.click = AsyncMock(return_value=None)
    page.get_by_text.return_value = text_locator

    result = await browser_click(page, selector="css=#missing", text="提交")
    assert result["success"] is True
    out = result["output"]
    assert out is not None
    assert '"selector_level": "text"' in out


async def test_browser_click_falls_back_to_role_when_css_and_text_fail():
    page = MagicMock()
    page.url = "https://x"
    page.evaluate = AsyncMock(return_value="Overview")
    page.wait_for_timeout = AsyncMock(return_value=None)

    css_locator = MagicMock()
    css_locator.click = AsyncMock(side_effect=PlaywrightError("css 未命中"))
    page.locator.return_value = css_locator

    text_locator = MagicMock()
    text_locator.click = AsyncMock(side_effect=PlaywrightError("text 未命中"))
    page.get_by_text.return_value = text_locator

    role_locator = MagicMock()
    role_locator.click = AsyncMock(return_value=None)
    page.get_by_role.return_value = role_locator

    result = await browser_click(page, selector="css=#missing", text="提交")
    assert result["success"] is True
    out = result["output"]
    assert out is not None
    assert '"selector_level": "role"' in out


async def test_browser_click_all_levels_fail_returns_failure():
    page = MagicMock()
    page.url = "https://x"
    page.evaluate = AsyncMock(return_value="Overview")
    page.wait_for_timeout = AsyncMock(return_value=None)

    css_locator = MagicMock()
    css_locator.click = AsyncMock(side_effect=PlaywrightError("css 未命中"))
    page.locator.return_value = css_locator

    text_locator = MagicMock()
    text_locator.click = AsyncMock(side_effect=PlaywrightError("text 未命中"))
    page.get_by_text.return_value = text_locator

    role_locator = MagicMock()
    role_locator.click = AsyncMock(side_effect=PlaywrightError("role 未命中"))
    page.get_by_role.return_value = role_locator

    result = await browser_click(page, selector="css=#missing", text="提交")
    assert result["success"] is False
    err = result["error_msg"]
    assert err is not None
    assert "三级降级均未命中" in err


async def test_browser_click_without_selector_or_text_fails_fast():
    page = MagicMock()
    page.url = "https://x"
    result = await browser_click(page)
    assert result["success"] is False
    err = result["error_msg"]
    assert err is not None
    assert "至少提供 selector 或 text" in err


async def test_browser_click_extracts_fallback_text_from_text_equals_selector():
    """selector='text=提交' 且未显式传 text 时，应拆出 '提交' 灌进 text/role 降级链。"""
    page = MagicMock()
    page.url = "https://x"
    page.evaluate = AsyncMock(return_value="Overview")
    page.wait_for_timeout = AsyncMock(return_value=None)

    css_locator = MagicMock()
    css_locator.click = AsyncMock(side_effect=PlaywrightError("css 未命中"))
    page.locator.return_value = css_locator

    text_locator = MagicMock()
    text_locator.click = AsyncMock(return_value=None)
    page.get_by_text.return_value = text_locator

    result = await browser_click(page, selector="text=提交")
    assert result["success"] is True
    page.get_by_text.assert_called_with("提交")


# ---------- browser_click page_changed semantics (DS-Y1) ----------


async def test_browser_click_page_changed_true_when_url_unchanged_but_text_changed():
    """正向验证：URL 不变、innerText 从 Overview 变成 Features → page_changed=True。"""
    page = MagicMock()
    page.url = "https://x"
    page.evaluate = AsyncMock(side_effect=["Overview", "Features"])
    page.wait_for_timeout = AsyncMock(return_value=None)

    locator = MagicMock()
    locator.click = AsyncMock(return_value=None)
    page.locator.return_value = locator

    result = await browser_click(page, selector="css=#tab-features")
    assert result["success"] is True
    assert result["page_changed"] is True
    # URL 未变化时应调用 wait_for_timeout 而非 wait_for_load_state
    page.wait_for_timeout.assert_awaited_once()
    page.wait_for_load_state.assert_not_called()


async def test_browser_click_page_changed_true_when_url_changed():
    """正向验证：URL 变化时 page_changed=True，且仍调用 wait_for_load_state。"""
    page = MagicMock()
    # page.url 被访问 4 次：url_before, _compute_fingerprint(before), url_changed判断, _compute_fingerprint(after)
    type(page).url = PropertyMock(side_effect=["https://x", "https://x", "https://y", "https://y"])
    page.evaluate = AsyncMock(side_effect=["text_before", "text_after"])
    page.wait_for_timeout = AsyncMock(return_value=None)
    page.wait_for_load_state = AsyncMock(return_value=None)

    locator = MagicMock()
    locator.click = AsyncMock(return_value=None)
    page.locator.return_value = locator

    result = await browser_click(page, selector="css=#link")
    assert result["success"] is True
    assert result["page_changed"] is True
    page.wait_for_load_state.assert_awaited_once_with("networkidle", timeout=5000)
    page.wait_for_timeout.assert_not_called()


async def test_browser_click_page_changed_false_when_url_and_text_unchanged():
    """回归检查：URL 不变且文本不变 → page_changed=False。"""
    page = MagicMock()
    page.url = "https://x"
    page.evaluate = AsyncMock(return_value="same text")
    page.wait_for_timeout = AsyncMock(return_value=None)

    locator = MagicMock()
    locator.click = AsyncMock(return_value=None)
    page.locator.return_value = locator

    result = await browser_click(page, selector="css=#btn")
    assert result["success"] is True
    assert result["page_changed"] is False


async def test_browser_click_fingerprint_failure_falls_back_to_url_changed(caplog):
    """指纹读取失败时回退为 url_changed，click 仍返回成功，记录 warning。"""
    import logging

    page = MagicMock()
    page.url = "https://x"
    page.evaluate = AsyncMock(side_effect=PlaywrightError("frame detached"))
    page.wait_for_timeout = AsyncMock(return_value=None)

    locator = MagicMock()
    locator.click = AsyncMock(return_value=None)
    page.locator.return_value = locator

    with caplog.at_level(logging.WARNING, logger="agent.browser_tools"):
        result = await browser_click(page, selector="css=#btn")
    assert result["success"] is True
    # URL 未变化且指纹读取失败 → 回退 url_changed=False
    assert result["page_changed"] is False
    assert any("状态指纹读取失败" in r.message for r in caplog.records)


async def test_browser_click_selector_level_output_unchanged():
    """回归检查：三层 selector 降级及 selector_level 输出不改变。"""
    page = MagicMock()
    page.url = "https://x"
    page.evaluate = AsyncMock(return_value="text")
    page.wait_for_timeout = AsyncMock(return_value=None)

    css_locator = MagicMock()
    css_locator.click = AsyncMock(side_effect=PlaywrightError("css 未命中"))
    page.locator.return_value = css_locator

    text_locator = MagicMock()
    text_locator.click = AsyncMock(return_value=None)
    page.get_by_text.return_value = text_locator

    result = await browser_click(page, selector="css=#missing", text="提交")
    assert result["success"] is True
    out = result["output"]
    assert out is not None
    assert '"selector_level": "text"' in out


# ---------- _parse_extract_response ----------

def _make_message(text: str) -> Message:
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


def test_parse_extract_response_valid_json():
    message = _make_message('{"title": "hello"}')
    assert _parse_extract_response(message) == {"title": "hello"}


def test_parse_extract_response_strips_markdown_code_fence():
    message = _make_message('```json\n{"title": "hello"}\n```')
    assert _parse_extract_response(message) == {"title": "hello"}


def test_parse_extract_response_invalid_json_raises_retry():
    message = _make_message("not valid json at all")
    with pytest.raises(LLMOutputRetry):
        _parse_extract_response(message)
