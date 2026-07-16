"""agent/exceptions.py 单元测试：覆盖消息格式化、上下文携带、
to_dict() 序列化，以及 LLMError 对超长 raw_response 的截断。
"""

from agent.exceptions import (
    BrowserError,
    EvalError,
    LLMError,
    SafetyError,
    WebAgentError,
)


def test_base_error_without_context_formats_message_only():
    err = WebAgentError("出错了")
    assert str(err) == "出错了"
    assert err.to_dict() == {"type": "WebAgentError", "message": "出错了"}


def test_base_error_with_context_appends_key_value_pairs():
    err = WebAgentError("出错了", foo="bar", n=1)
    assert str(err) == "出错了 (foo='bar', n=1)"
    d = err.to_dict()
    assert d["type"] == "WebAgentError"
    assert d["message"] == "出错了"
    assert d["foo"] == "bar"
    assert d["n"] == 1


def test_safety_error_carries_trigger_selector_url():
    err = SafetyError("检测到敏感字段", trigger="sensitive_field", selector="#pwd", url="https://x")
    assert err.trigger == "sensitive_field"
    assert err.selector == "#pwd"
    assert err.url == "https://x"
    assert err.to_dict()["trigger"] == "sensitive_field"


def test_browser_error_defaults_are_none():
    err = BrowserError("打开页面超时")
    assert err.action is None
    assert err.selector is None
    assert err.timeout_ms is None


def test_browser_error_carries_action_and_timeout():
    err = BrowserError("超时", action="goto", timeout_ms=15000)
    assert err.action == "goto"
    assert err.timeout_ms == 15000


def test_llm_error_truncates_raw_response():
    long_response = "x" * 1000
    err = LLMError("解析失败", stage="parse", raw_response=long_response, retry_count=2)
    assert len(err.raw_response) == 500
    assert err.stage == "parse"
    assert err.retry_count == 2


def test_llm_error_raw_response_none_stays_none():
    err = LLMError("网络错误", stage="request")
    assert err.raw_response is None


def test_eval_error_carries_case_id_and_field():
    err = EvalError("case 格式错误", case_id="L01", field="expected_output")
    assert err.case_id == "L01"
    assert err.field == "expected_output"


def test_all_error_types_are_webagenterror_subclasses():
    for cls in (SafetyError, BrowserError, LLMError, EvalError):
        assert issubclass(cls, WebAgentError)
