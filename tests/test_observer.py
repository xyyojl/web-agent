"""agent/observer.py 单元测试：只覆盖 _truncate_text 这个不依赖浏览器的
纯文本处理静态方法。observe() 本身依赖真实 Playwright Page，
更适合放在集成测试/e2e eval 套件里，不在这里做 mock 到失真的伪单测。
"""

from agent.observer import BrowserStateObserver


def test_truncate_text_collapses_horizontal_whitespace():
    text = "hello    world\t\tfoo"
    result = BrowserStateObserver._truncate_text(text, limit=1000)
    assert result == "hello world foo"


def test_truncate_text_collapses_multiple_blank_lines():
    text = "line1\n\n\n\nline2"
    result = BrowserStateObserver._truncate_text(text, limit=1000)
    assert result == "line1\nline2"


def test_truncate_text_preserves_single_newlines():
    text = "line1\nline2\nline3"
    result = BrowserStateObserver._truncate_text(text, limit=1000)
    assert result == "line1\nline2\nline3"


def test_truncate_text_truncates_to_limit():
    text = "a" * 5000
    result = BrowserStateObserver._truncate_text(text, limit=3000)
    assert len(result) == 3000


def test_truncate_text_strips_leading_trailing_whitespace():
    text = "   hello world   "
    result = BrowserStateObserver._truncate_text(text, limit=1000)
    assert result == "hello world"


def test_truncate_text_empty_string():
    assert BrowserStateObserver._truncate_text("", limit=100) == ""
