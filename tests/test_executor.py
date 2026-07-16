"""agent/executor.py 单元测试：只覆盖 PlaywrightExecutor 里不依赖真实
Chromium 进程的部分——execute()/_dispatch() 的字段校验与分发、
SafetyError 穿透、close() 的清理顺序与异常吞咽。

真正启动浏览器的 open() 成功路径需要 mock 掉 async_playwright() 三层
链式调用，价值有限（本质是在测 mock 本身），这里只覆盖它的异常兜底路径。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.config import AgentConfig
from agent.exceptions import SafetyError
from agent.executor import PlaywrightExecutor
from agent.types import LLMAction, ObserveResult, ToolResult


def _make_executor(tmp_path) -> PlaywrightExecutor:
    config = AgentConfig(trace_dir=str(tmp_path))
    return PlaywrightExecutor(config)


def _action(**overrides) -> LLMAction:
    base: LLMAction = {
        "action": "click",
        "selector": None,
        "text": None,
        "value": None,
        "reason": "test",
    }
    # 部分用例（如 action="teleport"）故意构造不合法的 action 类型，
    # 用于验证 execute() 对未知/畸形输入的拦截——这在真实场景里对应
    # LLM 工具调用解析出的原始 dict，本就不保证严格符合 Literal 约束，
    # 这里刻意放行，不是笔误。
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def _obs(**overrides) -> ObserveResult:
    base: ObserveResult = {
        "url": "https://example.com",
        "title": "Example",
        "visible_text_summary": "",
        "text_hash": "abc",
        "interactive_elements": [],
        "screenshot_path": "/tmp/step-001.png",
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def _error_msg(result: ToolResult) -> str:
    """从 ToolResult 里取出 error_msg 并断言其非 None。

    ToolResult.error_msg 的类型是 str | None，直接对它做 `in` 判断在
    静态类型层面不安全（None 没有 __contains__）；这里显式断言收窄类型，
    同时如果 error_msg 意外为 None，也能得到比 TypeError 更清晰的失败信息。
    """
    msg = result["error_msg"]
    assert msg is not None, "期望 error_msg 非 None，但实际为 None"
    return msg


# ---------- execute(): page 未初始化 / 未知 action ----------

async def test_execute_without_page_returns_failure(tmp_path):
    executor = _make_executor(tmp_path)
    assert executor.page is None

    result = await executor.execute(_action(action="click", selector="#a"))
    assert result["success"] is False
    assert "尚未初始化" in _error_msg(result)


async def test_execute_unknown_action_returns_failure(tmp_path):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()

    result = await executor.execute(_action(action="teleport"))
    assert result["success"] is False
    assert "未知的 action 类型" in _error_msg(result)


# ---------- click / type / scroll / select / extract 的字段校验 ----------

async def test_execute_click_dispatches_to_browser_click(tmp_path):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()

    with patch("agent.executor.browser_click", new=AsyncMock(return_value={"success": True})) as mock_click:
        result = await executor.execute(_action(action="click", selector="#submit", text="提交"))

    assert result["success"] is True
    mock_click.assert_awaited_once_with(
        executor.page, selector="#submit", text="提交", config=executor.config
    )


async def test_execute_type_missing_selector_or_text_fails_without_calling_browser_tools(tmp_path):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()

    with patch("agent.executor.browser_type", new=AsyncMock()) as mock_type:
        result = await executor.execute(_action(action="type", selector=None, text="hello"))
        assert result["success"] is False
        assert "缺少必填的 selector 或 text" in _error_msg(result)

        result2 = await executor.execute(_action(action="type", selector="#a", text=None))
        assert result2["success"] is False

    mock_type.assert_not_awaited()


async def test_execute_type_success_dispatches_with_positional_args(tmp_path):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()

    with patch("agent.executor.browser_type", new=AsyncMock(return_value={"success": True})) as mock_type:
        result = await executor.execute(_action(action="type", selector="#a", text="hello"))

    assert result["success"] is True
    mock_type.assert_awaited_once_with(executor.page, "#a", "hello")


@pytest.mark.parametrize("direction", ["up", "down"])
async def test_execute_scroll_valid_direction_dispatches(tmp_path, direction):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()

    with patch("agent.executor.browser_scroll", new=AsyncMock(return_value={"success": True})) as mock_scroll:
        result = await executor.execute(_action(action="scroll", value=direction))

    assert result["success"] is True
    mock_scroll.assert_awaited_once_with(executor.page, direction)


async def test_execute_scroll_invalid_direction_fails(tmp_path):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()

    with patch("agent.executor.browser_scroll", new=AsyncMock()) as mock_scroll:
        result = await executor.execute(_action(action="scroll", value="sideways"))

    assert result["success"] is False
    assert "value 字段非法" in _error_msg(result)
    mock_scroll.assert_not_awaited()


async def test_execute_select_missing_fields_fails(tmp_path):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()

    with patch("agent.executor.browser_select", new=AsyncMock()) as mock_select:
        result = await executor.execute(_action(action="select", selector=None, value="en"))
        assert result["success"] is False
        assert "缺少必填的 selector 或 value" in _error_msg(result)

    mock_select.assert_not_awaited()


async def test_execute_select_success_dispatches(tmp_path):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()

    with patch("agent.executor.browser_select", new=AsyncMock(return_value={"success": True})) as mock_select:
        result = await executor.execute(_action(action="select", selector="#lang", value="English"))

    assert result["success"] is True
    mock_select.assert_awaited_once_with(executor.page, "#lang", "English")


async def test_execute_extract_missing_instruction_fails(tmp_path):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()

    with patch("agent.executor.browser_extract", new=AsyncMock()) as mock_extract:
        result = await executor.execute(_action(action="extract", value=None))

    assert result["success"] is False
    assert "缺少必填的 value" in _error_msg(result)
    mock_extract.assert_not_awaited()


async def test_execute_extract_passes_provided_obs_without_reobserving(tmp_path):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()
    obs = _obs(url="https://x", text_hash="abc")

    with patch("agent.executor.browser_extract", new=AsyncMock(return_value={"success": True})) as mock_extract:
        result = await executor.execute(_action(action="extract", value="抓取标题"), obs=obs)

    assert result["success"] is True
    mock_extract.assert_awaited_once_with(
        executor.page, "抓取标题", executor.observer, executor.config, obs=obs
    )


async def test_execute_extract_without_obs_falls_back_to_internal_observe(tmp_path):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()

    with patch("agent.executor.browser_extract", new=AsyncMock(return_value={"success": True})) as mock_extract:
        result = await executor.execute(_action(action="extract", value="抓取标题"), obs=None)

    assert result["success"] is True
    mock_extract.assert_awaited_once_with(
        executor.page, "抓取标题", executor.observer, executor.config
    )


async def test_execute_screenshot_dispatches_with_tracer(tmp_path):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()

    with patch("agent.executor.browser_screenshot", new=AsyncMock(return_value={"success": True})) as mock_shot:
        result = await executor.execute(_action(action="screenshot"))

    assert result["success"] is True
    mock_shot.assert_awaited_once_with(executor.page, executor.tracer)


async def test_execute_done_returns_value_without_touching_browser(tmp_path):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()

    result = await executor.execute(_action(action="done", value="任务完成，标题为 XX"))
    assert result["success"] is True
    assert result["output"] == "任务完成，标题为 XX"
    assert result["page_changed"] is False


# ---------- SafetyError 穿透 ----------

async def test_execute_safety_error_propagates_not_swallowed(tmp_path):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()

    with patch("agent.executor.browser_click", new=AsyncMock(side_effect=SafetyError("检测到敏感字段"))):
        with pytest.raises(SafetyError):
            await executor.execute(_action(action="click", selector="css=#password"))


async def test_execute_unexpected_exception_converted_to_tool_result(tmp_path):
    executor = _make_executor(tmp_path)
    executor.page = MagicMock()

    with patch("agent.executor.browser_click", new=AsyncMock(side_effect=RuntimeError("页面已关闭"))):
        result = await executor.execute(_action(action="click", selector="#a"))

    assert result["success"] is False
    assert "页面已关闭" in _error_msg(result)


# ---------- open() 异常兜底路径 ----------

async def test_open_chromium_launch_failure_returns_tool_result_and_cleans_up(tmp_path):
    executor = _make_executor(tmp_path)

    fake_playwright_ctx = AsyncMock()
    fake_playwright_ctx.chromium.launch = AsyncMock(side_effect=RuntimeError("Chromium 启动失败"))

    fake_playwright_manager = MagicMock()
    fake_playwright_manager.start = AsyncMock(return_value=fake_playwright_ctx)

    with patch("agent.executor.async_playwright", return_value=fake_playwright_manager):
        result = await executor.open("https://example.com")

    assert result["success"] is False
    assert "启动浏览器失败" in _error_msg(result)
    # close() 应该已经把已创建的部分资源清理掉
    assert executor.page is None
    assert executor._browser is None


async def test_open_propagates_safety_error_from_browser_open(tmp_path):
    executor = _make_executor(tmp_path)

    fake_page = AsyncMock()
    fake_browser = AsyncMock()
    fake_browser.new_page = AsyncMock(return_value=fake_page)
    fake_playwright_ctx = AsyncMock()
    fake_playwright_ctx.chromium.launch = AsyncMock(return_value=fake_browser)
    fake_playwright_manager = MagicMock()
    fake_playwright_manager.start = AsyncMock(return_value=fake_playwright_ctx)

    with patch("agent.executor.async_playwright", return_value=fake_playwright_manager):
        with patch(
            "agent.executor.browser_open",
            new=AsyncMock(side_effect=SafetyError("登录页确认后用户选择中止")),
        ):
            with pytest.raises(SafetyError):
                await executor.open("https://example.com/login")


# ---------- close(): 顺序与异常吞咽 ----------

async def test_close_calls_page_browser_playwright_in_order(tmp_path):
    executor = _make_executor(tmp_path)

    call_order = []
    page = AsyncMock()
    page.close = AsyncMock(side_effect=lambda: call_order.append("page"))
    browser = AsyncMock()
    browser.close = AsyncMock(side_effect=lambda: call_order.append("browser"))
    playwright = AsyncMock()
    playwright.stop = AsyncMock(side_effect=lambda: call_order.append("playwright"))

    executor.page = page
    executor._browser = browser
    executor._playwright = playwright

    await executor.close()

    assert call_order == ["page", "browser", "playwright"]
    assert executor.page is None
    assert executor._browser is None
    assert executor._playwright is None


async def test_close_continues_cleanup_when_page_close_raises(tmp_path):
    executor = _make_executor(tmp_path)

    page = AsyncMock()
    page.close = AsyncMock(side_effect=RuntimeError("page 已经关闭"))
    browser = AsyncMock()
    playwright = AsyncMock()

    executor.page = page
    executor._browser = browser
    executor._playwright = playwright

    await executor.close()  # 不应抛出

    browser.close.assert_awaited_once()
    playwright.stop.assert_awaited_once()
    assert executor.page is None
    assert executor._browser is None
    assert executor._playwright is None


async def test_close_is_noop_safe_when_nothing_initialized(tmp_path):
    executor = _make_executor(tmp_path)
    await executor.close()  # 不应抛出
    assert executor.page is None
    assert executor._browser is None
    assert executor._playwright is None
