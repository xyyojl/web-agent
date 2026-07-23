"""真实浏览器集成测试：使用本地 HTML fixture 验证 Playwright 端到端行为。

覆盖 DS-Y4 验证策略中的正向与负向场景：
- 正向：tab 点击 DOM 状态变化（page_changed）；password 字段安全阻断；
  Observer 生成 PNG 截图 + text_hash 非空。
- 负向：不存在 selector 返回 ToolResult failure（不抛未处理异常）；
  LLMClient 不得被调用（由 conftest.py 的 autouse fixture 守卫）。

Implementation Contract:
- 所有页面通过 file:// 协议加载，不依赖公网、API key、当前日期或外部网页内容。
- trace/截图输出到 pytest tmp_path，不污染仓库。
"""

import os

import pytest

from agent.browser_tools import browser_click, browser_open, browser_type
from agent.config import AgentConfig
from agent.exceptions import SafetyError
from agent.observer import BrowserStateObserver
from agent.tracer import TraceLogger


# ---------- 正向验证 ----------


@pytest.mark.integration
async def test_tab_click_changes_dom_state(browser_page, pages_dir, config):
    """正向验证：tab_nav.html 点击 Features 后，页面文本变化且 page_changed=True。

    覆盖 DS-Y1 修复后的指纹比较逻辑——URL 不变但 innerText 变化时
    page_changed 必须为 True。
    """
    page = browser_page
    await page.goto(f"file://{pages_dir}/tab_nav.html")

    # 确认初始状态：Overview 面板可见
    initial_text = await page.evaluate("() => document.body.innerText")
    assert "专注于浏览器自动化的智能体框架" in initial_text

    # 点击 Features tab 按钮
    result = await browser_click(
        page, selector="css=.tab-btn[data-tab='features']", config=config
    )

    assert result["success"] is True
    assert result["page_changed"] is True

    # 验证 Features 面板内容已出现、Overview 面板内容已消失
    after_text = await page.evaluate("() => document.body.innerText")
    assert "结构化页面观察" in after_text
    assert "专注于浏览器自动化的智能体框架" not in after_text


@pytest.mark.integration
async def test_dynamic_noise_is_ignored_when_configured(browser_page, pages_dir):
    """无关动态区域更新不应被误记为业务页面变化。"""
    page = browser_page
    await page.goto(f"file://{pages_dir}/dynamic_noise.html")
    config = AgentConfig(noise_selectors=("#dynamic-noise",))

    result = await browser_click(page, selector="css=#noise-refresh", config=config)
    assert result["success"] is True
    assert result["page_changed"] is False
    assert await page.locator("#noise-value").inner_text() == "刷新计数：1"


@pytest.mark.integration
async def test_dynamic_noise_remains_visible_by_default(browser_page, pages_dir):
    """默认空配置保持保守：未声明为噪声的变化仍算页面变化。"""
    page = browser_page
    await page.goto(f"file://{pages_dir}/dynamic_noise.html")
    result = await browser_click(page, selector="css=#noise-refresh", config=AgentConfig())
    assert result["success"] is True
    assert result["page_changed"] is True


@pytest.mark.integration
async def test_password_field_write_blocked(browser_page, pages_dir):
    """正向验证：sensitive_field.html 中 type=password 字段写入被阻断。

    测试两个字段：
    - #credential-input: selector 不含敏感词，但 type=password + autocomplete=new-password
      → 被元素属性检查（DS-R2 第二道）拦截；
    - #password: selector 含 "password"
      → 被 selector 正则快速拒绝（DS-R2 第一道）。
    两者都不得调用 fill()，验证字段值保持为空。
    """
    page = browser_page
    await page.goto(f"file://{pages_dir}/sensitive_field.html")

    # 1. #credential-input: 基于元素真实属性的拦截
    with pytest.raises(SafetyError) as exc_info:
        await browser_type(page, "css=#credential-input", "secret123")
    assert exc_info.value.trigger == "sensitive_field"
    assert "type=password" in str(exc_info.value)

    # 验证 fill() 未被调用——字段值仍为空
    value = await page.evaluate(
        "() => document.getElementById('credential-input').value"
    )
    assert value == ""

    # 2. #password: selector 正则拦截
    with pytest.raises(SafetyError) as exc_info2:
        await browser_type(page, "css=#password", "secret456")
    assert exc_info2.value.trigger == "sensitive_field"

    value2 = await page.evaluate("() => document.getElementById('password').value")
    assert value2 == ""


@pytest.mark.integration
async def test_l11_localhost_fixture_reaches_sensitive_field_guard(browser_page, pages_dir, monkeypatch):
    """无人值守 localhost L11 不触发登录 EOF，实际到达 browser_type 拦截。"""
    page = browser_page
    url = "http://localhost:8080/sensitive_field.html"

    async def fulfill_l11(route):
        await route.fulfill(path=f"{pages_dir}/sensitive_field.html")

    await page.route(url, fulfill_l11)
    prompt_called = False

    async def forbidden_prompt(*_args, **_kwargs):
        nonlocal prompt_called
        prompt_called = True
        raise AssertionError("L11 local fixture must not ask for login confirmation")

    monkeypatch.setattr("agent.browser_tools.ask_human", forbidden_prompt)
    opened = await browser_open(page, url, AgentConfig())
    assert opened["success"] is True
    assert prompt_called is False

    with pytest.raises(SafetyError) as exc_info:
        await browser_type(page, "css=#credential-input", "secret123")
    assert exc_info.value.trigger == "sensitive_field"
    assert await page.locator("#credential-input").input_value() == ""


@pytest.mark.integration
async def test_aria_labelledby_multiple_ids_sensitive_label_blocked(browser_page):
    """aria-labelledby 是 IDREF 列表；后续敏感 label 也必须参与检查。"""
    page = browser_page
    await page.set_content("""
        <span id="normal">付款信息</span><span id="sensitive">CVV 安全码</span>
        <input id="field" type="text" aria-labelledby="normal sensitive">
    """)
    with pytest.raises(SafetyError) as exc_info:
        await browser_type(page, "css=#field", "123")
    assert exc_info.value.trigger == "sensitive_field"
    assert "CVV" in str(exc_info.value)
    assert await page.locator("#field").input_value() == ""


@pytest.mark.integration
async def test_label_read_failure_does_not_skip_other_associations(browser_page):
    """祖先 label 读取异常时，aria-labelledby 仍须继续检查并阻断。"""
    page = browser_page
    await page.set_content("""
        <span id="normal">付款信息</span><span id="sensitive">银行卡号</span>
        <input id="field" type="text" aria-labelledby="normal sensitive">
        <script>Element.prototype.closest = () => { throw new Error('broken label API'); };</script>
    """)
    with pytest.raises(SafetyError) as exc_info:
        await browser_type(page, "css=#field", "123")
    assert exc_info.value.trigger == "sensitive_field"
    assert "银行卡" in str(exc_info.value)
    assert await page.locator("#field").input_value() == ""


@pytest.mark.integration
async def test_observer_generates_png_and_text_hash(browser_page, pages_dir, tmp_path):
    """正向验证：Observer 在临时 trace dir 生成实际 PNG，text_hash 非空。

    覆盖真实 Playwright Page 的 observe() 流程：
    - 截图落盘为 PNG 文件（校验文件头魔数）；
    - text_hash 是 SHA-256（64 位十六进制）；
    - interactive_elements 非空（tab_nav.html 至少有 3 个 tab 按钮）。
    """
    page = browser_page
    await page.goto(f"file://{pages_dir}/tab_nav.html")

    config = AgentConfig()
    tracer = TraceLogger(base_dir=str(tmp_path / "traces"))
    observer = BrowserStateObserver(config, tracer, vision=False)

    result = await observer.observe(page)

    # text_hash 非空且为 64 位十六进制
    assert result["text_hash"]
    assert len(result["text_hash"]) == 64
    int(result["text_hash"], 16)  # 确认是合法十六进制

    # 截图文件存在且为真实 PNG
    assert os.path.isfile(result["screenshot_path"])
    with open(result["screenshot_path"], "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"

    # interactive_elements 非空
    assert len(result["interactive_elements"]) > 0

    # 截图落在 tmp_path 内，不污染仓库
    assert str(tmp_path) in result["screenshot_path"]


@pytest.mark.integration
async def test_observer_uses_unique_css_selector_for_duplicate_link_text(browser_page, tmp_path):
    """同名链接不能生成会触发 Playwright strict mode 的 text= selector。"""
    page = browser_page
    await page.set_content("""
        <main><a href="#first">asyncio — Asynchronous I/O</a></main>
        <footer><a href="#second">asyncio — Asynchronous I/O</a></footer>
    """)
    observer = BrowserStateObserver(AgentConfig(), TraceLogger(base_dir=str(tmp_path / "traces")))

    result = await observer.observe(page)
    duplicate_links = [
        el for el in result["interactive_elements"]
        if el["name"] == "asyncio — Asynchronous I/O"
    ]

    assert len(duplicate_links) == 2
    assert all(el["selector"].startswith("css=body >") for el in duplicate_links)
    for element in duplicate_links:
        assert await page.locator(element["selector"].removeprefix("css=")).count() == 1


# ---------- 负向验证 ----------


@pytest.mark.integration
async def test_nonexistent_selector_returns_failure(browser_page, pages_dir, config):
    """负向验证：模拟不存在 selector，必须返回 ToolResult failure，不得抛未处理异常。

    browser_click 三级降级（css → text → role）全部未命中时，
    应返回 success=False 的 ToolResult，而非向上抛异常。
    """
    page = browser_page
    await page.goto(f"file://{pages_dir}/tab_nav.html")

    result = await browser_click(
        page, selector="css=#nonexistent-element-12345", config=config
    )

    assert result["success"] is False
    error_msg = result["error_msg"]
    assert error_msg is not None
    assert "三级降级均未命中" in error_msg
