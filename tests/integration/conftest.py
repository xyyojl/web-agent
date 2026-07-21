"""集成测试共享 fixtures。

职责：
1. 提供真实 Chromium 浏览器 page 对象（通过 Playwright async API）；
2. 守卫 LLMClient：集成测试不得触发真实 LLM 调用（Implementation Contract），
   若调用则测试立即失败；
3. 提供 AgentConfig 和本地 HTML fixture 目录路径。
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from playwright.async_api import async_playwright

from agent.config import AgentConfig
from agent.llm_client import LLMClient

# 锚定到项目根目录，不依赖当前工作目录
_PROJECT_ROOT: str = str(Path(__file__).resolve().parents[2])
PAGES_DIR: str = str(Path(_PROJECT_ROOT) / "eval" / "pages")


@pytest.fixture
def pages_dir() -> str:
    """返回 eval/pages/ 的绝对路径，供集成测试通过 file:// 打开本地 fixture。"""
    return PAGES_DIR


@pytest.fixture
def config() -> AgentConfig:
    """使用纯默认值的 AgentConfig，不读取环境变量，确保无外部依赖。"""
    return AgentConfig()


@pytest.fixture(autouse=True)
def guard_no_llm_calls():
    """Implementation Contract: 集成测试不得调用 LLMClient。

    在类级别 patch call_with_retry，任何代码路径尝试发起 LLM 调用
    都会触发 AssertionError，使测试立即失败。
    """
    with patch.object(
        LLMClient,
        "call_with_retry",
        new=AsyncMock(
            side_effect=AssertionError(
                "LLMClient.call_with_retry must not be called in integration tests"
            )
        ),
    ):
        yield


@pytest.fixture
async def browser_page():
    """启动真实 Chromium 浏览器，提供一个 page 对象。

    测试结束后自动关闭 context 和 browser，不残留进程。
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context()
        page = await context.new_page()
        yield page
        await context.close()
        await browser.close()
