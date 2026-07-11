"""执行层：管理 Playwright/Chromium 生命周期，把 LLMAction 分发到
agent.browser_tools 中对应的工具函数。所有执行过程中的异常（除
SafetyError 外）统一转换为 ToolResult(success=False, error_msg=...)，
不向上抛出，避免单步失败打断整条 agent loop。
"""

import logging

from playwright.async_api import Browser, Page, Playwright, async_playwright

from agent.browser_tools import (
    browser_click,
    browser_extract,
    browser_open,
    browser_screenshot,
    browser_scroll,
    browser_select,
    browser_type,
)
from agent.config import AgentConfig
from agent.exceptions import SafetyError
from agent.observer import BrowserStateObserver
from agent.tracer import TraceLogger
from agent.types import LLMAction, ObserveResult, ToolResult

logger = logging.getLogger(__name__)

_KNOWN_ACTIONS = ("click", "type", "scroll", "extract", "screenshot", "select", "done")


class PlaywrightExecutor:
    """执行层：一个实例对应一次任务 run 的完整 Playwright 生命周期。

    tracer 支持从外部注入（CTRL-001 中 AgentController 会把自己持有的
    TraceLogger 传进来，让 observer 的截图、executor.execute() 里的
    screenshot/extract 分发、以及 controller 自己的 tracer.record() 全部
    落在同一个 traces/run-xxx/ 目录、截图编号连续）；不传时退化为独立
    创建一个新的 TraceLogger，保持 EXEC-001 单独使用场景下的原有行为。
    """

    def __init__(self, config: AgentConfig, tracer: TraceLogger | None = None) -> None:
        self.config = config
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self.page: Page | None = None

        self.tracer = tracer if tracer is not None else TraceLogger(base_dir=config.trace_dir)
        # 与 AgentController 同模式透传 vision：extract 降级路径（未收到外部 obs 时
        # 内部重新 observe）也要保持 vision 模式一致，避免该路径静默退化为纯 DOM。
        self.observer = BrowserStateObserver(config, self.tracer, vision=config.vision)

    async def open(self, url: str) -> ToolResult:
        """启动 headless Chromium 并打开 url。

        浏览器启动阶段（Playwright/Chromium 进程本身）的异常也会被捕获
        转成 ToolResult(success=False)，同时主动清理已创建的部分资源，
        避免启动失败一半时残留孤儿进程。
        """
        try:
            # 链式调用（.chromium/.new_page）作用在局部变量上——局部变量的
            # 非 None 类型窄化比 self.xxx 这种实例属性更可靠，能避免类型
            # 检查器把 self._playwright 仍当作 Playwright | None 处理而报
            # "None 没有 chromium 属性"之类的警告。
            # 但每一步成功后立刻同步给 self.*，而不是等三步都成功再一次性
            # 赋值——否则中途失败（比如 new_page() 抛错）时 self._playwright/
            # self._browser 仍是 None，下面的 self.close() 就清理不到已经
            # 启动的 Chromium 进程，造成资源泄漏。
            playwright = await async_playwright().start()
            self._playwright = playwright
            browser = await playwright.chromium.launch(headless=True)
            self._browser = browser
            page = await browser.new_page()
            self.page = page
        except Exception as exc:
            logger.warning("PlaywrightExecutor 启动浏览器失败: %s", exc)
            await self.close()
            return ToolResult(
                success=False,
                page_changed=False,
                output=None,
                error_msg=f"启动浏览器失败: {exc}",
            )

        try:
            return await browser_open(self.page, url, self.config)
        except SafetyError:
            # 安全拦截（如登录页确认后用户选择中止）：刻意不吞掉，
            # 与 agent/browser_tools.py 的设计保持一致，交由上层 agent loop 处理。
            raise
        except Exception as exc:
            logger.warning("open(%s) 时发生未预期异常: %s", url, exc)
            return ToolResult(
                success=False, page_changed=False, output=None, error_msg=str(exc)
            )

    async def execute(
        self, action: LLMAction, obs: ObserveResult | None = None, task: str | None = None
    ) -> ToolResult:
        """按 action['action'] 分发到对应的 browser_tools 函数。

        page 尚未初始化（未调用 open() 或已 close()）、action 类型未知、
        必填字段缺失，均返回 ToolResult(success=False)，不抛异常。

        obs：主循环这一步已经 observe() 过的结果，目前只有 extract 会用到
        ——避免 browser_extract 内部再重新 observe 一次（重复扫描页面 +
        额外占用一个截图编号，导致 trace.jsonl 里的 screenshot 字段和
        实际截图文件编号对不上）。其余动作不需要 obs，传 None 即可。

        task：原始任务文本（EvalCase.task），目前只有 extract 会用到——
        browser_extract 内部做字段名确定性校验时，优先用这个未经任何一次
        LLM 转述的原文解析"引号点名的必需字段名"，比只依赖 action 里
        ActionSelector 自己转述的 instruction 更可靠（模型转述时可能弄丢
        引号，导致校验被静默跳过）。其余动作不需要 task，传 None 即可。
        """
        if self.page is None:
            return ToolResult(
                success=False,
                page_changed=False,
                output=None,
                error_msg="page 尚未初始化，请先调用 open()",
            )

        action_name = action.get("action")
        if action_name not in _KNOWN_ACTIONS:
            return ToolResult(
                success=False,
                page_changed=False,
                output=None,
                error_msg=f"未知的 action 类型: {action_name!r}",
            )

        try:
            return await self._dispatch(action_name, action, obs, task)
        except SafetyError:
            # 同 open()：安全拦截刻意穿透，不在此处转换成 ToolResult。
            raise
        except Exception as exc:
            logger.warning("执行 action=%s 时发生未捕获异常: %s", action_name, exc)
            return ToolResult(
                success=False, page_changed=False, output=None, error_msg=str(exc)
            )

    async def _dispatch(
        self,
        action_name: str,
        action: LLMAction,
        obs: ObserveResult | None = None,
        task: str | None = None,
    ) -> ToolResult:
        """真正的分发逻辑，拆成独立方法便于 execute() 统一做异常兜底。"""
        assert self.page is not None  # execute() 已确保非空，帮助类型检查器收窄

        if action_name == "click":
            return await browser_click(
                self.page,
                selector=action.get("selector"),
                text=action.get("text"),
                config=self.config,
            )

        if action_name == "type":
            selector = action.get("selector")
            text = action.get("text")
            if not selector or not text:
                return ToolResult(
                    success=False,
                    page_changed=False,
                    output=None,
                    error_msg="type 动作缺少必填的 selector 或 text",
                )
            return await browser_type(self.page, selector, text)

        if action_name == "scroll":
            direction = action.get("value")
            if direction not in ("up", "down"):
                return ToolResult(
                    success=False,
                    page_changed=False,
                    output=None,
                    error_msg=f"scroll 动作的 value 字段非法: {direction!r}",
                )
            return await browser_scroll(self.page, direction)

        if action_name == "extract":
            instruction = action.get("value")
            if not instruction:
                return ToolResult(
                    success=False,
                    page_changed=False,
                    output=None,
                    error_msg="extract 动作缺少必填的 value（抽取指令）字段",
                )
            if obs is None:
                # 兜底：理论上主循环每一步都会先 observe 再 execute，
                # obs 不应为 None；如果调用方没传（比如未来新增的调用
                # 路径漏传），退化成原来的行为——由 browser_extract 自己
                # 重新 observe 一次，保证功能不中断，只是会重新出现
                # 多占一个截图编号的现象。
                logger.warning("execute(extract) 未收到 obs，退化为内部重新 observe")
                return await browser_extract(
                    self.page, instruction, self.observer, self.config, task=task
                )
            return await browser_extract(
                self.page, instruction, self.observer, self.config, obs=obs, task=task
            )

        if action_name == "select":
            selector = action.get("selector")
            value = action.get("value")
            if not selector or not value:
                return ToolResult(
                    success=False,
                    page_changed=False,
                    output=None,
                    error_msg="select 动作缺少必填的 selector 或 value",
                )
            return await browser_select(self.page, selector, value)

        if action_name == "screenshot":
            return await browser_screenshot(self.page, self.tracer)

        # action_name == "done"：不涉及浏览器操作，原样透出最终结果，
        # 由上层 agent loop 判断任务终止。
        return ToolResult(
            success=True,
            page_changed=False,
            output=action.get("value"),
            error_msg=None,
        )

    async def close(self) -> None:
        """依次关闭 page -> browser -> playwright（顺序不能反）。

        每一步单独 try/except：任何一步失败都不应阻止后续资源被清理，
        否则一次 page.close() 抛异常就可能导致 Chromium 进程永久残留。
        """
        try:
            if self.page is not None:
                await self.page.close()
        except Exception as exc:
            logger.warning("关闭 page 时出错（已忽略，继续清理其余资源）: %s", exc)
        finally:
            self.page = None

        try:
            if self._browser is not None:
                await self._browser.close()
        except Exception as exc:
            logger.warning("关闭 browser 时出错（已忽略，继续清理其余资源）: %s", exc)
        finally:
            self._browser = None

        try:
            if self._playwright is not None:
                await self._playwright.stop()
        except Exception as exc:
            logger.warning("关闭 playwright 时出错（已忽略）: %s", exc)
        finally:
            self._playwright = None
