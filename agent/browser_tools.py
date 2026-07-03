"""浏览器操作工具集：8 个供 LLM 调用的异步工具函数。

设计原则：
- 除 SafetyError（安全拦截）外，所有异常都在工具内部捕获，
  转换为 ToolResult(success=False, error_msg=...)，不向上抛出未捕获异常。
- SafetyError 是唯一允许穿透工具边界的异常类型，交由上层 agent loop
  决定是否终止任务 —— 这是刻意设计，安全拦截不应被静默吞掉。
"""

import json
import os
import re
from typing import Awaitable, Callable, Literal

import anthropic
from anthropic.types import Message, MessageParam, TextBlock
from dotenv import load_dotenv
from playwright.async_api import Error as PlaywrightError

from agent.config import AgentConfig
from agent.exceptions import BrowserError, LLMError, SafetyError
from agent.observer import BrowserStateObserver
from agent.prompts import EXTRACTOR_SYSTEM, EXTRACTOR_USER_TMPL
from agent.tracer import TraceLogger
from agent.types import ObserveResult, ToolResult

# 加载项目根目录下的 .env 文件（若存在），使 ANTHROPIC_API_KEY 等变量
# 能在未手动 export 的情况下被 os.environ.get() 读取到。
load_dotenv()

# 敏感字段正则：匹配到即认为是密码/卡号/证件号等字段，禁止 browser_type 写入。
SENSITIVE_PATTERNS = [
    r"password", r"passwd", r"pwd",
    r"credit.?card", r"card.?number", r"cvv",
    r"id.?number", r"national.?id", r"ssn",
    r"bank.?account",
]

# 登录页信号：browser_open 打开新页面后检测，命中则暂停等待人工确认。
LOGIN_SIGNALS = [
    "input[type=password]",
    'form[action*="login"]',
    'button:has-text("登录")',
    'button:has-text("Sign in")',
]

_SENSITIVE_RE = re.compile("|".join(SENSITIVE_PATTERNS), re.IGNORECASE)


def _check_sensitive(selector: str) -> None:
    """selector 命中敏感字段正则则抛出 SafetyError，不做任何吞掉/降级处理。"""
    if selector and _SENSITIVE_RE.search(selector):
        raise SafetyError(
            "检测到敏感字段，拒绝写入",
            trigger="sensitive_field",
            selector=selector,
        )


async def _detect_login_page(page) -> str | None:
    """检测当前页面是否命中 LOGIN_SIGNALS，返回命中的第一个信号，否则 None。"""
    for signal in LOGIN_SIGNALS:
        try:
            count = await page.locator(signal).count()
        except PlaywrightError:
            # 个别 signal 在部分浏览器引擎语法不兼容时跳过，不影响其余信号检测
            continue
        if count > 0:
            return signal
    return None


async def browser_open(page, url: str, config: AgentConfig) -> ToolResult:
    """打开 URL；若命中登录页信号，暂停请求人工确认是否继续。"""
    try:
        await page.goto(url, timeout=config.browser_timeout)
    except Exception as exc:
        return ToolResult(
            success=False,
            page_changed=False,
            output=None,
            error_msg=f"打开页面失败: {exc}",
        )

    signal = await _detect_login_page(page)
    if signal is not None:
        choice = await ask_human(
            reason=f"检测到登录页信号 `{signal}`，是否继续在该页面执行操作？",
            options=["continue", "abort"],
        )
        if choice != "continue":
            # 人工选择中止：作为安全拦截向上抛出，不在此处吞掉
            raise SafetyError(
                "用户在登录页确认环节选择中止",
                trigger="login_page",
                url=page.url,
            )

    return ToolResult(success=True, page_changed=True, output=page.url, error_msg=None)


async def browser_observe(page, observer: BrowserStateObserver) -> ObserveResult:
    """代理 observer.observe(page)，采集失败时包装为 BrowserError 向上抛出。"""
    try:
        return await observer.observe(page)
    except SafetyError:
        raise
    except Exception as exc:
        raise BrowserError("页面观察失败", action="observe") from exc


async def _try_click_css(page, selector: str, timeout: int) -> None:
    await page.locator(selector).click(timeout=timeout)


async def _try_click_text(page, text: str, timeout: int) -> None:
    await page.get_by_text(text).click(timeout=timeout)


async def _try_click_role(page, text: str, timeout: int) -> str:
    """依次尝试 button / link 两种常见 role，返回实际命中的 role 名。"""
    errors: list[Exception] = []
    for role in ("button", "link"):
        try:
            await page.get_by_role(role, name=text).click(timeout=timeout)
            return role
        except PlaywrightError as exc:
            errors.append(exc)
    if errors:
        raise errors[-1]
    raise BrowserError("role 降级未命中任何候选")


async def browser_click(
    page,
    selector: str | None = None,
    text: str | None = None,
    config: AgentConfig | None = None,
) -> ToolResult:
    """三级降级点击：CSS selector → get_by_text(text) → get_by_role()。

    ToolResult.output 中以 JSON 字符串记录实际命中的 selector_level，
    供 TraceLogger 写入 trace.jsonl 时使用。
    """
    timeout = config.browser_timeout if config else 15000
    url_before = page.url

    attempts: list[tuple[str, Callable[[], Awaitable[str | None]]]] = []
    if selector:
        selector_value: str = selector
        attempts.append(("css", lambda: _try_click_css(page, selector_value, timeout)))
    if text:
        text_value: str = text
        attempts.append(("text", lambda: _try_click_text(page, text_value, timeout)))
        attempts.append(("role", lambda: _try_click_role(page, text_value, timeout)))

    if not attempts:
        return ToolResult(
            success=False,
            page_changed=False,
            output=None,
            error_msg="browser_click 需要至少提供 selector 或 text 之一",
        )

    last_error: Exception | None = None
    for level, attempt in attempts:
        try:
            resolved_role = await attempt()
            page_changed = page.url != url_before
            output = json.dumps(
                {
                    "selector_level": level,
                    "resolved_role": resolved_role if level == "role" else None,
                },
                ensure_ascii=False,
            )
            return ToolResult(
                success=True,
                page_changed=page_changed,
                output=output,
                error_msg=None,
            )
        except Exception as exc:
            last_error = exc
            continue

    return ToolResult(
        success=False,
        page_changed=False,
        output=None,
        error_msg=f"三级降级均未命中可点击元素: {last_error}",
    )


async def browser_type(page, selector: str, text: str) -> ToolResult:
    """向指定 selector 填入文本；命中敏感字段直接抛出 SafetyError（不吞掉）。"""
    _check_sensitive(selector)  # 命中则在此处直接抛出，穿透工具边界

    try:
        await page.locator(selector).fill(text)
    except Exception as exc:
        return ToolResult(
            success=False,
            page_changed=False,
            output=None,
            error_msg=f"输入失败: {exc}",
        )

    return ToolResult(success=True, page_changed=False, output=selector, error_msg=None)


# 复用单一客户端实例，避免每次调用都重新建立连接池
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("未配置 ANTHROPIC_API_KEY，无法调用 LLM 抽取", stage="request")
        _client = anthropic.Anthropic(api_key=api_key)
    assert _client is not None  # 帮助静态类型检查器收窄为非 Optional
    return _client


def _call_llm_extract(prompt: str, config: AgentConfig, system: str | None = None) -> str:
    """调用 Anthropic Messages API（官方 SDK），返回模型原始文本输出。

    独立为模块级函数（而非内嵌在 browser_extract 中）便于测试时替换/打桩。
    SDK 自带超时与网络层重试，这里在其上再叠加一层业务级重试（config.llm_retry），
    用于覆盖偶发的可重试错误。

    system: 角色约束（如 agent.prompts.EXTRACTOR_SYSTEM），通过 Anthropic
        Messages API 的 system 参数单独传递，不与用户消息拼在一起，
        便于角色指令被模型稳定遵循，也便于不同调用方复用同一份 prompt 定义。
    """
    client = _get_client()

    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    last_exc: Exception | None = None
    for _ in range(max(1, config.llm_retry)):
        try:
            # 直接以关键字参数调用（而非 **kwargs 拼装 dict），才能让类型检查器
            # 根据重载签名正确推断出非流式返回类型 Message，而不是
            # Message | Stream[...] 联合类型；system 未提供时传 NOT_GIVEN 哨兵值
            # （而不是省略该参数），同样是为了保持这是一次静态可解析的直接调用。
            message: Message = client.messages.create(
                model=config.model,
                max_tokens=1024,
                timeout=config.llm_timeout,
                system=system if system else anthropic.NOT_GIVEN,
                messages=messages,
            )
        except anthropic.APIError as exc:
            last_exc = exc
            continue

        # 只有 TextBlock 才有 .text 属性，其余 block 类型（ThinkingBlock/ToolUseBlock 等）
        # 需要先用 isinstance 收窄类型，避免静态检查器报“成员不存在”。
        text_parts = [block.text for block in message.content if isinstance(block, TextBlock)]
        return "".join(text_parts)

    raise LLMError(f"LLM 请求失败: {last_exc}", stage="request", retry_count=config.llm_retry)


async def browser_extract(
    page,
    instruction: str,
    observer: BrowserStateObserver,
    config: AgentConfig,
) -> ToolResult:
    """依据 instruction 从当前页面文本中抽取结构化 JSON（保留来源 URL）。"""
    try:
        obs = await observer.observe(page)
    except Exception as exc:
        return ToolResult(
            success=False,
            page_changed=False,
            output=None,
            error_msg=f"观察页面失败，无法抽取: {exc}",
        )

    prompt = EXTRACTOR_USER_TMPL.format(
        instruction=instruction,
        title=obs["title"],
        visible_text_summary=obs["visible_text_summary"],
    )

    try:
        raw_output = _call_llm_extract(prompt, config, system=EXTRACTOR_SYSTEM)
    except LLMError as exc:
        return ToolResult(success=False, page_changed=False, output=None, error_msg=exc.message)
    except Exception as exc:
        return ToolResult(
            success=False, page_changed=False, output=None, error_msg=f"LLM 调用异常: {exc}"
        )

    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return ToolResult(
            success=False,
            page_changed=False,
            output=None,
            error_msg=f"LLM 输出无法解析为 JSON: {exc}",
        )

    result_payload = {"url": obs["url"], "data": parsed}
    return ToolResult(
        success=True,
        page_changed=False,
        output=json.dumps(result_payload, ensure_ascii=False),
        error_msg=None,
    )


async def browser_scroll(page, direction: Literal["up", "down"]) -> ToolResult:
    """按方向滚动一屏，通过对比 scrollY（页面纵向滚动偏移量）判断视口是否实际发生变化。"""
    key = "PageDown" if direction == "down" else "PageUp"
    try:
        before = await page.evaluate("() => window.scrollY")
        await page.keyboard.press(key)
        await page.wait_for_timeout(200)
        after = await page.evaluate("() => window.scrollY")
    except Exception as exc:
        return ToolResult(
            success=False,
            page_changed=False,
            output=None,
            error_msg=f"滚动失败: {exc}",
        )

    return ToolResult(
        success=True,
        page_changed=(before != after),
        output=str(after),
        error_msg=None,
    )


async def browser_screenshot(page, tracer: TraceLogger) -> ToolResult:
    """截图并保存到 tracer 分配的路径。"""
    screenshot_path = tracer.next_screenshot_path()
    try:
        await page.screenshot(path=screenshot_path)
    except Exception as exc:
        return ToolResult(
            success=False,
            page_changed=False,
            output=None,
            error_msg=f"截图失败: {exc}",
        )

    return ToolResult(
        success=True, page_changed=False, output=screenshot_path, error_msg=None
    )


async def ask_human(reason: str, options: list[str]) -> str:
    """终端打印提示，阻塞等待用户从 options 中选择一项并返回选择结果。"""
    print("\n[需要人工确认]")
    print(f"原因: {reason}")
    for idx, opt in enumerate(options, start=1):
        print(f"  {idx}. {opt}")

    while True:
        raw = input(f"请输入选项编号 (1-{len(options)}) 或直接输入选项文本: ").strip()
        if raw in options:
            return raw
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("输入无效，请重新输入。")
