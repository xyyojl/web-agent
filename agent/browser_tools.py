"""浏览器操作工具集：8 个供 LLM 调用的异步工具函数。

设计原则：
- 除 SafetyError（安全拦截）外，所有异常都在工具内部捕获，
  转换为 ToolResult(success=False, error_msg=...)，不向上抛出未捕获异常。
- SafetyError 是唯一允许穿透工具边界的异常类型，交由上层 agent loop
  决定是否终止任务 —— 这是刻意设计，安全拦截不应被静默吞掉。
"""

import asyncio
import json
import logging
import re
from typing import Awaitable, Callable, Literal

from anthropic.types import Message, TextBlock
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from agent.config import AgentConfig
from agent.exceptions import BrowserError, LLMError, SafetyError
from agent.llm_client import LLMClient, LLMOutputRetry
from agent.observer import BrowserStateObserver
from agent.prompts import EXTRACTOR_SYSTEM, EXTRACTOR_USER_TMPL
from agent.tracer import TraceLogger
from agent.types import ObserveResult, ToolResult

logger = logging.getLogger(__name__)

# 独立抓取页面上所有 <a> 标签的 text + href，供 browser_extract 使用。
_EXTRACT_LINKS_JS = """
() => {
    const isVisible = (el) => {
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        return (
            rect.width > 0 &&
            rect.height > 0 &&
            style.display !== "none" &&
            style.visibility !== "hidden"
        );
    };
    return Array.from(document.querySelectorAll("a"))
        .filter(isVisible)
        .map((el) => ({
            text: (el.innerText || "").trim(),
            href: el.getAttribute("href"),
        }));
}
"""

# 敏感字段正则：匹配到即认为是密码/卡号/证件号等字段，禁止 browser_type 写入。
SENSITIVE_PATTERNS = [
    r"password", r"passwd", r"pwd",
    r"credit.?card", r"card.?number", r"cvv",
    r"id.?number", r"national.?id", r"ssn",
    r"bank.?account",
    # 中文敏感字段（不含"密码"——密码类字段由 type=password / autocomplete 判定，
    # 避免"密码找回说明"等合法帮助文本被误伤，[R2-5]）
    r"银行卡", r"银行账号", r"银行账户",
    r"信用卡", r"安全码", r"身份证",
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


# autocomplete 值中明确标识密码类字段的取值
_PASSWORD_AUTOCOMPLETE_VALUES = {"current-password", "new-password"}

# 多元素匹配时记录 warning 的阈值 [R2-1]
_MAX_ELEMENT_WARNING_THRESHOLD = 50

# 读取元素安全相关属性的 JS 表达式（在页面上下文执行）
_CHECK_SENSITIVE_ATTRS_JS = """
(elements) => {
    const results = [];
    for (const el of elements) {
        const attrs = {
            tag_name: el.tagName.toLowerCase(),
            type: el.getAttribute('type'),
            id: el.getAttribute('id'),
            name: el.getAttribute('name'),
            autocomplete: el.getAttribute('autocomplete'),
            aria_label: el.getAttribute('aria-label'),
            placeholder: el.getAttribute('placeholder'),
            label_text: null,
        };

        // 关联 label 判定——三种方式 [R2-2]
        // 方式一：<label for=ID> 与目标 id 匹配
        if (attrs.id) {
            const label = document.querySelector('label[for="' + attrs.id + '"]');
            if (label) {
                attrs.label_text = (label.textContent || '').trim() || null;
            }
        }

        // 方式二：目标元素被 <label> 标签包裹
        if (!attrs.label_text) {
            const parentLabel = el.closest('label');
            if (parentLabel) {
                attrs.label_text = (parentLabel.textContent || '').trim() || null;
            }
        }

        // 方式三：aria-labelledby 指向的元素文本
        if (!attrs.label_text) {
            const labelledby = el.getAttribute('aria-labelledby');
            if (labelledby) {
                const labelEl = document.getElementById(labelledby);
                if (labelEl) {
                    attrs.label_text = (labelEl.textContent || '').trim() || null;
                }
            }
        }

        results.push(attrs);
    }
    return results;
}
"""


def _find_sensitive_evidence(attrs: dict[str, str | None]) -> list[str]:
    """检查单个元素的属性是否命中敏感特征，返回命中规则列表。"""
    hits = []

    # 1. type=password → 明确的密码输入框
    type_val = (attrs.get("type") or "").lower()
    if type_val == "password":
        hits.append("type=password")

    # 2. autocomplete 包含 current-password / new-password
    autocomplete_val = (attrs.get("autocomplete") or "").lower()
    if autocomplete_val in _PASSWORD_AUTOCOMPLETE_VALUES:
        hits.append(f"autocomplete={autocomplete_val}")

    # 3. 其他属性命中敏感正则
    for attr_name in ("id", "name", "aria_label", "placeholder", "label_text"):
        val = attrs.get(attr_name)
        if val and _SENSITIVE_RE.search(val):
            hits.append(f"{attr_name}={val!r}")

    return hits


async def _check_sensitive_attributes(page, selector: str) -> list[dict]:
    """读取目标元素的安全相关属性，返回命中敏感证据的元素列表。

    遍历所有匹配元素 [R2-1]，任意一个命中敏感特征即包含在返回列表中。
    属性读取失败时抛出 PlaywrightError，由调用方决定如何处理。

    不得写入页面、不得执行 fill()、不得吞掉 SafetyError。
    """
    locator = page.locator(selector)

    count = await locator.count()

    if count == 0:
        # 无匹配元素：不是安全问题，让 fill() 自然失败即可
        return []

    if count > _MAX_ELEMENT_WARNING_THRESHOLD:
        logger.warning(
            "selector=%s 匹配到 %d 个元素，超过阈值 %d，仍继续检查",
            selector, count, _MAX_ELEMENT_WARNING_THRESHOLD,
        )

    attrs_list = await locator.evaluate_all(_CHECK_SENSITIVE_ATTRS_JS)

    # count > 0 但 evaluate_all 返回空列表：元素可能位于 iframe 或
    # shadow DOM 内无法读取属性，按"属性读取失败"处理 [R2-3]
    if not attrs_list:
        raise PlaywrightError(
            f"selector={selector} 匹配到 {count} 个元素但无法读取属性"
            "（可能位于 iframe 或 shadow DOM 内）"
        )

    evidence = []
    for i, attrs in enumerate(attrs_list):
        hits = _find_sensitive_evidence(attrs)
        if hits:
            evidence.append({
                "element_index": i,
                "hits": hits,
            })

    return evidence


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
    """打开 URL；若命中登录页信号，暂停请求人工确认是否继续。

    页面加载/网络超时统一识别为 PlaywrightTimeoutError，包装为 BrowserError
    并记录 warning。超时属于典型的瞬时性故障（网络抖动、目标站点响应慢），
    单次失败不代表页面真的打不开，因此这里按 config.open_retry 做少量重试
    （固定 2s 间隔，不用指数退避——重试次数少，没必要让等待时间迅速拉长）；
    重试仍失败才返回 ToolResult(success=False, error_msg="timeout")。
    其他类型的错误（DNS 解析失败、证书错误等）大概率不是瞬时问题，重试
    也无济于事，不重试，直接返回失败并记录 error_msg。
    """
    max_attempts = max(1, config.open_retry + 1)
    for attempt in range(max_attempts):
        will_retry = attempt + 1 < max_attempts
        try:
            await page.goto(url, timeout=config.browser_timeout)
            break
        except PlaywrightTimeoutError as exc:
            browser_err = BrowserError(
                "打开页面超时",
                action="goto",
                selector=None,
                timeout_ms=config.browser_timeout,
            )
            logger.warning(
                "browser_open 超时（第 %d/%d 次尝试）: url=%s, timeout_ms=%d, detail=%s%s",
                attempt + 1,
                max_attempts,
                url,
                config.browser_timeout,
                browser_err,
                "，2s 后重试" if will_retry else "，已达重试上限",
                exc_info=exc,
            )
            if will_retry:
                await asyncio.sleep(2)
                continue
            return ToolResult(
                success=False,
                page_changed=False,
                output=None,
                error_msg="timeout",
            )
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
            # exact=True 按 role + 精确 name 命中唯一元素，避免子串匹配
            # 触发 Playwright strict mode violation（多个候选时报错）。
            await page.get_by_role(role, name=text, exact=True).click(timeout=timeout)
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
    # 这里在 selector 形如 "text=xxx" 且 text 字段为空时，拆出 xxx 灌进
    # text/role 降级链，让三级真正跑全；role 那级按 role+name 精准命中唯一按钮。
    fallback_text: str | None = None
    if selector and "=" in selector:
        prefix, _, raw_value = selector.partition("=")
        if prefix.strip() == "text" and raw_value.strip():
            fallback_text = raw_value.strip()
    effective_text = text or fallback_text

    if selector:
        selector_value: str = selector
        attempts.append(("css", lambda: _try_click_css(page, selector_value, timeout)))
    if effective_text:
        text_value: str = effective_text
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
            if page_changed:
                # 点击触发了跳转：等待新页面网络空闲，降低"跳转后元素消失/
                # 半成品 DOM"导致后续 observe/click 误判的概率。networkidle
                # 超时不视为点击失败——点击动作本身已经成功，这里只是尽力
                # 让页面稳定下来，超时就降级放行，记录 warning 供排查。
                try:
                    await page.wait_for_load_state("networkidle", timeout=5000)
                except PlaywrightTimeoutError as wait_exc:
                    logger.warning(
                        "browser_click 后等待 networkidle 超时（跳转已发生），降级继续: %s",
                        wait_exc,
                    )
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


async def browser_select(page, selector: str, value: str) -> ToolResult:
    """在 <select> 元素中选中指定 value/label 对应的选项。

    新增：此前的动作集合（click/type/scroll/extract/screenshot/done）里
    没有专门针对下拉框的操作——click 一个 <select> 元素在 headless
    Chromium 下不会像原生浏览器那样弹出可交互的选项列表，语义上就无法
    "选中某个 option"，无论 selector 换成什么、重试多少次都不可能成功。

    select_option 会依次尝试按 value / label / index 匹配，这里直接把
    ActionSelector 传来的显示文本（如"English"）交给 Playwright 处理，
    覆盖 value 属性与可见文本不一致的情况（如 value="en", label="English"）。
    """
    try:
        await page.locator(selector).select_option(label=value)
    except PlaywrightError:
        # label 匹配失败时，退化尝试直接按 value 属性匹配
        # （例如 LLM 传来的就是底层 value 而非显示文本）
        try:
            await page.locator(selector).select_option(value=value)
        except Exception as exc:
            return ToolResult(
                success=False,
                page_changed=False,
                output=None,
                error_msg=f"下拉框选择失败: {exc}",
            )

    return ToolResult(success=True, page_changed=False, output=value, error_msg=None)


async def browser_type(page, selector: str, text: str) -> ToolResult:
    """向指定 selector 填入文本；命中敏感字段直接抛出 SafetyError（不吞掉）。

    安全检查分两道：
    1. _check_sensitive(selector) —— selector 文本正则快速拒绝；
    2. _check_sensitive_attributes(page, selector) —— 读取元素真实属性，
       覆盖 selector 不含敏感词但元素实际是密码/银行卡等敏感字段的场景（DS-R2）。
    """
    _check_sensitive(selector)  # 第一道：selector 正则快速拒绝

    # 第二道：元素真实属性检查
    try:
        evidence = await _check_sensitive_attributes(page, selector)
    except PlaywrightError as exc:
        # 属性读取失败：不得默认放行，返回失败 ToolResult [R2-3]
        logger.warning("敏感字段安全检查失败，拒绝写入: selector=%s, error=%s", selector, exc)
        return ToolResult(
            success=False,
            page_changed=False,
            output=None,
            error_msg=f"无法完成敏感字段安全检查: {exc}",
        )

    if evidence:
        # 命中敏感属性：抛出 SafetyError，不调用 fill()
        evidence_str = "; ".join(
            f"element[{e['element_index']}]: {', '.join(e['hits'])}"
            for e in evidence
        )
        raise SafetyError(
            "检测到敏感字段，拒绝写入",
            trigger="sensitive_field",
            selector=selector,
            evidence=evidence_str,
        )

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


def _parse_extract_response(message: Message) -> object:
    """从 Extractor 返回的 Message 里解析出结构化 JSON（LLMClient 的 parse_response 回调）。

    合并了此前分两处处理的"这次输出不算数"情况：网络层失败已经由
    LLMClient.call_with_retry() 兜底，这里只处理"HTTP 成功但内容不可用"的
    情况——非法 JSON 通过 raise LLMOutputRetry 交给 call_with_retry() 同一组
    config.llm_retry 预算重试。

    此前 browser_extract 是自己在外层又套了一圈 max_attempts 循环包住
    对 LLMClient 的调用：LLMClient 内部失败已经重试 llm_retry 次，外层
    JSON 解析失败又触发一整轮新的 LLMClient 调用（内部又是 llm_retry
    次），最坏情况下实际请求次数被放大到 llm_retry² 次，且远超其余三处
    调用方（Planner/ActionSelector/Verifier）的重试预算，与"统一重试
    策略"的设计初衷相悖。合并成这一个回调、只让 call_with_retry() 跑
    一轮，就和其余三处调用方完全一致了。
    """
    # 只有 TextBlock 才有 .text 属性，其余 block 类型（ThinkingBlock/ToolUseBlock 等）
    # 需要先用 isinstance 收窄类型，避免静态检查器报“成员不存在”。
    text_parts = [block.text for block in message.content if isinstance(block, TextBlock)]
    raw_output = "".join(text_parts)

    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMOutputRetry(f"LLM 输出无法解析为 JSON（输出长度={len(raw_output)}）: {exc}") from exc


async def _format_links_info(page) -> str:
    """抓取页面上所有可见 <a> 标签的 text/href，渲染成供 Extractor 阅读的文本块。

    抓取失败（页面异常/超时）时静默降级返回空字符串，不影响抽取主流程——
    没有链接列表时 Extractor 退回到"看不到就填 null"的原有行为，
    比让整个 extract 工具失败更稳妥。
    """
    try:
        raw_links: list[dict] = await page.evaluate(_EXTRACT_LINKS_JS)
    except PlaywrightError:
        return ""

    if not raw_links:
        return ""

    lines = ["页面链接列表（text 与 href 均为真实属性，抽取链接相关字段时必须从此处原样取值，不得编造）："]
    for item in raw_links:
        text = (item.get("text") or "").strip() or "(无文本)"
        href = item.get("href")
        lines.append(f'  - text="{text}" href={href!r}')
    return "\n".join(lines)


async def browser_extract(
    page,
    instruction: str,
    observer: BrowserStateObserver,
    config: AgentConfig,
    obs: ObserveResult | None = None,
) -> ToolResult:
    """根据当前页面抽取结构化 JSON。

    只重试两类"客观可判定"的失败：LLM 调用本身失败、输出不是合法 JSON——
    这两类都统一交给 LLMClient.call_with_retry() 处理（前者是网络层异常，
    后者由 _parse_extract_response 转换成 LLMOutputRetry），一次调用跑完
    config.llm_retry 次预算，不再像此前那样在外层额外套一层重试循环。
    不再在这里对字段名做启发式校验——原实现从 instruction/task 里用正则
    抓取带引号的 token 当作"必须出现的字段名"，但 instruction 是
    Planner/ActionSelector 当场转述的自由文本，其中举例用的引号内容
    （比如 "字段 'name'（如 'Alice'）"里的 'Alice'）和真正的字段名在
    正则层面无法区分，会被一并当成"必需字段"，导致即使模型输出的
    name/score 完全正确，也会被误判为"缺字段"而不断重试、甚至把模型
    误导去输出根本不该存在的字段。是否符合预期的字段结构，交给任务
    结束后 Verifier 基于 EvalCase.verify_mode（如 json_schema）统一
    校验即可——那一层校验源头是 case 文件里确定性的 expected_output，
    不依赖对自由文本的正则猜测，更可靠。
    """
    if obs is None:
        try:
            obs = await observer.observe(page)
        except Exception as exc:
            return ToolResult(
                success=False,
                page_changed=False,
                output=None,
                error_msg=f"观察页面失败，无法抽取: {exc}",
            )

    base_prompt = EXTRACTOR_USER_TMPL.format(
        instruction=instruction,
        title=obs["title"],
        visible_text_summary=obs["visible_text_summary"],
        links_info=await _format_links_info(page),
    )

    try:
        parsed = await LLMClient(config).call_with_retry(
            caller_name="browser_extract",
            parse_response=_parse_extract_response,
            model=config.model,
            max_tokens=2048,
            system=EXTRACTOR_SYSTEM,
            messages=[{"role": "user", "content": base_prompt}],
        )
    except LLMError as exc:
        return ToolResult(success=False, page_changed=False, output=None, error_msg=exc.message)
    except Exception as exc:
        # call_with_retry() 正常情况下失败耗尽会统一包装成 LLMError，这里兜底的是
        # 边界情况：比如 parse_response（_parse_extract_response）内部抛出了
        # LLMOutputRetry 之外的异常、或底层 SDK 抛出了不属于 anthropic.APIError
        # 体系的异常，这类异常不会被 call_with_retry() 转换，会直接穿透到这里。
        return ToolResult(success=False, page_changed=False, output=None, error_msg=f"LLM 调用异常: {exc}")

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
    """终端打印提示，阻塞等待用户从 options 中选择一项并返回选择结果。

    input() 本身是同步阻塞调用，会独占整个事件循环；
    用 asyncio.to_thread 丢到线程池执行，让事件循环在等待用户输入期间
    仍可调度其他协程（如浏览器侧的后台任务）。
    """

    def _prompt_blocking() -> str:
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

    return await asyncio.to_thread(_prompt_blocking)
