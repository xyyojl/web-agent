"""决策层：ActionSelector 基于 Planner 给出的自然语言计划（plan），
通过 Anthropic Tool Calling 输出一次结构化动作（LLMAction）。

selector 的合法性完全依赖 SELECTOR_SYSTEM 的约束（必须从交互元素列表中
原样选取），本模块只负责把 tool_use block 解析成 LLMAction，不做
selector 是否真实存在于页面的校验——那是 browser_click/browser_type
执行时的职责。
"""

import asyncio
import logging
import os
from typing import Literal, cast

import anthropic
from anthropic.types import (
    Message,
    MessageParam,
    ToolChoiceAnyParam,
    ToolParam,
    ToolUseBlock,
)

from agent.config import AgentConfig
from agent.exceptions import LLMError
from agent.prompts import SELECTOR_SYSTEM, SELECTOR_TOOLS
from agent.types import LLMAction, ObserveResult
from agent.vision import build_vision_user_content

logger = logging.getLogger(__name__)

_VALID_ACTIONS = ("click", "type", "scroll", "extract", "screenshot", "select", "done")


def _format_observation(obs: ObserveResult) -> str:
    """给 ActionSelector 看的观察结果：显式列出完整 selector。

    与 WebPlanner._format_observation 刻意相反——ActionSelector 的职责
    就是从候选 selector 中原样挑一个填进 tool_use，必须能看到它们。
    """
    lines = [
        f"URL: {obs['url']}",
        f"标题: {obs['title']}",
        "当前页面交互元素（selector 必须从下列列表中原样选取，禁止臆造/拼接）：",
    ]
    elements = obs["interactive_elements"]
    if not elements:
        lines.append("  （未检测到交互元素，如需操作请优先考虑 scroll 或 screenshot）")
    else:
        for idx, el in enumerate(elements, start=1):
            name = el["name"] or "(无文本)"
            lines.append(
                f'  {idx}. role={el["role"]} name="{name}" selector={el["selector"]}'
            )
    return "\n".join(lines)


def _extract_tool_use(message: Message) -> ToolUseBlock | None:
    """从 message.content 中找出第一个 tool_use block。

    content 可能为空列表、只含 text block、或因模型异常未触发工具调用，
    这些情况统一返回 None，由调用方按“解析失败”处理并触发 retry，
    不在此处抛异常中断流程。
    """
    for block in message.content or []:
        if isinstance(block, ToolUseBlock):
            return block
    return None


def _get_str_field(raw_input: dict[str, object], key: str) -> str | None:
    """从 tool_use.input（类型是 Dict[str, object]）中取出一个字段并窄化为 str。

    直接用 raw_input.get(key) 拿到的是 object | None，无法赋给 LLMAction 里
    selector/text/value: str | None 的字段；这里统一做 isinstance 收窄，
    既满足静态类型检查器，也顺带拦掉了 LLM 返回非字符串类型（如数字/布尔）
    这种运行时异常情况，统一转成 ValueError 触发上层 retry。
    """
    value = raw_input.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"字段 {key} 期望是字符串类型，实际是 {type(value).__name__}")
    return value


def _build_llm_action(tool_use: ToolUseBlock) -> LLMAction:
    """把 tool_use block 转成 LLMAction。

    任何字段缺失/类型不对/取值非法都抛 ValueError，由调用方统一转换成
    LLMError 触发 retry，绝不让一个格式不对的 tool_use 直接崩溃整条任务链路。
    """
    name = tool_use.name
    if name not in _VALID_ACTIONS:
        raise ValueError(f"未知的 tool 名称: {name!r}")
    action = cast(
        Literal["click", "type", "scroll", "extract", "screenshot", "select", "done"], name
    )

    raw_input = tool_use.input
    if not isinstance(raw_input, dict):
        raise ValueError(f"tool_use.input 不是合法的 dict，实际是 {type(raw_input).__name__}")

    reason = _get_str_field(raw_input, "reason")
    if not reason or not reason.strip():
        raise ValueError(f"[{name}] 缺少非空的 reason 字段")

    selector: str | None = None
    text: str | None = None
    value: str | None = None

    if action == "click":
        selector = _get_str_field(raw_input, "selector")
        text = _get_str_field(raw_input, "text")
        if not selector and not text:
            raise ValueError("[click] selector 和 text 至少需要提供一个")
    elif action == "type":
        selector = _get_str_field(raw_input, "selector")
        text = _get_str_field(raw_input, "text")
        if not selector:
            raise ValueError("[type] 缺少必填的 selector 字段")
        if not text:
            raise ValueError("[type] 缺少必填的 text 字段")
    elif action == "scroll":
        direction = _get_str_field(raw_input, "direction")
        if direction not in ("up", "down"):
            raise ValueError(f"[scroll] direction 字段非法: {direction!r}")
        value = direction
    elif action == "extract":
        instruction = _get_str_field(raw_input, "instruction")
        if not instruction:
            raise ValueError("[extract] 缺少必填的 instruction 字段")
        value = instruction
    elif action == "select":
        selector = _get_str_field(raw_input, "selector")
        value = _get_str_field(raw_input, "value")
        if not selector:
            raise ValueError("[select] 缺少必填的 selector 字段")
        if not value:
            raise ValueError("[select] 缺少必填的 value 字段")
    elif action == "done":
        done_value = _get_str_field(raw_input, "value")
        if not done_value:
            raise ValueError("[done] 缺少必填的 value 字段")
        value = done_value
    # screenshot：除 reason 外无其他必填字段，selector/text/value 保持 None

    return LLMAction(action=action, selector=selector, text=text, value=value, reason=reason)


class ActionSelector:
    """决策层：把 plan + observation 转成一次 Tool Calling 结构化动作。"""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMError(
                    "未配置 ANTHROPIC_API_KEY，无法调用 ActionSelector", stage="request"
                )
            self._client = anthropic.AsyncAnthropic()
        assert self._client is not None  # 帮助静态类型检查器收窄为非 Optional
        return self._client

    async def select(
        self,
        plan: str,
        obs: ObserveResult,
        history: list[dict],
    ) -> LLMAction:
        """调用 LLM（Tool Calling 模式），返回一次结构化动作。

        失败时按 1s / 2s / 4s ... 指数退避重试，最多 config.llm_retry 次，
        超过后抛出 LLMError。以下情况均视为一次失败并触发 retry：
        网络/API 错误、response.content 为空或无 tool_use block、
        tool_use.name 不在合法工具集合内、input 字段缺失或取值非法。
        """
        client = self._get_client()

        user_content = build_vision_user_content(obs, f"行动计划：{plan}\n\n{_format_observation(obs)}")
        messages: list[MessageParam] = cast(
            "list[MessageParam]", [*history, {"role": "user", "content": user_content}]
        )
        tools: list[ToolParam] = cast("list[ToolParam]", SELECTOR_TOOLS)
        # 用 SDK 提供的 ToolChoiceAnyParam 类型构造（而非裸 dict），
        # 才能让类型检查器沿着正确的重载分支推断出非流式返回类型 Message，
        # 否则会退化成 Message | AsyncStream[...] 联合类型警告。
        tool_choice: ToolChoiceAnyParam = {"type": "any"}

        last_exc: Exception | None = None
        max_attempts = max(1, self.config.llm_retry)

        for attempt in range(max_attempts):
            will_retry = attempt + 1 < max_attempts

            try:
                # tool_choice={"type": "any"} 强制模型必须调用其中一个工具，
                # 而不是退化成纯文本回复——从源头降低“content 里没有 tool_use”的概率。
                message: Message = await client.messages.create(
                    model=self.config.model,
                    max_tokens=1024,
                    timeout=self.config.llm_timeout,
                    system=SELECTOR_SYSTEM,
                    tools=tools,
                    tool_choice=tool_choice,
                    messages=messages,
                )
            except anthropic.RateLimitError as exc:
                # 429 速率限制：免费模型速率窗口通常为 1 分钟，
                # 短时间指数退避反而加剧频率压力，改用固定的 rate_limit_delay 则更可靠。
                last_exc = exc
                wait_secs = self.config.rate_limit_delay
                logger.warning(
                    "ActionSelector LLM 调用失败（第 %d/%d 次尝试）: %s%s",
                    attempt + 1,
                    max_attempts,
                    exc,
                    f"，即将等待 {wait_secs}s 后重试" if will_retry else "，已达重试上限",
                )
                if will_retry:
                    await asyncio.sleep(wait_secs)
                continue
            except anthropic.APIError as exc:
                last_exc = exc
                logger.warning(
                    "ActionSelector LLM 调用失败（第 %d/%d 次尝试）: %s%s",
                    attempt + 1,
                    max_attempts,
                    exc,
                    "，即将重试" if will_retry else "，已达重试上限",
                )
                if will_retry:
                    await asyncio.sleep(2**attempt)  # 1s / 2s / 4s / ...
                continue

            tool_use = _extract_tool_use(message)
            if tool_use is None:
                last_exc = LLMError(
                    "response.content 为空或未找到 tool_use block", stage="parse"
                )
                logger.warning(
                    "ActionSelector 未解析到 tool_use（第 %d/%d 次尝试）%s",
                    attempt + 1,
                    max_attempts,
                    "，即将重试" if will_retry else "，已达重试上限",
                )
                if will_retry:
                    await asyncio.sleep(2**attempt)
                continue

            try:
                action = _build_llm_action(tool_use)
            except (ValueError, TypeError, KeyError) as exc:
                # ValueError: _get_str_field/字段校验主动抛出的“非法值”。
                # TypeError/KeyError: tool_use.input 结构本身不符合预期
                # （例如 SDK 返回的 input 不是预期的 dict 结构、字段缺失导致的
                # 底层 dict 操作异常）——同样视为一次“LLM 返回非法 action”，
                # 统一归一为 LLMError 触发 retry，而不是让整条任务链路崩溃。
                last_exc = LLMError(f"tool_use 解析失败: {exc}", stage="parse")
                logger.warning(
                    "ActionSelector tool_use 解析失败（第 %d/%d 次尝试）: %s%s",
                    attempt + 1,
                    max_attempts,
                    exc,
                    "，即将重试" if will_retry else "，已达重试上限",
                )
                if will_retry:
                    await asyncio.sleep(2**attempt)
                continue

            return action

        last_exc_desc = last_exc.message if isinstance(last_exc, LLMError) else str(last_exc)
        raise LLMError(
            f"ActionSelector LLM 请求连续失败 {max_attempts} 次: {last_exc_desc}",
            stage="request",
            retry_count=max_attempts,
        )
