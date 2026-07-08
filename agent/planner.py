"""推理层：WebPlanner 调用 LLM，把「当前页面状态 + 任务目标」转成一段
自然语言行动计划（plan），不涉及具体 selector —— selector 的解析工作
交给下游 ActionSelector（PROMPT-001 中 PLANNER_SYSTEM 已明确约束这一点）。
"""

import asyncio
import logging
import os
from typing import cast

import anthropic
from anthropic.types import Message, MessageParam, TextBlock

from agent.config import AgentConfig
from agent.exceptions import LLMError
from agent.prompts import PLANNER_SYSTEM, PLANNER_USER_TMPL
from agent.types import ObserveResult

logger = logging.getLogger(__name__)

# 用于在验收/日志中提示"plan 疑似混入了 selector 语法"的轻量检测，
# 不做硬性拦截（避免误判合法文本导致可用的 plan 被丢弃），仅记录 warning
# 供 PLAN-001 验收标准第 2 条人工核查使用。
_SELECTOR_SYNTAX_MARKERS = (">>", "[data-testid=", "css=", "xpath=")


def _format_observation(obs: ObserveResult) -> str:
    """把 ObserveResult 渲染成给 Planner 看的纯文本描述。

    刻意只展示交互元素的 role/name，不展示 selector 字符串——
    Planner 的上下文里根本不存在 selector，从源头上降低它在 plan 里
    "抄" 出 CSS/XPath 语法的概率，比单纯靠 Prompt 约束更可靠。
    """
    lines = [
        f"URL: {obs['url']}",
        f"标题: {obs['title']}",
        f"可见文本摘要: {obs['visible_text_summary']}",
        "当前页面交互元素（仅角色与文本，不含定位信息）：",
    ]
    elements = obs["interactive_elements"]
    if not elements:
        lines.append("  （未检测到交互元素）")
    else:
        for idx, el in enumerate(elements, start=1):
            name = el["name"] or "(无文本)"
            # Planner 判断"任务已完成/可直接给出答案"时能看到真实目标，
            # 而不是像此前那样看不到 href 只能靠 LLM 常识编造。
            href_info = f" href={el['href']}" if el.get("href") else ""
            lines.append(f"  {idx}. [{el['role']}] {name}{href_info}")
    return "\n".join(lines)


def _contains_selector_syntax(text: str) -> bool:
    return any(marker in text for marker in _SELECTOR_SYNTAX_MARKERS)


class WebPlanner:
    """推理层：把当前观察结果转成一段自然语言行动计划。"""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMError("未配置 ANTHROPIC_API_KEY，无法调用 Planner", stage="request")
            self._client = anthropic.AsyncAnthropic()
        assert self._client is not None  # 帮助静态类型检查器收窄为非 Optional
        return self._client

    async def plan(
        self,
        task: str,
        obs: ObserveResult,
        history: list[dict],
    ) -> str:
        """调用 LLM，返回下一步行动计划（纯文本，不含 selector）。

        messages 组装顺序：history（此前若干轮 plan/action 的对话记录）
        + 当前 observation 对应的最新一条 user 消息；system 固定为
        PLANNER_SYSTEM。失败时按 1s / 2s / 4s ... 指数退避重试，
        最多 config.llm_retry 次，超过后抛出 LLMError。
        """
        client = self._get_client()

        user_content = PLANNER_USER_TMPL.format(
            task=task, observation=_format_observation(obs)
        )
        # history 的公开签名是 list[dict]（调用方传入普通字典即可，不强制依赖
        # anthropic SDK 的类型），这里用 cast 显式告知类型检查器：
        # 运行时结构上符合 MessageParam（含 role/content），避免误报类型不匹配。
        messages: list[MessageParam] = cast(
            "list[MessageParam]", [*history, {"role": "user", "content": user_content}]
        )

        last_exc: Exception | None = None
        max_attempts = max(1, self.config.llm_retry)

        for attempt in range(max_attempts):
            try:
                message: Message = await client.messages.create(
                    model=self.config.model,
                    max_tokens=1024,
                    timeout=self.config.llm_timeout,
                    system=PLANNER_SYSTEM,
                    messages=messages,
                )
            except anthropic.RateLimitError as exc:
                # 429 速率限制：免费模型速率窗口通常为 1 分钟，
                # 短时间指数退避反而加剧频率压力，改用固定的 rate_limit_delay 则更可靠。
                last_exc = exc
                will_retry = attempt + 1 < max_attempts
                wait_secs = self.config.rate_limit_delay
                logger.warning(
                    "WebPlanner LLM 调用失败（第 %d/%d 次尝试）: %s%s",
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
                will_retry = attempt + 1 < max_attempts
                logger.warning(
                    "WebPlanner LLM 调用失败（第 %d/%d 次尝试）: %s%s",
                    attempt + 1,
                    max_attempts,
                    exc,
                    "，即将重试" if will_retry else "，已达重试上限",
                )
                if will_retry:
                    await asyncio.sleep(2**attempt)  # 1s / 2s / 4s / ...
                continue

            text_parts = [
                block.text for block in message.content if isinstance(block, TextBlock)
            ]
            plan_text = "".join(text_parts).strip()

            if not plan_text:
                last_exc = LLMError("LLM 返回了空的 plan 文本", stage="parse")
                will_retry = attempt + 1 < max_attempts
                logger.warning(
                    "WebPlanner 收到空 plan（第 %d/%d 次尝试）%s",
                    attempt + 1,
                    max_attempts,
                    "，即将重试" if will_retry else "，已达重试上限",
                )
                if will_retry:
                    await asyncio.sleep(2**attempt)
                continue

            if _contains_selector_syntax(plan_text):
                # 不硬性拦截：记录 warning 供人工核查/后续加固 Prompt 约束，
                # 同时仍然把 plan 返回给调用方，避免因误判而卡住整条任务链路。
                logger.warning(
                    "WebPlanner 输出疑似包含 selector 语法，请核查 PLANNER_SYSTEM 约束: %s",
                    plan_text,
                )

            return plan_text

        last_exc_desc = last_exc.message if isinstance(last_exc, LLMError) else str(last_exc)
        raise LLMError(
            f"WebPlanner LLM 请求连续失败 {max_attempts} 次: {last_exc_desc}",
            stage="request",
            retry_count=max_attempts,
        )
