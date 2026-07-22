"""推理层：WebPlanner 调用 LLM，把「当前页面状态 + 任务目标」转成一段
自然语言行动计划（plan），不涉及具体 selector —— selector 的解析工作
交给下游 ActionSelector（PROMPT-001 中 PLANNER_SYSTEM 已明确约束这一点）。
"""

import logging
from typing import cast

from anthropic.types import Message, MessageParam, TextBlock

from agent.config import AgentConfig
from agent.llm_client import LLMClient, LLMOutputRetry
from agent.prompts import PLANNER_SYSTEM, PLANNER_USER_TMPL, format_untrusted_page_content
from agent.types import ContentSafetyAssessment, ObserveResult
from agent.vision import build_vision_user_content

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
    return format_untrusted_page_content(
        {
            "url": obs["url"], "title": obs["title"],
            "visible_text_summary": obs["visible_text_summary"],
            "interactive_elements": [
                {"role": el["role"], "name": el["name"], "href": el.get("href")}
                for el in obs["interactive_elements"]
            ],
            "content_safety": obs.get("content_safety", ContentSafetyAssessment(status="clean", signals=[])),
        }
    )


def _contains_selector_syntax(text: str) -> bool:
    return any(marker in text for marker in _SELECTOR_SYNTAX_MARKERS)


class WebPlanner:
    """推理层：把当前观察结果转成一段自然语言行动计划。"""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._llm = LLMClient(config)

    async def plan(
        self,
        task: str,
        obs: ObserveResult,
        history: list[dict],
    ) -> str:
        """调用 LLM，返回下一步行动计划（纯文本，不含 selector）。

        messages 组装顺序：history（此前若干轮 plan/action 的对话记录）
        + 当前 observation 对应的最新一条 user 消息；system 固定为
        PLANNER_SYSTEM。失败时按 1s / 2s / 4s ... 指数退避重试（429 固定
        等待 config.rate_limit_delay 秒），最多 config.llm_retry 次，
        超过后抛出 LLMError（重试骨架见 agent/llm_client.py）。
        """
        user_content = build_vision_user_content(
            obs, PLANNER_USER_TMPL.format(task=task, observation=_format_observation(obs))
        )
        # history 的公开签名是 list[dict]（调用方传入普通字典即可，不强制依赖
        # anthropic SDK 的类型），这里用 cast 显式告知类型检查器：
        # 运行时结构上符合 MessageParam（含 role/content），避免误报类型不匹配。
        messages: list[MessageParam] = cast(
            "list[MessageParam]", [*history, {"role": "user", "content": user_content}]
        )

        def _parse_plan(message: Message) -> str:
            text_parts = [
                block.text for block in message.content if isinstance(block, TextBlock)
            ]
            plan_text = "".join(text_parts).strip()

            if not plan_text:
                raise LLMOutputRetry("LLM 返回了空的 plan 文本")

            if _contains_selector_syntax(plan_text):
                # 不硬性拦截：记录 warning 供人工核查/后续加固 Prompt 约束，
                # 同时仍然把 plan 返回给调用方，避免因误判而卡住整条任务链路。
                logger.warning(
                    "WebPlanner 输出疑似包含 selector 语法，请核查 PLANNER_SYSTEM 约束: %s",
                    plan_text,
                )

            return plan_text

        return await self._llm.call_with_retry(
            caller_name="WebPlanner",
            parse_response=_parse_plan,
            model=self.config.model,
            max_tokens=1024,
            system=PLANNER_SYSTEM,
            messages=messages,
        )
