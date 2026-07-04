"""编排层：AgentController 按 TRACE → OBS → PLAN → SEL → EXEC 顺序初始化
所有子组件，驱动"观察 -> 规划 -> 决策 -> 执行"主循环，并统一管理三种
终止条件（done / consecutive_fail / max_steps_exceeded）。

初始化顺序是硬约束：tracer 必须先于 observer 创建（observer 依赖 tracer
分配截图路径），tracer 还要被注入进 executor（否则 executor 内部
screenshot/extract 用的会是它自己另建的 TraceLogger，截图散落在两个不同
的 traces/run-xxx/ 目录，trace.jsonl 里记录的 screenshot 路径会对不上）。
"""

import asyncio
import logging

from agent.action_selector import ActionSelector
from agent.config import AgentConfig
from agent.exceptions import LLMError, SafetyError
from agent.executor import PlaywrightExecutor
from agent.observer import BrowserStateObserver
from agent.planner import WebPlanner
from agent.tracer import TraceLogger
from agent.types import AgentResult, LLMAction, ObserveResult, ToolResult

logger = logging.getLogger(__name__)

# 每步往 history 里追加一对 (user, assistant) 消息；只保留最近若干步，
# 避免任务步数一多，喂给 Planner/Selector 的上下文无限膨胀。
_MAX_HISTORY_STEPS = 5
_HISTORY_TEXT_LIMIT = 200  # 单条 history 摘要的字符上限，避免整页文本被塞进去


class AgentController:
    """编排层：协调 Planner / ActionSelector / PlaywrightExecutor / TraceLogger。"""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        # 初始化顺序不能打乱：TRACE -> OBS -> PLAN -> SEL -> EXEC
        self.tracer = TraceLogger(base_dir=config.trace_dir)
        self.observer = BrowserStateObserver(config, self.tracer)
        self.planner = WebPlanner(config)
        self.selector = ActionSelector(config)
        # 把 controller 自己的 tracer 注入 executor，保证同一次 run 的所有
        # 截图（无论来自主循环的 observe，还是 execute() 里的 screenshot/
        # extract 动作）都落在同一个 traces/run-xxx/ 目录、编号连续。
        self.executor = PlaywrightExecutor(config, tracer=self.tracer)

    async def run(self, task: str, url: str) -> AgentResult:
        """执行一次完整任务，返回 AgentResult。

        executor.close() 放在最外层 try/finally 里：无论是正常终止、
        三种预期的失败终止，还是 Planner/Selector/Executor 抛出未预期的
        异常，都能保证浏览器资源被释放，不因异常路径而残留 Chromium 进程。
        """
        history: list[dict] = []
        fail_count = 0

        try:
            open_result = await self.executor.open(url)
            if not open_result["success"]:
                return self._finalize(
                    task=task,
                    success=False,
                    output=None,
                    steps=0,
                    fail_reason=f"open_failed: {open_result['error_msg']}",
                )

            for step in range(self.config.max_steps):
                try:
                    obs, plan, action, result = await self._run_step(task, history)
                except SafetyError as exc:
                    logger.warning("任务在第 %d 步命中安全拦截，终止: %s", step, exc)
                    return self._finalize(
                        task=task,
                        success=False,
                        output=None,
                        steps=step + 1,
                        fail_reason=f"safety_violation: {exc.message}",
                    )
                except LLMError as exc:
                    logger.warning("任务在第 %d 步 LLM 调用失败，终止: %s", step, exc)
                    return self._finalize(
                        task=task,
                        success=False,
                        output=None,
                        steps=step + 1,
                        fail_reason=f"llm_error: {exc.message}",
                    )
                except Exception as exc:  # noqa: BLE001 - 编排层最后一道兜底，绝不能崩溃
                    logger.warning("任务在第 %d 步发生未预期异常，终止: %s", step, exc)
                    return self._finalize(
                        task=task,
                        success=False,
                        output=None,
                        steps=step + 1,
                        fail_reason=f"unexpected_error: {exc}",
                    )

                self.tracer.record(step, obs, plan, action, result)
                self._append_history(history, step, plan, action)

                if action["action"] == "done":
                    return self._finalize(
                        task=task,
                        success=True,
                        output=action["value"],
                        steps=step + 1,
                        fail_reason=None,
                    )

                # 连续失败计数：任意一步成功即清零，只统计"连续"失败，
                # 而不是整个任务生命周期内的累计失败次数。
                fail_count = 0 if result["success"] else fail_count + 1
                if fail_count >= self.config.max_fail:
                    return self._finalize(
                        task=task,
                        success=False,
                        output=None,
                        steps=step + 1,
                        fail_reason="consecutive_fail",
                    )

                await asyncio.sleep(self.config.step_delay)

            return self._finalize(
                task=task,
                success=False,
                output=None,
                steps=self.config.max_steps,
                fail_reason="max_steps_exceeded",
            )
        finally:
            await self.executor.close()

    async def _run_step(
        self, task: str, history: list[dict]
    ) -> tuple[ObserveResult, str, LLMAction, ToolResult]:
        """单步执行：观察 -> 规划 -> 决策 -> 执行。抽成独立方法便于上层统一捕获异常。"""
        assert self.executor.page is not None  # run() 已确保 open() 成功，帮助类型收窄
        obs = await self.observer.observe(self.executor.page)
        plan = await self.planner.plan(task, obs, history)
        action = await self.selector.select(plan, obs, history)
        result = await self.executor.execute(action)
        return obs, plan, action, result

    @staticmethod
    def _append_history(history: list[dict], step: int, plan: str, action: LLMAction) -> None:
        """追加一对 (user, assistant) 消息，保持历史记录严格按角色交替。

        Planner/ActionSelector 会把 history 原样拼进 Anthropic messages
        数组末尾，再追加一条 user 消息发起调用——如果 history 里连续出现
        两条 assistant 消息（比如每步只 append 一条），下一次调用的
        messages 序列就变成 assistant, assistant, ..., user，不满足
        Anthropic API 对角色交替的要求，实测会被拒绝。这里每步固定追加
        一条 user + 一条 assistant，保证任何时候 history 都以合法的
        交替序列结尾。
        """
        summary = plan[:_HISTORY_TEXT_LIMIT]
        history.append(
            {"role": "user", "content": f"[Step {step}] 已获取页面观察结果，请给出下一步。"}
        )
        history.append(
            {
                "role": "assistant",
                "content": (
                    f"[Step {step}] 计划: {summary}；"
                    f"执行动作: {action['action']}（{action['reason'][:_HISTORY_TEXT_LIMIT]}）"
                ),
            }
        )
        # 每步产生一对消息，按"步"为单位截断，避免破坏 user/assistant 交替。
        max_entries = 2 * _MAX_HISTORY_STEPS
        if len(history) > max_entries:
            del history[: len(history) - max_entries]

    def _finalize(
        self,
        *,
        task: str,
        success: bool,
        output: str | dict | None,
        steps: int,
        fail_reason: str | None,
    ) -> AgentResult:
        """构造 AgentResult 并落盘 report.json，避免在 run() 里重复这 3 行样板代码。"""
        result = AgentResult(
            task=task,
            success=success,
            output=output,
            steps=steps,
            fail_reason=fail_reason,
            trace_dir=self.tracer.run_dir,
        )
        self.tracer.write_report(task, result)
        return result
