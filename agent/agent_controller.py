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

# 连续 N 次动作完全相同（action/selector/text/value 四元组一致）时，
# 在下一次调用前往 history 里注入一条纠偏提示；如果纠偏后仍然连续
# 重复到 ABORT 阈值，直接提前终止任务，不再耗到 max_steps——既避免
# 陷入死循环浪费步数，也直接减少了打给 LLM API 的无效请求次数
_STAGNATION_NUDGE_THRESHOLD = 3
_STAGNATION_ABORT_THRESHOLD = 6


def _action_signature(action: LLMAction) -> tuple:
    """把一个 LLMAction 压缩成可比较的签名，用于判断"是否和上一次完全相同"。"""
    return action.get("action"), action.get("selector"), action.get("text"), action.get("value")


def _summarize_result(action: LLMAction, result: ToolResult) -> str:
    """把 ToolResult 摘要成一句话，写进 history 供下一步 Planner 参考。

    click/type/select/scroll 这类动作本身没有独立的"产出"，成功与否
    从下一次 observe() 的页面变化里就能看出来，这里简单标注成功/失败即可；
    extract/screenshot 这类动作的意义就是"产出点什么"（数据或截图路径），
    如果不把这个产出写进 history，Planner 下一步根本不知道刚才有没有拿到
    数据、拿到的是什么，只能靠重新观察页面去猜。
    """
    if not result["success"]:
        return f"执行结果: 失败（{result.get('error_msg') or '未知错误'}）"

    if action["action"] == "extract":
        output = result.get("output")
        if output:
            return f"执行结果: 成功，extract 已返回数据: {output[:_HISTORY_TEXT_LIMIT]}"
        return "执行结果: extract 调用成功，但未返回任何数据（output 为空）"

    if action["action"] == "screenshot":
        return f"执行结果: 成功，截图已保存到 {result.get('output') or '(路径未知)'}"

    return "执行结果: 成功"


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
        # 记录最近的动作签名，用于检测"连续重复同一个动作、页面毫无进展"的死循环。
        recent_action_signatures: list[tuple] = []
        # 任务开始后第一次 observe 的可见文本，作为"页面是否真的发生过变化"
        # 的基线。注意：不能只用"上一步 vs 这一步"做增量对比——像 tab 切换
        # 这种一次性生效的动作，切换完成后再点几次同一个按钮，相邻两步的
        # 文本本来就不会再变，若只看相邻步差异，依然会得出"没有进展"的
        # 错误结论。必须对比"现在"和"任务刚开始时"，才能识别出"其实早就
        # 已经变了，只是后面几次重复点击没有再带来新变化"这种情况。
        baseline_visible_text: str | None = None

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
                except Exception as exc:  # 编排层最后一道兜底，绝不能崩溃
                    logger.warning("任务在第 %d 步发生未预期异常，终止: %s", step, exc)
                    return self._finalize(
                        task=task,
                        success=False,
                        output=None,
                        steps=step + 1,
                        fail_reason=f"unexpected_error: {exc}",
                    )

                self.tracer.record(step, obs, plan, action, result)

                if baseline_visible_text is None:
                    baseline_visible_text = obs["visible_text_summary"]
                    content_changed_from_baseline = False
                else:
                    content_changed_from_baseline = (
                        obs["visible_text_summary"] != baseline_visible_text
                    )

                # 死循环检测：如果这一步和之前连续几步的动作完全相同
                # （同一个 action + selector + text + value），说明 Planner/
                # ActionSelector 陷入了"重复执行同一动作但毫无进展"的循环。
                signature = _action_signature(action)
                stagnant_streak = 1
                for prev_sig in reversed(recent_action_signatures):
                    if prev_sig == signature:
                        stagnant_streak += 1
                    else:
                        break
                recent_action_signatures.append(signature)

                if action["action"] != "done" and stagnant_streak >= _STAGNATION_ABORT_THRESHOLD:
                    logger.warning(
                        "任务在第 %d 步检测到连续 %d 次重复同一动作，判定为死循环，提前终止",
                        step,
                        stagnant_streak,
                    )
                    return self._finalize(
                        task=task,
                        success=False,
                        output=None,
                        steps=step + 1,
                        fail_reason="stuck_loop",
                    )

                self._append_history(
                    history, step, plan, action, result, stagnant_streak, content_changed_from_baseline
                )

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
        result = await self.executor.execute(action, obs=obs)
        return obs, plan, action, result

    @staticmethod
    def _append_history(
        history: list[dict],
        step: int,
        plan: str,
        action: LLMAction,
        result: ToolResult,
        stagnant_streak: int,
        content_changed_from_baseline: bool,
    ) -> None:
        """追加一对 (user, assistant) 消息，保持历史记录严格按角色交替。

        Planner/ActionSelector 会把 history 原样拼进 Anthropic messages
        数组末尾，再追加一条 user 消息发起调用——如果 history 里连续出现
        两条 assistant 消息（比如每步只 append 一条），下一次调用的
        messages 序列就变成 assistant, assistant, ..., user，不满足
        Anthropic API 对角色交替的要求，实测会被拒绝。这里每步固定追加
        一条 user + 一条 assistant，保证任何时候 history 都以合法的
        交替序列结尾。

        stagnant_streak 达到 _STAGNATION_NUDGE_THRESHOLD 时，在 user 消息
        里追加一条纠偏提示。

        依据 content_changed_from_baseline（当前页面可见文本是否
        已经不同于任务刚开始时）分两种措辞：真的没变化 → 保留原提示；
        其实已经变了 → 明确告诉模型"重复点击同一个元素不等于页面没反应，
        请重新检查当前观察结果是否已经满足任务要求"，避免它被"重复动作"
        这个表象误导，忽略掉已经到手的正确答案。

        把result 的关键信息（成功与否、output 或 error_msg）也写进
        assistant 消息，Planner 下一步就能直接看到"extract 已经成功，
        返回的数据是 XXX"，不需要再靠反复重试去确认。
        """
        user_content = f"[Step {step}] 已获取页面观察结果，请给出下一步。"
        if stagnant_streak >= _STAGNATION_NUDGE_THRESHOLD:
            action_desc = f"{action['action']} {action.get('selector') or action.get('value') or ''}"
            if content_changed_from_baseline:
                user_content += (
                    f"\n注意：系统检测到你已经连续 {stagnant_streak} 次选择了完全相同的动作"
                    f"（{action_desc}）。但页面当前的可见文本摘要，和任务刚开始时相比其实"
                    "已经不一样了——也就是说，之前某一次点击很可能已经真实生效过，只是你"
                    "没有意识到。请不要仅仅因为'我又选择了同一个元素'就判断这是无意义的"
                    "重复、或判断点击没有响应。请重新仔细阅读当前的可见文本摘要，确认其中"
                    "是否已经包含完成任务所需的信息；如果已经包含，请直接调用 done 并把该"
                    "信息作为结果给出，不要再次点击同一个元素或宣称任务无法完成。"
                )
            else:
                user_content += (
                    f"\n注意：系统检测到你已经连续 {stagnant_streak} 次执行了完全相同的动作"
                    f"（{action_desc}），页面没有产生任何新进展。请不要再重复这个动作——如果"
                    "当前观察结果已经包含完成任务所需的信息，请直接说明任务已完成并给出该"
                    "信息；否则请改用其他元素或其他动作。"
                )

        summary = plan[:_HISTORY_TEXT_LIMIT]
        history.append({"role": "user", "content": user_content})
        history.append(
            {
                "role": "assistant",
                "content": (
                    f"[Step {step}] 计划: {summary}；"
                    f"执行动作: {action['action']}（{action['reason'][:_HISTORY_TEXT_LIMIT]}）；"
                    f"{_summarize_result(action, result)}"
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
