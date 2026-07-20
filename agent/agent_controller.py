"""编排层：AgentController 按 TRACE → OBS → PLAN → SEL → EXEC 顺序初始化
所有子组件，驱动"观察 -> 规划 -> 决策 -> 执行"主循环，并统一管理三种
终止条件（done / consecutive_fail / max_steps_exceeded）。

初始化顺序是硬约束：tracer 必须先于 observer 创建（observer 依赖 tracer
分配截图路径），tracer 还要被注入进 executor（否则 executor 内部
screenshot/extract 用的会是它自己另建的 TraceLogger，截图散落在两个不同
的 traces/run-xxx/ 目录，trace.jsonl 里记录的 screenshot 路径会对不上）。
"""

import asyncio
import json
import logging
import re
import time

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
_HISTORY_TEXT_LIMIT = 200  # 单条 history 摘要（plan/reason 等自然语言文本）的字符上限
# extract 动作的 output 是结构化 JSON（title/authors/abstract 等），
# 复用上面 200 字符的通用上限会把 output 拦腰截断。
# extract 的成功产出必须给到远大于普通文本摘要的预算，才能让 Planner
# 在下一步真正看到"任务已经可以 done 了"。
_EXTRACT_OUTPUT_HISTORY_LIMIT = 3000

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
        return f"失败（{result.get('error_msg') or '未知错误'}）"

    if action["action"] == "extract":
        output = result.get("output")
        if output:
            # 用 _EXTRACT_OUTPUT_HISTORY_LIMIT（远大于通用 _HISTORY_TEXT_LIMIT）
            # 截断，确保 title/authors/abstract 这类结构化字段大概率能完整
            # 保留在 history 里，Planner 才能据此判断任务是否已经可以 done。
            truncated = output[:_EXTRACT_OUTPUT_HISTORY_LIMIT]
            suffix = "...(已截断)" if len(output) > _EXTRACT_OUTPUT_HISTORY_LIMIT else ""
            return f"成功，extract 已返回数据: {truncated}{suffix}"
        return "extract 调用成功，但未返回任何数据（output 为空）"

    if action["action"] == "screenshot":
        return f"成功，截图已保存到 {result.get('output') or '(路径未知)'}"

    return "成功"


# observer.py 的 _EXTRACT_TEXT_JS 只对 H1-6/BUTTON/LI 这三类标签加前缀
# （"# "/"[按钮] "/"- "），且每个节点只加一次；这里剥离时用同样的三种
# 前缀、只在字符串开头做一次非贪婪匹配，和注入端的行为严格对称——
# 剥离规则的正确性依赖于"注入规则我们自己完全掌控"这个前提，不是
# 泛化的 Markdown 清洗。
_STRUCTURAL_PREFIX_RE = re.compile(r"^(?:#{1,6}\s+|-\s+|\[按钮]\s*)")


def _strip_structural_prefix(text: str) -> str:
    """done.value 定型前统一剥离一次已知的三种
    observer 结构标注开头。只剥离一次、只在开头剥，不循环剥离——observer
    对每个节点只加一层前缀，模型即使自己额外叠了一层 Markdown 装饰，正常
    情况也不会无限嵌套；循环剥离反而有把页面真实存在的、以连续短横线开头
    的内容（如原文本来就是"-- 说明 --"）越剥越秃的风险。
    """
    return _STRUCTURAL_PREFIX_RE.sub("", text, count=1)


def _find_element_role(obs: ObserveResult, selector: str | None) -> str | None:
    """按 selector 从当前观察结果的交互元素列表中查出对应的 role。

    selector 不在列表中（模型臆造/元素已消失）时返回 None，调用方应放弃
    判断、按原有流程放行——这里只处理"selector 命中列表、但 role 与
    动作类型不匹配"这种可确定性判断的情况，不代替 selector 的合法性校验。
    """
    if not selector:
        return None
    for el in obs["interactive_elements"]:
        if el["selector"] == selector:
            return el["role"]
    return None


def _extract_page_key(obs: ObserveResult) -> tuple[str, str]:
    """用 (url, text_hash) 作为"页面状态是否发生变化"的判定依据。

    extract 是纯读取动作，不改变页面；只要这两项都没变，说明自上次成功
    extract 以来页面没有任何新信息出现——在这种前提下，无论 Planner/
    ActionSelector 这次给 extract 起的 instruction 措辞跟上次差多少
    （遣词造句差异不代表页面上真的有"新的可提取内容"），第二次调用都
    不可能从同一份静态文本里榨出实质不同的正确数据，纯属重复劳动。

    这里特意用 text_hash 而不是 visible_text_summary：后者会被
    observer 截断到 obs_text_limit（默认 3000）字符，如果两次页面
    截断后的前缀恰好相同，但真实变化发生在截断点之后（比如表格分页
    追加了新行），拿截断后的文本比较会把这次变化误判为"页面没变"，
    错误地跳过本该重新执行的 extract。text_hash 是在 observer 截断
    之前，对完整可见文本算出来的 hash，覆盖不到截断丢掉的部分不存在，
    能避免这种假阳性。
    """
    return obs["url"], obs["text_hash"]


def _unwrap_extract_data(output: str | None) -> object | None:
    """从 browser_extract 按 TOOL-001 契约返回的 {"url":..., "data":...}
    中取出 data 部分。取不出（output 为空/非法 JSON/没有 data 键）时返回 None，
    调用方应放弃走自动短路路径，退回原有流程。
    """
    if not output:
        return None
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "data" not in parsed:
        return None
    return parsed["data"]


class AgentController:
    """编排层：协调 Planner / ActionSelector / PlaywrightExecutor / TraceLogger。"""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        # 初始化顺序不能打乱：TRACE -> OBS -> PLAN -> SEL -> EXEC
        self.tracer = TraceLogger(base_dir=config.trace_dir)
        # vision 开关完全由 AgentConfig 决定（config.vision，默认 False），而不是在这里手写死值
        self.observer = BrowserStateObserver(config, self.tracer, vision=config.vision)
        self.planner = WebPlanner(config)
        self.selector = ActionSelector(config)
        # 把 controller 自己的 tracer 注入 executor，保证同一次 run 的所有
        # 截图（无论来自主循环的 observe，还是 execute() 里的 screenshot/
        # extract 动作）都落在同一个 traces/run-xxx/ 目录、编号连续。
        self.executor = PlaywrightExecutor(config, tracer=self.tracer)

        # 由 run() 在每次任务开始时写入，_finalize() 读取用来补全
        # AgentResult / report.json 里的 task_id / url / duration_s。
        # 这里先给出安全的兜底默认值，避免 _finalize 在 run() 尚未执行
        # （理论上不应发生）时因属性不存在而抛 AttributeError。
        self._task_id: str | None = None
        self._url: str = ""
        self._run_start_time: float = time.monotonic()

    async def run(self, task: str, url: str, task_id: str | None = None) -> AgentResult:
        """执行一次完整任务，返回 AgentResult。

        task_id：可选的任务标识（如 eval case 的 "L03"），供批量评测场景
        在 report.json 里区分是哪一条 case 产生的记录；main.py 单次 CLI
        运行不提供 case 概念，可以不传，report.json 里对应字段落为 null。

        executor.close() 放在最外层 try/finally 里：无论是正常终止、
        三种预期的失败终止，还是 Planner/Selector/Executor 抛出未预期的
        异常，都能保证浏览器资源被释放，不因异常路径而残留 Chromium 进程。
        """
        self._task_id = task_id
        self._url = url
        # duration_s 从任务真正开始（含 open() 耗时）算到 _finalize() 落盘
        # 那一刻为止，覆盖整个任务生命周期，口径与 trace.jsonl 里逐步的
        # duration_ms（从 reset_step_timer() 之后才开始计）刻意不同——
        # 后者是"主循环单步耗时"，前者是"这条 report 对应的总耗时"。
        self._run_start_time = time.monotonic()

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
        # extract 去重缓存：记录"最近一次成功 extract 时的页面状态 + 拿到的
        # 干净数据"。传给 _run_step，供其在真正调用 LLM/浏览器执行 extract
        # 之前做确定性拦截判断。None 表示还没有任何成功的 extract。
        extract_cache: dict[str, object] | None = None

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

            # open() 可能耗时不短（page.goto 重试、登录页信号命中时
            # ask_human() 阻塞等真人从终端输入选择），这些时间和"agent
            # 这一步跑得快不快"无关，重置计时基准，避免被计入第一步的
            # duration_ms。
            self.tracer.reset_step_timer()

            for step in range(self.config.max_steps):
                try:
                    obs, plan, action, result, extract_cache = await self._run_step(
                        step, task, history, extract_cache
                    )
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
                    done_value = action["value"]
                    assert done_value is not None  # 'done' 分支的 value 由 SEL-001 保证必填，帮助类型收窄
                    return self._finalize(
                        task=task,
                        success=True,
                        output=_strip_structural_prefix(done_value),
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
        self,
        step: int,
        task: str,
        history: list[dict],
        extract_cache: dict[str, object] | None,
    ) -> tuple[ObserveResult, str, LLMAction, ToolResult, dict[str, object] | None]:
        """单步执行：观察 -> 规划 -> 决策 -> [去重拦截] -> 执行。

        extract 去重拦截是本方法与纯 Prompt 方案的本质区别：Prompt 提示
        （PLANNER_SYSTEM / history 提醒）只能降低模型重复调用 extract 的
        概率，无法保证杜绝——模型仍然可能在措辞、上下文理解上出现偏差而
        忽略提示。这里在 ActionSelector 已经选出 action，但*执行之前*
        插入一层确定性判断：只要页面状态（url + visible_text_summary）
        与上一次成功 extract 时完全一致，无论这次 instruction 措辞如何，
        都直接判定为重复请求——不再消耗一次真实的 LLM 抽取调用，而是复用
        缓存数据，把这一步的 action 强制改写成 done，交给 run() 的主循环
        正常收尾。这一步是纯代码判断，不依赖也不信任模型自己的判断，
        因此不会因为模型行为的不确定性而失效。

        step：run() 主循环里的步数索引，只在 execute() 抛出 SafetyError
        时才用到——那种情况下 _run_step 不会正常返回，run() 里统一记录
        trace 的那一行代码到不了，只能在这里补记一笔，需要 step 拼进
        trace 记录里。正常返回路径不使用这个参数。
        """
        assert self.executor.page is not None  # run() 已确保 open() 成功，帮助类型收窄
        obs = await self.observer.observe(self.executor.page)
        plan = await self.planner.plan(task, obs, history)
        action = await self.selector.select(plan, obs, history)

        # 确定性拦截：click 目标元素若在 observe() 结果里被标注为
        # role=select（原生 <select> 下拉框），直接判定这一步失败，不真的执行点击。
        if action["action"] == "click":
            target_role = _find_element_role(obs, action.get("selector"))
            if target_role == "select":
                forced_result = ToolResult(
                    success=False,
                    page_changed=False,
                    output=None,
                    error_msg=(
                        "目标元素是原生 <select> 下拉框（role=select），click 不会"
                        "展开或切换任何选项，无论重试多少次都不可能生效。必须改用"
                        "select 动作（selector 保持不变，value 填目标选项的显示"
                        "文本或底层 value）。"
                    ),
                )
                return obs, plan, action, forced_result, extract_cache

        if action["action"] == "extract" and extract_cache is not None:
            if _extract_page_key(obs) == extract_cache["page_key"]:
                logger.info(
                    "检测到重复的 extract 请求（页面状态与上次成功 extract 时"
                    "完全一致），跳过真实抽取调用，直接用缓存数据收尾任务。"
                )
                forced_done = LLMAction(
                    action="done",
                    selector=None,
                    text=None,
                    value=json.dumps(extract_cache["data"], ensure_ascii=False),
                    reason=(
                        "系统检测到与上一次成功 extract 完全重复的请求"
                        "（页面状态未发生任何变化），数据已在上一步获取，"
                        "自动判定任务完成，避免无意义的重复抽取。"
                    ),
                )
                forced_result = ToolResult(
                    success=True,
                    page_changed=False,
                    output=forced_done["value"],
                    error_msg=None,
                )
                return obs, plan, forced_done, forced_result, extract_cache

        try:
            result = await self.executor.execute(action, obs=obs)
        except SafetyError as exc:
            # SafetyError 会穿透 _run_step 交给 run() 的主循环去终止任务，
            # 但 obs/plan/action 在这里都已经拿到了——不趁现在记一笔，
            # 这一步就会永远没有 trace.jsonl 记录（run() 里的
            # tracer.record() 只在 _run_step 正常返回时才会被执行到）。
            # 唯一被安全拦截的这一步，反而应该是最值得留痕的一步。
            blocked_result = ToolResult(
                success=False,
                page_changed=False,
                output=None,
                error_msg=f"safety_violation: {exc.message}",
            )
            self.tracer.record(step, obs, plan, action, blocked_result)
            raise

        updated_cache = extract_cache
        if action["action"] == "extract" and result["success"]:
            data = _unwrap_extract_data(result.get("output"))
            if data is not None:
                updated_cache = {"page_key": _extract_page_key(obs), "data": data}

        return obs, plan, action, result, updated_cache

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
        user_content = f"<系统记录> 第 {step} 步页面观察结果已就绪，请给出下一步计划。"
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
                    "（系统事后记录，非输出模板，仅陈述已发生的客观事实）\n"
                    f"上一步意图: {summary}\n"
                    f"已执行: {action['action']}（理由: {action['reason'][:_HISTORY_TEXT_LIMIT]}）\n"
                    f"执行反馈: {_summarize_result(action, result)}"
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
        """构造 AgentResult 并落盘 report.json，避免在 run() 里重复这几行样板代码。

        task_id / url 直接取 run() 开始时记下的 self._task_id / self._url；
        duration_s 用 self._run_start_time 到此刻的耗时（秒，保留 2 位小数）；
        last_screenshot 取 tracer 目前为止分配出去的最后一张截图路径——三者
        都不需要调用方（run() 里 7 处 _finalize 调用点）逐个传参。
        """
        result = AgentResult(
            task_id=self._task_id,
            task=task,
            url=self._url,
            success=success,
            output=output,
            steps=steps,
            duration_s=round(time.monotonic() - self._run_start_time, 2),
            fail_reason=fail_reason,
            trace_dir=self.tracer.run_dir,
            last_screenshot=self.tracer.last_screenshot_path,
        )
        self.tracer.write_report(task, result)
        return result
