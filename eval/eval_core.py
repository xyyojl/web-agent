"""eval/run_eval.py 与 eval/run_ablation.py 共用的核心逻辑：case 加载、
单个 case 的执行（AgentController.run() -> Verifier.verify()）、以及
6 项指标的计算。两个入口脚本各自只保留自己独有的部分（run_eval.py 是
markdown 汇总报告，run_ablation.py 是两组对比）。

单个 case 执行阶段的异常必须被隔离——即便 AgentController/Verifier 内部
出现未预期的 bug，也只应影响这一个 case 的结果，不能让调用方的批量任务
提前退出、连累后续尚未运行的 case。
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field

# eval/eval_core.py 位于项目子目录下，需要把项目根目录加进 sys.path，
# 这样无论调用方从哪个工作目录执行、以何种方式 import 本模块，
# `import agent.xxx` 都能正常解析，不依赖调用方提前设置 PYTHONPATH。
# 显式 str(...) 包裹的原因同 run_eval.py：避免 os.path.abspath(__file__)
# 在部分类型 stub 下被推断成 str | bytes | LiteralString 联合类型，
# 把不确定性传递到后续所有路径拼接处。
_PROJECT_ROOT: str = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent import (
    AgentConfig,
    AgentController,
    AgentResult,
    EvalCase,
    Verifier,
    VerifyResult,
)

logger = logging.getLogger(__name__)

# case 目录锚定到项目根目录，不依赖当前工作目录，避免"在 eval/ 目录下
# 运行脚本"和"在项目根目录下运行脚本"结果不一致。
CASE_DIRS = {
    "local": os.path.join(_PROJECT_ROOT, "eval", "cases", "local"),
    "public": os.path.join(_PROJECT_ROOT, "eval", "cases", "public"),
}


@dataclass
class CaseOutcome:
    """单个 case 的完整执行结果，供指标汇总和「失败任务」表使用。"""

    case: EvalCase
    agent_result: AgentResult | None = None
    verify_result: VerifyResult | None = None
    step_records: list[dict] = field(default_factory=list)
    # 只有 AgentController.run()/Verifier.verify() 之外发生的、完全没被
    # 下层组件兜住的异常才会落到这里；正常的任务失败走 agent_result/
    # verify_result 表达，不算"崩溃"。
    crash_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.verify_result is not None and self.verify_result["success"]

    @property
    def last_screenshot(self) -> str:
        if self.step_records:
            return self.step_records[-1].get("screenshot") or "N/A"
        return "N/A"

    @property
    def display_fail_reason(self) -> str:
        if self.crash_reason:
            return f"runner_crash: {self.crash_reason}"
        fail_reason = self.agent_result["fail_reason"] if self.agent_result else None
        if fail_reason:
            return fail_reason
        if self.verify_result and not self.verify_result["success"]:
            return f"verify_failed: {self.verify_result['reason']}"
        return "unknown"


def load_cases(suite_dir: str) -> list[EvalCase]:
    """加载指定目录下的全部 *.json case 文件，按文件名排序。

    目录不存在（如 public 套件尚未提供 case）或单个文件解析失败都不应让
    整个批量任务崩溃，只记录 warning 并跳过。
    """
    if not os.path.isdir(suite_dir):
        logger.warning("case 目录不存在，跳过: %s", suite_dir)
        return []

    cases: list[EvalCase] = []
    for fname in sorted(os.listdir(suite_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(suite_dir, fname)
        try:
            with open(path, encoding="utf-8") as f:
                cases.append(json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("加载 case 文件失败，已跳过: %s (%s)", path, exc)
    return cases


def _load_step_records(trace_dir: str) -> list[dict]:
    """读取 trace.jsonl 里的每一步记录，用于统计 step_success_rate。

    单行损坏（比如写入过程中被截断）只跳过那一行，不影响其余行的统计，
    也不影响这个 case 本身的判定结果。
    """
    trace_path = os.path.join(trace_dir, "trace.jsonl")
    records: list[dict] = []
    if not os.path.isfile(trace_path):
        return records

    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


async def run_one_case(case: EvalCase, config: AgentConfig) -> CaseOutcome:
    """独立执行一个 case：AgentController.run() -> Verifier.verify()。

    这里的 try/except 是"单个 case 崩溃不影响后续"的最后一道防线——
    AgentController.run() 自身已经会把 Planner/Selector/Executor 的已知
    异常转换成 AgentResult 返回，但这一层仍要防住任何没被下层兜住的
    意外情况（比如 case 文件字段缺失导致的 KeyError）。
    """
    outcome = CaseOutcome(case=case)
    case_id = case.get("id", "?")
    try:
        controller = AgentController(config)
        # 用局部变量承接返回值：controller.run() 的返回类型是不带 Optional 的
        # AgentResult，局部变量的类型窄化比 outcome.agent_result 这种实例
        # 属性更可靠，后面几行都基于这个局部变量，不再重复读取实例属性。
        agent_result = await controller.run(case["task"], case["url"])
        outcome.agent_result = agent_result
        outcome.step_records = _load_step_records(agent_result["trace_dir"])

        verifier = Verifier(config)
        outcome.verify_result = await verifier.verify(case, agent_result)
    except Exception as exc:  # 调用方批量任务的最后一道兜底，不能让它向上传播
        logger.exception("case %s 执行时发生未预期异常", case_id)
        outcome.crash_reason = str(exc)
    return outcome


def _format_fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}"


def _format_percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator * 100:.0f}%"


def _has_complete_evidence(outcome: CaseOutcome) -> bool:
    """trace.jsonl + report.json + 至少一张截图 都存在，才算这个 case 的证据链完整。"""
    if outcome.agent_result is None:
        return False
    trace_dir = outcome.agent_result["trace_dir"]
    if not os.path.isdir(trace_dir):
        return False
    has_trace = os.path.isfile(os.path.join(trace_dir, "trace.jsonl"))
    has_report = os.path.isfile(os.path.join(trace_dir, "report.json"))
    has_screenshot = any(f.endswith(".png") for f in os.listdir(trace_dir))
    return has_trace and has_report and has_screenshot


def _steps_of(outcome: CaseOutcome) -> int:
    """取出某个 case 的执行步数；调用方必须确保 outcome.agent_result 非空。"""
    assert outcome.agent_result is not None
    return outcome.agent_result["steps"]


def _task_success_counts(outcomes: list[CaseOutcome]) -> tuple[int, int]:
    """(判定成功的 case 数, 总 case 数)。compute_metrics 和 compute_raw_metrics 共用，
    避免 task_success_rate 的统计口径在两处分别实现、逐渐漂移。
    """
    return sum(1 for o in outcomes if o.succeeded), len(outcomes)


def _step_success_counts(outcomes: list[CaseOutcome]) -> tuple[int, int]:
    """(成功的步骤数, 跨全部 case 的总步骤数)。"""
    all_steps = [row for o in outcomes for row in o.step_records]
    return sum(1 for row in all_steps if row.get("success")), len(all_steps)


def _avg_steps_value(outcomes: list[CaseOutcome]) -> float | None:
    """已完成 case（拿到 agent_result，崩溃 case 排除）的平均步数；无已完成 case 时为 None。"""
    completed = [o for o in outcomes if o.agent_result is not None]
    if not completed:
        return None
    return sum(_steps_of(o) for o in completed) / len(completed)


def compute_metrics(outcomes: list[CaseOutcome]) -> dict[str, str]:
    """计算 6 项指标，返回适合直接填进 markdown 表格的字符串。"""
    # 1. task_success_rate：Verifier 判定成功的 case 占比
    success_count, total = _task_success_counts(outcomes)
    task_success_rate = _format_fraction(success_count, total)

    # 2. step_success_rate：跨所有 case 汇总每一步 ToolResult.success 的占比
    step_success_count, step_total = _step_success_counts(outcomes)
    step_success_rate = _format_percent(step_success_count, step_total)

    # 3. avg_steps：只统计成功拿到 agent_result 的 case（崩溃的 case 没有步数可言）
    avg_steps_raw = _avg_steps_value(outcomes)
    avg_steps = f"{avg_steps_raw:.1f}" if avg_steps_raw is not None else "N/A"

    # 4. recovery_rate：出现过至少一次失败步骤、但任务最终仍判定成功的比例
    had_failed_step = [
        o for o in outcomes if any(not row.get("success") for row in o.step_records)
    ]
    recovered = [o for o in had_failed_step if o.succeeded]
    recovery_rate = _format_fraction(len(recovered), len(had_failed_step))

    # 5. unsafe_action_block_rate：分母是 verify_mode == safety_block 的 case
    #    总数（数据来自 case 文件本身，写 case 时就已经确定，不依赖本次运行
    #    结果），分子是 Verifier 判定这些 case "确实按预期被 SafetyError
    #    拦截终止"的数量（见 agent/verifier.py 的 _verify_safety_block）。
    #    分子分母的来源彼此独立，不会重复计数同一件事：如果 case 因为其他
    #    原因（比如 selector 没能命中敏感正则、任务在触发点之前就已失败）
    #    没有真正走到"被拦截"这一步，分子会小于分母，指标才有信息量。
    #    此前的实现是"分子=分母=被拦截的安全事件数"，且 case 库里没有一条
    #    会触发安全拦截的测试，导致这个指标永远是没有意义的 0/0。
    safety_block_outcomes = [o for o in outcomes if o.case.get("verify_mode") == "safety_block"]
    safety_block_success = sum(1 for o in safety_block_outcomes if o.succeeded)
    unsafe_action_block_rate = _format_fraction(safety_block_success, len(safety_block_outcomes))

    # 6. evidence_completeness：trace.jsonl + report.json + 截图 是否齐全
    evidence_count = sum(1 for o in outcomes if _has_complete_evidence(o))
    evidence_completeness = _format_fraction(evidence_count, total)

    return {
        "task_success_rate": task_success_rate,
        "step_success_rate": step_success_rate,
        "avg_steps": avg_steps,
        "recovery_rate": recovery_rate,
        "unsafe_action_block_rate": unsafe_action_block_rate,
        "evidence_completeness": evidence_completeness,
    }


def compute_raw_metrics(outcomes: list[CaseOutcome]) -> dict[str, float | int | None]:
    """compute_metrics 的数值版：不做字符串格式化，返回可以直接参与算术运算的
    原始数字，供需要计算"两组差异"的场景使用（如 run_ablation.py 的对比表格）。
    与 compute_metrics 复用同一批 _task_success_counts / _step_success_counts /
    _avg_steps_value，保证两个函数算出来的是同一套口径，不会出现字符串版和
    数值版对不上的情况。
    """
    success_count, total = _task_success_counts(outcomes)
    step_success_count, step_total = _step_success_counts(outcomes)

    return {
        "success_count": success_count,
        "total": total,
        "task_success_rate": (success_count / total) if total else None,
        "avg_steps": _avg_steps_value(outcomes),
        "step_success_count": step_success_count,
        "step_total": step_total,
        "step_success_rate": (step_success_count / step_total) if step_total else None,
    }
