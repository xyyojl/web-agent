"""批量 Eval 执行脚本：加载 eval case，逐个跑 AgentController + Verifier，
汇总 6 项指标并生成 eval_summary.md。

用法：
    uv run python eval/run_eval.py --suite local     # 本地 10 条
    uv run python eval/run_eval.py --suite public    # 公开 5 条
    uv run python eval/run_eval.py --suite all       # 全部

单个 case 执行阶段的异常必须被隔离——即便 AgentController/Verifier 内部
出现未预期的 bug，也只应影响这一个 case 的结果，不能让整个批量任务提前
退出、连累后续尚未运行的 case。
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# eval/run_eval.py 位于项目子目录下，需要把项目根目录加进 sys.path，
# 这样无论从哪个工作目录执行 `python eval/run_eval.py`，`import agent.xxx`
# 都能正常解析，不依赖调用者提前设置 PYTHONPATH。
# 显式 str(...) 包裹：os.path.abspath(__file__) 在部分类型 stub 下会被
# 推断成 str | bytes | LiteralString 的联合类型，逐层 dirname 会把这个
# 不确定性传递下去，导致后面所有基于 _PROJECT_ROOT 拼接的路径全部被判定
# 为可能不是 str，进而在 os.path.join / sys.path.insert 处报类型不匹配。
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

# case 目录与汇总报告路径都锚定到项目根目录，不依赖当前工作目录，
# 避免"在 eval/ 目录下运行脚本"和"在项目根目录下运行脚本"结果不一致。
_CASE_DIRS = {
    "local": os.path.join(_PROJECT_ROOT, "eval", "cases", "local"),
    "public": os.path.join(_PROJECT_ROOT, "eval", "cases", "public"),
}
_SUMMARY_PATH = os.path.join(_PROJECT_ROOT, "eval_summary.md")
_SUITE_LABELS = {"local": "本地任务", "public": "公开网页"}
_METRIC_ORDER = (
    "task_success_rate",
    "step_success_rate",
    "avg_steps",
    "recovery_rate",
    "unsafe_action_block_rate",
    "evidence_completeness",
)


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


def _load_cases(suite_dir: str) -> list[EvalCase]:
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


async def _run_one_case(case: EvalCase, config: AgentConfig) -> CaseOutcome:
    """独立执行一个 case：AgentController.run() -> Verifier.verify()。

    这里的 try/except 是整个批量脚本"单个 case 崩溃不影响后续"的最后一道
    防线——AgentController.run() 自身已经会把 Planner/Selector/Executor
    的已知异常转换成 AgentResult 返回，但脚本这一层仍要防住任何没被
    下层兜住的意外情况（比如 case 文件字段缺失导致的 KeyError）。
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
    except Exception as exc:  # 批量脚本的最后一道兜底，不能让它向上传播
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


def _compute_metrics(outcomes: list[CaseOutcome]) -> dict[str, str]:
    """计算 6 项指标，返回适合直接填进 markdown 表格的字符串。"""
    total = len(outcomes)

    # 1. task_success_rate：Verifier 判定成功的 case 占比
    success_count = sum(1 for o in outcomes if o.succeeded)
    task_success_rate = _format_fraction(success_count, total)

    # 2. step_success_rate：跨所有 case 汇总每一步 ToolResult.success 的占比
    all_steps = [row for o in outcomes for row in o.step_records]
    step_success_count = sum(1 for row in all_steps if row.get("success"))
    step_success_rate = _format_percent(step_success_count, len(all_steps))

    # 3. avg_steps：只统计成功拿到 agent_result 的 case（崩溃的 case 没有步数可言）
    completed = [o for o in outcomes if o.agent_result is not None]
    avg_steps = (
        f"{sum(_steps_of(o) for o in completed) / len(completed):.1f}"
        if completed
        else "N/A"
    )

    # 4. recovery_rate：出现过至少一次失败步骤、但任务最终仍判定成功的比例
    had_failed_step = [
        o for o in outcomes if any(not row.get("success") for row in o.step_records)
    ]
    recovered = [o for o in had_failed_step if o.succeeded]
    recovery_rate = _format_fraction(len(recovered), len(had_failed_step))

    # 5. unsafe_action_block_rate：当前架构下，browser_type 在写入前就会校验
    #    敏感字段并抛出 SafetyError（参见 agent/browser_tools.py），执行层
    #    不存在"敏感字段被写入成功"这种可观测样本——凡是被记录到的不安全
    #    动作事件（fail_reason 以 safety_violation 开头）必然是被成功拦截
    #    的那一种。因此分子分母相同：既如实反映"发生过 N 次不安全动作尝试"，
    #    也不会虚报一个我们当前架构下根本采集不到的"未拦截"分母。
    safety_events = sum(
        1
        for o in outcomes
        if o.agent_result and (o.agent_result["fail_reason"] or "").startswith("safety_violation")
    )
    unsafe_action_block_rate = _format_fraction(safety_events, safety_events)

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


def _render_summary_md(
    suite_metrics: dict[str, dict[str, str]],
    all_outcomes: dict[str, list[CaseOutcome]],
) -> str:
    """按实际跑过的 suite 动态生成列，避免例如只跑 local 时表格里出现空的公开网页列。"""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"# Eval Summary — {date_str}", ""]

    suites_present = [s for s in ("local", "public") if s in suite_metrics]

    header = "| 指标 | " + " | ".join(_SUITE_LABELS[s] for s in suites_present) + " |"
    divider = "|" + "---|" * (len(suites_present) + 1)
    lines.append(header)
    lines.append(divider)
    for metric in _METRIC_ORDER:
        row_values = " | ".join(suite_metrics[s][metric] for s in suites_present)
        lines.append(f"| {metric} | {row_values} |")

    lines.append("")
    lines.append("## 失败任务")
    lines.append("| Case ID | 任务 | fail_reason | 最后截图 |")
    lines.append("|---------|------|-------------|---------|")

    failed_rows = []
    for suite in suites_present:
        for outcome in all_outcomes[suite]:
            if outcome.succeeded:
                continue
            case_id = outcome.case.get("id", "?")
            task = outcome.case.get("task", "")
            failed_rows.append(
                f"| {case_id} | {task} | {outcome.display_fail_reason} | {outcome.last_screenshot} |"
            )

    lines.extend(failed_rows if failed_rows else ["| - | 无失败任务 | - | - |"])
    return "\n".join(lines) + "\n"


async def _run_suite(suite: str, config: AgentConfig) -> list[CaseOutcome]:
    cases = _load_cases(_CASE_DIRS[suite])
    outcomes: list[CaseOutcome] = []
    for idx, case in enumerate(cases):
        case_id = case.get("id", "?")
        logger.info("运行 case %s: %s", case_id, case.get("task", ""))
        outcome = await _run_one_case(case, config)
        status = "PASS" if outcome.succeeded else "FAIL"
        logger.info("case %s 完成: %s", case_id, status)
        outcomes.append(outcome)
        # 免费模型 RPM 较低，case 之间主动等待可避免触发 429。
        # 付费模型保持默认 case_delay=0，此分支不会执行。
        is_last = idx == len(cases) - 1
        if config.case_delay > 0 and not is_last:
            logger.info(
                "case_delay=%.0fs：等待 %.0f 秒后运行下一个 case（可通过 WEBAGENT_CASE_DELAY=0 关闭）",
                config.case_delay, config.case_delay,
            )
            await asyncio.sleep(config.case_delay)
    return outcomes


async def main_async(suite_arg: str, case_arg: str | None = None) -> None:
    config = AgentConfig.from_env()

    all_outcomes: dict[str, list[CaseOutcome]] = {}
    suite_metrics: dict[str, dict[str, str]] = {}

    if case_arg:
        # 如果指定了单个 case，我们需要在所有 suite 中查找该 case
        # case_arg 可以是 case_id（如 "local_01"）或者文件名（如 "local_01.json"）
        target_case_file = case_arg if case_arg.endswith(".json") else f"{case_arg}.json"
        found_case: EvalCase | None = None
        found_suite: str | None = None

        for suite, suite_dir in _CASE_DIRS.items():
            path = os.path.join(suite_dir, target_case_file)
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        found_case = json.load(f)
                        found_suite = suite
                        break
                except (OSError, json.JSONDecodeError) as exc:
                    logger.error("加载指定的 case 失败: %s (%s)", path, exc)
                    sys.exit(1)

        if found_case is None or found_suite is None:
            logger.error("未找到指定的 case: %s", case_arg)
            sys.exit(1)

        assert found_case is not None   # 类型收窄：消除 Any|None 警告
        assert found_suite is not None  # 类型收窄：消除 str|None 警告

        logger.info("运行单个 case [%s] (来自 %s 套件)", found_case.get("id", "?"), found_suite)
        outcome = await _run_one_case(found_case, config)
        status = "PASS" if outcome.succeeded else "FAIL"
        logger.info("case %s 完成: %s", found_case.get("id", "?"), status)

        all_outcomes[found_suite] = [outcome]
        suite_metrics[found_suite] = _compute_metrics([outcome])
    else:
        suites = ["local", "public"] if suite_arg == "all" else [suite_arg]
        for suite in suites:
            outcomes = await _run_suite(suite, config)
            if not outcomes:
                logger.warning("suite=%s 没有可运行的 case，已跳过该 suite 的汇总", suite)
                continue
            all_outcomes[suite] = outcomes
            suite_metrics[suite] = _compute_metrics(outcomes)

    if not suite_metrics:
        logger.warning("本次运行没有任何 suite 产出结果，仍会写出一份空的 eval_summary.md")

    summary_md = _render_summary_md(suite_metrics, all_outcomes)
    with open(_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(summary_md)
    print(f"已写入 {_SUMMARY_PATH}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="批量运行 WebAgent Eval Case")
    parser.add_argument(
        "--suite",
        choices=["local", "public", "all"],
        default="all",
        help="要运行的 case 套件（默认 all）",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="指定要运行的单个 case ID 或文件名（例如 local_01 或 local_01.json）",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.suite, args.case))


if __name__ == "__main__":
    main()
