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
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone

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
from agent.privacy import redact_data

logger = logging.getLogger(__name__)

# DS-R3: Artifact 序列化相关常量
_ARTIFACT_FORMAT_VERSION = 1

# suite 标签和指标顺序——run_eval.py 的 eval_summary.md 渲染和
# eval_core.py 的 artifact summary 渲染共用，避免两处分别定义导致漂移。
_SUITE_LABELS = {"local": "本地任务", "public": "公开网页"}
_METRIC_ORDER = (
    "task_success_rate",
    "step_success_rate",
    "avg_steps",
    "recovery_rate",
    "unsafe_action_block_rate",
    "evidence_completeness",
)


class ArtifactError(Exception):
    """DS-R3: artifact 生成过程中的错误（目录已存在、case 不存在等）。

    写 artifact 失败不得伪装成 eval 成功，调用方应捕获此异常并以非零状态退出。
    """


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
        agent_result = await controller.run(case["task"], case["url"], task_id=case.get("id"))
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
    """trace.jsonl + report.json + 至少一张截图 + trace_schema_version >= 2 都满足，才算证据链完整。

    DS-Y3 [Y3-2]: 旧格式 trace（无 trace_schema_version 或版本 < 2）缺少 observation
    和 tool_output 字段，无法完整复盘页面状态和执行输出，因此不计为完整证据。
    这不会导致 runner 崩溃——旧 trace 仍可正常读取，只是 evidence_completeness
    指标会反映其不完整。
    """
    if outcome.agent_result is None:
        return False
    trace_dir = outcome.agent_result["trace_dir"]
    if not os.path.isdir(trace_dir):
        return False
    has_trace = os.path.isfile(os.path.join(trace_dir, "trace.jsonl"))
    report_path = os.path.join(trace_dir, "report.json")
    has_report = os.path.isfile(report_path)
    has_screenshot = any(f.endswith(".png") for f in os.listdir(trace_dir))
    if not (has_trace and has_report and has_screenshot):
        return False
    # v2 不是仅有版本号：每一条记录都必须具备完整观察/执行字段。
    try:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if report.get("privacy_redaction_version") != 1:
        return False
    return _records_have_complete_v2_evidence(outcome.step_records)


def _records_have_complete_v2_evidence(records: list[dict]) -> bool:
    if not records:
        return False
    required_observation = {"title", "text_hash", "visible_text_summary", "interactive_elements"}
    required_execution = {"action", "selector", "reason", "tool_output", "tool_output_truncated", "tool_output_sha256"}
    for record in records:
        observation = record.get("observation")
        if record.get("trace_schema_version") != 2 or record.get("privacy_redaction_version") != 1:
            return False
        if not isinstance(observation, dict) or not required_observation.issubset(observation):
            return False
        if not required_execution.issubset(record):
            return False
    return True


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


def _recovery_counts(outcomes: list[CaseOutcome]) -> tuple[int, int]:
    """(自愈成功的 case 数, 出现过失败步骤的 case 数)。

    排除 verify_mode == "safety_block" 的 case：这类 case 里 SafetyError
    产生的"失败步骤"是安全机制预期内的主动拦截终止，不是 agent 犯错后
    靠自己重新观察/规划/决策纠正回来的。如果不排除，这类 case 会被误记为
    一次"完美自愈"，让 recovery_rate 这个指标失去意义（参考 L11 场景：
    report.json success=false + fail_reason=safety_violation，但因为
    verify_mode=safety_block，Verifier 反向判定 success=true，若不过滤，
    这一个 case 就会同时进入分子和分母，虚报出 100% 的自愈率）。
    """
    had_failed_step = [
        o for o in outcomes
        if o.case.get("verify_mode") != "safety_block"
        and any(not row.get("success") for row in o.step_records)
    ]
    recovered = [o for o in had_failed_step if o.succeeded]
    return len(recovered), len(had_failed_step)


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
    recovered_count, had_failed_count = _recovery_counts(outcomes)
    recovery_rate = _format_fraction(recovered_count, had_failed_count)

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


# ===========================================================================
# DS-R3: Artifact 序列化——生成可提交、可复核的评测证据
# ===========================================================================
#
# 当 run_eval.py 传入 --artifact-dir 时，调用 write_artifact() 生成：
#   eval/artifacts/<dir>/
#     summary.md         人类可读汇总（含指标表、失败任务、基准有效性声明）
#     results.json       每个 case 的完整结构化结果（agent_result / verify_result 等）
#     provenance.json    运行参数、模型、vision、git commit 等溯源信息
#     traces/<case_id>/  归档的 trace.jsonl / report.json / 截图（仅 --archive-case-traces）
#
# 错误处理规则（Design Spec）：
#   - artifact 目录已存在且非空 → ArtifactError（不覆盖旧证据）
#   - --archive-case-traces 指定不存在 case 或不存在 trace 目录 → ArtifactError
#   - 写 artifact 失败不得伪装成 eval 成功；进程以非零状态退出


def _get_git_info() -> tuple[str | None, bool]:
    """获取当前 git 仓库的 commit SHA 和工作区是否 dirty。

    返回 (commit_sha, is_dirty)。git 不可用或不在 git 仓库中时，
    commit_sha=None, is_dirty=False（Spec: 若不可获取则显式记录 unknown）。

    [R3-2] is_dirty 用于 provenance.json 的 git_dirty 字段和 summary.md 的工作区提示。
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, False


def _build_case_record(outcome: CaseOutcome, suite: str) -> dict:
    """构建 results.json 中单个 case 的记录。

    DS-R3: 每个 case 必须包含 case_id / suite / succeeded / agent_result /
    verify_result / crash_reason / last_screenshot / trace_dir。
    """
    last_screenshot = outcome.last_screenshot
    if last_screenshot == "N/A":
        last_screenshot = None
    return {
        "case_id": outcome.case.get("id", "?"),
        "suite": suite,
        "succeeded": outcome.succeeded,
        "agent_result": redact_data(outcome.agent_result),
        "verify_result": redact_data(outcome.verify_result),
        "crash_reason": redact_data(outcome.crash_reason),
        "last_screenshot": last_screenshot,
        "trace_dir": outcome.agent_result["trace_dir"] if outcome.agent_result else None,
    }


def build_results_json(
    all_outcomes: dict[str, list[CaseOutcome]],
) -> dict:
    """构建 results.json 的完整结构。

    遍历 local 和 public 两个 suite 的全部 outcomes，每个 case 生成一条记录。
    """
    cases = []
    for suite in ("local", "public"):
        if suite in all_outcomes:
            for outcome in all_outcomes[suite]:
                cases.append(_build_case_record(outcome, suite))
    return {"cases": cases}


def build_provenance(
    config: AgentConfig,
    suite_arg: str,
    case_arg: str | None,
    case_ids: list[str],
    git_commit: str | None,
    git_dirty: bool,
) -> dict:
    """构建 provenance.json 结构。

    DS-R3: 必须包含 generated_at / suite_argument / case_argument / case_ids /
    model / vision / max_steps / max_fail / git_commit / artifact_format_version。
    [R3-2] 新增 git_dirty 字段，记录工作区是否有未提交改动。
    """
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite_argument": suite_arg,
        "case_argument": case_arg,
        "case_ids": case_ids,
        "model": config.model,
        "vision": config.vision,
        "max_steps": config.max_steps,
        "max_fail": config.max_fail,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "artifact_format_version": _ARTIFACT_FORMAT_VERSION,
    }


def render_artifact_summary(
    suite_metrics: dict[str, dict[str, str]],
    all_outcomes: dict[str, list[CaseOutcome]],
    provenance: dict,
) -> str:
    """构建 artifact 的 summary.md 内容。

    DS-R3: 基于 eval_summary.md 的格式，额外包含：
    - [R3-2] git_dirty 时顶部工作区提示
    - [R3-3] "基准有效性"小节，含 generated_at 和固定措辞声明
    """
    lines: list[str] = []

    # [R3-2] git_dirty 工作区提示
    if provenance.get("git_dirty"):
        lines.append(
            "> ⚠️ **工作区状态提示**：生成此 artifact 时工作区存在未提交改动，"
            "结果可能不完全对应 `git_commit` 所指向的代码状态。"
        )
        lines.append("")

    generated_at = provenance["generated_at"]
    date_str = generated_at[:10]
    lines.append(f"# Eval Artifact Summary — {date_str}")
    lines.append(f"- `generated_at`: {generated_at}")
    lines.append(f"- `git_commit`: {provenance.get('git_commit') or 'unknown'}")
    lines.append("")

    # 指标表格（与 eval_summary.md 格式一致）
    suites_present = [s for s in ("local", "public") if s in suite_metrics]
    if suites_present:
        header = "| 指标 | " + " | ".join(_SUITE_LABELS[s] for s in suites_present) + " |"
        divider = "|" + "---|" * (len(suites_present) + 1)
        lines.append(header)
        lines.append(divider)
        for metric in _METRIC_ORDER:
            row_values = " | ".join(suite_metrics[s][metric] for s in suites_present)
            lines.append(f"| {metric} | {row_values} |")

    # 失败任务表
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

    # [R3-3] 基准有效性
    lines.append("")
    lines.append("## 基准有效性")
    lines.append(f"`generated_at`: {generated_at}")
    lines.append("")
    lines.append(
        "public suite 结果基于外部网站在生成时刻的实际内容，"
        "网站变化后本 artifact 不再代表当前行为，请以最新 artifact 为准。"
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def _archive_case_traces(
    artifact_dir: str,
    all_outcomes: dict[str, list[CaseOutcome]],
    archive_case_ids: list[str],
) -> None:
    """将指定 case 的 trace 文件复制到 artifact 目录。

    DS-R3: --archive-case-traces 指定的 case 的 trace.jsonl、report.json 和
    截图（*.png）复制到 eval/artifacts/<dir>/traces/<case_id>/。

    [R3-1] 前置条件：DS-Y3（trace 脱敏与字段补全）已完成，归档的 trace
    已包含 trace_schema_version=2 的完整字段。L11（安全拦截 case）的
    trace 在 DS-Y3 Implementation Contract 下已不记录敏感输入值，可安全归档。

    错误处理：
    - case ID 不在 outcomes 中 → ArtifactError
    - case 的 trace_dir 不存在 → ArtifactError
    """
    # 构建 case_id → outcome 的映射
    id_to_outcome: dict[str, CaseOutcome] = {}
    for suite in ("local", "public"):
        if suite in all_outcomes:
            for outcome in all_outcomes[suite]:
                cid = outcome.case.get("id", "?")
                id_to_outcome[cid] = outcome

    # 验证所有 archive case ID 都存在
    missing = [cid for cid in archive_case_ids if cid not in id_to_outcome]
    if missing:
        raise ArtifactError(
            f"--archive-case-traces 指定的 case 不存在: {', '.join(missing)}"
        )

    # 在复制前完成全部输入的 schema / 隐私契约预检，避免把不可信 trace
    # 混入 artifact；_has_complete_evidence 同时检查 v2 必填字段和脱敏版本。
    for case_id in archive_case_ids:
        outcome = id_to_outcome[case_id]
        if outcome.agent_result is None:
            raise ArtifactError(
                f"--archive-case-traces 指定的 case {case_id} 没有产生 agent_result，"
                f"无法归档 trace"
            )
        trace_dir = outcome.agent_result["trace_dir"]
        if not trace_dir or not os.path.isdir(trace_dir):
            raise ArtifactError(
                f"--archive-case-traces 指定的 case {case_id} 的 trace 目录不存在: {trace_dir}"
            )
        disk_records = _load_step_records(trace_dir)
        disk_outcome = CaseOutcome(case=outcome.case, agent_result=outcome.agent_result, step_records=disk_records)
        if not _has_complete_evidence(disk_outcome):
            raise ArtifactError(
                f"--archive-case-traces 指定的 case {case_id} 的 trace 未通过 v2 完整性/隐私契约预检"
            )

    traces_dir = os.path.join(artifact_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    for case_id in archive_case_ids:
        outcome = id_to_outcome[case_id]
        assert outcome.agent_result is not None
        trace_dir = outcome.agent_result["trace_dir"]

        # 复制 trace.jsonl、report.json 和截图
        dest_dir = os.path.join(traces_dir, case_id)
        os.makedirs(dest_dir, exist_ok=True)

        for fname in os.listdir(trace_dir):
            # 只复制 trace.jsonl、report.json 和 *.png 截图
            if fname == "trace.jsonl" or fname == "report.json" or fname.endswith(".png"):
                src = os.path.join(trace_dir, fname)
                dst = os.path.join(dest_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)


def write_artifact(
    artifact_dir: str,
    all_outcomes: dict[str, list[CaseOutcome]],
    suite_metrics: dict[str, dict[str, str]],
    config: AgentConfig,
    suite_arg: str,
    case_arg: str | None,
    archive_case_ids: list[str] | None,
    git_info: tuple[str | None, bool] | None = None,
) -> None:
    """DS-R3: 生成可提交的评测 artifact。

    在指定目录下生成：
    - summary.md：人类可读的汇总报告
    - results.json：每个 case 的完整结构化结果
    - provenance.json：运行参数、模型信息、git commit 等
    - traces/<case_id>/：归档的 trace 文件（仅当 archive_case_ids 非空时）

    错误处理：
    - artifact 目录已存在且非空 → ArtifactError（不覆盖旧证据）
    - archive case 不存在或 trace 目录缺失 → ArtifactError
    - 任何写入失败 → ArtifactError（不得伪装成 eval 成功）
    """
    # 转为绝对路径
    artifact_dir = os.path.abspath(artifact_dir)

    # 覆盖保护：目录已存在且非空时拒绝（忽略 .gitkeep）
    if os.path.exists(artifact_dir) and os.path.isdir(artifact_dir):
        existing = [f for f in os.listdir(artifact_dir) if f != ".gitkeep"]
        if existing:
            raise ArtifactError(
                f"artifact 目录已存在且非空，拒绝覆盖: {artifact_dir}"
            )

    # 收集 case_ids
    case_ids: list[str] = []
    for suite in ("local", "public"):
        if suite in all_outcomes:
            for outcome in all_outcomes[suite]:
                case_ids.append(outcome.case.get("id", "?"))

    git_commit, git_dirty = git_info if git_info is not None else _get_git_info()

    # 先预检归档输入；这一步不得创建最终 artifact 目录。
    if archive_case_ids:
        _archive_case_traces_preflight(all_outcomes, archive_case_ids)

    # 在同一父目录的临时目录完成全部写入，成功后一次发布。任何异常都会
    # 清理临时目录，最终目录不会留下可被误认为有效证据的半成品。
    parent_dir = os.path.dirname(artifact_dir)
    os.makedirs(parent_dir, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix=".artifact-staging-", dir=parent_dir)
    try:
        # provenance.json
        provenance = build_provenance(
            config, suite_arg, case_arg, case_ids, git_commit, git_dirty,
        )
        provenance_path = os.path.join(staging_dir, "provenance.json")
        with open(provenance_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, ensure_ascii=False, indent=2)

        # results.json
        results = build_results_json(all_outcomes)
        results_path = os.path.join(staging_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # summary.md
        summary_md = render_artifact_summary(suite_metrics, all_outcomes, provenance)
        summary_path = os.path.join(staging_dir, "summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_md)

        # trace 归档
        if archive_case_ids:
            _archive_case_traces(staging_dir, all_outcomes, archive_case_ids)

        # 仅 .gitkeep 的预建目录允许被发布目录替换。
        if os.path.isdir(artifact_dir):
            gitkeep = os.path.join(artifact_dir, ".gitkeep")
            if os.path.isfile(gitkeep):
                os.unlink(gitkeep)
            os.rmdir(artifact_dir)
        os.replace(staging_dir, artifact_dir)

        logger.info("artifact 已生成: %s", artifact_dir)

    except ArtifactError:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise ArtifactError(f"写 artifact 失败: {exc}") from exc


def _archive_case_traces_preflight(
    all_outcomes: dict[str, list[CaseOutcome]], archive_case_ids: list[str]
) -> None:
    """在写任何 artifact 文件前验证所有待归档 trace。"""
    # 复用复制函数的校验路径，但不让它触碰最终目录；空临时目录只用于调用
    # 前的防御性检查不会满足“写最终文件”的条件。
    id_to_outcome = {
        outcome.case.get("id", "?"): outcome
        for suite in ("local", "public")
        for outcome in all_outcomes.get(suite, [])
    }
    missing = [case_id for case_id in archive_case_ids if case_id not in id_to_outcome]
    if missing:
        raise ArtifactError(f"--archive-case-traces 指定的 case 不存在: {', '.join(missing)}")
    for case_id in archive_case_ids:
        outcome = id_to_outcome[case_id]
        if outcome.agent_result is None:
            raise ArtifactError(f"--archive-case-traces 指定的 case {case_id} 没有产生 agent_result，无法归档 trace")
        trace_dir = outcome.agent_result["trace_dir"]
        if not trace_dir or not os.path.isdir(trace_dir):
            raise ArtifactError(f"--archive-case-traces 指定的 case {case_id} 的 trace 目录不存在: {trace_dir}")
        disk_records = _load_step_records(trace_dir)
        disk_outcome = CaseOutcome(case=outcome.case, agent_result=outcome.agent_result, step_records=disk_records)
        if not _has_complete_evidence(disk_outcome):
            raise ArtifactError(f"--archive-case-traces 指定的 case {case_id} 的 trace 未通过 v2 完整性/隐私契约预检")
