"""ABLAT-001 消融实验脚本：同一批 case 分别用 DOM-only（vision=False）和
DOM+Vision（vision=True）跑一遍，两组除 vision 外的 AgentConfig 完全一致，
对比 task_success_rate / avg_steps 等指标差异。

DS-W2: 扩展为可提交、可复核的 versioned artifact。每次可展示的消融实验
必须提交原始结果、实验配置、明确 case 列表、每组运行次数、模型/vision/
commit/prompt fingerprint，以及每 case 的成功/步数/trace 路径/verifier 结果。
报告中统计的 case 集必须与原始结果显式对应。

用法：
    uv run python eval/run_ablation.py --suite local
    uv run python eval/run_ablation.py --suite local --cases L01,L02,L03,L04,L05,L06,L07,L08,L09,L10
    uv run python eval/run_ablation.py --suite local --artifact-dir eval/artifacts/ablation-20260721

case 加载 / 单 case 执行 / 指标计算的实现在 eval/eval_core.py 里，与
eval/run_eval.py 共用，不重复实现。
"""

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# 把项目根目录加进 sys.path，使得无论从哪个工作目录执行
# `python eval/run_ablation.py`，`from eval.eval_core import ...` /
# `from agent import ...` 都能正常解析。
_PROJECT_ROOT: str = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eval.eval_core import (  #（sys.path 必须先于这行执行）
    CASE_DIRS,
    AgentConfig,
    ArtifactError,
    CaseOutcome,
    compute_metrics,
    compute_raw_metrics,
    _get_git_info,
    load_cases,
    run_one_case,
)
from agent.types import EvalCase
from agent.prompts import (
    PLANNER_SYSTEM,
    SELECTOR_SYSTEM,
    EXTRACTOR_SYSTEM,
    JUDGE_SYSTEM,
)

logger = logging.getLogger(__name__)

_RESULTS_PATH = os.path.join(_PROJECT_ROOT, "eval", "ablation_results.json")

# DS-W2: ablation artifact 格式版本
_ABLATION_ARTIFACT_FORMAT_VERSION = 1

# (分组标签, trace_dir 子目录, vision 取值) —— 两组唯一的受控变量差异
# 就是 vision；trace_dir 额外按分组拆子目录，是为了不依赖 TraceLogger
# 内部时间戳精度来保证两组 run_id 不冲突（ABLAT-001 风险备注里点名的那条），
# 物理隔离比"确认时间戳够细"更可靠，也不需要改动 tracer.py。
_GROUPS: tuple[tuple[str, str, bool], ...] = (
    ("dom_only", "ablation_dom_only", False),
    ("vision", "ablation_vision", True),
)

# DS-W2: 默认不参与 Vision 效率均值的 case。
# L11 是安全回归 case（verify_mode=safety_block），其"失败"是安全机制预期
# 内的主动拦截，不是 Vision 信号能否解题的体现，混入 avg_steps 会让消融
# 结论失真。允许运行（记录在 excluded_case_ids），但不参与均值计算。
_DEFAULT_EXCLUDE_FROM_AVG: list[str] = ["L11"]

_EXCLUDE_REASON_TEMPLATE = (
    "安全回归 case（verify_mode=safety_block），其失败是安全机制预期内的"
    "主动拦截，不反映 Vision 信号对任务解题的影响，不参与 Vision 效率均值"
)

# DS-W2 [W2-2]: 各 LLM 调用角色的采样参数（取实际调用值）。
# 这些值来自各调用方模块中 call_with_retry() 的 create_kwargs：
#   - planner: agent/planner.py max_tokens=1024
#   - selector: agent/action_selector.py max_tokens=1024
#   - extractor: agent/browser_tools.py max_tokens=2048
#   - judge: agent/verifier.py max_tokens=1024
# temperature / top_p 均未显式设置（使用 Anthropic API 默认值），记为 null。
# 修改各调用方的 max_tokens 后须同步更新此处，否则 prompt_fingerprint 与
# 实际调用不一致。
SAMPLING_PARAMS: dict[str, dict[str, int | float | None]] = {
    "planner": {"max_tokens": 1024, "temperature": None, "top_p": None},
    "selector": {"max_tokens": 1024, "temperature": None, "top_p": None},
    "extractor": {"max_tokens": 2048, "temperature": None, "top_p": None},
    "judge": {"max_tokens": 1024, "temperature": None, "top_p": None},
}

# DS-W2 [W2-1]: 模型别名指向的底层模型可能被供应商未公示更新，
# 历史 artifact 的可复现性受此限制。ablation 脚本层面无法获取 API
# 响应元数据中的更细粒度模型标识，因此显式记录此 caveat。
_MODEL_PINNING_CAVEAT = (
    "记录的是供应商别名，供应商可能对别名指向的底层模型做未公示更新，"
    "历史 artifact 的可复现性受此限制。"
)


def _resolve_suites(suite_arg: str) -> list[str]:
    return ["local", "public"] if suite_arg == "all" else [suite_arg]


def _select_cases(suite_arg: str, case_ids: list[str] | None) -> list:
    """加载指定 suite（可为 all）下的 case，按 --cases 过滤。

    找不到的 case id 只记 warning 不中断——和 run_eval.py 里
    "单个文件/单个 case 出问题不影响整体"的一贯处理方式保持一致。
    """
    cases = []
    for suite in _resolve_suites(suite_arg):
        cases.extend(load_cases(CASE_DIRS[suite]))

    if case_ids is None:
        return cases

    wanted = set(case_ids)
    selected = [c for c in cases if c.get("id") in wanted]
    found_ids = {c.get("id") for c in selected}
    missing = wanted - found_ids
    if missing:
        logger.warning("以下 case id 未在 suite=%s 中找到，已跳过: %s", suite_arg, sorted(missing))
    return selected


def _group_config(base_config: AgentConfig, vision: bool, trace_subdir: str) -> AgentConfig:
    """基于同一个 base_config 派生分组配置：只改 vision + trace_dir，
    其余字段原样保留。用 dataclasses.replace 而不是手动重建 AgentConfig(...)，
    是为了在 AgentConfig 以后新增字段时不会因为这里漏写某个字段而让两组
    "唯一差异是 vision" 这个前提悄悄被破坏。
    """
    return dataclasses.replace(
        base_config,
        vision=vision,
        trace_dir=os.path.join(base_config.trace_dir, trace_subdir),
    )


async def _run_group(
    cases: list, config: AgentConfig, label: str, run_count: int,
) -> list[list[CaseOutcome]]:
    """运行一个分组的全部 case，支持多次运行。

    DS-W2 [W2-3]: 当 run_count > 1 时，外层 list 每个元素是一轮完整运行
    的 outcomes，后续 _group_summary 会把同一 case 的多次运行归入 runs[] 数组。
    先跑完所有 case 的第 1 轮，再跑第 2 轮，使同一轮内 case 间的环境差异
    最小化。
    """
    all_runs: list[list[CaseOutcome]] = []
    for run_idx in range(run_count):
        outcomes: list[CaseOutcome] = []
        for idx, case in enumerate(cases):
            case_id = case.get("id", "?")
            if run_count > 1:
                logger.info(
                    "[%s] 第 %d/%d 轮，运行 case %s: %s",
                    label, run_idx + 1, run_count, case_id, case.get("task", ""),
                )
            else:
                logger.info("[%s] 运行 case %s: %s", label, case_id, case.get("task", ""))
            outcome = await run_one_case(case, config)
            status = "PASS" if outcome.succeeded else "FAIL"
            logger.info("[%s] case %s 完成: %s", label, case_id, status)
            outcomes.append(outcome)
            is_last = idx == len(cases) - 1
            if config.case_delay > 0 and not is_last:
                await asyncio.sleep(config.case_delay)
        all_runs.append(outcomes)
    return all_runs


def _single_run_record(outcome: CaseOutcome, run_index: int) -> dict:
    """构建单次运行的 case 记录，供 runs[] 数组使用。

    DS-W2: 每个 run 记录包含 success / steps / trace_dir / fail_reason /
    verify_result / verify_confidence。
    """
    agent_result = outcome.agent_result
    verify_result = outcome.verify_result
    return {
        "run_index": run_index,
        "success": outcome.succeeded,
        "steps": agent_result["steps"] if agent_result else None,
        "trace_dir": agent_result["trace_dir"] if agent_result else None,
        "fail_reason": None if outcome.succeeded else outcome.display_fail_reason,
        "verify_result": verify_result,
        "verify_confidence": verify_result["confidence"] if verify_result else None,
    }


def _case_record(
    case: EvalCase,
    runs_outcomes: list[CaseOutcome],
    excluded: bool,
) -> dict:
    """构建一个 case 的完整记录，包含 runs[] 数组。

    DS-W2 [W2-3]: 无论 run_count 是多少，每个 case 都存放一个 runs: [] 数组，
    逐次记录该 case 每次运行的完整结果。报告中展示的均值/成功率等聚合指标
    必须能从 runs[] 原始数据重新计算得出。
    """
    return {
        "case_id": case.get("id", "?"),
        "task": case.get("task", ""),
        "excluded": excluded,
        "runs": [
            _single_run_record(outcome, run_idx)
            for run_idx, outcome in enumerate(runs_outcomes)
        ],
    }


def _group_summary(
    runs_outcomes: list[list[CaseOutcome]],
    config: AgentConfig,
    label: str,
    included_ids: set[str],
    excluded_ids: set[str],
) -> dict:
    """构建一个分组的汇总信息。

    DS-W2: avg_steps / task_success_rate / step_success_rate 只基于
    included_case_ids 计算（excluded case 不参与均值）。cases[] 包含全部
    运行的 case（含 excluded），每个 case 标注 excluded 字段。

    聚合指标跨所有 run 计算：将所有 run 的 included outcomes 展平后统一统计，
    确保均值/成功率能从 runs[] 原始数据重新计算得出。
    """
    # 展平所有 run 的 outcomes，用于全局聚合指标计算
    all_outcomes_flat = [o for run in runs_outcomes for o in run]
    included_outcomes_flat = [
        o for o in all_outcomes_flat if o.case.get("id") in included_ids
    ]

    raw = compute_raw_metrics(included_outcomes_flat)

    # 按 case_id 聚合多次运行
    case_order: list[str] = []
    case_runs: dict[str, list[CaseOutcome]] = {}
    for run in runs_outcomes:
        for outcome in run:
            cid = outcome.case.get("id", "?")
            if cid not in case_runs:
                case_runs[cid] = []
                case_order.append(cid)
            case_runs[cid].append(outcome)

    cases = []
    for cid in case_order:
        # 从最后一次 run 的 outcome 中取 case 定义（case 定义在多 run 间不变）
        last_outcome = case_runs[cid][-1]
        excluded = cid in excluded_ids
        cases.append(_case_record(last_outcome.case, case_runs[cid], excluded))

    return {
        "label": label,
        "vision": config.vision,
        "trace_dir": config.trace_dir,
        "run_count": len(runs_outcomes),
        # 分母基于 included cases × run_count
        "total": raw["total"],
        "success_count": raw["success_count"],
        "task_success_rate": raw["task_success_rate"],
        "avg_steps": raw["avg_steps"],
        "step_success_count": raw["step_success_count"],
        "step_total": raw["step_total"],
        "step_success_rate": raw["step_success_rate"],
        "metrics": compute_metrics(included_outcomes_flat),
        "cases": cases,
    }


def _print_table(dom_summary: dict, vision_summary: dict) -> None:
    """打印可直接复制进消融实验报告的 Markdown 对比表：
    | 指标 | DOM-only | DOM+Vision | 差异 |
    3 行分别是 task_success_rate / avg_steps / step_success_rate。

    DS-W2: 表格分母来自 included_case_ids（已排除 L11 等安全回归 case）。
    """

    def _frac(count: int, total: int) -> str:
        return f"{count}/{total}"

    def _pct(rate: float | None) -> str:
        return f"{rate * 100:.0f}%" if rate is not None else "N/A"

    def _one_decimal(value: float | None) -> str:
        return f"{value:.1f}" if value is not None else "N/A"

    def _diff_count(dom_count: int, vision_count: int) -> str:
        return f"{vision_count - dom_count:+d}"

    def _diff_float(dom_value: float | None, vision_value: float | None) -> str:
        if dom_value is None or vision_value is None:
            return "N/A"
        return f"{vision_value - dom_value:+.1f}"

    def _diff_pct(dom_rate: float | None, vision_rate: float | None) -> str:
        if dom_rate is None or vision_rate is None:
            return "N/A"
        return f"{(vision_rate - dom_rate) * 100:+.0f}%"

    rows = (
        (
            "task_success_rate",
            _frac(dom_summary["success_count"], dom_summary["total"]),
            _frac(vision_summary["success_count"], vision_summary["total"]),
            _diff_count(dom_summary["success_count"], vision_summary["success_count"]),
        ),
        (
            "avg_steps",
            _one_decimal(dom_summary["avg_steps"]),
            _one_decimal(vision_summary["avg_steps"]),
            _diff_float(dom_summary["avg_steps"], vision_summary["avg_steps"]),
        ),
        (
            "step_success_rate",
            _pct(dom_summary["step_success_rate"]),
            _pct(vision_summary["step_success_rate"]),
            _diff_pct(dom_summary["step_success_rate"], vision_summary["step_success_rate"]),
        ),
    )

    lines = [
        "",
        "| 指标 | DOM-only | DOM+Vision | 差异 |",
        "|------|----------|------------|------|",
    ]
    lines.extend(f"| {name} | {dom_val} | {vision_val} | {delta} |" for name, dom_val, vision_val, delta in rows)
    print("\n".join(lines))


def _print_divergence(dom_summary: dict, vision_summary: dict) -> None:
    """逐 case 标出两组结果不一致的地方，这是消融实验最直接想看的信息，
    附在 Markdown 表格之后打印，不混进表格本身（不影响表格被直接复制使用）。
    """
    dom_by_id = {c["case_id"]: c for c in dom_summary["cases"]}
    vision_by_id = {c["case_id"]: c for c in vision_summary["cases"]}
    diverged = [
        case_id
        for case_id in dom_by_id
        if case_id in vision_by_id and dom_by_id[case_id]["runs"][0]["success"] != vision_by_id[case_id]["runs"][0]["success"]
    ]

    lines = [""]
    if diverged:
        lines.append("两组结果不一致的 case：")
        for case_id in diverged:
            dom_ok = "PASS" if dom_by_id[case_id]["runs"][0]["success"] else "FAIL"
            vision_ok = "PASS" if vision_by_id[case_id]["runs"][0]["success"] else "FAIL"
            lines.append(f"  {case_id}: DOM-only={dom_ok}  DOM+Vision={vision_ok}")
    else:
        lines.append("两组结果没有出现分歧的 case。")
    print("\n".join(lines))


# ===========================================================================
# DS-W2: prompt_fingerprint 与采样参数
# ===========================================================================


def _compute_prompt_fingerprint(sampling_params: dict) -> str:
    """DS-W2 [W2-2]: Planner/Selector/Extractor/Judge prompt 文本与采样参数
    拼接后取 SHA-256。

    采样参数包括各角色的 max_tokens / temperature / top_p（取实际调用值）。
    不得记录 API key。
    """
    parts = [
        PLANNER_SYSTEM,
        SELECTOR_SYSTEM,
        EXTRACTOR_SYSTEM,
        JUDGE_SYSTEM,
        json.dumps(sampling_params, sort_keys=True, ensure_ascii=False),
    ]
    return hashlib.sha256("\n---\n".join(parts).encode("utf-8")).hexdigest()


# ===========================================================================
# DS-W2: 一致性验证
# ===========================================================================


class AblationConsistencyError(Exception):
    """DS-W2: 消融实验一致性验证失败。

    两组 case 集不一致、included/excluded 与实际运行不匹配时抛出。
    生成过程必须失败，禁止输出差异结论。
    """


def _validate_case_consistency(
    dom_runs: list[list[CaseOutcome]],
    vision_runs: list[list[CaseOutcome]],
    included_ids: list[str],
    excluded_ids: list[str],
) -> None:
    """DS-W2 验证策略：两组 case ID 集不一致时必须失败。

    额外验证：
    - included + excluded = 实际运行的全部 case ID
    - included 和 excluded 不交叉
    - 每轮 run 的 case 集一致
    """
    # 取第一轮的 case ID 集作为基准（多轮间一致性在 _run_group 内由 cases 列表保证）
    if not dom_runs or not vision_runs:
        raise AblationConsistencyError("至少一个分组没有产出任何运行结果")

    dom_ids = {o.case.get("id", "?") for o in dom_runs[0]}
    vision_ids = {o.case.get("id", "?") for o in vision_runs[0]}

    if dom_ids != vision_ids:
        only_dom = dom_ids - vision_ids
        only_vision = vision_ids - dom_ids
        raise AblationConsistencyError(
            f"两组 case ID 集不一致: dom_only 独有={sorted(only_dom)}, "
            f"vision 独有={sorted(only_vision)}"
        )

    all_run_ids = dom_ids  # dom_ids == vision_ids
    included_set = set(included_ids)
    excluded_set = set(excluded_ids)

    # included + excluded 必须覆盖全部运行 case
    if included_set | excluded_set != all_run_ids:
        missing = all_run_ids - (included_set | excluded_set)
        extra = (included_set | excluded_set) - all_run_ids
        raise AblationConsistencyError(
            f"included + excluded 与实际运行的 case 集不匹配: "
            f"未分类={sorted(missing)}, 多余={sorted(extra)}"
        )

    # included 和 excluded 不交叉
    if included_set & excluded_set:
        overlap = included_set & excluded_set
        raise AblationConsistencyError(
            f"included_case_ids 与 excluded_case_ids 存在交叉: {sorted(overlap)}"
        )


# ===========================================================================
# DS-W2: Artifact 序列化
# ===========================================================================


def render_ablation_summary(
    payload: dict,
    dom_summary: dict,
    vision_summary: dict,
) -> str:
    """DS-W2: 构建 artifact 的 summary.md 内容。

    包含对比表、逐 case 步数、分歧 case、基准有效性声明、溯源信息摘要。
    """
    lines: list[str] = []

    # [R3-2] git_dirty 工作区提示
    if payload.get("git_dirty"):
        lines.append(
            "> ⚠️ **工作区状态提示**：生成此 artifact 时工作区存在未提交改动，"
            "结果可能不完全对应 `git_commit` 所指向的代码状态。"
        )
        lines.append("")

    generated_at = payload["generated_at"]
    date_str = generated_at[:10]
    lines.append(f"# Ablation Artifact Summary — {date_str}")
    lines.append(f"- `generated_at`: {generated_at}")
    lines.append(f"- `git_commit`: {payload.get('git_commit') or 'unknown'}")
    lines.append(f"- `model`: {payload.get('model')}")
    lines.append(f"- `prompt_fingerprint`: {payload.get('prompt_fingerprint')}")
    lines.append(f"- `run_count_per_group`: {payload.get('run_count_per_group')}")
    lines.append("")

    # 对比表
    lines.append("## 结果对比")
    lines.append("")
    lines.append("| 指标 | DOM-only | DOM+Vision | 差异 |")
    lines.append("|------|----------|------------|------|")

    def _frac(count, total):
        return f"{count}/{total}"

    def _pct(rate):
        return f"{rate * 100:.0f}%" if rate is not None else "N/A"

    def _one_decimal(value):
        return f"{value:.1f}" if value is not None else "N/A"

    dom = dom_summary
    vis = vision_summary
    lines.append(
        f"| task_success_rate | {_frac(dom['success_count'], dom['total'])} | "
        f"{_frac(vis['success_count'], vis['total'])} | "
        f"{vis['success_count'] - dom['success_count']:+d} |"
    )
    lines.append(
        f"| avg_steps | {_one_decimal(dom['avg_steps'])} | "
        f"{_one_decimal(vis['avg_steps'])} | "
        f"{(vis['avg_steps'] - dom['avg_steps']) if dom['avg_steps'] is not None and vis['avg_steps'] is not None else 'N/A':+,.1f} |"
    )
    lines.append(
        f"| step_success_rate | {_pct(dom['step_success_rate'])} | "
        f"{_pct(vis['step_success_rate'])} | "
        f"{((vis['step_success_rate'] - dom['step_success_rate']) * 100) if dom['step_success_rate'] is not None and vis['step_success_rate'] is not None else 'N/A':+,.0f}% |"
    )

    # case 集说明
    lines.append("")
    lines.append("## Case 集")
    lines.append(f"- `included_case_ids`（参与均值）: {payload.get('included_case_ids')}")
    lines.append(f"- `excluded_case_ids`（不参与均值）: {payload.get('excluded_case_ids')}")
    reasons = payload.get("excluded_case_reasons", {})
    if reasons:
        lines.append("- 排除原因：")
        for cid, reason in reasons.items():
            lines.append(f"  - `{cid}`: {reason}")

    # 逐 case 步数
    lines.append("")
    lines.append("## 逐 case 步数对比")
    lines.append("| case_id | excluded | DOM-only steps | DOM+Vision steps | 差值 |")
    lines.append("|---------|----------|-----------------|-------------------|------|")
    dom_by_id = {c["case_id"]: c for c in dom_summary["cases"]}
    vis_by_id = {c["case_id"]: c for c in vision_summary["cases"]}
    for cid in dom_by_id:
        dom_runs = dom_by_id[cid]["runs"]
        vis_runs = vis_by_id.get(cid, {}).get("runs", [])
        dom_steps_list = [r["steps"] for r in dom_runs if r["steps"] is not None]
        vis_steps_list = [r["steps"] for r in vis_runs if r["steps"] is not None]
        dom_avg = 0.0
        vis_avg = 0.0
        if dom_steps_list:
            dom_avg = sum(dom_steps_list) / len(dom_steps_list)
            dom_display = f"{dom_avg:.2f}" if len(dom_steps_list) > 1 else str(dom_steps_list[0])
        else:
            dom_display = "N/A"
        if vis_steps_list:
            vis_avg = sum(vis_steps_list) / len(vis_steps_list)
            vis_display = f"{vis_avg:.2f}" if len(vis_steps_list) > 1 else str(vis_steps_list[0])
        else:
            vis_display = "N/A"
        if dom_steps_list and vis_steps_list:
            diff = vis_avg - dom_avg
            diff_str = f"{diff:+.2f}"
        else:
            diff_str = "N/A"
        excl = "是" if dom_by_id[cid].get("excluded") else "否"
        lines.append(
            f"| {cid} | {excl} | {dom_display} | {vis_display} | {diff_str} |"
        )

    # 分歧 case
    lines.append("")
    lines.append("## 分歧 case")
    diverged = [
        cid for cid in dom_by_id
        if cid in vis_by_id
        and dom_by_id[cid]["runs"][0]["success"] != vis_by_id[cid]["runs"][0]["success"]
    ]
    if diverged:
        for cid in diverged:
            dom_ok = "PASS" if dom_by_id[cid]["runs"][0]["success"] else "FAIL"
            vis_ok = "PASS" if vis_by_id[cid]["runs"][0]["success"] else "FAIL"
            lines.append(f"- `{cid}`: DOM-only={dom_ok}  DOM+Vision={vis_ok}")
    else:
        lines.append("两组结果没有出现分歧的 case。")

    # 基准有效性
    lines.append("")
    lines.append("## 基准有效性")
    lines.append(f"`generated_at`: {generated_at}")
    lines.append("")
    lines.append(
        "消融实验结果基于生成时刻的代码状态和 LLM 模型行为，"
        "模型版本或代码变更后本 artifact 不再代表当前行为，请以最新 artifact 为准。"
    )
    lines.append("")
    lines.append(
        f"**模型快照说明**：{payload.get('model_pinning_caveat', '')}"
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def write_ablation_artifact(
    artifact_dir: str,
    payload: dict,
    summary_md: str,
) -> None:
    """DS-W2: 生成可提交的消融实验 artifact。

    在指定目录下生成：
    - ablation_results.json：包含原始结果 + provenance 字段的 versioned artifact
    - summary.md：人类可读的对比表 + 基准有效性声明

    错误处理：
    - artifact 目录已存在且非空 → ArtifactError（不覆盖旧证据）
    - 任何写入失败 → ArtifactError（不得伪装成成功）
    """
    artifact_dir = os.path.abspath(artifact_dir)

    # 覆盖保护：目录已存在且非空时拒绝（忽略 .gitkeep）
    if os.path.exists(artifact_dir) and os.path.isdir(artifact_dir):
        existing = [f for f in os.listdir(artifact_dir) if f != ".gitkeep"]
        if existing:
            raise ArtifactError(
                f"artifact 目录已存在且非空，拒绝覆盖: {artifact_dir}"
            )

    os.makedirs(artifact_dir, exist_ok=True)

    try:
        # ablation_results.json
        results_path = os.path.join(artifact_dir, "ablation_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # summary.md
        summary_path = os.path.join(artifact_dir, "summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_md)

        logger.info("ablation artifact 已生成: %s", artifact_dir)
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"写 ablation artifact 失败: {exc}") from exc


async def main_async(
    suite_arg: str,
    case_arg: str | None,
    artifact_dir: str | None,
    run_count: int,
    exclude_from_avg: list[str],
) -> None:
    git_commit, git_dirty = _get_git_info() if artifact_dir else (None, False)
    case_ids = [c.strip() for c in case_arg.split(",") if c.strip()] if case_arg else None
    cases = _select_cases(suite_arg, case_ids)
    if not cases:
        logger.error("没有可运行的 case（suite=%s, cases=%s），已终止", suite_arg, case_arg)
        sys.exit(1)

    base_config = AgentConfig.from_env()

    # DS-W2: 确定 included / excluded case 集合
    all_case_ids = [c.get("id", "?") for c in cases]
    excluded_set = set(exclude_from_avg) & set(all_case_ids)
    included_set = set(all_case_ids) - excluded_set

    excluded_reasons = {
        cid: _EXCLUDE_REASON_TEMPLATE for cid in excluded_set
    }

    # 传入 exclude_from_avg 但不在运行集中的 case，记 warning
    not_run_excluded = set(exclude_from_avg) - set(all_case_ids)
    if not_run_excluded:
        logger.warning(
            "--exclude-from-avg 中的 case 未在运行集中，已忽略: %s",
            sorted(not_run_excluded),
        )

    summaries: dict[str, dict] = {}
    runs_data: dict[str, list[list[CaseOutcome]]] = {}

    for idx, (label, trace_subdir, vision) in enumerate(_GROUPS):
        group_config = _group_config(base_config, vision, trace_subdir)
        logger.info(
            "=== 开始分组 [%s] vision=%s，共 %d 个 case，每组运行 %d 次 ===",
            label, vision, len(cases), run_count,
        )
        group_runs = await _run_group(cases, group_config, label, run_count)
        runs_data[label] = group_runs
        summaries[label] = _group_summary(
            group_runs, group_config, label, included_set, excluded_set,
        )

        is_last_group = idx == len(_GROUPS) - 1
        if base_config.case_delay > 0 and not is_last_group:
            await asyncio.sleep(base_config.case_delay)

    # DS-W2 验证策略：两组 case 集一致性验证
    _validate_case_consistency(
        runs_data["dom_only"], runs_data["vision"],
        sorted(included_set), sorted(excluded_set),
    )

    # DS-W2: 构建 provenance 字段
    if not artifact_dir:
        git_commit, git_dirty = _get_git_info()
    prompt_fingerprint = _compute_prompt_fingerprint(SAMPLING_PARAMS)

    included_case_ids = sorted(included_set)
    excluded_case_ids = sorted(excluded_set)

    result_payload = {
        "artifact_format_version": _ABLATION_ARTIFACT_FORMAT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "model": base_config.model,
        "model_pinning_caveat": _MODEL_PINNING_CAVEAT,
        "prompt_fingerprint": prompt_fingerprint,
        "sampling_params": SAMPLING_PARAMS,
        "run_count_per_group": run_count,
        "suite": suite_arg,
        "case_ids": all_case_ids,
        "included_case_ids": included_case_ids,
        "excluded_case_ids": excluded_case_ids,
        "excluded_case_reasons": excluded_reasons,
        # 实现者判断点：成本字段
        # 当前 API 响应未稳定暴露全部 token/cost 口径，llm_call_count 在
        # ablation 脚本层面无法获取（需在 run_one_case 层面埋点），记为 null。
        # 宁可字段为 null 并注明未采集，也不要伪造成本。
        "llm_call_count": None,
        "input_tokens": None,
        "output_tokens": None,
        "groups": summaries,
    }

    # 默认仍写本地忽略的 ablation_results.json
    os.makedirs(os.path.dirname(_RESULTS_PATH), exist_ok=True)
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    _print_table(summaries["dom_only"], summaries["vision"])
    _print_divergence(summaries["dom_only"], summaries["vision"])
    print(f"\n已写入 {_RESULTS_PATH}")

    # DS-W2: 生成可提交 artifact
    if artifact_dir:
        summary_md = render_ablation_summary(
            result_payload, summaries["dom_only"], summaries["vision"],
        )
        try:
            write_ablation_artifact(artifact_dir, result_payload, summary_md)
            print(f"ablation artifact 已生成: {os.path.abspath(artifact_dir)}")
        except ArtifactError as exc:
            logger.error("生成 ablation artifact 失败: %s", exc)
            sys.exit(1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="DOM-only vs DOM+Vision 消融实验")
    parser.add_argument(
        "--suite",
        choices=["local", "public", "all"],
        required=True,
        help="要运行的 case 套件",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="逗号分隔的 case id 列表（如 L01,L02,L03），缺省表示跑该 suite 下全部 case",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="指定 artifact 输出目录（相对路径），生成可提交的消融实验证据",
    )
    parser.add_argument(
        "--run-count",
        type=int,
        default=1,
        help="每组每个 case 的运行次数（默认 1），>1 时每个 case 存 runs[] 数组",
    )
    parser.add_argument(
        "--exclude-from-avg",
        default=None,
        help=(
            "逗号分隔的 case id 列表，指定运行但不参与 Vision 效率均值的 case"
            "（默认 L11）"
        ),
    )
    args = parser.parse_args()

    # 解析 exclude-from-avg
    if args.exclude_from_avg is not None:
        exclude_from_avg = [c.strip() for c in args.exclude_from_avg.split(",") if c.strip()]
    else:
        exclude_from_avg = list(_DEFAULT_EXCLUDE_FROM_AVG)

    if args.run_count < 1:
        logger.error("--run-count 必须 >= 1")
        sys.exit(1)

    asyncio.run(main_async(
        args.suite,
        args.cases,
        artifact_dir=args.artifact_dir,
        run_count=args.run_count,
        exclude_from_avg=exclude_from_avg,
    ))


if __name__ == "__main__":
    main()
