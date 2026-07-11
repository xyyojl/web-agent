"""ABLAT-001 消融实验脚本：同一批 case 分别用 DOM-only（vision=False）和
DOM+Vision（vision=True）跑一遍，两组除 vision 外的 AgentConfig 完全一致，
对比 task_success_rate / avg_steps 等指标差异。

用法：
    uv run python eval/run_ablation.py --suite local
    uv run python eval/run_ablation.py --suite local --cases L01,L02,L03,L04,L05,L06,L07,L08,L09,L10

case 加载 / 单 case 执行 / 指标计算的实现在 eval/eval_core.py 里，与
eval/run_eval.py 共用，不重复实现。
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
from datetime import datetime, timezone

# 与 eval_core.py 同一套路径约定：把项目根目录加进 sys.path，使得无论从
# 哪个工作目录执行 `python eval/run_ablation.py`，`import eval_core` /
# `from agent import ...` 都能正常解析。
_PROJECT_ROOT: str = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# eval/run_ablation.py 与 eval/eval_core.py 同目录，脚本自身所在目录会被
# Python 自动加进 sys.path[0]，因此可以直接 `import eval_core`。
from eval_core import (  # noqa: E402  （sys.path 必须先于这行执行）
    CASE_DIRS,
    AgentConfig,
    CaseOutcome,
    compute_metrics,
    compute_raw_metrics,
    load_cases,
    run_one_case,
)

logger = logging.getLogger(__name__)

_RESULTS_PATH = os.path.join(_PROJECT_ROOT, "eval", "ablation_results.json")

# (分组标签, trace_dir 子目录名, vision 取值) —— 两组唯一的受控变量差异
# 就是 vision；trace_dir 额外按分组拆子目录，是为了不依赖 TraceLogger
# 内部时间戳精度来保证两组 run_id 不冲突（ABLAT-001 风险备注里点名的那条），
# 物理隔离比"确认时间戳够细"更可靠，也不需要改动 tracer.py。
_GROUPS: tuple[tuple[str, str, bool], ...] = (
    ("dom_only", "ablation_dom_only", False),
    ("vision", "ablation_vision", True),
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


async def _run_group(cases: list, config: AgentConfig, label: str) -> list[CaseOutcome]:
    outcomes: list[CaseOutcome] = []
    for idx, case in enumerate(cases):
        case_id = case.get("id", "?")
        logger.info("[%s] 运行 case %s: %s", label, case_id, case.get("task", ""))
        outcome = await run_one_case(case, config)
        status = "PASS" if outcome.succeeded else "FAIL"
        logger.info("[%s] case %s 完成: %s", label, case_id, status)
        outcomes.append(outcome)
        is_last = idx == len(cases) - 1
        if config.case_delay > 0 and not is_last:
            await asyncio.sleep(config.case_delay)
    return outcomes


def _outcome_record(outcome: CaseOutcome) -> dict:
    """单个 case 的可序列化结果，供 ablation_results.json 使用。"""
    agent_result = outcome.agent_result
    verify_result = outcome.verify_result
    return {
        "case_id": outcome.case.get("id", "?"),
        "task": outcome.case.get("task", ""),
        "success": outcome.succeeded,
        "steps": agent_result["steps"] if agent_result else None,
        "trace_dir": agent_result["trace_dir"] if agent_result else None,
        "fail_reason": None if outcome.succeeded else outcome.display_fail_reason,
        "verify_confidence": verify_result["confidence"] if verify_result else None,
    }


def _group_summary(outcomes: list[CaseOutcome], config: AgentConfig, label: str) -> dict:
    raw = compute_raw_metrics(outcomes)
    return {
        "label": label,
        "vision": config.vision,
        "trace_dir": config.trace_dir,
        "total": raw["total"],
        "success_count": raw["success_count"],
        # 数值型字段供后续脚本/表格直接取用（含差异计算），metrics 里同名
        # 指标是给人看的格式化字符串（"8/10" 这种）——两者口径一致，都来自
        # compute_raw_metrics/compute_metrics 共用的同一套统计，不会对不上。
        "task_success_rate": raw["task_success_rate"],
        "avg_steps": raw["avg_steps"],
        "step_success_count": raw["step_success_count"],
        "step_total": raw["step_total"],
        "step_success_rate": raw["step_success_rate"],
        "metrics": compute_metrics(outcomes),
        "cases": [_outcome_record(o) for o in outcomes],
    }


def _print_table(dom_summary: dict, vision_summary: dict) -> None:
    """打印可直接复制进消融实验报告的 Markdown 对比表：
    | 指标 | DOM-only | DOM+Vision | 差异 |
    3 行分别是 task_success_rate / avg_steps / step_success_rate。
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
        if case_id in vision_by_id and dom_by_id[case_id]["success"] != vision_by_id[case_id]["success"]
    ]

    lines = [""]
    if diverged:
        lines.append("两组结果不一致的 case：")
        for case_id in diverged:
            dom_ok = "PASS" if dom_by_id[case_id]["success"] else "FAIL"
            vision_ok = "PASS" if vision_by_id[case_id]["success"] else "FAIL"
            lines.append(f"  {case_id}: DOM-only={dom_ok}  DOM+Vision={vision_ok}")
    else:
        lines.append("两组结果没有出现分歧的 case。")
    print("\n".join(lines))




async def main_async(suite_arg: str, case_arg: str | None) -> None:
    case_ids = [c.strip() for c in case_arg.split(",") if c.strip()] if case_arg else None
    cases = _select_cases(suite_arg, case_ids)
    if not cases:
        logger.error("没有可运行的 case（suite=%s, cases=%s），已终止", suite_arg, case_arg)
        sys.exit(1)

    base_config = AgentConfig.from_env()

    summaries: dict[str, dict] = {}
    for idx, (label, trace_subdir, vision) in enumerate(_GROUPS):
        group_config = _group_config(base_config, vision, trace_subdir)
        logger.info("=== 开始分组 [%s] vision=%s，共 %d 个 case ===", label, vision, len(cases))
        outcomes = await _run_group(cases, group_config, label)
        summaries[label] = _group_summary(outcomes, group_config, label)

        is_last_group = idx == len(_GROUPS) - 1
        if base_config.case_delay > 0 and not is_last_group:
            await asyncio.sleep(base_config.case_delay)

    result_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": suite_arg,
        "case_ids": [c.get("id", "?") for c in cases],
        "groups": summaries,
    }

    os.makedirs(os.path.dirname(_RESULTS_PATH), exist_ok=True)
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    _print_table(summaries["dom_only"], summaries["vision"])
    _print_divergence(summaries["dom_only"], summaries["vision"])
    print(f"\n已写入 {_RESULTS_PATH}")


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
    args = parser.parse_args()
    asyncio.run(main_async(args.suite, args.cases))


if __name__ == "__main__":
    main()
