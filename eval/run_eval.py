"""批量 Eval 执行脚本：加载 eval case，逐个跑 AgentController + Verifier，
汇总 6 项指标并生成 eval_summary.md。

用法：
    uv run python eval/run_eval.py --suite local     # 本地 10 条
    uv run python eval/run_eval.py --suite public    # 公开 5 条
    uv run python eval/run_eval.py --suite all       # 全部

case 加载 / 单 case 执行 / 指标计算的实现在 eval/eval_core.py 里，本文件
只保留 markdown 汇总报告相关的部分（与 eval/run_ablation.py 共用 core，
避免同一份逻辑两处维护、行为漂移）。
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# 同 eval_core.py 的路径约定：把项目根目录加进 sys.path，这样无论从哪个
# 工作目录执行 `python eval/run_eval.py`，`import eval_core` / `from agent
# import ...` 都能正常解析，不依赖调用者提前设置 PYTHONPATH。
_PROJECT_ROOT: str = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# eval/run_eval.py 与 eval/eval_core.py 同目录，脚本自身所在目录会被
# Python 自动加进 sys.path[0]，因此可以直接 `import eval_core`。
from eval_core import (  #（sys.path 必须先于这行执行）
    CASE_DIRS,
    AgentConfig,
    CaseOutcome,
    EvalCase,
    compute_metrics,
    load_cases,
    run_one_case,
)

logger = logging.getLogger(__name__)

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
    cases = load_cases(CASE_DIRS[suite])
    outcomes: list[CaseOutcome] = []
    for idx, case in enumerate(cases):
        case_id = case.get("id", "?")
        logger.info("运行 case %s: %s", case_id, case.get("task", ""))
        outcome = await run_one_case(case, config)
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

        for suite, suite_dir in CASE_DIRS.items():
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
        outcome = await run_one_case(found_case, config)
        status = "PASS" if outcome.succeeded else "FAIL"
        logger.info("case %s 完成: %s", found_case.get("id", "?"), status)

        all_outcomes[found_suite] = [outcome]
        suite_metrics[found_suite] = compute_metrics([outcome])
    else:
        suites = ["local", "public"] if suite_arg == "all" else [suite_arg]
        for suite in suites:
            outcomes = await _run_suite(suite, config)
            if not outcomes:
                logger.warning("suite=%s 没有可运行的 case，已跳过该 suite 的汇总", suite)
                continue
            all_outcomes[suite] = outcomes
            suite_metrics[suite] = compute_metrics(outcomes)

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
