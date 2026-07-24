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
import time
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from dotenv import load_dotenv

load_dotenv()

# 把项目根目录加进 sys.path，这样无论从哪个工作目录执行
# `python eval/run_eval.py`，`from eval.eval_core import ...` /
# `from agent import ...` 都能正常解析，不依赖调用者提前设置 PYTHONPATH。
_PROJECT_ROOT: str = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eval.eval_core import (  #（sys.path 必须先于这行执行）
    CASE_DIRS,
    AgentConfig,
    ArtifactError,
    CaseOutcome,
    EvalCase,
    _METRIC_ORDER,
    _SUITE_LABELS,
    _get_git_info,
    compute_metrics,
    load_cases,
    redact_outcome_failure_fields,
    run_one_case,
    write_artifact,
)

logger = logging.getLogger(__name__)

_SUMMARY_PATH = os.path.join(_PROJECT_ROOT, "eval", "eval_summary.md")

# [R3-5] public suite 礼貌性约束：同一域名请求间最小间隔（秒）。
# 实现者判断点：2 秒足以避免对目标站点造成突发请求压力，同时不过分拖慢 eval。
_PUBLIC_REQUEST_MIN_INTERVAL = 2.0

# 记录各域名最近一次访问时间，用于同域名礼貌性间隔控制
_last_domain_access: dict[str, float] = {}


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
            task, fail_reason = redact_outcome_failure_fields(outcome)
            failed_rows.append(
                f"| {case_id} | {task} | {fail_reason} | {outcome.last_screenshot} |"
            )

    lines.extend(failed_rows if failed_rows else ["| - | 无失败任务 | - | - |"])
    return "\n".join(lines) + "\n"


def _check_robots_txt(url: str) -> bool:
    """[R3-5] 检查 URL 是否被目标站点的 robots.txt 允许。

    public suite 礼貌性约束之一。使用 httpx 同步获取 robots.txt（设置超时），
    用 RobotFileParser 判定是否允许 WebAgent 访问。
    无法获取 robots.txt（网络错误、404 等）时保守地返回 True（允许），
    避免因 robots.txt 不可达而阻止正常评测。
    """
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        resp = httpx.get(robots_url, timeout=5, follow_redirects=True)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
            return rp.can_fetch("WebAgent", url)
        # 404 或其他状态码：没有 robots.txt 限制，允许
        return True
    except (httpx.RequestError, OSError, ValueError) as exc:
        logger.warning("无法获取 robots.txt: %s，保守允许访问（%s）", robots_url, exc)
        return True


async def _enforce_domain_politeness(url: str) -> None:
    """[R3-5] 同一域名请求间增加最小间隔。

    public suite 礼貌性约束之一。记录各域名最近一次访问时间，
    若距上次访问不足 _PUBLIC_REQUEST_MIN_INTERVAL 秒则等待。
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    now = time.monotonic()
    last = _last_domain_access.get(domain)
    if last is not None:
        elapsed = now - last
        if elapsed < _PUBLIC_REQUEST_MIN_INTERVAL:
            wait = _PUBLIC_REQUEST_MIN_INTERVAL - elapsed
            logger.info("域名 %s 礼貌性等待 %.1f 秒", domain, wait)
            await asyncio.sleep(wait)
    _last_domain_access[domain] = time.monotonic()


async def _run_suite(suite: str, config: AgentConfig) -> list[CaseOutcome]:
    cases = load_cases(CASE_DIRS[suite])
    outcomes: list[CaseOutcome] = []
    for idx, case in enumerate(cases):
        case_id = case.get("id", "?")

        # [R3-5] public suite 礼貌性约束：robots.txt 检查 + 域名间隔
        if suite == "public":
            url = case.get("url", "")
            if url and not _check_robots_txt(url):
                logger.warning("case %s 的 URL 被 robots.txt 禁止，跳过: %s", case_id, url)
                outcome = CaseOutcome(case=case)
                outcome.crash_reason = "robots_txt_disallowed"
                outcomes.append(outcome)
                continue
            if url:
                await _enforce_domain_politeness(url)

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


async def main_async(
    suite_arg: str,
    case_arg: str | None = None,
    artifact_dir: str | None = None,
    archive_case_ids: list[str] | None = None,
) -> None:
    git_info = _get_git_info() if artifact_dir else None
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

        # [R3-5] public suite 礼貌性约束
        if found_suite == "public":
            url = found_case.get("url", "")
            if url and not _check_robots_txt(url):
                logger.warning("case %s 的 URL 被 robots.txt 禁止，跳过", case_arg)
                outcome = CaseOutcome(case=found_case)
                outcome.crash_reason = "robots_txt_disallowed"
            else:
                if url:
                    await _enforce_domain_politeness(url)
                outcome = await run_one_case(found_case, config)
        else:
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

    # DS-R3: 生成可提交 artifact
    if artifact_dir:
        try:
            write_artifact(
                artifact_dir=artifact_dir,
                all_outcomes=all_outcomes,
                suite_metrics=suite_metrics,
                config=config,
                suite_arg=suite_arg,
                case_arg=case_arg,
                archive_case_ids=archive_case_ids,
                git_info=git_info,
            )
            print(f"artifact 已生成: {os.path.abspath(artifact_dir)}")
        except ArtifactError as exc:
            logger.error("生成 artifact 失败: %s", exc)
            sys.exit(1)


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
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="指定 artifact 输出目录（相对路径），生成可提交的评测证据",
    )
    parser.add_argument(
        "--archive-case-traces",
        default=None,
        help="逗号分隔的 case ID 列表，归档指定 case 的 trace（仅在 --artifact-dir 有效时生效）",
    )
    args = parser.parse_args()

    # 解析 archive-case-traces
    archive_case_ids: list[str] | None = None
    if args.archive_case_traces:
        archive_case_ids = [c.strip() for c in args.archive_case_traces.split(",") if c.strip()]
        if not args.artifact_dir:
            logger.warning("--archive-case-traces 仅在指定 --artifact-dir 时有效，已忽略")
            archive_case_ids = None

    asyncio.run(main_async(
        args.suite,
        args.case,
        artifact_dir=args.artifact_dir,
        archive_case_ids=archive_case_ids,
    ))


if __name__ == "__main__":
    main()
