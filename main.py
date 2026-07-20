"""CLI 入口：单次运行一个 WebAgent 任务。

用法示例：
    uv run python main.py --task "找到页面上的版本号" --url "http://localhost:8080/text_find.html"
    uv run python main.py --task "..." --url "https://example.com" --vision
    uv run python main.py --task "..." --url "https://example.com" --max-steps 20 --model claude-sonnet-4-6
"""

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from agent import AgentConfig, AgentController, LLMError, SafetyError
from rich.console import Console
from rich.panel import Panel

try:
    _console: "Console | None" = Console()
except ImportError:  # rich 是项目依赖，理论上总能导入；留个兜底防止环境缺依赖时脚本直接崩掉
    _console = None

logger = logging.getLogger(__name__)


def _print(msg: str, *, style: str | None = None) -> None:
    """统一的输出封装：有 rich 就用 rich 上色，没有就退化成普通 print。"""
    if _console is not None:
        _console.print(msg, style=style)
    else:
        print(msg)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WebAgent CLI：单次运行一个浏览器任务",
    )
    parser.add_argument("--task", required=True, help="任务描述，用自然语言说明要 Agent 做什么")
    parser.add_argument("--url", required=True, help="任务起始页面 URL")
    parser.add_argument(
        "--task-id",
        default=None,
        help="可选的任务标识，写入 report.json 的 task_id 字段（如 'L03'）。不传则为 null",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        default=None,
        help="开启视觉模态（截图喂给 LLM）。不传则使用 AgentConfig 默认/环境变量配置",
    )
    parser.add_argument("--model", default=None, help="覆盖使用的模型名，默认取 AgentConfig 配置")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖单任务最大步数")
    parser.add_argument("--max-fail", type=int, default=None, help="覆盖允许的连续失败次数")
    parser.add_argument("--trace-dir", default=None, help="覆盖 trace 输出根目录，默认 traces/")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="打印 DEBUG 级别日志（默认 INFO）",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> AgentConfig:
    """AgentConfig.from_env() 打底，命令行显式传入的参数再逐项覆盖。

    只覆盖用户真正传了的字段（不是 argparse 默认值 None），
    避免"没传 --vision"被误当成"显式关闭 vision"覆盖掉环境变量里的设置。
    """
    config = AgentConfig.from_env()
    overrides: dict = {}
    if args.vision is not None:
        overrides["vision"] = args.vision
    if args.model is not None:
        overrides["model"] = args.model
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.max_fail is not None:
        overrides["max_fail"] = args.max_fail
    if args.trace_dir is not None:
        overrides["trace_dir"] = args.trace_dir
    if overrides:
        config = AgentConfig(**{**config.__dict__, **overrides})
    return config


async def run_task(task: str, url: str, config: AgentConfig, task_id: str | None = None) -> int:
    """跑一次任务并打印结果 + trace 路径。返回进程退出码（0 成功 / 1 失败）。"""
    controller = AgentController(config)
    try:
        result = await controller.run(task, url, task_id=task_id)
    except SafetyError as exc:
        _print(f"[安全拦截] {exc}", style="bold red")
        return 1
    except LLMError as exc:
        _print(f"[LLM 调用失败] {exc}", style="bold red")
        return 1

    status = "✅ 成功" if result["success"] else "❌ 失败"
    style = "bold green" if result["success"] else "bold red"
    body = (
        f"任务 ID: {result['task_id'] or '-'}\n"
        f"任务: {result['task']}\n"
        f"状态: {status}\n"
        f"步数: {result['steps']}\n"
        f"耗时: {result['duration_s']}s\n"
        f"输出: {result['output']}\n"
        f"失败原因: {result['fail_reason'] or '-'}\n"
        f"Trace 目录: {result['trace_dir']}\n"
        f"最后截图: {result['last_screenshot'] or '-'}"
    )
    if _console is not None:
        _console.print(Panel(body, title="WebAgent 运行结果", style=style))
    else:
        _print(body)

    return 0 if result["success"] else 1


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = build_config(args)
    exit_code = asyncio.run(run_task(args.task, args.url, config, task_id=args.task_id))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
