"""执行轨迹记录层。

每步以一行 JSON 追加写入 trace.jsonl；任务结束后汇总生成 report.json。
写入策略：拼好整行字符串一次性 write + flush + fsync，
不使用 json.dump 直接写文件句柄，避免并发写入时出现半行截断。
"""

import json
import os
import time
from datetime import datetime, timezone

from agent.types import AgentResult, LLMAction, ObserveResult, ToolResult


class TraceLogger:
    """负责单次 run 的截图路径分配、逐步 trace 记录与最终报告生成。"""

    def __init__(self, base_dir: str = "traces") -> None:
        self.base_dir = base_dir
        self.run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.run_dir = os.path.join(base_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.trace_path = os.path.join(self.run_dir, "trace.jsonl")
        self.report_path = os.path.join(self.run_dir, "report.json")

        self._screenshot_step = 0
        # 记录最近一次分配出去的截图路径，供 write_report() 写入
        # report.json 的 last_screenshot 字段——next_screenshot_path()
        # 是全局唯一的截图路径分配入口（主循环 observe() 每步一次、
        # 以及 browser_screenshot 工具的显式 screenshot 动作都经它分配），
        # 因此这里天然就是"整个 run 里最后一张截图"的路径，不需要
        # AgentController 再额外传参数进来同步维护一份。
        self._last_screenshot_path: str | None = None
        # 用于计算每步 duration_ms：以上一次 record() 结束时刻为基准。
        # 这里在构造时先设一个兜底值——如果调用方在 open() 成功后没有调用
        # reset_step_timer()，第一步的 duration_ms 会从这一刻（TraceLogger
        # 构造时刻）算起，包含构造之后到第一次 record() 之间的全部耗时。
        self._last_time = time.monotonic()

    def reset_step_timer(self) -> None:
        """把 duration_ms 的计时基准重置为当前时刻。

        必须在 browser_open()（page.goto + 重试 + 登录页人工确认）成功
        结束、真正进入"观察->规划->决策->执行"主循环之前调用一次，否则
        第一步的 duration_ms 会把 open() 的耗时也算进去——包括登录页信号
        命中时 ask_human() 阻塞等待真人从终端输入选择的时间，这个等待
        时长和 agent 本身跑得快不快毫无关系，混进第一步的耗时里会让它
        变成一个和"这一步实际花了多久"脱节的离群值。
        """
        self._last_time = time.monotonic()

    def next_screenshot_path(self) -> str:
        """分配下一张截图路径，步数从 001 开始自动递增。"""
        self._screenshot_step += 1
        filename = f"step-{self._screenshot_step:03d}.png"
        path = os.path.join(self.run_dir, filename)
        self._last_screenshot_path = path
        return path

    @property
    def last_screenshot_path(self) -> str | None:
        """整个 run 目前为止分配出去的最后一张截图路径；一张都还没有时为 None。"""
        return self._last_screenshot_path

    @staticmethod
    def _parse_selector_level(selector: str | None) -> str | None:
        """从 selector 字符串解析定位策略层级（兜底方案）。

        例如 "text=Quickstart" -> "text"；"css=#submit" -> "css"；
        无 "=" 分隔符的原始 selector 归类为 "raw"；selector 为空返回 None。

        注意：这只是按前缀猜的，不代表最终真的是这一级命中的——比如
        "text=xxx" 本身就是合法的 Playwright locator 语法，可能在
        browser_click 三级降级的第一级（css/locator）就直接命中了，
        根本没走到 text 这一级。record() 会优先用 _resolve_selector_level
        读取 ToolResult.output 里的真实结果，只有拿不到时才退回这个猜测。
        """
        if not selector:
            return None
        if "=" in selector:
            level, _ = selector.split("=", 1)
            return level
        return "raw"

    @classmethod
    def _resolve_selector_level(cls, action: LLMAction, result: ToolResult) -> str | None:
        """解析这一步实际生效的 selector_level。

        click 动作的 browser_click() 会把三级降级里真正命中的那一级
        （"css"/"text"/"role"）写进 ToolResult.output（一段 JSON 字符串），
        这是唯一可信的来源；只有 output 缺失/非 click 动作/解析失败时，
        才退回 _parse_selector_level 按 selector 字符串前缀做的猜测。
        """
        if action.get("action") == "click":
            output = result.get("output")
            if isinstance(output, str):
                try:
                    parsed = json.loads(output)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict) and isinstance(parsed.get("selector_level"), str):
                    return parsed["selector_level"]
        return cls._parse_selector_level(action.get("selector"))

    def record(
        self,
        step: int,
        obs: ObserveResult,
        plan: str,
        action: LLMAction,
        result: ToolResult,
    ) -> None:
        """追加写入一行 trace 记录。"""
        now = time.monotonic()
        duration_ms = int((now - self._last_time) * 1000)
        self._last_time = now

        entry = {
            "run_id": self.run_id,
            "step": step,
            "timestamp": self._now_iso(),
            "url": obs.get("url"),
            "plan": plan,
            "action": action.get("action"),
            "selector": action.get("selector"),
            "selector_level": self._resolve_selector_level(action, result),
            "success": result.get("success"),
            "page_changed": result.get("page_changed"),
            "error_msg": result.get("error_msg"),
            "reason": action.get("reason"),
            "screenshot": obs.get("screenshot_path"),
            "duration_ms": duration_ms,
        }

        # 先在内存中拼好完整一行字符串，再一次性写入并 flush，
        # 避免多次 write 调用之间被打断导致 jsonl 单行截断。
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def write_report(self, task: str, result: AgentResult) -> None:
        """写入本次 run 的汇总报告 report.json。

        字段顺序 / 取值均对齐设计文档中的 report.json schema：
        run_id / task_id / task / url / success / output / steps /
        duration_s / fail_reason / trace_file / last_screenshot。
        task_id、url、duration_s、last_screenshot 由 AgentController._finalize()
        在构造 AgentResult 时一并写入，这里只负责原样落盘，不重新计算。
        """
        report = {
            "run_id": self.run_id,
            "task_id": result.get("task_id"),
            "task": task,
            "url": result.get("url"),
            "success": result.get("success"),
            "output": result.get("output"),
            "steps": result.get("steps"),
            "duration_s": result.get("duration_s"),
            "fail_reason": result.get("fail_reason"),
            "trace_file": self.trace_path,
            "last_screenshot": result.get("last_screenshot"),
        }
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _now_iso() -> str:
        """返回毫秒精度的 UTC ISO8601 时间戳，形如 2024-12-01T14:30:25.123Z。"""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
