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
        # 用于计算每步 duration_ms：以上一次 record() 结束时刻为基准
        self._last_time = time.monotonic()

    def next_screenshot_path(self) -> str:
        """分配下一张截图路径，步数从 001 开始自动递增。"""
        self._screenshot_step += 1
        filename = f"step-{self._screenshot_step:03d}.png"
        return os.path.join(self.run_dir, filename)

    @staticmethod
    def _parse_selector_level(selector: str | None) -> str | None:
        """从 selector 字符串解析定位策略层级。

        例如 "text=Quickstart" -> "text"；"css=#submit" -> "css"；
        无 "=" 分隔符的原始 selector 归类为 "raw"；selector 为空返回 None。
        """
        if not selector:
            return None
        if "=" in selector:
            level, _ = selector.split("=", 1)
            return level
        return "raw"

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
            "selector_level": self._parse_selector_level(action.get("selector")),
            "success": result.get("success"),
            "page_changed": result.get("page_changed"),
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
        """写入本次 run 的汇总报告 report.json。"""
        report = {
            "run_id": self.run_id,
            "task": task,
            "success": result.get("success"),
            "steps": result.get("steps"),
            "fail_reason": result.get("fail_reason"),
            "output": result.get("output"),
            "trace_file": self.trace_path,
        }
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _now_iso() -> str:
        """返回毫秒精度的 UTC ISO8601 时间戳，形如 2024-12-01T14:30:25.123Z。"""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
