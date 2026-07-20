"""agent/tracer.py 单元测试：使用 tmp_path 作为 base_dir，覆盖截图路径
分配、selector_level 解析优先级、trace.jsonl / report.json 的写入内容。
"""

import json

from agent.tracer import TraceLogger
from agent.types import AgentResult, LLMAction, ObserveResult, ToolResult


def _make_tracer(tmp_path):
    return TraceLogger(base_dir=str(tmp_path))


def test_run_dir_created_on_init(tmp_path):
    tracer = _make_tracer(tmp_path)
    assert tmp_path.joinpath(tracer.run_id).is_dir()


def test_run_id_unique_across_consecutive_instances(tmp_path):
    """同一进程内连续创建 1,000 个 TraceLogger 时，run_id 和 run_dir 必须全部唯一。

    uuid.uuid4().hex[:8] 提供 32 bit 随机性，碰撞概率可忽略，
    足以覆盖同进程或跨进程的实际并发规模（[W1-1] 已评估并接受）。
    """
    run_ids = set()
    run_dirs = set()
    for _ in range(1000):
        tracer = TraceLogger(base_dir=str(tmp_path))
        run_ids.add(tracer.run_id)
        run_dirs.add(tracer.run_dir)
    assert len(run_ids) == 1000
    assert len(run_dirs) == 1000


def test_run_id_format_includes_timestamp_and_suffix(tmp_path):
    """run_id 仍以 run- 开头，包含时间戳和 8 位十六进制后缀。"""
    tracer = _make_tracer(tmp_path)
    assert tracer.run_id.startswith("run-")
    # 格式: run-YYYYMMDD-HHMMSS-a1b2c3d4
    parts = tracer.run_id.split("-")
    assert len(parts) == 4
    assert len(parts[3]) == 8
    int(parts[1])  # YYYYMMDD 可 parse 为整数
    int(parts[2])  # HHMMSS 可 parse 为整数


def test_next_screenshot_path_increments(tmp_path):
    tracer = _make_tracer(tmp_path)
    p1 = tracer.next_screenshot_path()
    p2 = tracer.next_screenshot_path()
    p3 = tracer.next_screenshot_path()
    assert p1.endswith("step-001.png")
    assert p2.endswith("step-002.png")
    assert p3.endswith("step-003.png")


def test_parse_selector_level_variants():
    assert TraceLogger._parse_selector_level(None) is None
    assert TraceLogger._parse_selector_level("") is None
    assert TraceLogger._parse_selector_level("text=Quickstart") == "text"
    assert TraceLogger._parse_selector_level("css=#submit") == "css"
    assert TraceLogger._parse_selector_level("#submit") == "raw"


def test_resolve_selector_level_prefers_tool_result_output_for_click():
    action: LLMAction = {"action": "click", "selector": "css=#submit", "text": None, "value": None, "reason": "x"}
    result: ToolResult = {"success": True, "page_changed": False, "output": json.dumps({"selector_level": "role"}), "error_msg": None}
    # ToolResult.output 里记录的真实命中层级（role）应优先于 selector 前缀猜测（css）
    assert TraceLogger._resolve_selector_level(action, result) == "role"


def test_resolve_selector_level_falls_back_when_output_not_json():
    action: LLMAction = {"action": "click", "selector": "css=#submit", "text": None, "value": None, "reason": "x"}
    result: ToolResult = {"success": True, "page_changed": False, "output": "not a json string", "error_msg": None}
    assert TraceLogger._resolve_selector_level(action, result) == "css"


def test_resolve_selector_level_falls_back_when_output_missing():
    action: LLMAction = {"action": "click", "selector": "text=OK", "text": None, "value": None, "reason": "x"}
    result: ToolResult = {"success": True, "page_changed": False, "output": None, "error_msg": None}
    assert TraceLogger._resolve_selector_level(action, result) == "text"


def test_resolve_selector_level_non_click_action_uses_selector_prefix():
    action: LLMAction = {"action": "type", "selector": "css=#input", "text": None, "value": None, "reason": "x"}
    result: ToolResult = {"success": True, "page_changed": False, "output": json.dumps({"selector_level": "role"}), "error_msg": None}
    # 非 click 动作忽略 output 里的 selector_level，直接按 selector 前缀猜测
    assert TraceLogger._resolve_selector_level(action, result) == "css"


def test_record_appends_jsonl_line(tmp_path):
    tracer = _make_tracer(tmp_path)
    obs: ObserveResult = {
        "url": "https://x",
        "title": "Test",
        "visible_text_summary": "",
        "text_hash": "h1",
        "interactive_elements": [],
        "screenshot_path": "/tmp/step-001.png",
    }
    plan = "点击提交按钮"
    action: LLMAction = {"action": "click", "selector": "css=#submit", "text": None, "value": None, "reason": "提交表单"}
    result: ToolResult = {"success": True, "page_changed": True, "error_msg": None, "output": None}

    tracer.record(0, obs, plan, action, result)
    tracer.record(1, obs, plan, action, result)

    lines = tracer.trace_path
    with open(lines, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 2
    assert rows[0]["step"] == 0
    assert rows[1]["step"] == 1
    assert rows[0]["url"] == "https://x"
    assert rows[0]["action"] == "click"
    assert rows[0]["success"] is True
    assert rows[0]["run_id"] == tracer.run_id
    assert "duration_ms" in rows[0]


def test_write_report_contains_expected_fields(tmp_path):
    tracer = _make_tracer(tmp_path)
    # 分配一张截图，模拟主循环 observe() 至少跑过一步，验证
    # last_screenshot 会原样透传进 report.json。
    screenshot_path = tracer.next_screenshot_path()
    result: AgentResult = {
        "task_id": "L03",
        "task": "测试任务",
        "url": "http://localhost:8080/tab_nav.html",
        "success": True,
        "steps": 3,
        "duration_s": 12.3,
        "fail_reason": None,
        "output": "done",
        "trace_dir": tracer.run_dir,
        "last_screenshot": screenshot_path,
    }
    tracer.write_report("测试任务", result)

    with open(tracer.report_path, encoding="utf-8") as f:
        report = json.load(f)

    assert report["run_id"] == tracer.run_id
    assert report["task_id"] == "L03"
    assert report["task"] == "测试任务"
    assert report["url"] == "http://localhost:8080/tab_nav.html"
    assert report["success"] is True
    assert report["output"] == "done"
    assert report["steps"] == 3
    assert report["duration_s"] == 12.3
    assert report["fail_reason"] is None
    assert report["trace_file"] == tracer.trace_path
    assert report["last_screenshot"] == screenshot_path


def test_write_report_task_id_and_last_screenshot_default_to_none(tmp_path):
    """task_id 未传（如 main.py CLI 单次运行场景）、且从未分配过截图时，
    两个字段应落为 null，而不是抛异常或被静默省略。
    """
    tracer = _make_tracer(tmp_path)
    result: AgentResult = {
        "task_id": None,
        "task": "测试任务",
        "url": "https://example.com",
        "success": False,
        "steps": 0,
        "duration_s": 0.5,
        "fail_reason": "open_failed: timeout",
        "output": None,
        "trace_dir": tracer.run_dir,
        "last_screenshot": None,
    }
    tracer.write_report("测试任务", result)

    with open(tracer.report_path, encoding="utf-8") as f:
        report = json.load(f)

    assert report["task_id"] is None
    assert report["last_screenshot"] is None


def test_last_screenshot_path_updates_as_screenshots_allocated(tmp_path):
    tracer = _make_tracer(tmp_path)
    assert tracer.last_screenshot_path is None
    p1 = tracer.next_screenshot_path()
    assert tracer.last_screenshot_path == p1
    p2 = tracer.next_screenshot_path()
    assert tracer.last_screenshot_path == p2
    assert p1 != p2
