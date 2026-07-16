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
    result: AgentResult = {
        "task": "测试任务",
        "success": True,
        "steps": 3,
        "fail_reason": None,
        "output": "done",
        "trace_dir": tracer.run_dir,
    }
    tracer.write_report("测试任务", result)

    with open(tracer.report_path, encoding="utf-8") as f:
        report = json.load(f)

    assert report["task"] == "测试任务"
    assert report["success"] is True
    assert report["steps"] == 3
    assert report["output"] == "done"
    assert report["run_id"] == tracer.run_id
