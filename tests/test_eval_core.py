"""eval/eval_core.py 单元测试：覆盖 6 项指标的计算口径（尤其是
safety_block case 从 recovery_rate 分子分母中排除的过滤逻辑）、
case 加载的容错行为，以及 CaseOutcome 的派生属性。
"""

import json

import pytest

from agent.types import AgentResult, EvalCase, VerifyResult
from eval.eval_core import (
    CaseOutcome,
    _avg_steps_value,
    _has_complete_evidence,
    _recovery_counts,
    _step_success_counts,
    _task_success_counts,
    compute_metrics,
    compute_raw_metrics,
    load_cases,
)


def _case(**overrides) -> EvalCase:
    base: EvalCase = {
        "id": "L01",
        "type": "local",
        "task_type": "extract",
        "task": "抓取标题",
        "url": "https://example.com",
        "expected_output": "hello",
        "verify_mode": "exact",
        "difficulty": "easy",
    }
    base.update(overrides)
    return base


def _agent_result(**overrides) -> AgentResult:
    base: AgentResult = {
        "task_id": "T01",
        "task": "抓取标题",
        "url": "https://example.com",
        "success": True,
        "output": "hello",
        "steps": 2,
        "duration_s": 1.0,
        "fail_reason": None,
        "trace_dir": "/tmp/run-x",
        "last_screenshot": None,
    }
    base.update(overrides)
    return base


def _outcome(*, case=None, agent_result=None, verify_success=True, steps_records=None, crash=None):
    outcome = CaseOutcome(case=case or _case())
    outcome.crash_reason = crash
    if agent_result is not None:
        outcome.agent_result = agent_result
        verify_result: VerifyResult = {
            "case_id": outcome.case["id"],
            "success": verify_success,
            "reason": "ok" if verify_success else "no",
            "confidence": 1.0,
        }
        outcome.verify_result = verify_result
    outcome.step_records = steps_records or []
    return outcome


# ---------- load_cases ----------

def test_load_cases_missing_directory_returns_empty(tmp_path):
    missing_dir = str(tmp_path / "nonexistent")
    assert load_cases(missing_dir) == []


def test_load_cases_reads_and_sorts_json_files(tmp_path):
    (tmp_path / "L02.json").write_text(json.dumps(_case(id="L02")), encoding="utf-8")
    (tmp_path / "L01.json").write_text(json.dumps(_case(id="L01")), encoding="utf-8")
    (tmp_path / "readme.txt").write_text("not a case", encoding="utf-8")

    cases = load_cases(str(tmp_path))
    assert [c["id"] for c in cases] == ["L01", "L02"]


def test_load_cases_skips_invalid_json_file(tmp_path):
    (tmp_path / "L01.json").write_text(json.dumps(_case(id="L01")), encoding="utf-8")
    (tmp_path / "L02.json").write_text("{not valid json", encoding="utf-8")

    cases = load_cases(str(tmp_path))
    assert [c["id"] for c in cases] == ["L01"]


# ---------- CaseOutcome derived properties ----------

def test_case_outcome_succeeded_true_when_verify_success():
    outcome = _outcome(agent_result=_agent_result(), verify_success=True)
    assert outcome.succeeded is True


def test_case_outcome_succeeded_false_without_verify_result():
    outcome = CaseOutcome(case=_case())
    assert outcome.succeeded is False


def test_case_outcome_last_screenshot_returns_na_when_empty():
    outcome = CaseOutcome(case=_case())
    assert outcome.last_screenshot == "N/A"


def test_case_outcome_last_screenshot_returns_latest_step():
    outcome = CaseOutcome(case=_case())
    outcome.step_records = [{"screenshot": "step-001.png"}, {"screenshot": "step-002.png"}]
    assert outcome.last_screenshot == "step-002.png"


def test_case_outcome_display_fail_reason_prefers_crash():
    outcome = CaseOutcome(case=_case())
    outcome.crash_reason = "KeyError: url"
    assert outcome.display_fail_reason == "runner_crash: KeyError: url"


def test_case_outcome_display_fail_reason_uses_agent_result_fail_reason():
    outcome = _outcome(agent_result=_agent_result(fail_reason="max_steps_exceeded", success=False))
    assert outcome.display_fail_reason == "max_steps_exceeded"


def test_case_outcome_display_fail_reason_uses_verify_reason_when_no_fail_reason():
    outcome = _outcome(agent_result=_agent_result(fail_reason=None), verify_success=False)
    assert "verify_failed" in outcome.display_fail_reason


def test_case_outcome_display_fail_reason_unknown_fallback():
    outcome = CaseOutcome(case=_case())
    assert outcome.display_fail_reason == "unknown"


# ---------- metric helper functions ----------

def test_task_success_counts():
    outcomes = [
        _outcome(agent_result=_agent_result(), verify_success=True),
        _outcome(agent_result=_agent_result(), verify_success=False),
    ]
    assert _task_success_counts(outcomes) == (1, 2)


def test_step_success_counts():
    outcomes = [
        _outcome(agent_result=_agent_result(), steps_records=[{"success": True}, {"success": False}]),
        _outcome(agent_result=_agent_result(), steps_records=[{"success": True}]),
    ]
    assert _step_success_counts(outcomes) == (2, 3)


def test_avg_steps_value_ignores_crashed_cases():
    outcomes = [
        _outcome(agent_result=_agent_result(steps=4)),
        _outcome(agent_result=_agent_result(steps=6)),
        CaseOutcome(case=_case(), crash_reason="boom"),  # 无 agent_result，应被忽略
    ]
    assert _avg_steps_value(outcomes) == 5.0


def test_avg_steps_value_returns_none_when_no_completed_cases():
    outcomes = [CaseOutcome(case=_case(), crash_reason="boom")]
    assert _avg_steps_value(outcomes) is None


def test_recovery_counts_excludes_safety_block_cases():
    """safety_block case 即使出现失败步骤且最终判定成功，也不能计入自愈率分子分母。"""
    safety_case = _case(id="L11", verify_mode="safety_block")
    outcomes = [
        _outcome(
            case=safety_case,
            agent_result=_agent_result(fail_reason="safety_violation: x", success=False),
            verify_success=True,  # Verifier 反向判定为成功
            steps_records=[{"success": False}],
        ),
        _outcome(
            agent_result=_agent_result(),
            verify_success=True,
            steps_records=[{"success": False}, {"success": True}],
        ),
    ]
    recovered, had_failed = _recovery_counts(outcomes)
    assert had_failed == 1  # 只统计非 safety_block 的那一条
    assert recovered == 1


def test_recovery_counts_zero_when_no_failed_steps():
    outcomes = [_outcome(agent_result=_agent_result(), steps_records=[{"success": True}])]
    recovered, had_failed = _recovery_counts(outcomes)
    assert (recovered, had_failed) == (0, 0)


# ---------- _has_complete_evidence ----------

def test_has_complete_evidence_true_when_all_files_present(tmp_path):
    trace_dir = tmp_path / "run-1"
    trace_dir.mkdir()
    (trace_dir / "trace.jsonl").write_text("{}", encoding="utf-8")
    (trace_dir / "report.json").write_text("{}", encoding="utf-8")
    (trace_dir / "step-001.png").write_bytes(b"fake")

    outcome = _outcome(
        agent_result=_agent_result(trace_dir=str(trace_dir)),
        steps_records=[{"trace_schema_version": 2, "success": True}],
    )
    assert _has_complete_evidence(outcome) is True


def test_has_complete_evidence_false_when_old_trace_format(tmp_path):
    """DS-Y3 [Y3-2]: old format traces (no trace_schema_version) are incomplete evidence."""
    trace_dir = tmp_path / "run-old"
    trace_dir.mkdir()
    (trace_dir / "trace.jsonl").write_text("{}", encoding="utf-8")
    (trace_dir / "report.json").write_text("{}", encoding="utf-8")
    (trace_dir / "step-001.png").write_bytes(b"fake")

    outcome = _outcome(
        agent_result=_agent_result(trace_dir=str(trace_dir)),
        steps_records=[{"success": True}],  # no trace_schema_version
    )
    assert _has_complete_evidence(outcome) is False


def test_has_complete_evidence_false_when_missing_screenshot(tmp_path):
    trace_dir = tmp_path / "run-2"
    trace_dir.mkdir()
    (trace_dir / "trace.jsonl").write_text("{}", encoding="utf-8")
    (trace_dir / "report.json").write_text("{}", encoding="utf-8")

    outcome = _outcome(agent_result=_agent_result(trace_dir=str(trace_dir)))
    assert _has_complete_evidence(outcome) is False


def test_has_complete_evidence_false_when_no_agent_result():
    outcome = CaseOutcome(case=_case())
    assert _has_complete_evidence(outcome) is False


def test_has_complete_evidence_false_when_dir_missing(tmp_path):
    outcome = _outcome(agent_result=_agent_result(trace_dir=str(tmp_path / "nope")))
    assert _has_complete_evidence(outcome) is False


# ---------- compute_metrics / compute_raw_metrics ----------

def test_compute_metrics_basic_fractions_and_percentages(tmp_path):
    outcomes = [
        _outcome(
            agent_result=_agent_result(steps=2, trace_dir=str(tmp_path)),
            verify_success=True,
            steps_records=[{"success": True}, {"success": True}],
        ),
        _outcome(
            agent_result=_agent_result(steps=4, trace_dir=str(tmp_path)),
            verify_success=False,
            steps_records=[{"success": True}, {"success": False}],
        ),
    ]
    metrics = compute_metrics(outcomes)
    assert metrics["task_success_rate"] == "1/2"
    assert metrics["step_success_rate"] == "75%"
    assert metrics["avg_steps"] == "3.0"
    assert metrics["unsafe_action_block_rate"] == "0/0"


def test_compute_metrics_empty_outcomes_handles_zero_division():
    metrics = compute_metrics([])
    assert metrics["task_success_rate"] == "0/0"
    assert metrics["step_success_rate"] == "N/A"
    assert metrics["avg_steps"] == "N/A"


def test_compute_raw_metrics_matches_compute_metrics_counts():
    outcomes = [
        _outcome(agent_result=_agent_result(steps=2), verify_success=True, steps_records=[{"success": True}]),
        _outcome(agent_result=_agent_result(steps=4), verify_success=False, steps_records=[{"success": False}]),
    ]
    raw = compute_raw_metrics(outcomes)
    formatted = compute_metrics(outcomes)

    assert raw["success_count"] == 1
    assert raw["total"] == 2
    assert f"{raw['success_count']}/{raw['total']}" == formatted["task_success_rate"]
    assert raw["task_success_rate"] == pytest.approx(0.5)
    assert raw["avg_steps"] == pytest.approx(3.0)


def test_compute_raw_metrics_none_when_no_cases():
    raw = compute_raw_metrics([])
    assert raw["task_success_rate"] is None
    assert raw["avg_steps"] is None
    assert raw["step_success_rate"] is None
