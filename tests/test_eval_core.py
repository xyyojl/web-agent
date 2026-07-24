"""eval/eval_core.py 单元测试：覆盖 6 项指标的计算口径（尤其是
safety_block case 从 recovery_rate 分子分母中排除的过滤逻辑）、
case 加载的容错行为，以及 CaseOutcome 的派生属性。
"""

import json
import os

import pytest

from agent.types import AgentResult, EvalCase, VerifyResult
from agent.config import AgentConfig
from eval.eval_core import (
    ArtifactError,
    CaseOutcome,
    _avg_steps_value,
    _build_case_record,
    _has_complete_evidence,
    _recovery_counts,
    _step_success_counts,
    _task_success_counts,
    build_provenance,
    build_results_json,
    compute_metrics,
    compute_raw_metrics,
    load_cases,
    render_artifact_summary,
    write_artifact,
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


def _complete_v2_record():
    return {
        "trace_schema_version": 2, "privacy_redaction_version": 1,
        "action": "click", "selector": "css=#ok", "reason": "submit",
        "tool_output": None, "tool_output_truncated": False, "tool_output_sha256": None,
        "observation": {"title": "T", "text_hash": "h", "visible_text_summary": "text", "interactive_elements": []},
        "success": True,
    }


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
    (trace_dir / "report.json").write_text('{"privacy_redaction_version": 1}', encoding="utf-8")
    (trace_dir / "step-001.png").write_bytes(b"fake")

    outcome = _outcome(
        agent_result=_agent_result(trace_dir=str(trace_dir)),
        steps_records=[_complete_v2_record()],
    )
    assert _has_complete_evidence(outcome) is True


def test_has_complete_evidence_false_when_old_trace_format(tmp_path):
    """DS-Y3 [Y3-2]: old format traces (no trace_schema_version) are incomplete evidence."""
    trace_dir = tmp_path / "run-old"
    trace_dir.mkdir()
    (trace_dir / "trace.jsonl").write_text("{}", encoding="utf-8")
    (trace_dir / "report.json").write_text('{"privacy_redaction_version": 1}', encoding="utf-8")
    (trace_dir / "step-001.png").write_bytes(b"fake")

    outcome = _outcome(
        agent_result=_agent_result(trace_dir=str(trace_dir)),
        steps_records=[{"success": True}],  # no trace_schema_version
    )
    assert _has_complete_evidence(outcome) is False


def test_has_complete_evidence_false_for_v2_missing_required_fields(tmp_path):
    trace_dir = tmp_path / "run-incomplete-v2"
    trace_dir.mkdir()
    (trace_dir / "trace.jsonl").write_text("{}", encoding="utf-8")
    (trace_dir / "report.json").write_text('{"privacy_redaction_version": 1}', encoding="utf-8")
    (trace_dir / "step-001.png").write_bytes(b"fake")
    fake_v2 = {"trace_schema_version": 2, "privacy_redaction_version": 1, "observation": {"title": "T"}}
    outcome = _outcome(agent_result=_agent_result(trace_dir=str(trace_dir)), steps_records=[fake_v2])
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


# ---------- DS-R3: Artifact 序列化 ----------


def test_build_case_record_contains_all_required_fields():
    """DS-R3: results.json per case must have case_id/suite/succeeded/agent_result/
    verify_result/crash_reason/last_screenshot/trace_dir."""
    outcome = _outcome(agent_result=_agent_result())
    record = _build_case_record(outcome, "local")
    assert record["case_id"] == "L01"
    assert record["suite"] == "local"
    assert record["succeeded"] is True
    assert record["agent_result"] is not None
    assert record["verify_result"] is not None
    assert record["crash_reason"] is None
    assert "last_screenshot" in record
    assert record["trace_dir"] is not None


def test_build_case_record_crash_case_not_pass():
    """DS-R3 负向: crashed case must have succeeded=False, crash_reason set, agent_result=None."""
    outcome = _outcome(crash="KeyError: url")
    record = _build_case_record(outcome, "local")
    assert record["succeeded"] is False
    assert record["crash_reason"] == "KeyError: url"
    assert record["agent_result"] is None
    assert record["verify_result"] is None


def test_build_results_json_case_count_matches_outcomes():
    """DS-R3: results.json case count must match outcomes count."""
    all_outcomes = {
        "local": [_outcome(agent_result=_agent_result()), _outcome(agent_result=_agent_result())],
        "public": [_outcome(agent_result=_agent_result())],
    }
    results = build_results_json(all_outcomes)
    assert len(results["cases"]) == 3
    assert results["cases"][0]["suite"] == "local"
    assert results["cases"][2]["suite"] == "public"


def test_build_results_json_redacts_sensitive_task_values():
    secret = "TEST_PASSWORD_DO_NOT_USE"
    outcome = _outcome(agent_result=_agent_result(task=f"将密码修改为 {secret}"))
    results = build_results_json({"local": [outcome]})
    assert secret not in json.dumps(results, ensure_ascii=False)


def test_build_provenance_contains_all_required_fields():
    """DS-R3: provenance.json must have all required fields including git_dirty [R3-2]."""
    config = AgentConfig()
    provenance = build_provenance(
        config=config,
        suite_arg="all",
        case_arg=None,
        case_ids=["L01", "L02"],
        git_commit="abc123",
        git_dirty=True,
    )
    required = {
        "generated_at", "suite_argument", "case_argument", "case_ids",
        "model", "vision", "max_steps", "max_fail", "git_commit",
        "git_dirty", "artifact_format_version",
    }
    assert required.issubset(provenance.keys())
    assert provenance["git_dirty"] is True
    assert provenance["artifact_format_version"] == 1


def test_render_artifact_summary_includes_baselines_section():
    """DS-R3 [R3-3]: summary.md must include 基准有效性 section."""
    config = AgentConfig()
    provenance = build_provenance(config, "all", None, ["L01"], "abc", False)
    outcome = _outcome(agent_result=_agent_result())
    suite_metrics = {"local": compute_metrics([outcome])}
    all_outcomes = {"local": [outcome]}
    summary = render_artifact_summary(suite_metrics, all_outcomes, provenance)
    assert "基准有效性" in summary
    assert "generated_at" in summary
    assert "网站变化后本 artifact 不再代表当前行为" in summary


def test_render_artifact_summary_git_dirty_warning():
    """DS-R3 [R3-2]: summary.md must show git_dirty warning when dirty."""
    config = AgentConfig()
    provenance = build_provenance(config, "all", None, ["L01"], "abc", True)
    outcome = _outcome(agent_result=_agent_result())
    suite_metrics = {"local": compute_metrics([outcome])}
    all_outcomes = {"local": [outcome]}
    summary = render_artifact_summary(suite_metrics, all_outcomes, provenance)
    assert "工作区状态提示" in summary
    assert "未提交改动" in summary


def test_render_artifact_summary_no_warning_when_clean():
    """DS-R3 [R3-2]: summary.md must NOT show git_dirty warning when clean."""
    config = AgentConfig()
    provenance = build_provenance(config, "all", None, ["L01"], "abc", False)
    outcome = _outcome(agent_result=_agent_result())
    suite_metrics = {"local": compute_metrics([outcome])}
    all_outcomes = {"local": [outcome]}
    summary = render_artifact_summary(suite_metrics, all_outcomes, provenance)
    assert "工作区状态提示" not in summary


def test_artifact_summary_redacts_failed_task(tmp_path):
    """失败摘要需脱敏任务和失败原因，同时保留普通失败证据。"""
    secret = "TEST_PASSWORD_DO_NOT_USE"
    sensitive = _outcome(
        case=_case(id="L01", task=f"将登录密码修改为 {secret}，然后保存"),
        agent_result=_agent_result(
            task=f"将登录密码修改为 {secret}，然后保存",
            success=False,
            fail_reason=f"保存 {secret} 时被拒绝",
        ),
        verify_success=False,
        steps_records=[{"screenshot": "step-sensitive.png"}],
    )
    ordinary = _outcome(
        case=_case(id="L02", task="点击取消按钮"),
        agent_result=_agent_result(success=False, fail_reason="element_not_found"),
        verify_success=False,
        steps_records=[{"screenshot": "step-ordinary.png"}],
    )
    all_outcomes = {"local": [sensitive, ordinary]}
    artifact_dir = str(tmp_path / "redacted-artifact")

    write_artifact(
        artifact_dir, all_outcomes, {"local": compute_metrics([sensitive, ordinary])},
        AgentConfig(), "local", None, None,
    )

    summary = open(os.path.join(artifact_dir, "summary.md"), encoding="utf-8").read()
    results = json.loads(open(os.path.join(artifact_dir, "results.json"), encoding="utf-8").read())
    assert secret not in summary
    assert secret not in json.dumps(results, ensure_ascii=False)
    assert "[REDACTED:browser_type_input]" in summary
    assert "| L01 |" in summary
    assert "step-sensitive.png" in summary
    assert "| L02 | 点击取消按钮 | element_not_found | step-ordinary.png |" in summary
    assert results["cases"][0]["succeeded"] is False


def test_write_artifact_generates_all_files(tmp_path):
    """DS-R3 正向: write_artifact must generate summary.md, results.json, provenance.json."""
    artifact_dir = str(tmp_path / "test-artifact")
    config = AgentConfig()
    outcome = _outcome(agent_result=_agent_result())
    all_outcomes = {"local": [outcome]}
    suite_metrics = {"local": compute_metrics([outcome])}

    write_artifact(artifact_dir, all_outcomes, suite_metrics, config, "local", None, None)

    assert os.path.isfile(os.path.join(artifact_dir, "summary.md"))
    assert os.path.isfile(os.path.join(artifact_dir, "results.json"))
    assert os.path.isfile(os.path.join(artifact_dir, "provenance.json"))

    # Verify results case count matches metrics denominator (1 case)
    with open(os.path.join(artifact_dir, "results.json"), encoding="utf-8") as f:
        results = json.load(f)
    assert len(results["cases"]) == 1


def test_write_artifact_nonempty_dir_fails(tmp_path):
    """DS-R3 负向: artifact dir already exists and non-empty → ArtifactError."""
    artifact_dir = str(tmp_path / "existing-artifact")
    os.makedirs(artifact_dir)
    with open(os.path.join(artifact_dir, "old.txt"), "w") as f:
        f.write("old")

    config = AgentConfig()
    outcome = _outcome(agent_result=_agent_result())
    all_outcomes = {"local": [outcome]}
    suite_metrics = {"local": compute_metrics([outcome])}

    with pytest.raises(ArtifactError, match="已存在且非空"):
        write_artifact(artifact_dir, all_outcomes, suite_metrics, config, "local", None, None)


def test_write_artifact_gitkeep_only_dir_succeeds(tmp_path):
    """DS-R3: artifact dir exists but only has .gitkeep → should succeed."""
    artifact_dir = str(tmp_path / "gitkeep-only")
    os.makedirs(artifact_dir)
    with open(os.path.join(artifact_dir, ".gitkeep"), "w") as f:
        f.write("")

    config = AgentConfig()
    outcome = _outcome(agent_result=_agent_result())
    all_outcomes = {"local": [outcome]}
    suite_metrics = {"local": compute_metrics([outcome])}

    write_artifact(artifact_dir, all_outcomes, suite_metrics, config, "local", None, None)
    assert os.path.isfile(os.path.join(artifact_dir, "results.json"))


def test_write_artifact_archive_traces_copies_files(tmp_path):
    """DS-R3 正向: --archive-case-traces copies trace.jsonl, report.json, screenshots."""
    # Create a fake trace dir
    trace_dir = str(tmp_path / "fake-trace")
    os.makedirs(trace_dir)
    with open(os.path.join(trace_dir, "trace.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(_complete_v2_record()) + "\n")
    with open(os.path.join(trace_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"success": True, "privacy_redaction_version": 1}, f)
    with open(os.path.join(trace_dir, "step-001.png"), "wb") as f:
        f.write(b"fake png")

    artifact_dir = str(tmp_path / "artifact-with-traces")
    config = AgentConfig()
    outcome = _outcome(agent_result=_agent_result(trace_dir=trace_dir), steps_records=[_complete_v2_record()])
    all_outcomes = {"local": [outcome]}
    suite_metrics = {"local": compute_metrics([outcome])}

    write_artifact(artifact_dir, all_outcomes, suite_metrics, config, "local", None, ["L01"])

    # Verify trace files copied
    dest_trace = os.path.join(artifact_dir, "traces", "L01")
    assert os.path.isfile(os.path.join(dest_trace, "trace.jsonl"))
    assert os.path.isfile(os.path.join(dest_trace, "report.json"))
    assert os.path.isfile(os.path.join(dest_trace, "step-001.png"))


def test_write_artifact_archive_case_not_found_fails(tmp_path):
    """DS-R3 负向: --archive-case-traces with non-existent case → ArtifactError."""
    artifact_dir = str(tmp_path / "artifact-missing-case")
    config = AgentConfig()
    outcome = _outcome(agent_result=_agent_result())
    all_outcomes = {"local": [outcome]}
    suite_metrics = {"local": compute_metrics([outcome])}

    with pytest.raises(ArtifactError, match="不存在"):
        write_artifact(artifact_dir, all_outcomes, suite_metrics, config, "local", None, ["X99"])


def test_write_artifact_archive_trace_dir_missing_fails(tmp_path):
    """DS-R3 负向: --archive-case-traces with missing trace dir → ArtifactError."""
    artifact_dir = str(tmp_path / "artifact-missing-trace")
    config = AgentConfig()
    outcome = _outcome(agent_result=_agent_result(trace_dir="/nonexistent/path"))
    all_outcomes = {"local": [outcome]}
    suite_metrics = {"local": compute_metrics([outcome])}

    with pytest.raises(ArtifactError, match="trace 目录不存在"):
        write_artifact(artifact_dir, all_outcomes, suite_metrics, config, "local", None, ["L01"])
    assert not os.path.exists(artifact_dir)


def test_write_artifact_archive_preflight_failure_leaves_no_artifact(tmp_path):
    """归档 trace 伪装为 v2 但缺字段时，最终目录不能留下半成品。"""
    trace_dir = tmp_path / "bad-trace"
    trace_dir.mkdir()
    (trace_dir / "trace.jsonl").write_text('{"trace_schema_version": 2}\n', encoding="utf-8")
    (trace_dir / "report.json").write_text('{"privacy_redaction_version": 1}', encoding="utf-8")
    (trace_dir / "step-001.png").write_bytes(b"fake")
    outcome = _outcome(agent_result=_agent_result(trace_dir=str(trace_dir)), steps_records=[{"trace_schema_version": 2}])
    artifact_dir = str(tmp_path / "must-not-exist")
    with pytest.raises(ArtifactError, match="完整性/隐私契约"):
        write_artifact(artifact_dir, {"local": [outcome]}, {"local": compute_metrics([outcome])}, AgentConfig(), "local", None, ["L01"])
    assert not os.path.exists(artifact_dir)


def test_write_artifact_crash_reason_recorded(tmp_path):
    """DS-R3 负向: crashed case must have crash_reason in results, not marked as pass."""
    artifact_dir = str(tmp_path / "artifact-crash")
    config = AgentConfig()
    outcome = _outcome(crash="runner exploded")
    all_outcomes = {"local": [outcome]}
    suite_metrics = {"local": compute_metrics([outcome])}

    write_artifact(artifact_dir, all_outcomes, suite_metrics, config, "local", None, None)

    with open(os.path.join(artifact_dir, "results.json"), encoding="utf-8") as f:
        results = json.load(f)
    assert results["cases"][0]["crash_reason"] == "runner exploded"
    assert results["cases"][0]["succeeded"] is False
    assert results["cases"][0]["agent_result"] is None
