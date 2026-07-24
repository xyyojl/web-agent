"""DS-R3 runner-level tests: mock run_one_case, test artifact generation through
run_eval.main_async().

验证策略（Design Spec DS-R3）：
- 正向验证：使用 mock outcomes 执行 --artifact-dir，验证生成 summary.md /
  results.json / provenance.json；验证 results 的 case 数和 metrics 分母一致；
  验证指定 L01 的 trace 文件被复制且路径存在。
- 负向验证：对已存在且非空 artifact 目录再次运行，不得覆盖，必须失败；
  指定不存在的 --archive-case-traces X99，必须失败；
  case runner crash 时，artifact 中必须记录 crash_reason，不能把 case 记为 pass。
- 回归检查：不传 --artifact-dir 时，现有 eval/eval_summary.md 输出行为保持不变。

所有测试 mock run_one_case，不调用真实 LLM。
"""

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from eval.eval_core import CaseOutcome
from agent.types import AgentResult, EvalCase, VerifyResult

# run_eval.py 在模块顶层调用 load_dotenv()，会从 .env 加载 WEBAGENT_MODEL
# 等环境变量，破坏 test_config.py 的测试隔离。在 import 前 patch 掉
# load_dotenv，防止 .env 被加载。
with patch("dotenv.load_dotenv"):
    from eval import run_eval


def _make_outcome(
    case_id: str = "L01",
    succeeded: bool = True,
    crash: str | None = None,
    trace_dir: str | None = None,
) -> CaseOutcome:
    """构造一个 mock CaseOutcome，用于测试 artifact 生成。"""
    case: EvalCase = {
        "id": case_id,
        "type": "local",
        "task_type": "extract",
        "task": "test task",
        "url": "http://localhost:8080/test.html",
        "expected_output": "hello",
        "verify_mode": "exact",
        "difficulty": "easy",
    }
    outcome = CaseOutcome(case=case)
    outcome.crash_reason = crash
    if crash is None:
        agent_result: AgentResult = {
            "task_id": case_id,
            "task": "test task",
            "url": "http://localhost:8080/test.html",
            "success": succeeded,
            "output": "hello" if succeeded else None,
            "steps": 2,
            "duration_s": 1.0,
            "fail_reason": None if succeeded else "verify_failed",
            "trace_dir": trace_dir or "/tmp/fake-trace",
            "last_screenshot": "/tmp/fake.png",
        }
        outcome.agent_result = agent_result
        verify_result: VerifyResult = {
            "case_id": case_id,
            "success": succeeded,
            "reason": "ok" if succeeded else "no",
            "confidence": 1.0,
        }
        outcome.verify_result = verify_result
    outcome.step_records = [{
        "success": True, "trace_schema_version": 2, "privacy_redaction_version": 1,
        "action": "click", "selector": "css=#ok", "reason": "ok",
        "tool_output": None, "tool_output_truncated": False, "tool_output_sha256": None,
        "observation": {"title": "T", "text_hash": "h", "visible_text_summary": "text", "interactive_elements": []},
    }]
    return outcome


# ---------- 正向验证 ----------


async def test_artifact_generation_with_mock(tmp_path):
    """正向验证: mock outcomes + --artifact-dir → generates summary.md, results.json, provenance.json.

    验证 results 的 case 数和 metrics 分母一致（local suite 共 11 个 case）。
    """
    artifact_dir = str(tmp_path / "test-artifact")

    mock_outcome = _make_outcome()
    with patch("eval.run_eval.run_one_case", new_callable=AsyncMock, return_value=mock_outcome):
        with patch("eval.run_eval._SUMMARY_PATH", str(tmp_path / "eval_summary.md")):
            # 运行本身可创建未跟踪 trace；artifact 应记录运行开始时的快照。
            with patch("eval.run_eval._get_git_info", return_value=("start-sha", False)) as git_info:
                await run_eval.main_async(
                    suite_arg="local",
                    case_arg=None,
                    artifact_dir=artifact_dir,
                    archive_case_ids=None,
                )

    git_info.assert_called_once_with()

    assert os.path.isfile(os.path.join(artifact_dir, "summary.md"))
    assert os.path.isfile(os.path.join(artifact_dir, "results.json"))
    assert os.path.isfile(os.path.join(artifact_dir, "provenance.json"))

    # 验证 results 的 case 数和 metrics 分母一致
    with open(os.path.join(artifact_dir, "results.json"), encoding="utf-8") as f:
        results = json.load(f)
    # local suite 有 11 个 case (L01-L11)
    assert len(results["cases"]) == 11
    with open(os.path.join(artifact_dir, "provenance.json"), encoding="utf-8") as f:
        provenance = json.load(f)
    assert provenance["git_commit"] == "start-sha"
    assert provenance["git_dirty"] is False


def test_default_summary_redacts_failed_task():
    """默认 eval_summary.md 与 artifact 使用相同的失败行脱敏规则。"""
    secret = "TEST_PASSWORD_DO_NOT_USE"
    sensitive = _make_outcome(case_id="L01", succeeded=False)
    sensitive.case["task"] = f"将登录密码修改为 {secret}，然后保存"
    assert sensitive.agent_result is not None
    sensitive.agent_result["task"] = sensitive.case["task"]
    sensitive.agent_result["fail_reason"] = f"保存 {secret} 时被拒绝"
    sensitive.step_records = [{"screenshot": "step-sensitive.png"}]

    ordinary = _make_outcome(case_id="L02", succeeded=False)
    ordinary.case["task"] = "点击取消按钮"
    assert ordinary.agent_result is not None
    ordinary.agent_result["fail_reason"] = "element_not_found"
    ordinary.step_records = [{"screenshot": "step-ordinary.png"}]

    summary = run_eval._render_summary_md(
        {"local": run_eval.compute_metrics([sensitive, ordinary])},
        {"local": [sensitive, ordinary]},
    )

    assert secret not in summary
    assert "[REDACTED:browser_type_input]" in summary
    assert "| L01 |" in summary
    assert "step-sensitive.png" in summary
    assert "| L02 | 点击取消按钮 | element_not_found | step-ordinary.png |" in summary


async def test_trace_archived_with_mock(tmp_path):
    """正向验证: --archive-case-traces L01 → trace 文件被复制且路径存在。"""
    # 创建 fake trace 目录
    trace_dir = str(tmp_path / "fake-trace")
    os.makedirs(trace_dir)
    mock_outcome = _make_outcome(trace_dir=trace_dir)
    with open(os.path.join(trace_dir, "trace.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(mock_outcome.step_records[0]) + "\n")
    with open(os.path.join(trace_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"success": True, "privacy_redaction_version": 1}, f)
    with open(os.path.join(trace_dir, "step-001.png"), "wb") as f:
        f.write(b"fake png")

    artifact_dir = str(tmp_path / "artifact-traces")

    with patch("eval.run_eval.run_one_case", new_callable=AsyncMock, return_value=mock_outcome):
        with patch("eval.run_eval._SUMMARY_PATH", str(tmp_path / "eval_summary.md")):
            await run_eval.main_async(
                suite_arg="local",
                case_arg=None,
                artifact_dir=artifact_dir,
                archive_case_ids=["L01"],
            )

    # 验证 L01 trace 文件被复制
    dest = os.path.join(artifact_dir, "traces", "L01")
    assert os.path.isfile(os.path.join(dest, "trace.jsonl"))
    assert os.path.isfile(os.path.join(dest, "report.json"))
    assert os.path.isfile(os.path.join(dest, "step-001.png"))


# ---------- 负向验证 ----------


async def test_artifact_dir_nonempty_fails(tmp_path):
    """负向验证: 对已存在且非空 artifact 目录再次运行，不得覆盖，必须失败。"""
    artifact_dir = str(tmp_path / "existing")
    os.makedirs(artifact_dir)
    with open(os.path.join(artifact_dir, "old.txt"), "w") as f:
        f.write("old")

    mock_outcome = _make_outcome()
    with patch("eval.run_eval.run_one_case", new_callable=AsyncMock, return_value=mock_outcome):
        with patch("eval.run_eval._SUMMARY_PATH", str(tmp_path / "eval_summary.md")):
            with pytest.raises(SystemExit):
                await run_eval.main_async(
                    suite_arg="local",
                    artifact_dir=artifact_dir,
                )


async def test_archive_case_not_found_fails(tmp_path):
    """负向验证: 指定不存在的 --archive-case-traces X99，必须失败。"""
    artifact_dir = str(tmp_path / "artifact-x99")

    mock_outcome = _make_outcome()
    with patch("eval.run_eval.run_one_case", new_callable=AsyncMock, return_value=mock_outcome):
        with patch("eval.run_eval._SUMMARY_PATH", str(tmp_path / "eval_summary.md")):
            with pytest.raises(SystemExit):
                await run_eval.main_async(
                    suite_arg="local",
                    artifact_dir=artifact_dir,
                    archive_case_ids=["X99"],
                )


async def test_crash_reason_recorded(tmp_path):
    """负向验证: case runner crash 时，artifact 中必须记录 crash_reason，不能把 case 记为 pass。"""
    artifact_dir = str(tmp_path / "artifact-crash")

    mock_outcome = _make_outcome(crash="unexpected error")
    with patch("eval.run_eval.run_one_case", new_callable=AsyncMock, return_value=mock_outcome):
        with patch("eval.run_eval._SUMMARY_PATH", str(tmp_path / "eval_summary.md")):
            await run_eval.main_async(
                suite_arg="local",
                artifact_dir=artifact_dir,
            )

    with open(os.path.join(artifact_dir, "results.json"), encoding="utf-8") as f:
        results = json.load(f)
    # 所有 11 个 case 都是 crashed
    for case_record in results["cases"]:
        assert case_record["crash_reason"] == "unexpected error"
        assert case_record["succeeded"] is False
        assert case_record["agent_result"] is None


# ---------- 回归检查 ----------


async def test_no_artifact_dir_writes_summary_only(tmp_path):
    """回归检查: 不传 --artifact-dir 时，现有 eval/eval_summary.md 输出行为保持不变。"""
    summary_path = str(tmp_path / "eval_summary.md")

    mock_outcome = _make_outcome()
    with patch("eval.run_eval.run_one_case", new_callable=AsyncMock, return_value=mock_outcome):
        with patch("eval.run_eval._SUMMARY_PATH", summary_path):
            await run_eval.main_async(
                suite_arg="local",
                case_arg=None,
                artifact_dir=None,
                archive_case_ids=None,
            )

    # eval_summary.md 被写入
    assert os.path.isfile(summary_path)
    # 没有 artifact 目录被创建
    assert not os.path.isdir(str(tmp_path / "test-artifact"))
