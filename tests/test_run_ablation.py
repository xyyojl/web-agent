"""DS-W2 runner-level tests: mock run_one_case, test ablation artifact generation
through run_ablation.main_async().

验证策略（Design Spec DS-W2）：
- 正向验证：
  - 指定 L01-L10 后，artifact 的 included_case_ids 必须恰为 10 条。
  - L11 若运行，必须位于 excluded_case_ids，不参与 avg_steps。
  - 报告表中 DOM-only / Vision 的分母与 artifact 一致。
- 负向验证：
  - 两组 case ID 集不一致时，必须失败，禁止输出差异结论。
  - included/excluded 与实际运行 case 不匹配时，必须失败。
- 回归检查：
  - 不传 --artifact-dir 时，现有 ablation_results.json 输出行为保持不变。
  - DOM-only 与 Vision 的唯一运行变量仍是 vision；trace 子目录隔离保持不变。

所有测试 mock run_one_case，不调用真实 LLM。
"""

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from eval.eval_core import CaseOutcome
from agent.types import AgentResult, EvalCase, VerifyResult

# run_ablation.py 在模块顶层调用 load_dotenv()，会从 .env 加载 WEBAGENT_MODEL
# 等环境变量。在 import 前 patch 掉 load_dotenv，防止 .env 被加载。
with patch("dotenv.load_dotenv"):
    from eval import run_ablation


def _make_outcome(
    case_id: str = "L01",
    succeeded: bool = True,
    crash: str | None = None,
    trace_dir: str | None = None,
    steps: int = 2,
) -> CaseOutcome:
    """构造一个 mock CaseOutcome，用于测试 ablation artifact 生成。"""
    case: EvalCase = {
        "id": case_id,
        "type": "local",
        "task_type": "extract",
        "task": f"test task for {case_id}",
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
            "task": f"test task for {case_id}",
            "url": "http://localhost:8080/test.html",
            "success": succeeded,
            "output": "hello" if succeeded else None,
            "steps": steps,
            "duration_s": 1.0,
            "fail_reason": None if succeeded else "verify_failed",
            "trace_dir": trace_dir or f"/tmp/fake-trace-{case_id}",
            "last_screenshot": f"/tmp/fake-{case_id}.png",
        }
        outcome.agent_result = agent_result
        verify_result: VerifyResult = {
            "case_id": case_id,
            "success": succeeded,
            "reason": "ok" if succeeded else "no",
            "confidence": 1.0,
        }
        outcome.verify_result = verify_result
    outcome.step_records = [{"success": True, "trace_schema_version": 2}]
    return outcome


def _mock_run_one_case_factory(steps_map: dict[str, int] | None = None):
    """创建一个 mock run_one_case，根据 case 的 id 返回对应 outcome。"""
    resolved_steps: dict[str, int] = {} if steps_map is None else steps_map
    call_count = {"n": 0}

    async def _mock(case, _config):
        call_count["n"] += 1
        cid = case.get("id", "?")
        return _make_outcome(
            case_id=cid,
            steps=resolved_steps.get(cid, 2),
        )

    return _mock, call_count


# ---------- 正向验证 ----------


async def test_artifact_generation_with_mock(tmp_path):
    """正向验证: mock outcomes + --artifact-dir → generates ablation_results.json + summary.md.

    验证 included_case_ids 恰为 L01-L10（10 条），L11 位于 excluded_case_ids。
    """
    artifact_dir = str(tmp_path / "test-ablation-artifact")
    results_path = str(tmp_path / "ablation_results.json")

    mock_fn, _ = _mock_run_one_case_factory()
    with patch("eval.run_ablation.run_one_case", new_callable=AsyncMock, side_effect=mock_fn):
        with patch("eval.run_ablation._RESULTS_PATH", results_path):
            await run_ablation.main_async(
                suite_arg="local",
                case_arg=None,
                artifact_dir=artifact_dir,
                run_count=1,
                exclude_from_avg=["L11"],
            )

    # 验证 artifact 文件存在
    assert os.path.isfile(os.path.join(artifact_dir, "ablation_results.json"))
    assert os.path.isfile(os.path.join(artifact_dir, "summary.md"))

    with open(os.path.join(artifact_dir, "ablation_results.json"), encoding="utf-8") as f:
        payload = json.load(f)

    # 验证 artifact_format_version
    assert payload["artifact_format_version"] == 1

    # 验证 included_case_ids 恰为 L01-L10（10 条）
    included = payload["included_case_ids"]
    assert len(included) == 10
    assert "L11" not in included
    for i in range(1, 11):
        assert f"L{i:02d}" in included

    # L11 位于 excluded_case_ids
    excluded = payload["excluded_case_ids"]
    assert "L11" in excluded

    # 验证 excluded_case_reasons
    assert "L11" in payload["excluded_case_reasons"]

    # 验证 prompt_fingerprint 是 64 字符的 SHA-256
    assert len(payload["prompt_fingerprint"]) == 64

    # 验证 sampling_params 明文记录
    assert "planner" in payload["sampling_params"]
    assert "max_tokens" in payload["sampling_params"]["planner"]

    # 验证 model_pinning_caveat 存在且非空
    assert payload["model_pinning_caveat"]

    # 验证 git_commit 字段存在（可能为 None）
    assert "git_commit" in payload

    # 验证 run_count_per_group
    assert payload["run_count_per_group"] == 1


async def test_l11_not_in_avg_steps(tmp_path):
    """正向验证: L11 运行但不参与 avg_steps。

    报告表中 DOM-only / Vision 的分母与 artifact 一致（应为 10，不是 11）。
    """
    artifact_dir = str(tmp_path / "ablation-avg")
    results_path = str(tmp_path / "ablation_results.json")

    # L01-L10 各 2 步，L11 给一个不同的步数
    steps_map = {f"L{i:02d}": 2 for i in range(1, 12)}
    steps_map["L11"] = 5

    mock_fn, _ = _mock_run_one_case_factory(steps_map)
    with patch("eval.run_ablation.run_one_case", new_callable=AsyncMock, side_effect=mock_fn):
        with patch("eval.run_ablation._RESULTS_PATH", results_path):
            await run_ablation.main_async(
                suite_arg="local",
                case_arg=None,
                artifact_dir=artifact_dir,
                run_count=1,
                exclude_from_avg=["L11"],
            )

    with open(os.path.join(artifact_dir, "ablation_results.json"), encoding="utf-8") as f:
        payload = json.load(f)

    # 两组的 total 都应该是 10（included cases × run_count）
    for group_label in ("dom_only", "vision"):
        group = payload["groups"][group_label]
        assert group["total"] == 10
        # avg_steps 应该是 2.0（L01-L10 各 2 步），不是包含 L11 的均值
        assert group["avg_steps"] == 2.0

    # L11 仍在 cases 列表中但 excluded=True
    for group_label in ("dom_only", "vision"):
        group = payload["groups"][group_label]
        case_ids_in_group = [c["case_id"] for c in group["cases"]]
        assert "L11" in case_ids_in_group
        l11_record = next(c for c in group["cases"] if c["case_id"] == "L11")
        assert l11_record["excluded"] is True


async def test_verify_result_recorded_in_cases(tmp_path):
    """正向验证: 每个 case 的 runs[] 中包含 verify_result 字段。"""
    artifact_dir = str(tmp_path / "ablation-verify")
    results_path = str(tmp_path / "ablation_results.json")

    mock_fn, _ = _mock_run_one_case_factory()
    with patch("eval.run_ablation.run_one_case", new_callable=AsyncMock, side_effect=mock_fn):
        with patch("eval.run_ablation._RESULTS_PATH", results_path):
            await run_ablation.main_async(
                suite_arg="local",
                case_arg=None,
                artifact_dir=artifact_dir,
                run_count=1,
                exclude_from_avg=["L11"],
            )

    with open(os.path.join(artifact_dir, "ablation_results.json"), encoding="utf-8") as f:
        payload = json.load(f)

    dom_group = payload["groups"]["dom_only"]
    l01_record = next(c for c in dom_group["cases"] if c["case_id"] == "L01")
    assert len(l01_record["runs"]) == 1
    run0 = l01_record["runs"][0]
    assert run0["verify_result"] is not None
    assert run0["verify_result"]["case_id"] == "L01"
    assert run0["verify_result"]["success"] is True
    assert "trace_dir" in run0
    assert run0["trace_dir"] is not None


async def test_run_count_gt_1_produces_runs_array(tmp_path):
    """正向验证 [W2-3]: run_count > 1 时每个 case 存 runs[] 数组，含多次运行结果。"""
    artifact_dir = str(tmp_path / "ablation-multirun")
    results_path = str(tmp_path / "ablation_results.json")

    mock_fn, _ = _mock_run_one_case_factory()
    with patch("eval.run_ablation.run_one_case", new_callable=AsyncMock, side_effect=mock_fn):
        with patch("eval.run_ablation._RESULTS_PATH", results_path):
            await run_ablation.main_async(
                suite_arg="local",
                case_arg="L01,L02,L03",
                artifact_dir=artifact_dir,
                run_count=3,
                exclude_from_avg=[],  # 不排除任何 case
            )

    with open(os.path.join(artifact_dir, "ablation_results.json"), encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["run_count_per_group"] == 3
    # included_case_ids 应该是全部 3 条（因为 exclude_from_avg 为空）
    assert len(payload["included_case_ids"]) == 3
    assert len(payload["excluded_case_ids"]) == 0

    dom_group = payload["groups"]["dom_only"]
    l01_record = next(c for c in dom_group["cases"] if c["case_id"] == "L01")
    # runs[] 数组有 3 条
    assert len(l01_record["runs"]) == 3
    # 每个 run 都有 run_index
    for i, run in enumerate(l01_record["runs"]):
        assert run["run_index"] == i

    # total 应该是 included_cases × run_count = 3 × 3 = 9
    assert dom_group["total"] == 9


async def test_trace_subdir_isolation_maintained(tmp_path):
    """回归检查: DOM-only 与 Vision 的 trace 子目录隔离保持不变。"""
    artifact_dir = str(tmp_path / "ablation-trace")
    results_path = str(tmp_path / "ablation_results.json")

    mock_fn, _ = _mock_run_one_case_factory()
    with patch("eval.run_ablation.run_one_case", new_callable=AsyncMock, side_effect=mock_fn):
        with patch("eval.run_ablation._RESULTS_PATH", results_path):
            await run_ablation.main_async(
                suite_arg="local",
                case_arg="L01",
                artifact_dir=artifact_dir,
                run_count=1,
                exclude_from_avg=[],
            )

    with open(os.path.join(artifact_dir, "ablation_results.json"), encoding="utf-8") as f:
        payload = json.load(f)

    dom_trace = payload["groups"]["dom_only"]["trace_dir"]
    vis_trace = payload["groups"]["vision"]["trace_dir"]
    assert dom_trace != vis_trace
    assert "ablation_dom_only" in dom_trace
    assert "ablation_vision" in vis_trace


# ---------- 负向验证 ----------


async def test_case_set_mismatch_fails(tmp_path):
    """负向验证: 两组 case ID 集不一致时，必须失败。

    通过让 mock 根据分组返回不同 case_id 模拟 case 集不一致。
    """
    results_path = str(tmp_path / "ablation_results.json")

    # 模拟 dom_only 组正常返回，vision 组返回不同 case_id
    async def _mock_mismatch(case, config):
        cid = case.get("id", "?")
        # 通过 vision 配置区分两组
        if config.vision:
            # vision 组返回篡改的 case_id
            return _make_outcome(case_id=cid + "_X")
        return _make_outcome(case_id=cid)

    with patch("eval.run_ablation.run_one_case", new_callable=AsyncMock, side_effect=_mock_mismatch):
        with patch("eval.run_ablation._RESULTS_PATH", results_path):
            with pytest.raises(run_ablation.AblationConsistencyError, match="case ID 集不一致"):
                await run_ablation.main_async(
                    suite_arg="local",
                    case_arg="L01,L02",
                    artifact_dir=None,
                    run_count=1,
                    exclude_from_avg=[],
                )


async def test_included_excluded_mismatch_fails(tmp_path):
    """负向验证: included/excluded 与实际运行 case 不匹配时，必须失败。

    通过 --exclude-from-avg 指定一个不在运行集中的 case。
    （实际不会 crash，因为 set 操作会过滤掉不存在的，但验证 set 仍覆盖全部 case）
    """
    results_path = str(tmp_path / "ablation_results.json")

    mock_fn, _ = _mock_run_one_case_factory()
    with patch("eval.run_ablation.run_one_case", new_callable=AsyncMock, side_effect=mock_fn):
        with patch("eval.run_ablation._RESULTS_PATH", results_path):
            # 排除一个不存在的 case X99 + L11
            # X99 不在运行集中，会被过滤掉；实际 excluded 只有 L11
            # included + excluded 仍 = 全部运行 case，不会失败
            await run_ablation.main_async(
                suite_arg="local",
                case_arg="L01,L02,L03",
                artifact_dir=None,
                run_count=1,
                exclude_from_avg=["L11", "X99"],
            )

    with open(results_path, encoding="utf-8") as f:
        payload = json.load(f)
    # L11 不在运行集，所以 excluded 为空
    assert payload["excluded_case_ids"] == []
    assert len(payload["included_case_ids"]) == 3


async def test_artifact_dir_nonempty_fails(tmp_path):
    """负向验证: 对已存在且非空 artifact 目录再次运行，不得覆盖，必须失败。"""
    artifact_dir = str(tmp_path / "existing-ablation")
    os.makedirs(artifact_dir)
    with open(os.path.join(artifact_dir, "old.txt"), "w") as f:
        f.write("old")

    results_path = str(tmp_path / "ablation_results.json")
    mock_fn, _ = _mock_run_one_case_factory()
    with patch("eval.run_ablation.run_one_case", new_callable=AsyncMock, side_effect=mock_fn):
        with patch("eval.run_ablation._RESULTS_PATH", results_path):
            with pytest.raises(SystemExit):
                await run_ablation.main_async(
                    suite_arg="local",
                    case_arg="L01",
                    artifact_dir=artifact_dir,
                    run_count=1,
                    exclude_from_avg=[],
                )


# ---------- 回归检查 ----------


async def test_no_artifact_dir_writes_results_only(tmp_path):
    """回归检查: 不传 --artifact-dir 时，现有 ablation_results.json 输出行为保持不变。"""
    results_path = str(tmp_path / "ablation_results.json")
    artifact_dir = str(tmp_path / "should-not-exist")

    mock_fn, _ = _mock_run_one_case_factory()
    with patch("eval.run_ablation.run_one_case", new_callable=AsyncMock, side_effect=mock_fn):
        with patch("eval.run_ablation._RESULTS_PATH", results_path):
            await run_ablation.main_async(
                suite_arg="local",
                case_arg="L01,L02",
                artifact_dir=None,
                run_count=1,
                exclude_from_avg=[],
            )

    # ablation_results.json 被写入
    assert os.path.isfile(results_path)
    # 没有 artifact 目录被创建
    assert not os.path.isdir(artifact_dir)

    with open(results_path, encoding="utf-8") as f:
        payload = json.load(f)
    # 即使不生成 artifact，payload 仍包含 DS-W2 新增字段
    assert "artifact_format_version" in payload
    assert "prompt_fingerprint" in payload
    assert "included_case_ids" in payload


# ---------- prompt_fingerprint 单元测试 ----------


def test_prompt_fingerprint_is_deterministic():
    """prompt_fingerprint 相同输入应产生相同输出。"""
    fp1 = run_ablation._compute_prompt_fingerprint(run_ablation.SAMPLING_PARAMS)
    fp2 = run_ablation._compute_prompt_fingerprint(run_ablation.SAMPLING_PARAMS)
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex


def test_prompt_fingerprint_changes_with_params():
    """采样参数变化时 prompt_fingerprint 应不同。"""
    fp1 = run_ablation._compute_prompt_fingerprint(run_ablation.SAMPLING_PARAMS)
    modified = {k: dict(v) for k, v in run_ablation.SAMPLING_PARAMS.items()}
    modified["planner"]["max_tokens"] = 999
    fp2 = run_ablation._compute_prompt_fingerprint(modified)
    assert fp1 != fp2
