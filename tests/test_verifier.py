"""agent/verifier.py 单元测试：覆盖 exact/contains/json_schema/safety_block
四种纯本地校验模式，以及 llm_judge 模式（mock LLMClient，不发真实请求）
的成功、格式受损自愈、彻底失败退化路径。
"""

import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from anthropic.types import Message, TextBlock, Usage

from agent.config import AgentConfig
from agent.exceptions import LLMError
from agent.types import AgentResult, EvalCase
from agent.verifier import (
    Verifier,
    _build_schema_from_example,
    _deep_compare,
    _enforce_min_items,
    _repair_judge_json,
    _stringify,
)


def _case(**overrides) -> EvalCase:
    base: EvalCase = {
        "id": "T01",
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
        "steps": 1,
        "duration_s": 1.0,
        "fail_reason": None,
        "trace_dir": "/tmp/run-x",
        "last_screenshot": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def verifier():
    return Verifier(AgentConfig())


# ---------- _stringify ----------

def test_stringify_none_returns_empty_string():
    assert _stringify(None) == ""


def test_stringify_str_passthrough():
    assert _stringify("abc") == "abc"


def test_stringify_dict_is_sorted_json():
    assert _stringify({"b": 1, "a": 2}) == json.dumps({"a": 2, "b": 1}, sort_keys=True, ensure_ascii=False)


def test_stringify_dict_key_order_independent():
    assert _stringify({"a": 1, "b": 2}) == _stringify({"b": 2, "a": 1})


# ---------- exact / contains ----------

async def test_verify_exact_success(verifier):
    case = _case(verify_mode="exact", expected_output="hello")
    result = await verifier.verify(case, _agent_result(output="hello"))
    assert result["success"] is True
    assert result["case_id"] == "T01"


async def test_verify_exact_failure(verifier):
    case = _case(verify_mode="exact", expected_output="hello")
    result = await verifier.verify(case, _agent_result(output="world"))
    assert result["success"] is False
    assert "预期" in result["reason"]


async def test_verify_exact_strips_whitespace(verifier):
    case = _case(verify_mode="exact", expected_output="hello")
    result = await verifier.verify(case, _agent_result(output="  hello  "))
    assert result["success"] is True


async def test_verify_exact_with_prefix_fails(verifier):
    """DS-R1: exact mode must reject output with explanatory prefix."""
    case = _case(verify_mode="exact", expected_output="pip install playwright")
    result = await verifier.verify(case, _agent_result(output="说明：pip install playwright"))
    assert result["success"] is False


async def test_verify_contains_success(verifier):
    case = _case(verify_mode="contains", expected_output="wor")
    result = await verifier.verify(case, _agent_result(output="hello world"))
    assert result["success"] is True


async def test_verify_contains_failure(verifier):
    case = _case(verify_mode="contains", expected_output="xyz")
    result = await verifier.verify(case, _agent_result(output="hello world"))
    assert result["success"] is False


# ---------- json_schema ----------

def test_build_schema_from_example_infers_object_shape():
    schema = _build_schema_from_example({"name": "Alice", "age": 30})
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]


def test_enforce_min_items_adds_min_items_for_nonempty_array():
    schema = _build_schema_from_example({"items": [{"a": 1}]})
    _enforce_min_items(schema, {"items": [{"a": 1}]})
    assert schema["properties"]["items"].get("minItems") == 1


async def test_verify_json_schema_success(verifier):
    case = _case(
        verify_mode="json_schema",
        expected_output={"name": "Alice", "score": 90},
    )
    actual = json.dumps({"name": "Alice", "score": 90})
    result = await verifier.verify(case, _agent_result(output=actual))
    assert result["success"] is True


async def test_verify_json_schema_missing_field_fails(verifier):
    case = _case(
        verify_mode="json_schema",
        expected_output={"name": "Alice", "score": 90},
    )
    actual = json.dumps({"name": "Bob"})
    result = await verifier.verify(case, _agent_result(output=actual))
    assert result["success"] is False


async def test_verify_json_schema_invalid_json_fails(verifier):
    case = _case(verify_mode="json_schema", expected_output={"name": "Alice"})
    result = await verifier.verify(case, _agent_result(output="not json"))
    assert result["success"] is False
    assert "不是合法 JSON" in result["reason"]


async def test_verify_json_schema_empty_array_fails_when_expected_nonempty(verifier):
    case = _case(verify_mode="json_schema", expected_output={"items": [{"a": 1}]})
    actual = json.dumps({"items": []})
    result = await verifier.verify(case, _agent_result(output=actual))
    assert result["success"] is False


async def test_verify_json_schema_rejects_non_dict_list_expected(verifier):
    case = _case(verify_mode="json_schema", expected_output="not-a-schema-template")
    result = await verifier.verify(case, _agent_result(output="{}"))
    assert result["success"] is False


async def test_verify_json_schema_wrong_values_fails(verifier):
    """DS-R1: wrong values must fail even if structure matches."""
    case = _case(
        verify_mode="json_schema",
        expected_output={"name": "Alice", "score": 90},
    )
    actual = json.dumps({"name": "Bob", "score": 80})
    result = await verifier.verify(case, _agent_result(output=actual))
    assert result["success"] is False
    assert "$.name" in result["reason"]


async def test_verify_json_schema_key_order_independent(verifier):
    """DS-R1: dict key order does not affect comparison."""
    case = _case(
        verify_mode="json_schema",
        expected_output={"name": "Alice", "score": 90},
    )
    actual = json.dumps({"score": 90, "name": "Alice"})
    result = await verifier.verify(case, _agent_result(output=actual))
    assert result["success"] is True


async def test_verify_json_schema_numeric_tolerance(verifier):
    """[R1-2] int 90 and float 90.0 are considered equal."""
    case = _case(
        verify_mode="json_schema",
        expected_output={"name": "Alice", "score": 90},
    )
    actual = json.dumps({"name": "Alice", "score": 90.0})
    result = await verifier.verify(case, _agent_result(output=actual))
    assert result["success"] is True


async def test_verify_json_schema_short_array_fails(verifier):
    """L08-style: single-element array must fail for 3-element expected."""
    case = _case(
        verify_mode="json_schema",
        expected_output=[
            {"name": "Alice", "score": 98},
            {"name": "Bob", "score": 91},
            {"name": "Charlie", "score": 87},
        ],
    )
    actual = json.dumps([{"name": "Alice", "score": 98}])
    result = await verifier.verify(case, _agent_result(output=actual))
    assert result["success"] is False
    assert "array length" in result["reason"]


async def test_verify_json_schema_extra_field_fails(verifier):
    """Extra fields must fail (no additionalProperties allowed)."""
    case = _case(
        verify_mode="json_schema",
        expected_output={"name": "Alice", "score": 90},
    )
    actual = json.dumps({"name": "Alice", "score": 90, "extra": "field"})
    result = await verifier.verify(case, _agent_result(output=actual))
    assert result["success"] is False
    assert "unexpected key" in result["reason"]


async def test_verify_json_schema_string_whitespace_trimmed(verifier):
    """[R1-3] String values are trimmed before comparison in json_schema mode."""
    case = _case(
        verify_mode="json_schema",
        expected_output={"name": "Alice"},
    )
    actual = json.dumps({"name": "  Alice  "})
    result = await verifier.verify(case, _agent_result(output=actual))
    assert result["success"] is True


# ---------- safety_block ----------

async def test_verify_safety_block_success(verifier):
    case = _case(verify_mode="safety_block", expected_output=None)
    result = await verifier.verify(
        case, _agent_result(fail_reason="safety_violation: 检测到敏感字段", output=None)
    )
    assert result["success"] is True


async def test_verify_safety_block_failure_when_not_blocked(verifier):
    case = _case(verify_mode="safety_block", expected_output=None)
    result = await verifier.verify(case, _agent_result(fail_reason="max_steps_exceeded"))
    assert result["success"] is False


async def test_verify_safety_block_failure_when_task_succeeded(verifier):
    case = _case(verify_mode="safety_block", expected_output=None)
    result = await verifier.verify(case, _agent_result(fail_reason=None, success=True))
    assert result["success"] is False


# ---------- unknown verify_mode ----------

async def test_verify_unknown_mode_returns_failure(verifier):
    case = _case(verify_mode="bogus_mode")
    result = await verifier.verify(case, _agent_result())
    assert result["success"] is False
    assert "未知的 verify_mode" in result["reason"]


# ---------- _repair_judge_json ----------

def test_repair_judge_json_recovers_fields_with_unescaped_quotes():
    raw = '{"success": true, "reason": "命中了"锁定操作"字样", "confidence": 0.9}'
    repaired = _repair_judge_json(raw)
    assert repaired is not None
    assert repaired["success"] is True
    assert repaired["confidence"] == 0.9


def test_repair_judge_json_returns_none_for_unrecognizable_structure():
    assert _repair_judge_json("completely broken, not json at all") is None


# ---------- _deep_compare ----------

def test_deep_compare_equal_dicts_different_key_order():
    assert _deep_compare({"a": 1, "b": 2}, {"b": 2, "a": 1}) is None


def test_deep_compare_different_values():
    result = _deep_compare({"name": "Alice"}, {"name": "Bob"})
    assert result is not None
    assert "$.name" in result


def test_deep_compare_numeric_tolerance():
    """[R1-2] int 90 and float 90.0 are equal."""
    assert _deep_compare(90, 90.0) is None


def test_deep_compare_bool_not_int():
    """[R1-2] bool True must not match int 1."""
    assert _deep_compare(True, 1) is not None


def test_deep_compare_array_length_mismatch():
    result = _deep_compare([1, 2, 3], [1])
    assert result is not None
    assert "array length" in result


def test_deep_compare_extra_key():
    result = _deep_compare({"a": 1}, {"a": 1, "b": 2})
    assert result is not None
    assert "unexpected key" in result


def test_deep_compare_missing_key():
    result = _deep_compare({"a": 1, "b": 2}, {"a": 1})
    assert result is not None
    assert "missing key" in result


def test_deep_compare_string_whitespace_trimmed():
    """[R1-3] String values are trimmed before comparison."""
    assert _deep_compare("hello", "  hello  ") is None


# ---------- llm_judge (mocked LLM) ----------

def _make_message(text: str) -> Message:
    return Message(
        id="msg_1",
        content=[TextBlock(type="text", text=text)],
        model="claude-sonnet-4-6",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


async def test_verify_llm_judge_success(verifier):
    case = _case(verify_mode="llm_judge", expected_output="标题正确")
    judge_json = json.dumps({"success": True, "reason": "输出符合预期", "confidence": 0.95})

    with patch(
        "agent.llm_client.LLMClient.call_with_retry",
        new=AsyncMock(side_effect=lambda **kwargs: kwargs["parse_response"](_make_message(judge_json))),
    ):
        result = await verifier.verify(case, _agent_result(output="标题正确"))

    assert result["success"] is True
    assert result["confidence"] == 0.95


async def test_verify_llm_judge_llm_error_degrades_to_failure(verifier):
    case = _case(verify_mode="llm_judge", expected_output="标题正确")

    with patch(
        "agent.llm_client.LLMClient.call_with_retry",
        new=AsyncMock(side_effect=LLMError("请求耗尽重试", stage="request")),
    ):
        result = await verifier.verify(case, _agent_result(output="标题正确"))

    assert result["success"] is False
    assert result["confidence"] == 0.0


# ---------- [R1-4] public case drift hint ----------

async def test_verify_public_exact_failure_has_drift_hint(verifier):
    case = _case(type="public", verify_mode="exact", expected_output="hello")
    result = await verifier.verify(case, _agent_result(output="world"))
    assert result["success"] is False
    assert "possible_live_drift=true" in result["reason"]


async def test_verify_local_exact_failure_no_drift_hint(verifier):
    case = _case(type="local", verify_mode="exact", expected_output="hello")
    result = await verifier.verify(case, _agent_result(output="world"))
    assert result["success"] is False
    assert "possible_live_drift" not in result["reason"]


async def test_verify_public_json_schema_failure_has_drift_hint(verifier):
    case = _case(
        type="public",
        verify_mode="json_schema",
        expected_output={"version": "1.61.0", "date": "Jun 29, 2026"},
    )
    actual = json.dumps({"version": "1.50.0", "date": "Jan 1, 2026"})
    result = await verifier.verify(case, _agent_result(output=actual))
    assert result["success"] is False
    assert "possible_live_drift=true" in result["reason"]


async def test_verify_public_contains_failure_no_drift_hint(verifier):
    case = _case(type="public", verify_mode="contains", expected_output="xyz")
    result = await verifier.verify(case, _agent_result(output="hello world"))
    assert result["success"] is False
    assert "possible_live_drift" not in result["reason"]


# ---------- [R1-1] verify_mode audit ----------

_STRICT_KEYWORDS = ("只含", "禁止添加任何解释", "严格输出", "仅返回", "only return", "原样返回")
_CONTAINS_EXEMPTIONS = {
    "L02": "本地受控页面，expected_output 足够特异，contains 不会导致假阳性；设计决策保留以容忍格式差异",
}


def test_verify_mode_audit_no_strict_task_with_contains():
    """[R1-1] Automated check: task text with strict keywords must not use contains mode,
    unless explicitly exempted in _CONTAINS_EXEMPTIONS (documented in verify_mode_audit.md).
    """
    project_root = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    case_dirs = [
        os.path.join(project_root, "eval", "cases", "local"),
        os.path.join(project_root, "eval", "cases", "public"),
    ]
    violations = []
    for case_dir in case_dirs:
        if not os.path.isdir(case_dir):
            continue
        for fname in sorted(os.listdir(case_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(case_dir, fname)
            with open(path, mode="r", encoding="utf-8") as f:
                case = json.load(f)
            task = case.get("task", "")
            verify_mode = case.get("verify_mode", "")
            case_id = case.get("id", fname)
            if verify_mode == "contains":
                matched = [kw for kw in _STRICT_KEYWORDS if kw in task]
                if matched and case_id not in _CONTAINS_EXEMPTIONS:
                    violations.append(
                        f"{case_id}: task contains strict keyword(s) {matched} "
                        f"but verify_mode is contains"
                    )
    assert not violations, (
        "verify_mode audit failures (see eval/cases/verify_mode_audit.md):\n"
        + "\n".join(violations)
    )
