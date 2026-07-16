"""agent/verifier.py 单元测试：覆盖 exact/contains/json_schema/safety_block
四种纯本地校验模式，以及 llm_judge 模式（mock LLMClient，不发真实请求）
的成功、格式受损自愈、彻底失败退化路径。
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from anthropic.types import Message, TextBlock, Usage

from agent.config import AgentConfig
from agent.exceptions import LLMError
from agent.types import AgentResult, EvalCase
from agent.verifier import (
    Verifier,
    _build_schema_from_example,
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
        "task": "抓取标题",
        "success": True,
        "output": "hello",
        "steps": 1,
        "fail_reason": None,
        "trace_dir": "/tmp/run-x",
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
    actual = json.dumps({"name": "Bob", "score": 80})
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
