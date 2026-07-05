"""验证层：Verifier 根据 EvalCase.verify_mode 自动判断任务是否成功。

四种模式里只有 llm_judge 会调用 LLM，其余三种都是纯本地字符串/结构比对，
不发起网络请求；即使 llm_judge 模式下 LLM 输出不是合法 JSON、缺字段、
调用失败，也一律退化为 VerifyResult(success=False, ...) 返回，
verify() 本身不会抛出未捕获异常。
"""

import json
import logging
import os

import anthropic
from anthropic.types import Message, MessageParam, TextBlock

from agent.config import AgentConfig
from agent.prompts import JUDGE_SYSTEM
from agent.types import AgentResult, EvalCase, VerifyResult

logger = logging.getLogger(__name__)

_VALID_VERIFY_MODES = ("exact", "contains", "json_schema", "llm_judge")


def _stringify(value: str | dict | None) -> str:
    """把 expected_output / result.output（str | dict | None）统一转成字符串。

    dict 用 json.dumps(sort_keys=True) 序列化，保证同一份数据无论键顺序
    如何都得到一致的字符串，便于 exact/contains 两种模式做文本级比较。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_type_category(value: object) -> str:
    """把 Python 值映射成一个粗粒度的 JSON 类型分类。

    bool 必须在 int 之前判断——Python 里 bool 是 int 的子类，
    isinstance(True, int) 为 True，顺序反了会把布尔值误判成数字。
    int/float 统一归为 "number"：不同 LLM 抽取结果里同一个分数字段
    可能被解析成 int 或 float，只校验字段名和类型时不应因此判定失败。
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return "unknown"


def _schema_matches(expected: object, actual: object, path: str = "$") -> tuple[bool, str]:
    """递归比较 expected 与 actual 的字段名与类型，不比较具体值。

    - object: expected 里的每个 key 必须在 actual 里存在，且递归匹配；
      actual 里的多余字段不算错误（宽松处理 LLM 抽取可能带的额外信息）。
    - array: 用 expected[0] 作为每个元素的类型模板，actual 中每个元素都要
      能匹配这个模板；不检查数组长度是否相等（长度属于"值"而非"schema"）。
    - 标量（string/number/bool/null）：只比较类型分类，不比较具体值。

    返回 (是否匹配, 不匹配时的原因说明)。
    """
    exp_type = _json_type_category(expected)
    act_type = _json_type_category(actual)
    if exp_type != act_type:
        return False, f"{path}: 期望类型 {exp_type}，实际类型 {act_type}"

    if exp_type == "object":
        assert isinstance(expected, dict) and isinstance(actual, dict)
        for key, exp_val in expected.items():
            if key not in actual:
                return False, f"{path}.{key}: 缺少该字段"
            ok, reason = _schema_matches(exp_val, actual[key], f"{path}.{key}")
            if not ok:
                return False, reason
        return True, ""

    if exp_type == "array":
        assert isinstance(expected, list) and isinstance(actual, list)
        if not expected:
            return True, ""  # 期望是空数组，没有模板可比对，跳过逐项校验
        if not actual:
            return False, f"{path}: 期望非空数组，实际为空数组"
        template = expected[0]
        for idx, item in enumerate(actual):
            ok, reason = _schema_matches(template, item, f"{path}[{idx}]")
            if not ok:
                return False, reason
        return True, ""

    return True, ""  # 标量类型：类型分类已在上面比较过，值本身不参与判断


class Verifier:
    """验证层：支持 exact / contains / json_schema / llm_judge 四种模式。"""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("未配置 ANTHROPIC_API_KEY，无法调用 LLM Judge")
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        assert self._client is not None  # 帮助静态类型检查器收窄为非 Optional
        return self._client

    async def verify(self, case: EvalCase, result: AgentResult) -> VerifyResult:
        """按 case['verify_mode'] 分发到对应的校验方法。"""
        mode = case["verify_mode"]
        if mode == "exact":
            return self._verify_exact(case, result)
        if mode == "contains":
            return self._verify_contains(case, result)
        if mode == "json_schema":
            return self._verify_json_schema(case, result)
        if mode == "llm_judge":
            return await self._verify_llm_judge(case, result)

        # Literal 类型只在静态检查阶段约束取值；运行时 case 数据来自 JSON
        # 文件，仍可能出现非法 verify_mode，这里兜底而不是让 KeyError/
        # AttributeError 直接抛出。
        return VerifyResult(
            case_id=case["id"],
            success=False,
            reason=f"未知的 verify_mode: {mode!r}",
            confidence=1.0,
        )

    @staticmethod
    def _verify_exact(case: EvalCase, result: AgentResult) -> VerifyResult:
        expected = _stringify(case["expected_output"]).strip()
        actual = _stringify(result["output"]).strip()
        success = expected == actual
        reason = (
            "实际输出与预期完全一致"
            if success
            else f"实际输出与预期不一致: 预期={expected!r}, 实际={actual!r}"
        )
        return VerifyResult(case_id=case["id"], success=success, reason=reason, confidence=1.0)

    @staticmethod
    def _verify_contains(case: EvalCase, result: AgentResult) -> VerifyResult:
        expected = _stringify(case["expected_output"]).strip()
        actual = _stringify(result["output"])
        success = expected in actual
        reason = (
            "实际输出包含预期子串"
            if success
            else f"实际输出未包含预期子串: 预期子串={expected!r}"
        )
        return VerifyResult(case_id=case["id"], success=success, reason=reason, confidence=1.0)

    @staticmethod
    def _verify_json_schema(case: EvalCase, result: AgentResult) -> VerifyResult:
        expected = case["expected_output"]
        if not isinstance(expected, dict):
            return VerifyResult(
                case_id=case["id"],
                success=False,
                reason="case.expected_output 不是合法的 schema 模板（应为 dict）",
                confidence=1.0,
            )

        raw_actual = result["output"]
        try:
            actual = json.loads(raw_actual) if isinstance(raw_actual, str) else raw_actual
        except (json.JSONDecodeError, TypeError) as exc:
            return VerifyResult(
                case_id=case["id"],
                success=False,
                reason=f"实际输出不是合法 JSON: {exc}",
                confidence=1.0,
            )

        ok, reason = _schema_matches(expected, actual)
        return VerifyResult(
            case_id=case["id"],
            success=ok,
            reason=reason or "字段名与类型均匹配",
            confidence=1.0,
        )

    async def _verify_llm_judge(self, case: EvalCase, result: AgentResult) -> VerifyResult:
        """调用 LLM Judge 打分。任何环节失败（调用失败/输出非 JSON/字段
        缺失或类型不对）都统一退化为 VerifyResult(success=False, ...)，
        不抛出异常——这是 llm_judge 模式相对其他三种模式唯一的额外风险点。
        """
        try:
            client = self._get_client()
        except RuntimeError as exc:
            return VerifyResult(case_id=case["id"], success=False, reason=str(exc), confidence=0.0)

        prompt = (
            f"任务描述: {case['task']}\n"
            f"预期输出: {_stringify(case['expected_output'])}\n"
            f"实际输出: {_stringify(result['output'])}\n"
        )

        messages: list[MessageParam] = [MessageParam(role="user", content=prompt)]

        try:
            message: Message = await client.messages.create(
                model=self.config.model,
                max_tokens=512,
                timeout=self.config.llm_timeout,
                system=JUDGE_SYSTEM,
                messages=messages,
            )
        except anthropic.APIError as exc:
            logger.warning("LLM Judge 调用失败: %s", exc)
            return VerifyResult(
                case_id=case["id"], success=False, reason=f"LLM Judge 调用失败: {exc}", confidence=0.0
            )

        raw_text = "".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        ).strip()

        cleaned = raw_text
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("LLM Judge 输出无法解析为 JSON: %s；原始输出: %r", exc, raw_text)
            return VerifyResult(
                case_id=case["id"],
                success=False,
                reason=f"LLM Judge 输出不是合法 JSON: {raw_text[:200]!r}",
                confidence=0.0,
            )

        if not isinstance(parsed, dict):
            return VerifyResult(
                case_id=case["id"],
                success=False,
                reason=f"LLM Judge 输出不是 JSON 对象: {parsed!r}",
                confidence=0.0,
            )

        judged_success = parsed.get("success")
        judged_reason = parsed.get("reason")
        judged_confidence = parsed.get("confidence")

        if (
            not isinstance(judged_success, bool)
            or not isinstance(judged_reason, str)
            or not isinstance(judged_confidence, (int, float))
            or isinstance(judged_confidence, bool)
        ):
            return VerifyResult(
                case_id=case["id"],
                success=False,
                reason=f"LLM Judge 输出字段缺失或类型不对: {parsed!r}",
                confidence=0.0,
            )

        return VerifyResult(
            case_id=case["id"],
            success=judged_success,
            reason=judged_reason,
            confidence=float(judged_confidence),
        )
