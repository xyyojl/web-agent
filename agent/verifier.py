"""验证层：Verifier 根据 EvalCase.verify_mode 自动判断任务是否成功。

四种模式里只有 llm_judge 会调用 LLM，其余三种都是纯本地字符串/结构比对，
不发起网络请求；即使 llm_judge 模式下 LLM 输出不是合法 JSON、缺字段、
调用失败，也一律退化为 VerifyResult(success=False, ...) 返回，
verify() 本身不会抛出未捕获异常。
"""

import asyncio
import json
import logging
import os
import re

import anthropic
from anthropic.types import Message, MessageParam, TextBlock

from agent.config import AgentConfig
from agent.prompts import JUDGE_SYSTEM
from agent.types import AgentResult, EvalCase, VerifyResult

logger = logging.getLogger(__name__)

_VALID_VERIFY_MODES = ("exact", "contains", "json_schema", "llm_judge")

# 兜底修复：Judge 输出严格按 {"success": bool, "reason": str, "confidence": float}
# 三个固定字段、固定顺序（JUDGE_SYSTEM 里约定的格式），即使 reason 字符串内部
# 出现了未转义的双引号（模型习惯性地用英文双引号引用具体词语，例如
# reason 里写 "锁定操作"，把外层字符串提前"戳穿"）导致 json.loads 失败，
# 也能按这个已知的三段式结构把值抠出来，不依赖 reason 内部本身是合法 JSON字符串这个前提。
_JUDGE_REPAIR_RE = re.compile(
    r'"success"\s*:\s*(true|false)\s*,\s*"reason"\s*:\s*"(.*)"\s*,\s*"confidence"\s*:\s*([0-9]*\.?[0-9]+)\s*}',
    re.DOTALL,
)


def _repair_judge_json(raw_text: str) -> dict[str, object] | None:
    """从格式受损（内部含未转义引号）的 Judge 输出里尽力抠出三个字段。

    只处理"整体结构没坏、只是 reason 内部有未转义引号"这一种已知的、
    真实复现过的受损模式；如果连 success/confidence这两个锚点字段都对不上，
    说明输出损坏得更严重，直接返回 None，交由上层退化为 success=False，不做进一步猜测。
    """
    match = _JUDGE_REPAIR_RE.search(raw_text)
    if not match:
        return None
    success_str, reason_raw, confidence_str = match.groups()
    try:
        confidence = float(confidence_str)
    except ValueError:
        return None
    return {
        "success": success_str == "true",
        "reason": reason_raw,
        "confidence": confidence,
    }


def _stringify(value: str | dict | list | None) -> str:
    """把 expected_output / result.output（str | dict | list | None）统一转成字符串。

    dict / list 用 json.dumps(sort_keys=True) 序列化，保证同一份数据无论键顺序
    或数组元素顺序如何都得到一致的字符串，便于 exact/contains 两种模式做文本级比较。
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
            self._client = anthropic.AsyncAnthropic()
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
        if not isinstance(expected, (dict, list)):
            return VerifyResult(
                case_id=case["id"],
                success=False,
                reason="case.expected_output 不是合法的 schema 模板（应为 dict 或 list）",
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
        """调用 LLM Judge 打分。调用异常、返回内容为空、JSON 解析失败、
        字段缺失或类型不对——这四类失败都视为一次"格式抖动"，统一纳入
        同一套重试逻辑；只有 config.llm_retry 次全部用尽仍未拿到合法结果，
        才最终退化为 VerifyResult(success=False, ...)。verify() 本身
        不会因 llm_judge 抛出未捕获异常。

        对 429 速率限制错误会等待 config.rate_limit_delay 秒后重试；
        其余情况按指数退避（2 ** attempt 秒）重试。
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
        max_attempts = max(1, self.config.llm_retry)
        last_failure_reason = "未知错误"

        for attempt in range(max_attempts):
            will_retry = attempt + 1 < max_attempts

            try:
                message: Message = await client.messages.create(
                    model=self.config.model,
                    # 512 tokens 在实测中不够用。
                    # 调大到 1024 留出安全余量，仍远小于会显著增加延迟/成本的量级。
                    max_tokens=1024,
                    timeout=self.config.llm_timeout,
                    system=JUDGE_SYSTEM,
                    messages=messages,
                )
            except anthropic.RateLimitError as exc:
                last_failure_reason = f"LLM Judge 调用失败: {exc}"
                wait_secs = self.config.rate_limit_delay
                logger.warning(
                    "LLM Judge 调用失败（第 %d/%d 次尝试）: %s%s",
                    attempt + 1, max_attempts, exc,
                    f"，即将等待 {wait_secs}s 后重试" if will_retry else "，已达重试上限",
                )
                if will_retry:
                    await asyncio.sleep(wait_secs)
                continue
            except anthropic.APIError as exc:
                last_failure_reason = f"LLM Judge 调用失败: {exc}"
                logger.warning("LLM Judge 调用失败（第 %d/%d 次尝试）: %s%s",
                               attempt + 1, max_attempts, exc,
                               "，即将重试" if will_retry else "，已达重试上限")
                if will_retry:
                    await asyncio.sleep(2 ** attempt)
                continue

            # 以下都是"API 调用成功，但输出内容有格式问题"的场景：
            # 空内容 / 非法 JSON（且兜底正则也未命中）/ 字段缺失或类型不对。
            # 三种情况都视为一次可重试的格式抖动，而不是立即判负。

            raw_text = "".join(
                block.text for block in message.content if isinstance(block, TextBlock)
            ).strip()

            if not raw_text:
                block_types = [type(block).__name__ for block in message.content]
                last_failure_reason = "LLM Judge 多次重试后仍返回空内容"
                logger.warning(
                    "LLM Judge 返回内容为空（第 %d/%d 次尝试），stop_reason=%s，content=%s%s",
                    attempt + 1, max_attempts, message.stop_reason, block_types,
                    "，即将重试" if will_retry else "，已达重试上限",
                )
                if will_retry:
                    await asyncio.sleep(2 ** attempt)
                continue

            cleaned = raw_text
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned

            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                repaired = _repair_judge_json(raw_text)
                if repaired is not None:
                    logger.warning(
                        "LLM Judge 输出不是合法 JSON（%s），已通过兜底修复正则恢复三个字段，"
                        "原始输出: %r",
                        exc,
                        raw_text,
                    )
                    parsed = repaired
                else:
                    last_failure_reason = f"LLM Judge 输出不是合法 JSON: {raw_text[:200]!r}"
                    logger.warning(
                        "LLM Judge 输出无法解析为 JSON 且兜底修复也未命中（第 %d/%d 次尝试）: "
                        "%s；原始输出: %r%s",
                        attempt + 1, max_attempts, exc, raw_text,
                        "，即将重试" if will_retry else "，已达重试上限",
                    )
                    if will_retry:
                        await asyncio.sleep(2 ** attempt)
                    continue

            if not isinstance(parsed, dict):
                last_failure_reason = f"LLM Judge 输出不是 JSON 对象: {parsed!r}"
                logger.warning(
                    "LLM Judge 输出不是 JSON 对象（第 %d/%d 次尝试）: %r%s",
                    attempt + 1, max_attempts, parsed,
                    "，即将重试" if will_retry else "，已达重试上限",
                )
                if will_retry:
                    await asyncio.sleep(2 ** attempt)
                continue

            judged_success = parsed.get("success")
            judged_reason = parsed.get("reason")
            judged_confidence = parsed.get("confidence")

            if (
                not isinstance(judged_success, bool)
                or not isinstance(judged_reason, str)
                or not isinstance(judged_confidence, (int, float))
                or isinstance(judged_confidence, bool)
            ):
                last_failure_reason = f"LLM Judge 输出字段缺失或类型不对: {parsed!r}"
                logger.warning(
                    "LLM Judge 输出字段缺失或类型不对（第 %d/%d 次尝试）: %r%s",
                    attempt + 1, max_attempts, parsed,
                    "，即将重试" if will_retry else "，已达重试上限",
                )
                if will_retry:
                    await asyncio.sleep(2 ** attempt)
                continue

            return VerifyResult(
                case_id=case["id"],
                success=judged_success,
                reason=judged_reason,
                confidence=float(judged_confidence),
            )

        # 全部重试耗尽，仍未拿到一次完全合法的 Judge 输出
        return VerifyResult(
            case_id=case["id"],
            success=False,
            reason=last_failure_reason,
            confidence=0.0,
        )
