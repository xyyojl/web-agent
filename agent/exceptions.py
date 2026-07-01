"""Web Agent 统一异常层级。

所有异常均携带结构化上下文信息（而非仅一条消息字符串），
便于上层调用者（agent/loop.py、eval/runner.py 等）在 except 块中
直接读取字段做分支处理或写入 trace 日志，而不必解析异常文本。
"""

from typing import Any


class WebAgentError(Exception):
    """所有 Web Agent 异常的基类。

    Attributes:
        message: 人类可读的错误描述。
        context: 附加的结构化上下文（如 selector、case_id 等），
            用于日志记录和调试，不参与异常匹配逻辑。
    """

    def __init__(self, message: str, **context: Any) -> None:
        self.message = message
        self.context = context
        super().__init__(self._format())

    def _format(self) -> str:
        if not self.context:
            return self.message
        ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} ({ctx_str})"

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict，便于写入 trace / 结构化日志。"""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            **self.context,
        }


class SafetyError(WebAgentError):
    """安全拦截 — 触发：敏感字段检测、登录页检测。

    Attributes:
        trigger: 触发拦截的原因标识，如 "sensitive_field" / "login_page"。
        selector: 触发拦截的目标元素 selector（若适用）。
        url: 触发拦截时所在页面 URL（若适用）。
    """

    def __init__(
        self,
        message: str,
        trigger: str | None = None,
        selector: str | None = None,
        url: str | None = None,
        **context: Any,
    ) -> None:
        self.trigger = trigger
        self.selector = selector
        self.url = url
        super().__init__(
            message,
            trigger=trigger,
            selector=selector,
            url=url,
            **context,
        )


class BrowserError(WebAgentError):
    """浏览器操作失败 — 触发：超时、元素不存在、页面崩溃。

    Attributes:
        action: 失败时正在执行的动作（click/type/scroll/... ）。
        selector: 目标元素 selector（若适用）。
        timeout_ms: 触发超时的阈值（若因超时失败）。
    """

    def __init__(
        self,
        message: str,
        action: str | None = None,
        selector: str | None = None,
        timeout_ms: int | None = None,
        **context: Any,
    ) -> None:
        self.action = action
        self.selector = selector
        self.timeout_ms = timeout_ms
        super().__init__(
            message,
            action=action,
            selector=selector,
            timeout_ms=timeout_ms,
            **context,
        )


class LLMError(WebAgentError):
    """LLM 调用或解析失败 — 触发：API 超时、JSON 解析失败。

    Attributes:
        stage: 失败发生的阶段，如 "request" / "parse"。
        raw_response: 解析失败时的原始返回内容（截断存储，便于排查）。
        retry_count: 失败前已重试的次数。
    """

    _RAW_RESPONSE_MAX_LEN = 500

    def __init__(
        self,
        message: str,
        stage: str | None = None,
        raw_response: str | None = None,
        retry_count: int = 0,
        **context: Any,
    ) -> None:
        self.stage = stage
        self.retry_count = retry_count
        self.raw_response = (
            raw_response[: self._RAW_RESPONSE_MAX_LEN]
            if raw_response is not None
            else None
        )
        super().__init__(
            message,
            stage=stage,
            retry_count=retry_count,
            raw_response=self.raw_response,
            **context,
        )


class EvalError(WebAgentError):
    """Eval 配置错误 — 触发：case 格式错误、URL 无法访问。

    Attributes:
        case_id: 出错的 EvalCase 的 id（若适用）。
        field: 格式错误所在的字段名（若适用）。
    """

    def __init__(
        self,
        message: str,
        case_id: str | None = None,
        field: str | None = None,
        **context: Any,
    ) -> None:
        self.case_id = case_id
        self.field = field
        super().__init__(
            message,
            case_id=case_id,
            field=field,
            **context,
        )
