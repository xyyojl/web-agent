"""持久化边界的输入值脱敏工具。"""

import re
from typing import Any

_REDACTED = "[REDACTED:browser_type_input]"
_SENSITIVE_ASSIGNMENT = re.compile(
    r"((?:password|passwd|pwd|secret|token|密码|口令|凭据)\s*(?:为|是|=|:|to|is|set to|修改为|设置为)\s*[「\"']?)([^\s，。！？,;；\"'」]+)",
    re.IGNORECASE,
)


def redact_text(value: str | None, secrets: set[str] | None = None) -> str | None:
    """返回可持久化文本，不保留 browser_type 输入值或任务中的敏感赋值。"""
    if value is None:
        return None
    result = value
    for secret in sorted(secrets or set(), key=len, reverse=True):
        if secret:
            result = result.replace(secret, _REDACTED)
    return _SENSITIVE_ASSIGNMENT.sub(r"\1" + _REDACTED, result)


def redact_data(value: Any, secrets: set[str] | None = None) -> Any:
    """递归清洗将写入 JSON/Markdown artifact 的数据。"""
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, list):
        return [redact_data(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: redact_data(item, secrets) for key, item in value.items()}
    return value
