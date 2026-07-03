"""WebAgent 核心包：浏览器自动化 Agent 的配置、类型、异常、执行工具集合。

对外公开导出的符号在此汇总，外部代码优先使用
`from agent import AgentConfig, TraceLogger, ...`
而不是深入各子模块路径，便于后续内部重构不影响调用方。
"""

from agent.browser_tools import (
    LOGIN_SIGNALS,
    SENSITIVE_PATTERNS,
    ask_human,
    browser_click,
    browser_extract,
    browser_observe,
    browser_open,
    browser_screenshot,
    browser_scroll,
    browser_type,
)
from agent.config import AgentConfig
from agent.exceptions import (
    BrowserError,
    EvalError,
    LLMError,
    SafetyError,
    WebAgentError,
)
from agent.observer import BrowserStateObserver
from agent.prompts import (
    EXTRACTOR_SYSTEM,
    EXTRACTOR_USER_TMPL,
    JUDGE_SYSTEM,
    PLANNER_SYSTEM,
    PLANNER_USER_TMPL,
    SELECTOR_SYSTEM,
    SELECTOR_TOOLS,
)
from agent.tracer import TraceLogger
from agent.types import (
    AgentResult,
    Element,
    EvalCase,
    LLMAction,
    ObserveResult,
    ToolResult,
    VerifyResult,
)

__all__ = [
    # 配置
    "AgentConfig",
    # 数据结构（TypedDict）
    "Element",
    "ObserveResult",
    "LLMAction",
    "ToolResult",
    "AgentResult",
    "EvalCase",
    "VerifyResult",
    # 异常
    "WebAgentError",
    "SafetyError",
    "BrowserError",
    "LLMError",
    "EvalError",
    # 执行层
    "TraceLogger",
    "BrowserStateObserver",
    # 浏览器工具
    "browser_open",
    "browser_observe",
    "browser_click",
    "browser_type",
    "browser_extract",
    "browser_scroll",
    "browser_screenshot",
    "ask_human",
    "SENSITIVE_PATTERNS",
    "LOGIN_SIGNALS",
    # Prompt 模板
    "PLANNER_SYSTEM",
    "PLANNER_USER_TMPL",
    "SELECTOR_SYSTEM",
    "SELECTOR_TOOLS",
    "EXTRACTOR_SYSTEM",
    "EXTRACTOR_USER_TMPL",
    "JUDGE_SYSTEM",
]
