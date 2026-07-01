from typing import TypedDict, Literal


class Element(TypedDict):
    role:     str
    name:     str
    selector: str


class ObserveResult(TypedDict):
    url:                  str
    title:                str
    visible_text_summary: str
    interactive_elements: list[Element]
    screenshot_path:      str


class LLMAction(TypedDict):
    action:   Literal["click", "type", "scroll", "extract", "screenshot", "done"]
    selector: str | None
    text:     str | None
    value:    str | None
    reason:   str


class ToolResult(TypedDict):
    success:      bool
    page_changed: bool
    output:       str | None
    error_msg:    str | None


class AgentResult(TypedDict):
    task:        str
    success:     bool
    output:      str | dict | None
    steps:       int
    fail_reason: str | None
    trace_dir:   str


class EvalCase(TypedDict):
    id:              str
    type:            str
    task_type:       str
    task:            str
    url:             str
    expected_output: str | dict
    verify_mode:     Literal["exact", "contains", "json_schema", "llm_judge"]
    difficulty:      str


class VerifyResult(TypedDict):
    case_id:    str
    success:    bool
    reason:     str
    confidence: float
