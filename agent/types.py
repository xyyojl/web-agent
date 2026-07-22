from typing import NotRequired, TypedDict, Literal


class Element(TypedDict):
    role:     str
    name:     str
    selector: str
    href:     str | None


class ContentSafetySignal(TypedDict):
    rule_id: str
    source: str
    content_sha256: str


class ContentSafetyAssessment(TypedDict):
    status: Literal["clean", "suspected", "blocked"]
    signals: list[ContentSafetySignal]


class ObserveResult(TypedDict):
    url:                  str
    title:                str
    visible_text_summary: str
    text_hash:            str
    interactive_elements: list[Element]
    screenshot_path:      str
    screenshot_b64:       NotRequired[str | None]
    content_safety:       NotRequired[ContentSafetyAssessment]


class LLMAction(TypedDict):
    action:   Literal["click", "type", "scroll", "extract", "screenshot", "select", "done"]
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
    task_id:         str | None
    task:            str
    url:             str
    success:         bool
    output:          str | dict | None
    steps:           int
    duration_s:      float
    fail_reason:     str | None
    trace_dir:       str
    last_screenshot: str | None


class EvalCase(TypedDict):
    id:              str
    type:            str
    task_type:       str
    task:            str
    url:             str
    expected_output: str | dict | list
    verify_mode:     Literal["exact", "contains", "json_schema", "llm_judge", "safety_block"]
    difficulty:      str


class VerifyResult(TypedDict):
    case_id:    str
    success:    bool
    reason:     str
    confidence: float
