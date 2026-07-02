from dataclasses import dataclass


@dataclass
class AgentConfig:
    model:            str   = "glm-4.7-flash"
    max_steps:        int   = 15
    max_fail:         int   = 3
    step_delay:       float = 0.5
    obs_max_elements: int   = 20
    obs_text_limit:   int   = 500
    llm_timeout:      int   = 30
    browser_timeout:  int   = 15000 # 单位：毫秒
    llm_retry:        int   = 3
    trace_dir:        str   = "traces"
