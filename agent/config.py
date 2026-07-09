import os
from dataclasses import dataclass, fields


@dataclass
class AgentConfig:
    model:              str   = "claude-sonnet-4-6"
    max_steps:          int   = 15
    max_fail:           int   = 3
    step_delay:         float = 0.5
    obs_max_elements:   int   = 20
    obs_text_limit:     int   = 3000
    llm_timeout:        int   = 30
    browser_timeout:    int   = 15000
    open_retry:         int   = 2
    llm_retry:          int   = 3
    trace_dir:          str   = "traces"
    rate_limit_delay:   int   = 60
    case_delay:         float = 0.0

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """从 WEBAGENT_<FIELD_NAME> 环境变量读取覆盖值，未设置的字段保留 dataclass 默认值。

        字段名与环境变量名的映射规则：全大写 + WEBAGENT_ 前缀，例如
        max_steps -> WEBAGENT_MAX_STEPS，与 .env.example 中列出的变量名一一对应。
        `AgentConfig()`（零参构造）的行为不受影响，仍然是纯默认值，
        只有显式调用 from_env() 才会读取环境变量，避免隐式的环境依赖。
        """
        overrides: dict = {}
        for field in fields(cls):
            env_name = f"WEBAGENT_{field.name.upper()}"
            raw_value = os.environ.get(env_name)
            if raw_value is None:
                continue
            overrides[field.name] = field.type(raw_value) if field.type in (int, float) else raw_value
        return cls(**overrides)
