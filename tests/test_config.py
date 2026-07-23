"""agent/config.py 单元测试：重点覆盖 from_env() 的类型转换逻辑，
尤其是 bool 字段容易踩的 `bool("false") == True` 陷阱。
"""

import pytest

from agent.config import AgentConfig


def test_default_construction_ignores_environment(monkeypatch):
    """AgentConfig() 零参构造不应读取任何环境变量，即使它们被设置了。"""
    monkeypatch.setenv("WEBAGENT_MAX_STEPS", "999")
    monkeypatch.setenv("WEBAGENT_VISION", "true")

    config = AgentConfig()

    assert config.max_steps == 15
    assert config.vision is False


def test_from_env_no_overrides_returns_defaults():
    """未设置任何 WEBAGENT_* 环境变量时，from_env() 应等价于默认构造。"""
    config = AgentConfig.from_env()
    assert config == AgentConfig()


@pytest.mark.parametrize(
    "raw_value,expected",
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("False", False),
        ("no", False),
        ("off", False),
        ("", False),
        ("garbage", False),
    ],
)
def test_from_env_bool_field_parsing(monkeypatch, raw_value, expected):
    """WEBAGENT_VISION 必须按真值词表解析，而不是 Python 的 bool(str) 语义。"""
    monkeypatch.setenv("WEBAGENT_VISION", raw_value)
    config = AgentConfig.from_env()
    assert config.vision is expected


def test_from_env_int_field_parsing(monkeypatch):
    monkeypatch.setenv("WEBAGENT_MAX_STEPS", "42")
    config = AgentConfig.from_env()
    assert config.max_steps == 42
    assert isinstance(config.max_steps, int)


def test_from_env_float_field_parsing(monkeypatch):
    monkeypatch.setenv("WEBAGENT_STEP_DELAY", "1.5")
    config = AgentConfig.from_env()
    assert config.step_delay == 1.5
    assert isinstance(config.step_delay, float)


def test_from_env_str_field_parsing(monkeypatch):
    monkeypatch.setenv("WEBAGENT_MODEL", "claude-test-model")
    config = AgentConfig.from_env()
    assert config.model == "claude-test-model"


def test_from_env_parses_noise_selectors(monkeypatch):
    monkeypatch.setenv("WEBAGENT_NOISE_SELECTORS", " #ticker, .ad-slot ,, ")
    assert AgentConfig.from_env().noise_selectors == ("#ticker", ".ad-slot")


def test_from_env_unset_fields_keep_dataclass_defaults(monkeypatch):
    monkeypatch.setenv("WEBAGENT_MAX_STEPS", "3")
    config = AgentConfig.from_env()
    assert config.max_steps == 3
    # 未设置的字段应保持 dataclass 默认值不变
    assert config.max_fail == 3 or config.max_fail == AgentConfig().max_fail
    assert config.model == AgentConfig().model
    assert config.trace_dir == AgentConfig().trace_dir
