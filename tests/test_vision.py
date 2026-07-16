"""agent/vision.py 单元测试：验证 screenshot_b64 缺失/None/存在
三种情况下 build_vision_user_content() 的分支行为。
"""

from agent.types import ObserveResult
from agent.vision import build_vision_user_content


def _make_obs(**overrides) -> ObserveResult:
    base: ObserveResult = {
        "url": "https://example.com",
        "title": "Example",
        "visible_text_summary": "hello",
        "text_hash": "abc",
        "interactive_elements": [],
        "screenshot_path": "/tmp/step-001.png",
    }
    base.update(overrides)
    return base


def test_no_screenshot_b64_key_returns_plain_text():
    obs = _make_obs()  # 未启用 vision，字段本身不存在
    result = build_vision_user_content(obs, "hello world")
    assert result == "hello world"


def test_screenshot_b64_none_returns_plain_text():
    obs = _make_obs(screenshot_b64=None)  # 启用 vision 但本步采集失败
    result = build_vision_user_content(obs, "hello world")
    assert result == "hello world"


def test_screenshot_b64_empty_string_returns_plain_text():
    obs = _make_obs(screenshot_b64="")
    result = build_vision_user_content(obs, "hello world")
    assert result == "hello world"


def test_screenshot_b64_present_returns_image_and_text_blocks():
    obs = _make_obs(screenshot_b64="ZmFrZWJhc2U2NA==")
    result = build_vision_user_content(obs, "hello world")

    assert isinstance(result, list)
    assert len(result) == 2

    image_block: dict = result[0]
    text_block: dict = result[1]
    assert image_block["type"] == "image"
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/jpeg"
    assert image_block["source"]["data"] == "ZmFrZWJhc2U2NA=="

    assert text_block == {"type": "text", "text": "hello world"}
