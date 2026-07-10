"""VISION-001 共享辅助：WebPlanner 与 ActionSelector 都需要把当前
ObserveResult 转成一次 API 调用的 user content——纯文本，或
[image_block, text_block] 列表。两边逻辑完全一致（不像
_format_observation 那样因为“Planner 隐藏 selector、Selector 显示
selector”而必须各写一份），因此抽成单独模块共享，避免日后改动
image block 格式时两处代码分别维护、行为漂移。
"""

from agent.types import ObserveResult


def build_vision_user_content(obs: ObserveResult, text: str) -> str | list[dict[str, object]]:
    """按 obs 里是否带有效 screenshot_b64 决定 user 消息内容。

    screenshot_b64 缺失（未启用 vision，字段本身不存在）或为 None
    （启用了 vision 但本步截图采集失败，observer.py 里已降级处理）
    时，统一退化为纯文本——不构造一个 data 字段为 None 的非法
    image block 传给 API。
    """
    screenshot_b64 = obs.get("screenshot_b64")
    if not screenshot_b64:
        return text
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": screenshot_b64,
            },
        },
        {"type": "text", "text": text},
    ]
