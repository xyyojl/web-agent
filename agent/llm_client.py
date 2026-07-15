"""共享的 Anthropic 客户端初始化 + 统一重试策略。

WebPlanner / ActionSelector / Verifier（llm_judge 模式）/ browser_tools
（browser_extract）过去各自实现了几乎一致的一段代码（客户端懒加载 +
429 固定等待 + 其余错误指数退避 + 耗尽后包装成 LLMError），大约
50~60 行 × 4 处。这段重试策略本身就是项目里声明过的统一策略
（见各调用方文档），不是"看起来像但各管各的"，所以值得收敛成一份：
一旦要调整重试次数、等待时间、日志格式，改一处即可，不会再出现
"抄了四份、改漏了一份"的情况——browser_tools.py 那份副本此前就已经
悄悄丢了 429 特殊处理和指数退避，这次顺带修好。

四处真正不同的只有"这次 LLM 返回算不算可用、怎么从 Message 里解析出
业务需要的结果"，这部分通过 parse_response 回调留在各调用方自己的
文件里，不塞进这个通用模块。
"""

import asyncio
import logging
import os
from typing import Callable, TypeVar, cast

import anthropic
from anthropic.types import Message

from agent.config import AgentConfig
from agent.exceptions import LLMError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 进程级单例：所有 LLMClient 实例共享同一个底层连接池。四个调用方
# 用的是同一个 ANTHROPIC_API_KEY、同一个 base_url，没有理由各建一份
# 连接池——这一点和"要不要共用重试策略"是两件独立的事，所以拆成
# 模块级函数单独维护，不和 LLMClient 实例的生命周期绑在一起。
_shared_anthropic_client: anthropic.AsyncAnthropic | None = None


def _get_shared_anthropic_client() -> anthropic.AsyncAnthropic:
    global _shared_anthropic_client
    if _shared_anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("未配置 ANTHROPIC_API_KEY，无法调用 LLM", stage="request")
        _shared_anthropic_client = anthropic.AsyncAnthropic()
    assert _shared_anthropic_client is not None  # 帮助静态类型检查器收窄为非 Optional
    return _shared_anthropic_client


class LLMOutputRetry(Exception):
    """调用方在 parse_response 里判定"这次输出不可用"时抛出，触发下一次重试。

    和网络层的 anthropic.APIError 走同一套指数退避逻辑——对
    LLMClient.call_with_retry() 来说，"HTTP 请求失败"和"HTTP 成功但
    模型输出内容有问题（空文本/JSON 解析失败/字段缺失等）"是同一类
    "这次尝试不算数，再试一次"事件，不需要调用方自己写 sleep/continue。
    """


class LLMClient:
    """封装"客户端懒加载 + 统一重试策略"，供各推理/决策/校验层复用。

    429 速率限制固定等待 config.rate_limit_delay 秒；其余 API 错误
    以及 parse_response 抛出的 LLMOutputRetry 按指数退避（2 ** attempt
    秒）重试；两者都在 config.llm_retry 次内耗尽后统一包装成
    LLMError 抛出，调用方按需自行捕获（比如 Verifier 会把它转成
    VerifyResult(success=False, ...)，而不是让异常继续往上抛）。
    """

    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    async def call_with_retry(
        self,
        *,
        caller_name: str,
        parse_response: Callable[[Message], T],
        **create_kwargs: object,
    ) -> T:
        """发起一次（含重试的）Messages API 调用，返回 parse_response 解析后的结果。

        caller_name：仅用于日志前缀（如 "WebPlanner"/"ActionSelector"），
            方便从日志里分辨是谁在重试，不影响重试逻辑本身。
        create_kwargs：透传给 client.messages.create() 的其余参数
            （model/system/messages/tools/tool_choice/max_tokens 等）；
            timeout 统一取 self._config.llm_timeout，调用方不需要重复传。
        """
        client = _get_shared_anthropic_client()
        max_attempts = max(1, self._config.llm_retry)
        last_exc: Exception | None = None

        # 下面用 cast(Message, ...) 收窄类型的前提是"调用方不会传 stream=True"，
        # 这里显式断言守住这个前提，避免以后哪个调用方悄悄传了 stream=True，
        # 却因为 cast 被静默当成 Message 处理，实际拿到的其实是 AsyncStream。
        assert "stream" not in create_kwargs, (
            "LLMClient.call_with_retry() 不支持流式响应（stream=True）"
        )

        for attempt in range(max_attempts):
            will_retry = attempt + 1 < max_attempts

            try:
                # client.messages.create() 是重载方法：stream=True 时返回
                # Message | AsyncStream[...] 联合类型，stream=False/省略时
                # 才返回单一的 Message。**create_kwargs 是一个普通
                # dict[str, object]，类型检查器没法据此静态确认调用方
                # 没传 stream=True，只能退化成联合类型——但运行时四个
                # 调用方（Planner/ActionSelector/Verifier/browser_tools）
                # 都不会传 stream，所以这里用 cast 显式收窄，而不是真的
                # 需要处理流式响应的分支逻辑。
                raw_message = await client.messages.create(
                    timeout=self._config.llm_timeout, **create_kwargs
                )
                message = cast(Message, raw_message)
            except anthropic.RateLimitError as exc:
                # 429 速率限制：免费模型速率窗口通常为 1 分钟，
                # 短时间指数退避反而加剧频率压力，改用固定的 rate_limit_delay 更可靠。
                last_exc = exc
                wait_secs = self._config.rate_limit_delay
                logger.warning(
                    "%s LLM 调用失败（第 %d/%d 次尝试）: %s%s",
                    caller_name, attempt + 1, max_attempts, exc,
                    f"，即将等待 {wait_secs}s 后重试" if will_retry else "，已达重试上限",
                )
                if will_retry:
                    await asyncio.sleep(wait_secs)
                continue
            except anthropic.APIError as exc:
                last_exc = exc
                logger.warning(
                    "%s LLM 调用失败（第 %d/%d 次尝试）: %s%s",
                    caller_name, attempt + 1, max_attempts, exc,
                    "，即将重试" if will_retry else "，已达重试上限",
                )
                if will_retry:
                    await asyncio.sleep(2**attempt)  # 1s / 2s / 4s / ...
                continue

            try:
                return parse_response(message)
            except LLMOutputRetry as exc:
                last_exc = exc
                logger.warning(
                    "%s 输出不可用（第 %d/%d 次尝试）: %s%s",
                    caller_name, attempt + 1, max_attempts, exc,
                    "，即将重试" if will_retry else "，已达重试上限",
                )
                if will_retry:
                    await asyncio.sleep(2**attempt)
                continue

        last_exc_desc = last_exc.message if isinstance(last_exc, LLMError) else str(last_exc)
        raise LLMError(
            f"{caller_name} LLM 请求连续失败 {max_attempts} 次: {last_exc_desc}",
            stage="request",
            retry_count=max_attempts,
        )
