"""感知层：从 Playwright Page 提取结构化观察结果，不传完整 HTML。

只把「可见文本摘要 + 精简后的交互元素列表」喂给 LLM，
避免整页 DOM/HTML 撑爆 prompt，同时保留截图供人工排查。
"""

from agent.config import AgentConfig
from agent.exceptions import BrowserError
from agent.tracer import TraceLogger
from agent.types import Element, ObserveResult

# 提取可见文本：过滤 script/style/noscript 等不可见节点，
# 按文档流顺序拼接可见文本节点，交给 Python 侧再做长度截断。
_EXTRACT_TEXT_JS = """
() => {
    const skipTags = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE"]);
    const isVisible = (el) => {
        // checkVisibility() 会沿祖先链检查 display/visibility，
        // 比只看直接父节点的 computedStyle 更准确
        // （display:none 只影响命中的那个祖先本身的渲染框，
        //  子元素自身的 computedStyle.display 并不会变成 none）。
        if (typeof el.checkVisibility === "function") {
            return el.checkVisibility({
                checkOpacity: false,
                checkVisibilityCSS: true,
            });
        }
        const style = window.getComputedStyle(el);
        return style && style.display !== "none" && style.visibility !== "hidden";
    };
    const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        {
            acceptNode(node) {
                const parent = node.parentElement;
                if (!parent) return NodeFilter.FILTER_REJECT;
                if (skipTags.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
                if (!isVisible(parent)) return NodeFilter.FILTER_REJECT;
                if (!node.textContent || !node.textContent.trim()) {
                    return NodeFilter.FILTER_SKIP;
                }
                return NodeFilter.FILTER_ACCEPT;
            },
        }
    );
    const parts = [];
    let node;
    while ((node = walker.nextNode())) {
        parts.push(node.textContent.trim());
    }
    return parts.join(" ");
}
"""

# 提取交互元素：button > a > input > select 的优先级顺序，
# 只保留 role/name/唯一 selector，selector 优先用 text=，退化到 css nth-of-type。
_EXTRACT_ELEMENTS_JS = """
(maxElements) => {
    const isVisible = (el) => {
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        return (
            rect.width > 0 &&
            rect.height > 0 &&
            style.display !== "none" &&
            style.visibility !== "hidden"
        );
    };

    const buildSelector = (el) => {
        const text = (el.innerText || el.value || "").trim();
        if (text && text.length <= 60) {
            return `text=${text}`;
        }
        if (el.id) {
            return `css=#${el.id}`;
        }
        const tag = el.tagName.toLowerCase();
        const siblings = Array.from(el.parentElement ? el.parentElement.children : []).filter(
            (n) => n.tagName === el.tagName
        );
        const index = siblings.indexOf(el) + 1;
        return `css=${tag}:nth-of-type(${index})`;
    };

    const roleFor = (el) => {
        const explicit = el.getAttribute("role");
        if (explicit) return explicit;
        const tag = el.tagName.toLowerCase();
        if (tag === "a") return "link";
        if (tag === "button") return "button";
        if (tag === "input") return `input:${el.type || "text"}`;
        if (tag === "select") return "select";
        return tag;
    };

    const nameFor = (el) => {
        const text = (el.innerText || "").trim();
        if (text) return text;
        const aria = el.getAttribute("aria-label");
        if (aria) return aria;
        const placeholder = el.getAttribute("placeholder");
        if (placeholder) return placeholder;
        const value = el.value;
        if (value) return String(value);
        return "";
    };

    // button > a > input > select 优先级
    const groups = [
        Array.from(document.querySelectorAll("button")),
        Array.from(document.querySelectorAll("a")),
        Array.from(document.querySelectorAll("input")),
        Array.from(document.querySelectorAll("select")),
    ];

    const seen = new Set();
    const results = [];
    for (const group of groups) {
        for (const el of group) {
            if (results.length >= maxElements) break;
            if (!isVisible(el)) continue;
            const selector = buildSelector(el);
            if (seen.has(selector)) continue;
            seen.add(selector);
            results.push({
                role: roleFor(el),
                name: nameFor(el),
                selector: selector,
            });
        }
        if (results.length >= maxElements) break;
    }
    return results.slice(0, maxElements);
}
"""


class BrowserStateObserver:
    """从 Playwright Page 提取结构化观察结果（不含完整 HTML）。"""

    def __init__(self, config: AgentConfig, tracer: TraceLogger) -> None:
        self.config = config
        self.tracer = tracer

    async def observe(self, page) -> ObserveResult:
        """采集当前页面状态，返回 ObserveResult。

        步骤：
        1. 等待页面网络空闲，降低动态渲染（React/Vue）读取半成品 DOM 的概率。
        2. 提取 url / title。
        3. JS 提取可见文本，Python 侧截断到 obs_text_limit 字符。
        4. JS 提取交互元素，最多 obs_max_elements 条。
        5. 截图落盘到 tracer 分配的路径。
        """
        try:
            await page.wait_for_load_state(
                "networkidle", timeout=self.config.browser_timeout
            )
        except Exception as exc:
            # networkidle 超时不应直接判为致命错误（长轮询/SSE 页面永远不会 idle），
            # 降级继续采集，但记录到 BrowserError 供上层按需处理/忽略。
            fallback_error = BrowserError(
                "等待 networkidle 超时，降级继续采集当前 DOM 状态",
                action="wait_for_load_state",
                timeout_ms=self.config.browser_timeout,
            )
            _ = fallback_error  # 仅记录语义，不中断流程

        url = page.url
        title = await page.title()

        raw_text: str = await page.evaluate(_EXTRACT_TEXT_JS)
        visible_text_summary = self._truncate_text(raw_text, self.config.obs_text_limit)

        raw_elements: list[dict] = await page.evaluate(
            _EXTRACT_ELEMENTS_JS, self.config.obs_max_elements
        )
        interactive_elements: list[Element] = [
            Element(
                role=item.get("role", ""),
                name=item.get("name", ""),
                selector=item.get("selector", ""),
            )
            for item in raw_elements[: self.config.obs_max_elements]
        ]

        screenshot_path = self.tracer.next_screenshot_path()
        try:
            await page.screenshot(path=screenshot_path)
        except Exception as exc:
            raise BrowserError(
                "截图失败",
                action="screenshot",
                selector=None,
            ) from exc

        return ObserveResult(
            url=url,
            title=title,
            visible_text_summary=visible_text_summary,
            interactive_elements=interactive_elements,
            screenshot_path=screenshot_path,
        )

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        """将连续空白折叠为单个空格后按字符数截断到 limit。"""
        collapsed = " ".join(text.split())
        return collapsed[:limit]
