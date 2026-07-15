"""感知层：从 Playwright Page 提取结构化观察结果，不传完整 HTML。

只把「可见文本摘要 + 精简后的交互元素列表」喂给 LLM，
避免整页 DOM/HTML 撑爆 prompt，同时保留截图供人工排查。
"""

import base64
import hashlib
import logging
import re

from playwright.async_api import Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError

from agent.config import AgentConfig
from agent.exceptions import BrowserError
from agent.tracer import TraceLogger
from agent.types import Element, ObserveResult

logger = logging.getLogger(__name__)

# 提取可见文本：过滤 script/style/noscript 等不可见节点，
# 按文档流顺序拼接可见文本节点，交给 Python 侧再做长度截断。
_EXTRACT_TEXT_JS = r"""
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
    // 这里把 <table> 单独摘出来，按行序列化成 Markdown 风格的
    // "| cell1 | cell2 |"，每一行的列边界都显式标出，不再需要模型自己
    // 去猜"这个数字属于哪一行哪一列"。
    const isTableDescendant = (el) => !!el.closest("table");
    // 根据常见的 class/id（如 banner、alert、toast）识别提示区域，
    // 避免仅依赖模型从页面文本中判断状态信息。
    // 沿祖先链查找，因为提示文本可能包裹在没有 class 的子元素中。
    const _HINT_CLASS_RE = /banner|result|alert|toast|notice|message|msg|status|tip|feedback/i;
    const isHintContainer = (el) => {
        let node = el;
        while (node && node !== document.body) {
            const cls = typeof node.className === "string" ? node.className : "";
            const id = node.id || "";
            if (_HINT_CLASS_RE.test(cls) || _HINT_CLASS_RE.test(id)) {
                return true;
            }
            node = node.parentElement;
        }
        return false;
    };
    const serializeTable = (table) => {
        const rows = Array.from(table.querySelectorAll("tr")).filter(isVisible);
        const lines = rows.map((tr) => {
            const cells = Array.from(tr.querySelectorAll("th, td")).filter(isVisible);
            const cellTexts = cells.map((c) =>
                (c.innerText || "").trim().replace(/\s+/g, " ")
            );
            return "| " + cellTexts.join(" | ") + " |";
        });
        return lines.join("\n");
    };
    const tables = Array.from(document.querySelectorAll("table")).filter(isVisible);
    const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        {
            acceptNode(node) {
                const parent = node.parentElement;
                if (!parent) return NodeFilter.FILTER_REJECT;
                if (skipTags.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
                if (!isVisible(parent)) return NodeFilter.FILTER_REJECT;
                if (isTableDescendant(parent)) return NodeFilter.FILTER_REJECT;
                if (!node.textContent || !node.textContent.trim()) {
                    return NodeFilter.FILTER_SKIP;
                }
                return NodeFilter.FILTER_ACCEPT;
            },
        }
    );
    // 给关键标签加最基础的结构前缀——标题用
    // "# "，按钮/tab 用 "[按钮] "，列表项用 "- "，命中"提示/结果反馈容器"
    // 命名模式的用 "[提示] "——不引入完整 DOM 树，只是让 LLM 不必再靠猜测
    // 区分"这是标题"/"这是提示"还是普通正文。
    const prefixFor = (tag, el) => {
        if (/^H[1-6]$/.test(tag)) return "# ";
        if (tag === "BUTTON") return "[按钮] ";
        if (tag === "LI") return "- ";
        if (isHintContainer(el)) return "[提示] ";
        return "";
    };
    const parts = [];
    let node;
    while ((node = walker.nextNode())) {
        const tag = node.parentElement.tagName;
        parts.push(prefixFor(tag, node.parentElement) + node.textContent.trim());
    }
    // 表格统一追加在最后，用 [表格] 标注开头，避免和上面的普通文本
    // 混在一起分不清；这里没有按表格在文档中的原始位置做精确插入
    // （对绝大多数任务场景足够，若页面有多段正文夹杂多个表格、且顺序
    // 很关键，需要进一步改成按 DOM 顺序插入）。
    tables.forEach((table) => {
        const serialized = serializeTable(table);
        if (serialized.trim()) {
            parts.push("[表格]\n" + serialized);
        }
    });
    return parts.join("\n");
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
        const tag = el.tagName.toLowerCase();
        // text= 定位只应保留给 button/a 这类靠可见文字辨识的元素；
        // 表单控件一律优先用 css=#id，其次退化到 nth-of-type。
        const isFormControl = tag === "input" || tag === "select" || tag === "textarea";
        if (!isFormControl) {
            const text = (el.innerText || "").trim();
            if (text && text.length <= 60) {
                return `text=${text}`;
            }
        }
        if (el.id) {
            return `css=#${el.id}`;
        }
        if (el.name) {
            return `css=${tag}[name='${el.name}']`;
        }
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
        // 对表单控件，必须优先看"当前实际值"，value 为空时才退化到
        // placeholder / aria-label 作为提示信息展示。
        const tag = el.tagName.toLowerCase();
        if (tag === "input" || tag === "textarea") {
            const value = el.value;
            if (value) return String(value);
            const placeholder = el.getAttribute("placeholder");
            if (placeholder) return placeholder;
            const aria = el.getAttribute("aria-label");
            if (aria) return aria;
            return "";
        }
        if (tag === "select") {
            const selected = el.selectedOptions && el.selectedOptions[0];
            if (selected) return selected.text.trim();
            return "";
        }
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

    // 新增：<a> 标签的 href 属性。textContent/innerText 采集不到这个
    // 不可见属性，此前 Planner/Extractor 都看不到链接指向哪里，只能靠 LLM 幻觉编造。
    const hrefFor = (el) => {
        if (el.tagName.toLowerCase() !== "a") return null;
        const href = el.getAttribute("href");
        return href === null ? null : href;
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
            const href = hrefFor(el);
            // 对 <a> 元素额外把 href 并入去重 key，文字相同但目标不同的链接不再被误判为重复。
            const dedupeKey = href !== null ? `${selector}::${href}` : selector;
            if (seen.has(dedupeKey)) continue;
            seen.add(dedupeKey);
            results.push({
                role: roleFor(el),
                name: nameFor(el),
                selector: selector,
                href: href,
            });
        }
        if (results.length >= maxElements) break;
    }
    return results.slice(0, maxElements);
}
"""


class BrowserStateObserver:
    """从 Playwright Page 提取结构化观察结果（不含完整 HTML）。"""

    def __init__(
        self, config: AgentConfig, tracer: TraceLogger, vision: bool = False
    ) -> None:
        self.config = config
        self.tracer = tracer
        # VISION-001：vision=True 时 observe() 额外采集一张 JPEG 截图并
        # base64 编码进 ObserveResult。默认 False，不改变现有纯文本观察行为
        self.vision = vision

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
        except PlaywrightTimeoutError as exc:
            # networkidle 超时不应直接判为致命错误（长轮询/SSE 页面永远不会 idle），
            # 降级继续采集，仅记录一条 warning 供排查，不中断流程。
            logger.warning(
                "等待 networkidle 超时（timeout_ms=%d），降级继续采集当前 DOM 状态: %s",
                self.config.browser_timeout,
                BrowserError(
                    "等待 networkidle 超时，降级继续采集当前 DOM 状态",
                    action="wait_for_load_state",
                    timeout_ms=self.config.browser_timeout,
                ),
                exc_info=exc,
            )

        # networkidle 只保证"没有活跃网络请求"，不保证 JS 渲染/动画/懒加载
        # 已完成写入 DOM（常见于 SPA 首屏渲染后仍有一次 setState 补数据）。
        # 额外等待固定 500ms 作为廉价的兜底，牺牲少量延迟换取更稳定的采集结果；
        # 若失败（页面已关闭等）不影响主流程，静默忽略即可。
        try:
            await page.wait_for_timeout(500)
        except PlaywrightError:
            # 页面已关闭等场景下静默忽略，不影响主流程
            pass

        url = page.url
        title = await page.title()

        raw_text: str = await page.evaluate(_EXTRACT_TEXT_JS)
        # 必须在截断之前对完整 raw_text 求 hash：visible_text_summary 会被
        # 截到 obs_text_limit（默认 3000）字符，如果拿截断后的文本去判断
        # "页面是否变化"（见 agent_controller._extract_page_key），两次
        # observe 只要前 3000 字符恰好相同，截断之后发生的真实变化
        # （比如表格追加了新行）就会被吞掉，出现假阳性。text_hash 覆盖
        # 完整文本，不受截断影响。
        text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        visible_text_summary = self._truncate_text(raw_text, self.config.obs_text_limit)

        raw_elements: list[dict] = await page.evaluate(
            _EXTRACT_ELEMENTS_JS, self.config.obs_max_elements
        )
        interactive_elements: list[Element] = [
            Element(
                role=item.get("role", ""),
                name=item.get("name", ""),
                selector=item.get("selector", ""),
                href=item.get("href"),
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

        result = ObserveResult(
            url=url,
            title=title,
            visible_text_summary=visible_text_summary,
            text_hash=text_hash,
            interactive_elements=interactive_elements,
            screenshot_path=screenshot_path,
        )

        if self.vision:
            # 单独用 JPEG（quality=80）而非落盘的 PNG trace 截图：JPEG 体积
            # 明显更小，risk 备注里 +500~2000 token 的估算就是按 JPEG 口径算的，
            # 复用 PNG trace 截图会让费用进一步上涨。
            # 这次采集失败不应该拖垮整步观察——文本侧数据已经拿到了，
            # 这里只记录 warning，把 screenshot_b64 显式置 None，交给
            # 下游 Planner/ActionSelector 按“没有图像”退化为纯文本请求。
            try:
                screenshot_bytes = await page.screenshot(type="jpeg", quality=80)
                result["screenshot_b64"] = base64.b64encode(screenshot_bytes).decode()
            except Exception as exc:
                logger.warning("vision 截图采集失败，本步退化为纯文本观察: %s", exc)
                result["screenshot_b64"] = None

        return result

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        """折叠行内多余空白，保留换行分隔符，再按字符数截断到 limit。"""
        # 只折叠横向空白（空格/Tab），保留换行，让 LLM 能感知列表项边界
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)  # 合并连续空行为单个换行
        return text.strip()[:limit]
