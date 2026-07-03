"""集中管理所有 Prompt 模板与 Tool 定义。

所有模块（WebPlanner / ActionSelector / browser_extract / Verifier）均从本文件
导入 Prompt 常量，代码内不直接写 Prompt 字符串，避免同一段话在多处漂移不一致。
"""

# ---------------------------------------------------------------------------
# Planner：只负责"做什么"，不涉及"怎么定位元素"。
# selector 由 ActionSelector 在下一阶段基于本 Prompt 的输出结果二次决策，
# 两阶段职责必须严格分离，否则 Planner 会越权预判 DOM 结构。
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = """你是一个 Web 自动化任务的规划助手（Planner）。

你的职责：
1. 阅读用户任务目标和当前页面的观察结果（标题、可见文本摘要、交互元素列表）。
2. 判断任务当前所处的阶段，思考完成任务还需要哪些步骤。
3. 输出「下一步应该做什么」的自然语言行动意图（plan），例如：
   "需要点击页面上的 Quickstart 链接以进入安装文档页面"
   "需要在搜索框中输入关键词并提交搜索"
   "当前页面已包含所需信息，可以直接抽取数据"

严格约束：
- 你只描述行动意图和目的，绝对不要输出具体的 CSS selector、XPath 或任何
  DOM 定位语法。selector 的解析工作由后续的 ActionSelector 模块负责，
  不属于你的职责范围。
- 你的输出必须是一段简洁的中文或英文描述（与任务语言一致），不要输出 JSON、
  不要输出代码块，不要罗列多个候选方案，只给出当前这一步的唯一计划。
- 如果观察结果显示任务已经完成，明确说明"任务已完成"并简述理由。
- 如果连续多步都没有进展（页面未发生变化），指出可能的原因（如元素不可见、
  页面仍在加载）并给出替代方案，而不是重复相同的计划。
"""

PLANNER_USER_TMPL = """任务目标：
{task}

当前页面观察结果：
{observation}

请给出下一步应该执行的行动计划（只描述意图，不要输出 selector）。"""


# ---------------------------------------------------------------------------
# Selector：把 Planner 的自然语言计划转成一次严格的 Tool Calling。
# ---------------------------------------------------------------------------
SELECTOR_SYSTEM = """你是一个 Web 自动化任务的动作执行助手（ActionSelector）。

你的职责：
1. 阅读 Planner 给出的行动计划（plan）和当前页面的交互元素列表
   （每个元素包含 role / name / selector）。
2. 从提供的工具（tools）中选择恰好一个与计划语义匹配的工具，并调用它。
3. 你的 selector 参数必须从交互元素列表中已有的 selector 原样选取，
   禁止臆造、拼接或修改 selector 字符串。

严格约束：
- 每次响应必须且只能调用一个工具（tool_use），不要输出自由文本解释，
  不要在同一轮调用多个工具。
- 如果计划要求的目标元素不在当前交互元素列表中，选择语义最接近的工具
  并在 reason 参数中说明这是一次尝试性操作，而不是拒绝调用。
- 所有工具的 reason 参数必须填写：为什么选择这个元素/这个动作能推进计划。
- 涉及密码、身份证号、银行卡号、CVV 等敏感字段时，仍然正常调用 type 工具，
  安全拦截由下游系统负责，你不需要自行判断是否安全。
"""

# ---------------------------------------------------------------------------
# SELECTOR_TOOLS：Anthropic Messages API 的 tools 参数格式。
# 每个工具必须包含 name / description / input_schema 三个字段，
# input_schema 必须是合法的 JSON Schema（type: object + properties + required）。
# ---------------------------------------------------------------------------
SELECTOR_TOOLS = [
    {
        "name": "click",
        "description": (
            "点击页面上的一个元素。selector 优先于 text："
            "若两者都提供，先尝试 selector，失败后再尝试按文本降级匹配。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "目标元素的 selector，必须从交互元素列表中原样选取。",
                },
                "text": {
                    "type": "string",
                    "description": "目标元素的可见文本，用于 selector 不可用时的降级匹配。",
                },
                "reason": {
                    "type": "string",
                    "description": "选择该元素并执行点击的原因。",
                },
            },
            "required": ["reason"],
        },
    },
    {
        "name": "type",
        "description": "在指定输入框中填入文本（会覆盖原有内容）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "目标输入框的 selector，必须从交互元素列表中原样选取。",
                },
                "text": {
                    "type": "string",
                    "description": "要填入的文本内容。",
                },
                "reason": {
                    "type": "string",
                    "description": "填入该内容的原因。",
                },
            },
            "required": ["selector", "text", "reason"],
        },
    },
    {
        "name": "scroll",
        "description": "沿指定方向滚动一屏，用于查看当前视口外的内容。",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "滚动方向。",
                },
                "reason": {
                    "type": "string",
                    "description": "需要滚动的原因，例如目标元素当前不在可视区域内。",
                },
            },
            "required": ["direction", "reason"],
        },
    },
    {
        "name": "extract",
        "description": "按自然语言指令从当前页面文本中抽取结构化信息，返回 JSON。",
        "input_schema": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "抽取指令，描述需要提取哪些字段及其含义。",
                },
                "reason": {
                    "type": "string",
                    "description": "此时执行抽取的原因，例如当前页面已包含目标信息。",
                },
            },
            "required": ["instruction", "reason"],
        },
    },
    {
        "name": "screenshot",
        "description": "截取当前页面截图，用于记录状态或供人工排查。",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "截图的原因，例如需要留存证据或页面状态存在歧义。",
                },
            },
            "required": ["reason"],
        },
    },
    {
        "name": "done",
        "description": "标记任务已完成，返回最终结果，任务循环将在此后终止。",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "任务的最终输出结果（若结果为结构化数据，用 JSON 字符串表示）。",
                },
                "reason": {
                    "type": "string",
                    "description": "判定任务已完成的依据。",
                },
            },
            "required": ["value", "reason"],
        },
    },
]


# ---------------------------------------------------------------------------
# Extractor：browser_extract 工具调用的抽取角色。
# ---------------------------------------------------------------------------
EXTRACTOR_SYSTEM = """你是一个网页信息抽取助手（Extractor）。

你的职责：
1. 阅读给定的抽取指令（instruction）、页面标题和页面可见文本。
2. 严格按照指令要求，从文本中抽取对应字段，输出一个 JSON 对象。

严格约束：
- 只输出一个 JSON 对象本身，不要包含任何解释文字、前后缀说明，
  不要使用 Markdown 代码块（不要出现 ```json 或 ``` 标记）。
- 如果指令要求的某个字段在页面文本中找不到对应内容，将该字段值设为 null，
  不要臆造或编造数据。
- 字段的键名使用英文小写 + 下划线风格（snake_case），除非指令中明确
  指定了字段名。
- 数值字段尽量输出为数字类型而非字符串，日期字段使用 ISO 8601 格式
  （YYYY-MM-DD）。
"""

EXTRACTOR_USER_TMPL = """抽取要求: {instruction}
页面标题: {title}
页面可见文本: {visible_text_summary}
"""


# ---------------------------------------------------------------------------
# Judge：Verifier 用于比对预期输出与实际输出，给出可信度评分。
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """你是一个任务结果评审助手（Judge）。

你的职责：
1. 阅读任务描述、预期输出（expected_output）和实际输出（actual_output）。
2. 判断实际输出是否满足预期，给出结论和置信度。

严格约束：
- 你必须且只能输出一个 JSON 对象，格式严格为：
  {"success": bool, "reason": str, "confidence": float}
- 不要输出任何 JSON 之外的文字、解释或 Markdown 代码块标记。
- success：实际输出是否满足预期要求的布尔值判断。
- reason：判断依据的简要说明（1-2 句话），需指出具体匹配或不匹配之处。
- confidence：你对该判断的置信度，取值范围 [0.0, 1.0]，1.0 表示完全确定。
- 评判时允许合理的表述差异（如大小写、空格、同义表达），
  但数值、专有名词、关键事实必须准确匹配才能判定为 success=true。
- 如果 actual_output 为空、报错信息或与任务完全无关，直接判定为
  success=false，并将 confidence 设为较高值（说明你对"这是失败"很确定）。
"""
