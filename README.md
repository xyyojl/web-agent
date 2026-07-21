# Web Agent — 浏览器任务自动化与评测

这是一个基于 LLM 的 Web Agent，用 AgentController 统一编排六层职责单一的子组件（感知 → 推理 → 决策 → 执行 → 记录 → 评测），覆盖信息查找、页面导航、表单填写、结构化抽取四类任务，含完整 Eval 体系与失败追溯机制。

---

## 📌 架构设计与系统分层

本项目采用清晰的模块化、分层架构设计，确保各个组件职责单一、低耦合，方便进行扩展、调试和多模态升级。

```
                  ┌─────────────────────────┐
                  │     用户任务 (Task)      │
                  └────────────┬────────────┘
                               │
                               ▼
L0 编排层
                  ┌─────────────────────────┐
                  │     AgentController     │
                  └────────────┬────────────┘
                               │ (1) 观察
                               ▼
L1 感知层
                  ┌─────────────────────────┐
                  │  BrowserStateObserver   │
                  └────────────┬────────────┘
                               │
                               ▼
L2 推理层
                  ┌─────────────────────────┐
                  │       WebPlanner        │
                  └────────────┬────────────┘
                               │ (2) 规划
                               ▼
L3 决策层
                  ┌─────────────────────────┐
                  │     ActionSelector      │
                  └────────────┬────────────┘
                               │ (3) 动作决策
                               ▼
L4 执行层
                  ┌─────────────────────────┐
                  │   PlaywrightExecutor    │
                  │    + BrowserTools       │
                  └────────────┬────────────┘
                               │ (4) 每步记录
                               ▼
L5 记录层
                  ┌─────────────────────────┐
                  │      TraceLogger        │  trace.jsonl + 截图
                  └─────────────────────────┘

L6 评测层
                  ┌─────────────────────────┐
                  │       Verifier          │  独立于主循环，评测任务结果
                  └─────────────────────────┘
```

### 1. 核心层说明

系统按职责分为七层（L0~L6），每层只做一件事，层间通过 TypedDict 数据结构单向传递：

*   **L0 编排层 · AgentController**：系统唯一入口与主循环控制器。管理任务生命周期、最大重试次数、多重终止条件判定（Done / Max Steps / Max Fail / Safety Blocked），协调所有子组件，不包含业务逻辑。
*   **L1 感知层 · BrowserStateObserver**：从浏览器提取结构化页面状态——URL、Title、可见文本摘要（3000 字）、交互元素列表（≤20 条）、截图路径，包装为 `ObserveResult` 喂给 LLM。
*   **L2 推理层 · WebPlanner**：分析任务目标与当前页面状态，输出自然语言行动计划（`plan: str`），说明意图但不指定 selector。
*   **L3 决策层 · ActionSelector**：基于 Planner 的计划做 Tool Calling，输出严格格式的结构化动作 `LLMAction`（action/selector/text/value/reason）。与 L2 拆为两次独立 LLM 调用，职责分离（推理需宽松格式、决策需严格 JSON，合并会互相干扰）。
*   **L4 执行层 · PlaywrightExecutor + BrowserTools**：将 LLMAction 翻译为 Playwright 底层浏览器操作，三级 selector 降级（CSS → get_by_text → get_by_role），返回 `ToolResult`。
*   **L5 记录层 · TraceLogger**：每步强制写入 `trace.jsonl`（含 Step / URL / observation 证据 / Action / tool_output / Reason / ToolResult）+ 截图 `step-N.png`，失败步骤尤其详尽，确保执行轨迹 100% 可复现可审计。trace_schema_version=2。
*   **L6 评测层 · Verifier**：独立于主循环，对任务结果做评测。支持 exact / contains / json_schema / llm_judge / safety_block 五种模式按 case 指定，输出 `VerifyResult` 并汇总为 `eval_summary.md`。

### 2. 关键设计决策

| 决策 | 选项 | 选择 | 理由 |
|---|---|---|---|
| 观察层传什么给 LLM | 完整 HTML / 结构化摘要 | 结构化摘要 | 完整 HTML 可达 100k+ token 成本高延迟大；LLM 只需"能看到什么 + 能点什么"，摘要压到 1~2k token |
| Planner 与 Selector 合并还是拆开 | 一次 LLM 调用 / 两次独立调用 | 两次独立调用 | 推理需宽松格式、Tool Calling 需严格 JSON，合并会让 Prompt 职责混乱；拆开后各自可独立优化 |
| 安全拦截靠 LLM 还是代码 | Prompt 告知 / 工具层代码拦截 | 工具层代码拦截 | LLM Prompt 遵从非 100% 可靠，不能作安全保障；`browser_type` 命中敏感正则即抛 `SafetyError`，确定性不可绕过 |

---

## 🚀 快速开始

### 1. 环境要求
*   **Python**: `>= 3.11`
*   **包管理工具**: [uv](https://github.com/astral-sh/uv) (极速 Python 包和项目管理器)

### 2. 安装与配置
首先，克隆项目并进入根目录：
```bash
git clone https://github.com/xyyojl/web-agent.git && cd web-agent
```

通过 `uv` 自动同步并安装所有核心依赖：
```bash
uv sync
```

安装 Playwright 所需的 Chromium 浏览器：
```bash
uv run playwright install chromium
```

配置环境变量：
```bash
cp .env.example .env
```
编辑 `.env` 文件，填入您真实的 API 密钥：
```env
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

### 3. 运行单个任务（CLI）
`main.py` 提供了一个开箱即用的 CLI 入口，单次运行一个 WebAgent 任务，无需接触 eval 体系：

```bash
uv run python main.py --task "找到页面上的版本号" --url "http://localhost:8080/text_find.html"
```

运行结束后会在终端打印结果面板（任务 ID / 状态 / 步数 / 耗时 / 输出 / 失败原因 / Trace 目录 / 最后截图），执行轨迹与截图会落在 `traces/run-<timestamp>/` 下，用法与 eval 体系共用同一套 `AgentConfig` / `TraceLogger`。

常用参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--task`（必填） | 任务描述，用自然语言说明要 Agent 做什么 | — |
| `--url`（必填） | 任务起始页面 URL | — |
| `--task-id` | 可选的任务标识，写入 `report.json` 的 `task_id` 字段（如批量脚本里按 `"L03"` 这类 case id 关联单次 CLI 运行） | 不传则为 `null` |
| `--vision` | 开启视觉模态（截图随观察结果一起喂给 LLM） | 不传则沿用 `AgentConfig`/环境变量配置 |
| `--model` | 覆盖使用的模型名 | `AgentConfig.model`（`claude-sonnet-4-6`） |
| `--max-steps` | 覆盖单任务最大步数 | `AgentConfig.max_steps`（15） |
| `--max-fail` | 覆盖允许的连续失败次数 | `AgentConfig.max_fail`（3） |
| `--trace-dir` | 覆盖 trace 输出根目录 | `traces/` |
| `-v` / `--verbose` | 打印 DEBUG 级别日志 | 默认 INFO |

> 💡 未显式传入的参数会退回 `AgentConfig.from_env()`，即优先读取 `.env` 里的 `WEBAGENT_*` 环境变量（完整列表见 `.env.example`），最后才是 dataclass 默认值——命令行参数 > 环境变量 > 默认值。

更多示例：
```bash
uv run python main.py --task "..." --url "https://example.com" --vision
uv run python main.py --task "..." --url "https://example.com" --max-steps 20 --model claude-sonnet-4-6
uv run python main.py --task "..." --url "https://example.com" --task-id "L03"
```

安全拦截（`SafetyError`）或 LLM 调用彻底失败（`LLMError`）时，CLI 会打印对应错误信息并以非零退出码结束，便于接入 CI 或 shell 脚本判断成败。

### 4. 运行任务评测
通过评测套件，您可以一键运行本地和公开任务评测，系统会自动统计成功率并在 `eval/` 目录生成 `eval_summary.md` 汇总报告：
```bash
uv run python eval/run_eval.py --suite local     # 本地 11 条
uv run python eval/run_eval.py --suite public    # 公开 5 条
uv run python eval/run_eval.py --suite all       # 全部
```

生成可提交、可复核的评测 artifact：
```bash
uv run python eval/run_eval.py --suite all \
  --artifact-dir eval/artifacts/<YYYY-MM-DD>-local-public \
  --archive-case-traces L01,L03,L11
```
- `--artifact-dir`：指定后生成 `summary.md` / `results.json` / `provenance.json`，不传则保持原有 `eval_summary.md` 行为。
- `--archive-case-traces`：归档指定 case 的 `trace.jsonl`、`report.json` 和截图到 artifact 目录的 `traces/<case_id>/` 下。

> ⚠️ **`--suite local` 需要先起本地静态服务**：本地 11 条 case 的目标页面是 `http://localhost:8080/*.html`，对应 `eval/pages/` 下的静态 HTML 集。跑 `--suite local`（或 `all`）前需另开一个终端起服务，否则所有 case 会因连接超时而集体判负：
>
> ```bash
> uv run python -m http.server 8080 --directory eval/pages   # 标准库自带，零额外依赖
> ```
>
> `--suite public` 的目标页面是公网真实网站，不依赖此服务。

### 5. 运行单元测试
项目在 `tests/` 目录下提供了覆盖各分层纯逻辑与关键分支（配置解析、异常体系、Planner/Selector/Verifier/Executor 的字段校验与分发、重试骨架、Eval 指标计算等）的单元测试，全部基于 mock，不依赖真实浏览器或网络请求：
```bash
uv run pytest tests/ -q
```

需要覆盖率报告可额外加 `pytest-cov`：
```bash
uv run pytest tests/ -q --cov=agent --cov=eval --cov-report=term-missing
```

> 与 `eval/run_eval.py` 的关系：单元测试验证的是各组件的内部逻辑正确性（不启动浏览器、不调用真实 LLM）；`eval/run_eval.py` 是端到端集成评测（真实跑 Playwright + LLM），两者互补，不是替代关系。

---

## 📊 Eval 评估结果

评测结果以可提交、可复核的 artifact 形式存放在 `eval/artifacts/` 目录下。每份 artifact 包含：

- `summary.md`：人类可读的汇总报告（指标表、失败任务、基准有效性声明）
- `results.json`：每个 case 的完整结构化结果（`agent_result` / `verify_result` / `crash_reason` 等）
- `provenance.json`：运行参数、模型信息、git commit 等溯源信息
- `traces/<case_id>/`：归档的 trace 文件（仅 `--archive-case-traces` 指定的代表 case）

最新一次全量运行（`--suite all`）的 artifact：

- **artifact 目录**：[eval/artifacts/2026-07-21-local-public/](eval/artifacts/2026-07-21-local-public)
- **汇总报告**：[summary.md](eval/artifacts/2026-07-21-local-public/summary.md)（`generated_at` 见文件顶部）
- **结构化结果**：[results.json](eval/artifacts/2026-07-21-local-public/results.json)
- **溯源信息**：[provenance.json](eval/artifacts/2026-07-21-local-public/provenance.json)

> ⚠️ **基准有效性**：public suite 结果基于外部网站在 `generated_at` 时刻的实际内容，网站变化后该 artifact 不再代表当前行为，请以最新 artifact 为准。

运行过程中的 `trace.jsonl`、`report.json` 以及每一步的视觉快照都将被安全持久化在 `traces/run-<timestamp>/` 下，确保执行轨迹 100% 可复现、可审计。

> 💡 **消融实验**：在本地 10 项任务上对照 DOM-only vs DOM+Vision 两组，任务成功率打平（10/10），平均步数减少 20%（2.5 → 2.0，以 DOM-only 为基准），差异出现在两类任务上——需要确认页面状态变化的（标签页导航、表单填写）和纯文本/结构化抽取的（文本查找、表格抽取），说明视觉信号主要提升的是多模态 Agent 的**执行效率**而非**准确率**。完整报告含逐 case 步数对比与分析：[ablation_report.md](eval/ablation_report.md)

### Trace 格式说明（trace_schema_version=2）

每步以一行 JSON 追加写入 `trace.jsonl`，包含以下字段：

| 字段 | 说明 |
|---|---|
| `trace_schema_version` | Trace schema 版本号（当前为 `2`），供下游代码区分新旧格式 |
| `run_id` / `step` / `timestamp` | 运行标识、步数、UTC 时间戳 |
| `url` / `screenshot` | 当前页面 URL 与截图路径（截图为观察前快照） |
| `plan` / `action` / `selector` / `selector_level` | Planner 计划、执行动作、定位策略 |
| `reason` | 动作选择理由 |
| `success` / `page_changed` / `error_msg` | 执行结果 |
| `duration_ms` | 本步耗时 |
| `observation.title` | 页面标题 |
| `observation.text_hash` | 可见文本 hash |
| `observation.visible_text_summary` | 可见文本摘要 |
| `observation.interactive_elements` | 交互元素列表（role/name/selector/href） |
| `tool_output` | ToolResult.output（单条最大 10,000 字符，超出时截断） |
| `tool_output_truncated` | `true` 表示 tool_output 已被截断 |
| `tool_output_sha256` | 截断时记录完整内容的 SHA-256 摘要 |

**安全约束**：
- `browser_type()` 的输入文本值（如密码）**不会**写入 trace。
- `.env`、认证 token、cookie、Authorization header 不会写入 trace。
- 截图始终是**观察前**快照，不是动作后截图。

> ⚠️ **第三方数据风险提示**：`tool_output` 和 `observation` 字段可能包含被抓取页面本身携带的第三方数据（如页面上展示的他人信息）。这类内容不属于用户输入，不在当前脱敏范围内。trace 中的 `url` 字段记录了数据来源页面。归档 trace 或 artifact 前，需人工检查是否包含敏感第三方内容。

### Artifact 治理与合规说明

**仓库体积治理**：

- 单次 `--archive-case-traces` 归档的 case 数量建议不超过 5 个，且需在 PR 描述中说明归档理由。
- `eval/artifacts/` 下超过 90 天且未被 README 引用的历史 artifact，允许在后续清理性 PR 中整体删除（不视为破坏可复核性，因为 README 只应引用当前有效 artifact）。

**公开网站访问礼貌性与合规性**：

`run_eval.py` 对 public suite 的请求遵守以下基本“礼貌性”约束：

- 同一域名请求间增加最小间隔（2 秒），避免突发请求压力。
- 遵守目标站点 `robots.txt` 中与自动化访问相关的限制；若 robots.txt 明确禁止访问对应路径，该 case 将被跳过并记录 `crash_reason=robots_txt_disallowed`。
- 失败时使用有界退避重试（`browser_open` 的 `open_retry`，默认 2 次重试），不无限重试。

归档到 Git 的截图/文本仅用于内部工程可复核目的。若目标站点存在明确的 robots/ToS 限制自动化抓取或存档，对应 case 应改为不归档 trace（仅保留 pass/fail 摘要），由实现者在归档前逐 case 确认。

---

## 🛠 技术栈

*   **Core LLM**: Anthropic Claude (claude-sonnet-4-6)
*   **Automation**: Playwright (Python)
*   **Environment**: Python 3.11+, uv, dotenv
*   **Testing & Mocking**: pytest, pytest-asyncio
*   **UI/CLI**: Rich (Terminal Formatting)

---

## 📁 项目结构

```
web-agent/
├── main.py              # CLI 入口：单次运行一个任务
├── agent/                # L0~L6 各分层实现
│   ├── agent_controller.py   # L0 编排层
│   ├── observer.py           # L1 感知层
│   ├── planner.py            # L2 推理层
│   ├── action_selector.py    # L3 决策层
│   ├── executor.py            # L4 执行层（+ browser_tools.py）
│   ├── tracer.py              # L5 记录层
│   ├── verifier.py            # L6 评测层
│   ├── config.py / types.py / exceptions.py / llm_client.py / vision.py / prompts.py
├── eval/                 # Eval 体系：case 定义、评测运行器、消融实验
│   ├── cases/                 # 本地 + 公开任务 case 定义（JSON）
│   ├── pages/                 # 本地 case 对应的静态 HTML 页面
│   ├── eval_core.py            # 指标计算与 case 加载
│   ├── run_eval.py             # 评测运行器（生成 eval_summary.md）
│   └── run_ablation.py         # DOM-only vs DOM+Vision 消融实验
├── tests/                # 单元测试（pytest，mock 掉浏览器与 LLM）
├── traces/               # 运行时生成：每次任务的 trace.jsonl / report.json / 截图
├── .env.example           # 环境变量配置示例（含全部 WEBAGENT_* 可选项说明）
└── pyproject.toml
```

---

## 📄 License

本项目基于 [MIT License](LICENSE) 开源。
