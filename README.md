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
*   **L5 记录层 · TraceLogger**：每步强制写入 `trace.jsonl`（含 Step / URL / Reason / Action / ToolResult）+ 截图 `step-N.png`，失败步骤尤其详尽，确保执行轨迹 100% 可复现可审计。
*   **L6 评测层 · Verifier**：独立于主循环，对任务结果做评测。支持 exact / contains / json_schema / llm_judge 四种模式按 case 指定，输出 `VerifyResult` 并汇总为 `eval_summary.md`。

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

### 3. 运行任务评测
通过评测套件，您可以一键运行本地和公开任务评测，系统会自动统计成功率并在 `eval/` 目录生成 `eval_summary.md` 汇总报告：
```bash
uv run python eval/run_eval.py --suite local     # 本地 10 条
uv run python eval/run_eval.py --suite public    # 公开 5 条
uv run python eval/run_eval.py --suite all       # 全部
```

> ⚠️ **`--suite local` 需要先起本地静态服务**：本地 10 条 case 的目标页面是 `http://localhost:8080/*.html`，对应 `eval/pages/` 下的静态 HTML 集。跑 `--suite local`（或 `all`）前需另开一个终端起服务，否则所有 case 会因连接超时而集体判负：
>
> ```bash
> uv run python -m http.server 8080 --directory eval/pages   # 标准库自带，零额外依赖
> ```
>
> `--suite public` 的目标页面是公网真实网站，不依赖此服务。

---

## 📊 Eval 评估结果

项目内置了自动化 Verifier 与指标评估机制，在运行评测后会在 `eval/` 目录生成直观的 `eval_summary.md` 报告。以下为最近一次 `--suite all` 的实测结果（2026-07-09）：

| 指标 | 本地任务 | 公开网页 | 目标值 |
| :--- | :--- | :--- | :--- |
| 任务成功率 (task_success_rate) | 10/10 | 5/5 | ≥ 70% |
| 步骤成功率 (step_success_rate) | 100% | 100% | — |
| 平均执行步数 (avg_steps) | 2.6 | 1.2 | — |
| 自恢复率 (recovery_rate) | 0/0 | 0/0 | — |
| 反安全动作拦截率 (unsafe_action_block_rate) | 0/0 | 0/0 | — |
| 证据完整性 (evidence_completeness) | 10/10 | 5/5 | — |

运行过程中的 `trace.jsonl`、`report.json` 以及每一步的视觉快照都将被安全持久化在 `traces/run-<timestamp>/` 下，确保执行轨迹 100% 可复现、可审计。

> 💡 **消融实验**：在本地 10 项任务上对照 DOM-only vs DOM+Vision 两组，任务成功率打平（10/10），平均步数减少 20%（2.5 → 2.0，以 DOM-only 为基准），差异出现在两类任务上——需要确认页面状态变化的（标签页导航、表单填写）和纯文本/结构化抽取的（文本查找、表格抽取），说明视觉信号主要提升的是多模态 Agent 的**执行效率**而非**准确率**。完整报告含逐 case 步数对比与分析：[ablation_report.md](eval/ablation_report.md)

---

## 🛠 技术栈

*   **Core LLM**: Anthropic Claude (claude-sonnet-4-6)
*   **Automation**: Playwright (Python)
*   **Environment**: Python 3.11+, uv, dotenv
*   **Testing & Mocking**: pytest, pytest-asyncio
*   **UI/CLI**: Rich (Terminal Formatting)
