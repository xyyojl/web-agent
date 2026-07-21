# 消融实验报告 — DOM-only vs DOM+Vision

> **可复核性声明**：本报告中的所有数字均来自对应的 ablation artifact（`ablation_results.json`），可从 artifact 的 `groups.<group>.cases[].runs[]` 原始数据重新计算得出。报告引用的 case 集与 artifact 的 `included_case_ids` / `excluded_case_ids` 显式对应。
>
> 生成 artifact 时使用的配置（模型、prompt fingerprint、采样参数、git commit 等）记录在 artifact 的顶层字段中，详见对应 artifact 目录下的 `ablation_results.json`。

## Artifact 信息

- **artifact 目录**：[eval/artifacts/ablation-20260721/](../eval/artifacts/ablation-20260721)
- **生成时间**（`generated_at`）：`2026-07-21T09:10:18.472445+00:00`
- **模型**（`model`）：`sensenova-6.7-flash-lite`
- **git commit**：`b7a3249db2fd90783bbe527d68d1a90b354af48a`
- **prompt_fingerprint**：`20987369637cfebca287b2d6611335f18b3ff2f11b580bcd1f7ffd11d3eb064`
- **run_count_per_group**：`3`

> **模型快照说明**：`model` 字段记录的是供应商别名，供应商可能对别名指向的底层模型做未公示更新，历史 artifact 的可复现性受此限制（详见 artifact 中的 `model_pinning_caveat` 字段）。

## 实验设计

- 任务集：本地 case（`included_case_ids` = L01~L10，共 10 条参与均值）
- 排除项：L11（安全回归 case，`verify_mode=safety_block`，其失败是安全机制预期内的主动拦截，不反映 Vision 信号对任务解题的影响，记录在 `excluded_case_ids` 中，允许运行但不参与 `avg_steps` / `task_success_rate` 均值）
- 控制变量：相同 task / 相同页面 / 相同 AgentConfig（仅 vision 字段不同）
- 唯一变量：BrowserStateObserver.vision（True / False）
- 每组运行次数：3（`run_count_per_group=3`），每组 10 cases × 3 runs = 30 次运行

> **L11 环境说明**：本次消融实验在非交互环境（后台脚本）中运行，L11 的 `sensitive_field.html` 页面会触发 `ask_human()` 登录页确认，因无交互输入导致 `EOF when reading a line`。L11 的 `fail_reason` 为 `open_failed: EOF when reading a line`，非 `SafetyError`。这不影响消融实验结论（L11 被 `excluded`），但说明 L11 的安全拦截验证需在交互环境下运行。

## 结果对比

| 指标               | DOM-only | DOM+Vision | 差异  |
|--------------------|----------|------------|-------|
| task_success_rate  | 30/30    | 30/30      | +0    |
| avg_steps          | 2.5      | 2.2        | -0.4  |
| step_success_rate  | 99%      | 100%       | +1%   |

> 表格分母（30）= `included_case_ids` 数量（10）× `run_count_per_group`（3），与 artifact 一致。L11 虽被运行但不在分母中。

两组 30 次运行的成功/失败结果完全一致，没有出现分歧的 case（即不存在某条 case 在一组成功、在另一组失败的情况）。

其余 3 项指标：`recovery_rate` DOM-only 为 1/1（30 次运行中有 1 次出现失败步但最终自愈）、Vision 为 0/0（无失败步）；`unsafe_action_block_rate` 均为 0/0（`included_case_ids` 中未触发安全拦截，L11 的记录在 `excluded_case_ids` 中不计入）；`evidence_completeness` 均为 30/30（两组数据均完整齐全，具备可比性）。

逐 case 步数对比（3 次运行的原始步数 / 均值 / 差值）：

| case_id | task_type      | DOM-only steps (3 runs) | DOM 均值 | Vision steps (3 runs) | Vision 均值 | 差值  |
|---------|----------------|--------------------------|----------|------------------------|-------------|------|
| L01     | 文本查找        | [2, 1, 1]                | 1.33     | [1, 1, 1]              | 1.00        | -0.33 |
| L02     | 文本查找        | [1, 1, 1]                | 1.00     | [2, 1, 1]              | 1.33        | +0.33 |
| L03     | 标签页导航      | [4, 6, 3]                | 4.33     | [2, 2, 2]              | 2.00        | -2.33 |
| L04     | 折叠面板        | [2, 2, 2]                | 2.00     | [2, 2, 2]              | 2.00        | 0.00  |
| L05     | 表单填写        | [4, 6, 6]                | 5.33     | [5, 4, 4]              | 4.33        | -1.00 |
| L06     | 下拉框选择      | [4, 3, 3]                | 3.33     | [3, 3, 3]              | 3.00        | -0.33 |
| L07     | 表格抽取        | [2, 2, 2]                | 2.00     | [2, 2, 2]              | 2.00        | 0.00  |
| L08     | 链接抽取        | [2, 2, 2]                | 2.00     | [2, 2, 2]              | 2.00        | 0.00  |
| L09     | 禁用按钮识别    | [2, 2, 2]                | 2.00     | [2, 2, 2]              | 2.00        | 0.00  |
| L10     | 页面跳转        | [2, 2, 2]                | 2.00     | [2, 2, 2]              | 2.00        | 0.00  |

> 原始步数数组对应 artifact 中 `groups.<group>.cases[].runs[].steps`，可逐条复核。

## 分析

**准确率维度：Vision 没有带来提升，也没有带来负面影响。** 两组的 `task_success_rate` 完全打平（30/30 = 100%），`run_count=3` 的重复运行进一步确认了这一结论的稳健性——不是单次偶然。说明在当前这批任务的难度范围内，纯 DOM 文本信号已经足以支撑模型做出正确决策，Vision 不是"能不能做对"的决定性因素。

**效率维度：Vision 在部分任务上减少了步数，但差异比单次运行更保守。** 10 条 case 里有 5 条出现步数差异，但拆开看稳定性差异很大：

- **L03（标签页导航）——最稳定且最大的差异来源**：DOM-only 均值 4.33 步（3 次运行分别为 4/6/3，波动大），Vision 稳定为 2 步（3 次完全一致）。这类任务的共同点是操作后页面发生了**状态变化**（Tab 高亮 + 内容整体切换），DOM-only 模式下模型往往需要额外一步重新观察确认"状态确实已经变化"才敢作答；Vision 模式下一张截图能同时呈现"操作前后的整体状态"，把"确认状态变化"和"读取内容"合并成一步。L03 的差异（-2.33）是所有 case 里最大的，和它涉及的状态变化最明显是吻合的。
- **L05（表单填写）——稳定的差异**：DOM-only 均值 5.33 步（3 次为 4/6/6），Vision 均值 4.33 步（3 次为 5/4/4）。同样涉及表单字段填入后的状态确认，Vision 减少了约 1 步。
- **L01、L06——小幅差异，源于 DOM-only 的偶发多步**：L01 的 DOM-only 第 1 次运行跑了 2 步（后两次都是 1 步），L06 的 DOM-only 第 1 次跑了 4 步（后两次都是 3 步），而 Vision 3 次都稳定在更少步数。这类差异更多体现为 DOM-only 偶发的"额外确认"被 Vision 消除，而非系统性差异。
- **L02——方向反转，说明单次差异不稳定**：DOM-only 3 次都是 1 步，Vision 第 1 次跑了 2 步（后两次 1 步），均值反而 DOM-only 更低。这与之前 `run_count=1` 时观察到的"Vision 省 1 步"方向相反，说明 L02 这类纯文本抽取任务的步数差异受 LLM 输出随机性影响大，不宜归因为 Vision 的系统性收益。
- **L07——之前单次的差异消失**：`run_count=1` 时 L07 出现 DOM-only 2 步 → Vision 1 步的差异，`run_count=3` 后两组都是稳定的 2 步，说明之前的差异是单次偶然。

**没有差异的 5 条 case**（L04、L07、L08、L09、L10）3 次运行步数完全一致，多是目标信息本身已经足够明确、DOM 文本一次观察就能给出高置信度答案的场景（如折叠面板展开后的纯文本、select 下拉框的确定性操作、链接抽取等），不依赖任何形式的"额外确认"，因此 Vision 没有额外收益。

**step_success_rate 的细微差异**：DOM-only 有 1 步失败（75/76 = 99%），但对应 case 最终自愈成功（`recovery_rate` = 1/1）；Vision 没有任何失败步（65/65 = 100%）。这一差异规模很小，不宜过度解读，但方向上与"Vision 减少 DOM-only 模式下的不确定决策"一致。

综合来看，`run_count=3` 的重复运行让结论比单次更保守也更可信：Vision 的系统性收益集中在**需要确认页面状态变化的任务**（L03/L05），这类任务 DOM-only 模式下模型需要额外观察确认状态变化，Vision 能把"确认"和"读取"合并成一步；而之前单次运行中观察到的 L02/L07 的差异在重复运行后消失或反转，说明不宜将所有步数差异都归因为 Vision 的收益——部分差异来自 LLM 输出的随机性。

> 提示：本轮两组均 30/30，不存在失败 case 可供对比，因此步数差异是本轮唯一能观察到的分化维度。若未来重跑（或扩大任务集难度）出现 DOM-only 失败、DOM+Vision 成功这类分歧 case，分析部分应优先讨论"Vision 是否救回了 DOM-only 的失败"，这比步数差异更能体现 Vision 的核心价值，应作为后续消融实验重点关注的问题。

## 结论

在本地 10 项 Web 自动化任务上（每组运行 3 次，共 60 次运行），DOM+Vision 与纯 DOM 模式的任务成功率打平（30/30），但平均步数从 2.53 降至 2.17（以 DOM-only 为基准，相对降幅约 14.5%）。差异主要稳定出现在需要确认页面状态变化的任务（标签页导航 L03、表单填写 L05）上，说明视觉信号主要提升的是多模态 Agent 的**执行效率**而非**准确率**。`run_count=3` 的重复运行确认了这一结论的稳健性，同时也修正了单次运行中因 LLM 输出随机性导致的过度归因（如 L02/L07 的差异在重复后消失或反转）。

## 复现说明

复现本次消融实验：

```bash
# 需先启动本地静态服务
uv run python -m http.server 8080 --directory eval/pages

# 运行消融实验（3 次重复）
uv run python eval/run_ablation.py --suite local \
  --artifact-dir eval/artifacts/ablation-<YYYYMMDD> \
  --run-count 3
```

- `--artifact-dir`：指定后生成可提交的 `ablation_results.json` + `summary.md`，不传则仅写本地忽略的 `eval/ablation_results.json`。
- `--run-count <n>`：每组每个 case 运行 n 次（默认 1），>1 时每个 case 存 `runs[]` 数组，聚合指标从原始数据重新计算。
- `--exclude-from-avg <case-ids>`：指定运行但不参与均值的 case（默认 `L11`）。

生成的 artifact 包含 `artifact_format_version` / `git_commit` / `model` / `prompt_fingerprint` / `sampling_params` / `run_count_per_group` / `included_case_ids` / `excluded_case_ids` / `excluded_case_reasons` 等溯源字段，确保结果可复核。
