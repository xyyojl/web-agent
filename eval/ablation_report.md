# 消融实验报告 — DOM-only vs DOM+Vision

> **可复核性声明**：本报告的所有数值均由 [eval/artifacts/ablation-20260723/ablation_results.json](artifacts/ablation-20260723/ablation_results.json) 中的 `groups.<group>.cases[].runs[]` 原始记录重新计算得出；若与 `summary.md` 不一致，以 JSON 为准。报告中的 case 集与 artifact 的 `included_case_ids` / `excluded_case_ids` 显式对应。
>
> 配置溯源（模型、prompt fingerprint、采样参数和 git commit 等）记录在该 JSON 顶层字段中。

## Artifact 信息

- **artifact 目录**：[eval/artifacts/ablation-20260723/](artifacts/ablation-20260723)
- **生成时间**（`generated_at`）：`2026-07-23T09:49:48.578418+00:00`
- **模型**（`model`）：`sensenova-6.7-flash-lite`
- **git commit**：`3c390207409422cf529b79cc217211e8ccefbd88`
- **prompt_fingerprint**：`15dc8e5d2a40dd80b7b0c1eb267e9e4acdaeb61c5fd9595063a5a0eb25f5906e`
- **run_count_per_group**：`3`

> **模型快照说明**：记录的是供应商别名，供应商可能对别名指向的底层模型做未公示更新，历史 artifact 的可复现性受此限制。

## 实验设计

- 任务集：本地 case；`included_case_ids` 为 L01–L10，共 10 条，参与成功率与步数统计。
- 排除项：`excluded_case_ids` 为 L11。顶层 `excluded_case_reasons` 将其标记为安全回归 case，说明其不参与 Vision 效率均值；这是预先配置的排除策略，而非本次运行结果。本次两组各 3 条 L11 `runs[]` 均为 `success=true`（合计 6/6），未发生安全拦截或失败；L11 仍不参与聚合指标。
- 控制变量：相同任务、页面与基础 `AgentConfig`；两组唯一的实验变量是 `vision`（DOM-only 为 `false`，DOM+Vision 为 `true`）。
- 重复次数：每组每个 case 运行 3 次。因此每组纳入统计的运行数为 10 × 3 = 30 次；两组合计 60 次。

## 结果对比

下表均由每个纳入 case 的 `runs[]` 重新计算。成功率分母 = `included_case_ids` 数量（10）× `run_count_per_group`（3）= 30；平均步数分母为这 30 条运行记录（两组每条记录均有非空 `steps`）。差异 = DOM+Vision − DOM-only。

| 指标 | DOM-only | DOM+Vision | 差异 |
|------|----------|------------|------|
| task_success_rate | 29/30（96.7%） | 30/30（100.0%） | +1/30（+3.3 个百分点） |
| avg_steps | 2.27 | 2.17 | -0.10 |

逐 case 步数同样取各自 `runs[].steps` 的算术均值；差值 = Vision 均值 − DOM-only 均值。L11 已排除，未列入此表。

| case_id | DOM-only steps（3 runs） | DOM-only 均值 | DOM+Vision steps（3 runs） | DOM+Vision 均值 | 差值 |
|---------|---------------------------|---------------|-----------------------------|-----------------|------|
| L01 | [1, 1, 1] | 1.00 | [1, 2, 1] | 1.33 | +0.33 |
| L02 | [2, 1, 1] | 1.33 | [1, 1, 1] | 1.00 | -0.33 |
| L03 | [4, 4, 2] | 3.33 | [3, 2, 2] | 2.33 | -1.00 |
| L04 | [2, 2, 2] | 2.00 | [2, 2, 2] | 2.00 | 0.00 |
| L05 | [4, 5, 4] | 4.33 | [4, 4, 4] | 4.00 | -0.33 |
| L06 | [3, 3, 3] | 3.00 | [3, 3, 3] | 3.00 | 0.00 |
| L07 | [2, 2, 2] | 2.00 | [2, 2, 2] | 2.00 | 0.00 |
| L08 | [2, 2, 1] | 1.67 | [2, 2, 2] | 2.00 | +0.33 |
| L09 | [2, 2, 2] | 2.00 | [2, 2, 2] | 2.00 | 0.00 |
| L10 | [2, 2, 2] | 2.00 | [2, 2, 2] | 2.00 | 0.00 |

## 分析

**效率方向与上次结果一致，但幅度明显更小。** DOM+Vision 的平均步数从 2.27 降至 2.17，绝对减少 0.10 步；以 DOM-only 为基准的相对降幅为 `(2.27 - 2.17) / 2.27 = 4.4%`（用四舍五入均值 2.2667 与 2.1667 计算）。主要正向差异来自 L03（-1.00 步），其次是 L02 和 L05（各 -0.33 步）；L01 与 L08 则各增加 0.33 步，说明收益并非在每个任务上稳定出现。

**成功率出现了一条分歧，但不宜直接归因于 Vision 提升准确率。** 差异 case 是 L09 的第 3 轮：DOM-only 为失败、Vision 为成功。原始 `verify_result.reason` 将 DOM-only 失败记录为 LLM Judge 连续 3 次输出缺少/类型不符的字段（返回内容只有 `success` 与 `confidence`），即判题输出格式失败；同一运行记录的 `steps` 仍为 2。该证据更符合判题器格式抖动，而不能单独证明 DOM-only 未完成页面任务。因此，表中的 29/30 与 30/30 是对 artifact 验证结果的准确报告，但这 1 次差异不应被解读为 Vision 已被证明带来 3.3 个百分点的任务能力提升。

**其余 case 的步数对比没有显示普遍性的稳定优势。** L04、L06、L07、L09、L10 两组均值相同；L01 和 L08 的方向对 Vision 不利。结合仅 3 次重复和 L09 的判题格式异常，本轮可支持的结论是：在这 10 条纳入任务上，Vision 的平均执行步数较低，但幅度为 4.4%，且需要在更大样本或更稳定的 verifier 下继续验证。

## 结论

在本地 10 项纳入任务上（每组 3 次、共 60 次运行），DOM+Vision 的 artifact 验证成功率为 30/30，DOM-only 为 29/30；其中唯一差异 L09 来自 DOM-only 第 3 轮的 LLM Judge 输出格式失败，不应过度归因于模型能力。DOM+Vision 的平均步数为 2.17，低于 DOM-only 的 2.27，按未四舍五入数据计算相对降幅约 4.4%。因此，本轮结果与上次“Vision 主要影响执行效率”的方向一致，但证据强度更保守。

## 复现说明

复现同类消融实验：

```bash
# 需先启动本地静态服务
uv run python -m http.server 8080 --directory eval/pages

# 每组每个 case 运行 3 次，并写入新的 artifact 目录
uv run python eval/run_ablation.py --suite local \
  --artifact-dir eval/artifacts/ablation-<YYYYMMDD> \
  --run-count 3
```

- `--artifact-dir`：指定后生成可提交的 `ablation_results.json` 与 `summary.md`；不传则仅写本地忽略的 `eval/ablation_results.json`。
- `--run-count <n>`：每组每个 case 运行 n 次（默认 1）；多次结果记录在每个 case 的 `runs[]`，聚合指标应从其原始字段重新计算。
- `--exclude-from-avg <case-ids>`：指定运行但不参与聚合指标的 case；当前默认排除 L11。

本报告对应的 artifact 是 `eval/artifacts/ablation-20260723/`，其顶层 `model_pinning_caveat` 说明模型别名可能被供应商更新，历史结果的可复现性受此限制。
