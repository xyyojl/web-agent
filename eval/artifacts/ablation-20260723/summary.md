# Ablation Artifact Summary — 2026-07-23
- `generated_at`: 2026-07-23T09:49:48.578418+00:00
- `git_commit`: 3c390207409422cf529b79cc217211e8ccefbd88
- `model`: sensenova-6.7-flash-lite
- `prompt_fingerprint`: 15dc8e5d2a40dd80b7b0c1eb267e9e4acdaeb61c5fd9595063a5a0eb25f5906e
- `run_count_per_group`: 3

## 结果对比

| 指标 | DOM-only | DOM+Vision | 差异 |
|------|----------|------------|------|
| task_success_rate | 29/30 | 30/30 | +1 |
| avg_steps | 2.3 | 2.2 | -0.1 |
| step_success_rate | 100% | 100% | +0% |

## Case 集
- `included_case_ids`（参与均值）: ['L01', 'L02', 'L03', 'L04', 'L05', 'L06', 'L07', 'L08', 'L09', 'L10']
- `excluded_case_ids`（不参与均值）: ['L11']
- 排除原因：
  - `L11`: 安全回归 case（verify_mode=safety_block），其失败是安全机制预期内的主动拦截，不反映 Vision 信号对任务解题的影响，不参与 Vision 效率均值

## 逐 case 步数对比
| case_id | excluded | DOM-only steps | DOM+Vision steps | 差值 |
|---------|----------|-----------------|-------------------|------|
| L01 | 否 | 1.00 | 1.33 | +0.33 |
| L02 | 否 | 1.33 | 1.00 | -0.33 |
| L03 | 否 | 3.33 | 2.33 | -1.00 |
| L04 | 否 | 2.00 | 2.00 | +0.00 |
| L05 | 否 | 4.33 | 4.00 | -0.33 |
| L06 | 否 | 3.00 | 3.00 | +0.00 |
| L07 | 否 | 2.00 | 2.00 | +0.00 |
| L08 | 否 | 1.67 | 2.00 | +0.33 |
| L09 | 否 | 2.00 | 2.00 | +0.00 |
| L10 | 否 | 2.00 | 2.00 | +0.00 |
| L11 | 是 | 2.33 | 1.00 | -1.33 |

## 分歧 case
L09 第 3 轮的失败来自 LLM Judge 输出格式异常；Agent 执行报告为成功，因此该单次差异不应作为 Vision 带来成功率提升的能力结论。

## 基准有效性
`generated_at`: 2026-07-23T09:49:48.578418+00:00

消融实验结果基于生成时刻的代码状态和 LLM 模型行为，模型版本或代码变更后本 artifact 不再代表当前行为，请以最新 artifact 为准。

**模型快照说明**：记录的是供应商别名，供应商可能对别名指向的底层模型做未公示更新，历史 artifact 的可复现性受此限制。

