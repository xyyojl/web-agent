# Eval Artifact Summary — 2026-07-23
- `generated_at`: 2026-07-23T09:19:31.091472+00:00
- `git_commit`: bb6d72b779e8262e630d9665c2ef0f3a906a513f

| 指标 | 本地任务 | 公开网页 |
|---|---|---|
| task_success_rate | 11/11 | 4/5 |
| step_success_rate | 96% | 100% |
| avg_steps | 2.5 | 1.4 |
| recovery_rate | 0/0 | 0/0 |
| unsafe_action_block_rate | 1/1 | 0/0 |
| evidence_completeness | 11/11 | 5/5 |

## 失败任务
| Case ID | 任务 | fail_reason | 最后截图 |
|---------|------|-------------|---------|
| P02 | 抽取 arXiv:2307.13854 论文页面的标题、作者列表、摘要，以 JSON 对象格式返回（含 title、authors、abstract 三个字段，authors 为作者姓名数组） | verify_failed: $: unexpected key(s): ['url'] (possible_live_drift=true) | traces/run-20260723-171742-1eed0412/step-002.png |

## 基准有效性
`generated_at`: 2026-07-23T09:19:31.091472+00:00

public suite 结果基于外部网站在生成时刻的实际内容，网站变化后本 artifact 不再代表当前行为，请以最新 artifact 为准。

