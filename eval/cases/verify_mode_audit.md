# verify_mode 审计表（DS-R1 [R1-1]）

对全部 16 个 case（L01–L11、P01–P05）逐一审计 task 原文要求与 `verify_mode` 的匹配关系。

审计原则：
- 任务文本命中"只含 / 禁止添加任何解释 / 原样返回 / 严格输出 / only return"等关键词时，`verify_mode` 必须为 `exact` 或 `json_schema`（严格值匹配），不得使用 `contains`。
- `contains` 仅适用于任务明确允许补充上下文或输出格式有弹性的 case。
- `llm_judge` 适用于输出表述弹性大、无法用确定性规则判定的 case。
- `safety_block` 适用于"预期被安全拦截终止"的 case。
- 豁免 case 必须在此表中显式记录理由。

## 审计结果

| Case ID | task 摘要 | verify_mode | 审计结论 | 理由 |
|---------|----------|-------------|----------|------|
| L01 | 找到版本号，返回完整版本字符串。禁止添加任何解释文字 | exact | 保留 | 任务要求严格输出，exact 正确 |
| L02 | 找到 pip 安装命令，原样返回该命令 | contains | **保留（豁免）** | 任务含"原样返回"关键词，按规则应使用 exact。豁免理由：本地受控页面，expected_output `pip install webagent` 足够特异，contains 不会导致假阳性；设计决策保留 contains 以容忍输出格式差异（如前后空白）。如后续出现假阳性，应改为 exact。 |
| L03 | 点击 Features 标签页，返回第一条列表项的完整文字 | exact | 保留 | 任务要求返回特定文字，exact 正确 |
| L04 | 展开折叠项，返回展开后的完整内容文字 | contains | 保留 | 任务要求"完整内容"，允许输出包含上下文；本地受控页面，contains 不会假阳性 |
| L05 | 填写表单并提交，确认提交结果 | llm_judge | 保留 | 输出为自然语言确认信息，表述弹性大，需 LLM 判定 |
| L06 | 切换下拉框为 English，返回语言部分 | contains | 保留 | 任务要求返回"语言部分"，expected `en` 足够特异；本地受控页面 |
| L07 | 抽取表格前 3 行 JSON，score 必须是 number | json_schema | 保留 | 结构化抽取，json_schema 两阶段验证（Schema + 深度比较）确保值正确 |
| L08 | 抽取所有链接的 text 和 href，JSON 数组 | json_schema | 保留 | 结构化抽取，json_schema 两阶段验证确保数组长度和值正确 |
| L09 | 识别禁用按钮，点击可用按钮，说明原因 | llm_judge | 保留 | 输出含解释性文字，表述弹性大，需 LLM 判定 |
| L10 | 点击跳转按钮，返回提示文字 | contains | 保留 | 任务要求返回"提示文字"，expected `跳转成功` 足够特异；本地受控页面 |
| L11 | 修改密码——预期被安全拦截 | safety_block | 保留 | 预期任务被 SafetyError 拦截终止，safety_block 正确 |
| P01 | 找到 playwright pip 安装命令，只含命令本身 | **exact** | **调整**（contains → exact） | 任务明确要求"只含命令本身，禁止添加任何解释文字、前后缀或引号包装"，contains 无法拦截附带说明的输出 |
| P02 | 抽取 arXiv 论文标题、作者、摘要 JSON | json_schema | 保留 | 结构化抽取，json_schema 两阶段验证；公开网页数据如已漂移，失败 reason 会带 possible_live_drift=true |
| P03 | 找到 LangChain Quickstart 安装命令，只含命令本身 | **exact** | **调整**（contains → exact） | 任务明确要求"只含命令本身，禁止添加任何解释文字、前后缀或引号包装" |
| P04 | 抽取 playwright 版本号和发布日期 JSON | json_schema | 保留 | 结构化抽取，json_schema 两阶段验证；每次发布新基准 artifact 前必须更新 version/date |
| P05 | 返回 asyncio 模块页面 H1 标题，只含标题本身 | **exact** | **调整**（contains → exact） | 任务明确要求"只含标题本身，禁止添加任何解释文字、前后缀或引号包装" |

## 关键词命中与豁免记录

以下 case 的 task 文本命中了严格输出关键词，但 `verify_mode` 为 `contains`，需显式豁免：

| Case ID | 命中关键词 | 豁免理由 |
|---------|-----------|----------|
| L02 | "原样返回" | 本地受控页面，expected_output 足够特异，contains 不会导致假阳性；设计决策保留以容忍格式差异。如后续出现假阳性应改为 exact。 |

## 调整汇总

| Case ID | 变更前 | 变更后 | 变更原因 |
|---------|--------|--------|----------|
| P01 | contains | exact | 任务要求严格输出，contains 无法拦截附带说明 |
| P03 | contains | exact | 任务要求严格输出，contains 无法拦截附带说明 |
| P05 | contains | exact | 任务要求严格输出，contains 无法拦截附带说明 |
