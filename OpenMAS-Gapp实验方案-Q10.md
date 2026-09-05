# OpenMAS-Gapp 实验方案 - Q10

## 任务定义

Q10 为纯金融场景的 Graph Harness case study，核心问题是：

> Graph Harness 能否面向金融分析构建专用 MAS application，并把文件/表格证据、数值分析、风险评估、合规审查、审计链路和最终报告组织到同一张图里？

本实验只使用金融数据集，不引入医学数据。

## 实验目标

1. 验证 Graph Harness 是否能把金融 row 直接转成领域化的 Requirement Model、Blueprint 和 Executable MAS。
2. 验证系统是否能同时组织 filing/table evidence、quantitative analysis、risk assessment、compliance review 和 audit trail。
3. 验证输出是否具备可追溯性、结构完整性和图式一致性。

## 数据集

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| FinanceBench | 财务/SEC | 高 | SEC filing 证据、财务事实抽取、证据对齐、风险与披露约束 | Accuracy / evidence match | 150（HF 样本）；全量 10,231 | CC BY-NC 4.0（需核验） | https://huggingface.co/datasets/PatronusAI/financebench |
| FinQA | 财务推理 | 中-高 | 表格推理、数值计算、单位处理、程序化审计 | Accuracy | 1,147 | CC BY 4.0 | https://huggingface.co/datasets/ibm-research/finqa |

## 输入输出约定

### 输入

每条 row 统一包含：

- `id`
- `question`
- `answer`
- `context`
- `choices`
- `source`
- `raw`

FinanceBench 额外保留 filing evidence snippets。FinQA 额外保留 table / narrative / program metadata。

### 输出

Q10 记录以下产物：

- `Requirement Model`
- `Application Blueprint`
- `Executable MAS`
- `execution trace`
- `answer score`
- `source / hash audit`

## 工作流程

1. 先冻结 `q10_datasets/manifests/` 和 `q10_datasets/normalized/`。
2. 再运行 `qualify_q10_datasets.py` 检查 row contract。
3. 用 `run_q10_smoke.py` 生成确定性或显式指定模型的 smoke 结果。
4. 用 `run_q10_full.py` 生成完整评测结果。
5. 用 `render_evolution_graph.py` 和 `render_harness_graph.py` 输出 16:9 图像。
6. 汇总到 `outputs/q10_financial/` 下的 runs、figures、full_evaluations、audit 与 trace。

## Graph Harness 结构要求

Q10 的图不是线性 QA pipeline，而是有治理约束的金融 MAS application：

- filing/table 证据与 risk/disclosure 证据并行；
- financial analysis 作为汇合点；
- risk assessment 和 compliance review 独立存在；
- audit trail 在 final report 之前完成；
- compliance / risk / auditability 约束门控最终输出。

## 验证标准

- 结构验证：Blueprint 与 Executable MAS 非空，节点与边对齐。
- 路由验证：多分支证据链和控制链存在。
- 图像验证：evolution 与 harness 图为 16:9，字号更大，文字尽量留在节点或箭头内部。
- 结果验证：full evaluation 产物可复现、可追溯、可归档。

## 当前工作区位置

- 数据集：`q10_datasets/`
- 结果：`outputs/q10_financial/`
- 方案总表：`OpenMAS-Gapp实验方案.md`
- 工作说明：`Q10_WORKSPACE.md`
- 数据分析：`Q10_DATASET_ANALYSIS.md`
