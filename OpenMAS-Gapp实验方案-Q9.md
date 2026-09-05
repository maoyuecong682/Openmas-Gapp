# OpenMAS-Gapp 实验方案 - Q9

## 任务定义

Q9 是医学领域的 Graph Harness case study。

> Graph Harness 能否构建面向医学领域的专用 MAS application？

本实验以医学问答和医学知识推理为主，验证 Graph Harness 是否能够把医学 row 转成包含检索、临床推理、安全检查、专业审核与最终回答的专用 MAS application。

## 实验目标

1. 验证 Graph Harness 是否能把医学 row 直接转成领域化的 Requirement Model、Blueprint 和 Executable MAS。
2. 验证系统是否能够组织医学证据检索、病例推理、风险/不确定性检查与人工审核流程。
3. 验证输出是否具备可追溯性、结构完整性和 16:9 图像可读性。

## 数据集

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| MedQA | 临床医学 | 高 | USMLE 风格临床病例推理，4 选 1 | Accuracy | 1,273 | 无显式许可 | https://huggingface.co/datasets/openlifescienceai/medqa |
| MedMCQA | 医学 | 中 | 多学科医学选择题，覆盖常见临床知识 | Accuracy | ~4.2K | CC BY-SA 4.0（以原始卡片为准） | https://huggingface.co/datasets/openlifescienceai/medmcqa |
| PubMedQA | 生物医学 | 中-高 | 文献问答与 yes/no/maybe 推理 | Accuracy | 1,000（专家标注） | MIT（以原始卡片为准） | https://huggingface.co/datasets/qiaojin/PubMedQA |
| MMLU-Medical | 医学子集 | 中 | 解剖、临床、遗传、护理等医学知识推理 | Accuracy | ~1,664 | MIT | https://huggingface.co/datasets/cais/mmlu |

## 输入与输出

### 输入

每条 normalized row 至少包含：

- `id`
- `question`
- `answer`
- `context`
- `choices`
- `source`
- `raw`

### 输出

每个数据集输出：

- `Requirement Model`
- `Application Blueprint`
- `Executable MAS`
- `execution trace`
- `primary score`
- `source / hash audit`
- harness 图像（16:9 PNG）

## 工作流程

1. 冻结 `q9_datasets/raw/medical/`、`normalized/` 和 `pilot/`。
2. 运行资格检查，确认 row contract 完整。
3. 运行 smoke / harness 生成流程。
4. 输出到 `outputs/q9_medical/figures/` 与 `outputs/q9_medical/runs/`。
5. 复核图像中的节点、箭头、控制点与文字位置。

## Graph Harness 结构要求

Q9 的图不是单纯 QA pipeline，而是带约束的医学 MAS application：

- 检索与病例推理分离；
- 安全检查与不确定性检查独立存在；
- 专业审核位于最终回答之前；
- 约束节点必须门控最终输出；
- 文字尽量保持在节点或箭头内部，不溢出。

## 验证标准

- 图像为 16:9；
- 节点和边非空；
- 医学控制节点存在；
- 输出路径可复现；
- 结果可追溯到原始 row 与 hash。

## 当前工作区

- 数据集根目录：`q9_datasets/`
- 输出目录：`outputs/q9_medical/`
- 工作说明：`Q9_WORKSPACE.md`
