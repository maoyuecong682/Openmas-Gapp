# OpenMAS-Gapp 实验方案 - Q11

## 任务定义

Q11 是科学发现领域的 Graph Harness case study。

> Graph Harness 能否构建面向科学研究领域的专用 MAS application？

本实验以科学发现、文献分析和实验设计为主，验证 Graph Harness 是否能够组织检索、推理、模拟和验证等科研能力，并把不同类型的科学问题编排成专用 MAS application。

## 实验目标

1. 验证 Graph Harness 是否能把科学研究 row 直接转成领域化的 Requirement Model、Blueprint 和 Executable MAS。
2. 验证系统是否能够组织多跳检索、证据链推理、实验/数值验证与结果审查。
3. 验证输出 harness 是否清晰表达科研能力分工和控制流程。

## 数据集

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| HotpotQA | 多跳 / 百科 | 中-高 | 两跳桥接型与比较型问答，支持事实监督 | F1 / EM | 7,405 | CC BY-SA 4.0 | https://huggingface.co/datasets/hotpotqa/hotpotqa |
| MuSiQue | 多跳 / 可控制 | 高 | 跳数可控，适合检索-推理-验证分工 | F1 / EM | 25K–27K | CC BY 4.0 | https://huggingface.co/datasets/StonyBrookNLP/musique |
| DROP | 阅读 + 数值推理 | 中-高 | 段落阅读后的离散运算、计数和验证 | F1 / EM | 9,535（val） | CC BY-SA 4.0 | https://huggingface.co/datasets/ucinlp/drop |

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

1. 冻结 `q11_datasets/raw/scientific/`、`normalized/` 和 `pilot/`。
2. 运行资格检查，确认 row contract 完整。
3. 生成 Q11 harness 记录和 PNG 图。
4. 输出到 `outputs/q11_scientific/figures/` 与 `outputs/q11_scientific/runs/`。
5. 复核图像中的检索、推理、实验设计和验证控制点。

## Graph Harness 结构要求

Q11 的图不是单纯问答流水线，而是科学研究 MAS application：

- 多跳检索与证据合成分离；
- 实验设计或数值验证与推理分工明确；
- 验证节点门控最终结论；
- 需要保留证据链、分支资源和结果审查；
- 文字尽量保持在节点或箭头内部，不溢出。

## 验证标准

- 图像为 16:9；
- 节点和边非空；
- 任务分工清晰；
- 输出可复现；
- 结果可追溯到原始 row 与 hash。

## 当前工作区

- 数据集根目录：`q11_datasets/`
- 输出目录：`outputs/q11_scientific/`
