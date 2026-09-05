# OpenMAS-Gapp 实验方案 - Q12

## 任务定义

Q12 是法律领域的 Graph Harness case study。

> Graph Harness 能否构建面向法律领域的专用 MAS application？

本实验以法律检索、案例分析和合规审查为主，验证 Graph Harness 是否能够处理法律知识、证据约束和专业审核流程，并构建面向法律问题的专用 MAS application。

## 实验目标

1. 验证 Graph Harness 是否能把法律 row 直接转成领域化的 Requirement Model、Blueprint 和 Executable MAS。
2. 验证系统是否能够组织判例检索、规则解释、证据分析、合规检查和专业审核。
3. 验证输出 harness 是否清楚展示法律任务链与控制门控。

## 数据集

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| CaseHOLD | 美国判例法 | 中-高 | 法律引用、要旨识别，5 选 1 | Accuracy | ~53K | 无显式许可 | https://huggingface.co/datasets/casehold/casehold |
| LegalBench | 法律 | 高（多样） | 分类、抽取、生成、蕴含等多任务法律推理 | 按任务各异 | 数万 | 任务级混合许可 | https://huggingface.co/datasets/nguha/legalbench |

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

1. 冻结 `q12_datasets/raw/legal/`、`normalized/` 和 `pilot/`。
2. 运行资格检查，确认 row contract 完整。
3. 生成 Q12 harness 记录和 PNG 图。
4. 输出到 `outputs/q12_legal/figures/` 与 `outputs/q12_legal/runs/`。
5. 复核图像中的法律检索、案例分析、规则解释和合规门控。

## Graph Harness 结构要求

Q12 的图不是单纯分类器，而是法律 MAS application：

- 判例/法条检索与事实分析分离；
- 规则解释与合规检查独立存在；
- 专业审核位于最终结论之前；
- 需要保留证据链、约束节点和审核节点；
- 文字尽量保持在节点或箭头内部，不溢出。

## 验证标准

- 图像为 16:9；
- 节点和边非空；
- 法律控制节点存在；
- 输出可复现；
- 结果可追溯到原始 row 与 hash。

## 当前工作区

- 数据集根目录：`q12_datasets/`
- 输出目录：`outputs/q12_legal/`
