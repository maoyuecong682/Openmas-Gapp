# OpenMAS-Gapp实验方案

## 总体实验设计

| 实验编号 | Research Question (Q)                                        | 实验名称                                    | 主要验证内容                                                 |
| :------: | :----------------------------------------------------------- | :------------------------------------------ | :----------------------------------------------------------- |
|    Q1    | 为什么 MAS Application Construction 需要 Harness-based Orchestration？ | Harness Paradigm Comparison                 | 对比 Direct Generation、Workflow-based Construction、Component-based Assembly 与 Graph Harness 等不同 application construction paradigm，验证 Harness 作为应用级编排层的必要性 |
|    Q2    | Graph Harness 中哪些模块对 MAS application construction 发挥关键作用？ | Pipeline Component-wise Ablation            | 通过移除 Requirement Grounding、Graph-based Orchestration、Application Blueprint 和 Application Realization 等关键模块，分析各阶段对于应用构建能力的贡献 |
|    Q3    | 为什么 MAS application construction 需要 Graph-based Orchestration？ | Orchestration Representation Comparison     | 对比 Flat Component、Sequence、Workflow、Agent Graph 与 Graph Harness 等不同组织表示方式，验证 graph 显式建模组件关系对于 application orchestration 的作用 |
|    Q4    | Application Blueprint 是否能够作为有效的 MAS application intermediate representation？ | Intermediate Representation Analysis        | 对比无中间表示、Task Representation、Workflow Representation、Agent Graph 和 Application Blueprint，验证 Blueprint 在需求保持、能力组织和架构实现中的优势 |
|    Q5    | Graph Harness 是否能够适应大规模组件生态下的 application construction？ | Large-scale Component Ecosystem Analysis    | 通过增加 agent、tool、resource 和 domain module 数量，分析 Graph Harness 在复杂组件空间中的构建能力、选择准确性和扩展性能 |
|    Q6    | Graph Harness 是否能够根据需求和环境变化动态调整已有 MAS application？ | Incremental Application Reconfiguration     | 在需求变化、组件失效和约束变化等场景下，验证 Graph Harness 相比重新生成方法的增量修改能力和功能保持能力 |
|    Q7    | Graph Harness 是否能够降低 MAS application construction 的资源消耗？ | Construction Efficiency Analysis            | 从构建时间、LLM 调用次数、token 消耗、组件搜索次数和重构成本等角度分析 Graph Harness 的构建效率 |
|    Q8    | Graph Harness 在不完整和噪声应用环境下是否具有稳定性？       | Construction Robustness Analysis            | 通过需求歧义、组件元数据错误、干扰组件和约束冲突等扰动条件，评估 Graph Harness 对应用构建不确定性的鲁棒性 |
|    Q9    | Graph Harness 能否构建面向医学领域的专用 MAS application？   | Medical Application Case Study              | 以医疗辅助诊断、临床决策支持等任务为例，分析 Graph Harness 如何组织医学知识、证据检索、推理和人工审核等能力 |
|   Q10    | Graph Harness 能否构建面向金融领域的专用 MAS application？   | Financial Application Case Study            | 以金融分析、风险评估和合规审查等任务为例，分析 Graph Harness 如何处理数据分析、风险控制和监管约束等应用需求 |
|   Q11    | Graph Harness 能否构建面向科学研究领域的专用 MAS application？ | Scientific Discovery Application Case Study | 以科学发现、文献分析和实验设计等任务为例，分析 Graph Harness 如何组织检索、推理、模拟和验证等科研能力 |
|   Q12    | Graph Harness 能否构建面向法律领域的专用 MAS application？   | Legal Application Case Study                | 以法律检索、案例分析和合规审查等任务为例，分析 Graph Harness 如何处理法律知识、证据约束和专业审核流程 |

整体实验形成：

$$
Harness\ Paradigm\ Validation
\rightarrow
Pipeline\ Mechanism\ Analysis
\rightarrow
System\ Capability\ Evaluation
\rightarrow
Domain\ Application\ Construction
$$

的完整验证体系。

## Q1：Harness Paradigm Comparison

### 实验目的

验证 MAS application construction 是否需要一个面向应用层的 Harness-based orchestration 范式。现有 MAS 方法通常关注 agent 能力提升、角色设计或任务 workflow 生成，但通常假设系统结构已经被预先定义，缺少从高层应用需求出发，对 agent、工具、资源和执行流程进行统一组织的能力。因此，本实验通过保持底层模型能力、组件生态和执行环境一致，仅改变 application construction paradigm，比较不同构建范式生成 MAS application 的能力，验证 Graph Harness 作为应用级编排层的必要性。

### 实验方案

给定相同的 application requirement 和 component ecosystem，仅改变系统如何组织 MAS application 的方式，设计以下不同 construction paradigm 进行对比：

| 方法                        | Construction Paradigm                                        | 主要特点                                                     |
| :-------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Direct MAS Generation       | Requirement → MAS                                            | 直接根据用户需求生成 agent 角色和 workflow，不进行显式 application modeling |
| Plan-based Construction     | Requirement → Task Plan → MAS                                | 通过任务分解规划执行流程，但缺少组件关系和应用约束建模       |
| Component-based Assembly    | Requirement → Component Retrieval → MAS                      | 根据需求检索 agent/tool 组件并组合，但缺少显式关系编排机制   |
| Workflow-based Construction | Requirement → Workflow → MAS                                 | 使用预定义或生成 workflow 组织任务执行，但难以表达复杂能力依赖和应用约束 |
| Graph Harness (Ours)        | Requirement → Requirement Model → Graph Orchestration → Blueprint → MAS | 通过 Harness 对需求、能力、工具、流程和约束进行统一编排      |

实验过程中，所有方法共享：

- 相同的 LLM backbone；
- 相同的 agent/tool/component repository；
- 相同的 application requirement；
- 相同的 execution environment。

仅改变 application construction 过程中的组织机制。


针对生成的 MAS application，从以下方面进行评价：

| 评价指标                | 实验设计                                                     | 验证内容                            |
| :---------------------- | :----------------------------------------------------------- | :---------------------------------- |
| Requirement Coverage    | 检查生成 application 是否覆盖需求中的功能目标、能力需求和约束条件 | 验证构建结果是否真正满足用户需求    |
| Capability Completeness | 比较生成系统是否包含完成任务所需的关键能力组件               | 验证 application 级能力组织能力     |
| Architecture Validity   | 分析任务、能力、工具和约束之间的关系是否合理                 | 验证 application structure 是否有效 |
| Constraint Satisfaction | 检查领域约束、安全要求和治理规则是否被正确加入 application   | 验证复杂应用需求处理能力            |
| Execution Performance   | 在真实任务环境中运行生成的 MAS application                   | 验证最终应用完成任务的能力          |

### 实验产出

1. 不同 application construction paradigm 下的构建性能对比结果，验证直接生成、workflow 构建和组件拼接方式在 application-level construction 中的不足；

2. 分析 Graph Harness 相比传统 MAS construction 方法在需求覆盖、能力组织和约束满足方面的优势；

3. 证明 MAS application construction 需要独立的 application-level orchestration layer，而 Graph Harness 能够提供该能力。

## Q2：Pipeline Component-wise Ablation

### 实验目的

验证 Graph Harness 框架中不同阶段和关键机制对于 MAS application construction 的贡献。Graph Harness 并非简单通过组件检索或 agent 生成完成应用构建，而是通过 Requirement Grounding、Graph-based Application Orchestration 和 Application Blueprint Realization 三个阶段逐步将高层应用需求转化为可执行 MAS application。因此，本实验通过逐步移除关键模块，分析不同设计对于需求理解、能力组织、应用架构生成以及最终执行效果的影响，验证 Graph Harness 各组成部分的必要性。

### 实验方案

在完整 Graph Harness pipeline 基础上，分别移除或替换关键模块，构建不同模型变式进行对比。

| 方法                               | 移除/修改模块                                                | 验证内容                                                     |
| :--------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Full Graph Harness (Ours)          | 完整 Requirement Grounding + Graph Orchestration + Blueprint Realization pipeline | 验证完整 application construction 能力                       |
| w/o Requirement Grounding          | 移除 Application Requirement Grounding，直接根据原始需求进行 application orchestration | 验证结构化需求理解对于减少需求遗漏和提升 application alignment 的作用 |
| w/o Graph Orchestration            | 保留需求理解和组件库，但使用简单组件检索或线性组合替代 graph-based orchestration | 验证显式关系建模对于能力组织和 workflow 构建的作用           |
| w/o Blueprint                      | 移除 Application Blueprint，将 orchestration 结果直接映射为 executable MAS | 验证 application-level intermediate representation 的必要性  |
| w/o Constraint-aware Orchestration | 保留 graph orchestration，但不显式建模应用约束和治理规则     | 验证约束编排对于复杂领域 application construction 的作用     |
| w/o Realization                    | 保留 Blueprint，但使用直接 MAS generation 替代 Blueprint-preserving realization | 验证架构保持式实例化对于最终执行一致性的作用                 |

针对不同模型变式，从 application construction 和 execution 两个层面进行评价：

| 评价指标                        | 实验设计                                                | 验证内容                                   |
| :------------------------------ | :------------------------------------------------------ | :----------------------------------------- |
| Requirement Satisfaction        | 比较生成 application 对目标、任务、能力和约束的覆盖程度 | 验证 Requirement Grounding 的贡献          |
| Capability Organization Quality | 分析能力选择、依赖关系和组件组合是否合理                | 验证 Graph Orchestration 的贡献            |
| Blueprint Fidelity              | 比较应用架构设计与最终执行系统之间的一致性              | 验证 Blueprint 作为中间表示的作用          |
| Constraint Satisfaction         | 统计领域规则、安全约束和治理要求满足情况                | 验证 Constraint-aware Orchestration 的作用 |
| Execution Performance           | 在真实任务环境中运行生成 MAS application                | 验证各模块对于最终任务完成效果的影响       |

### 实验产出

1. 不同 Graph Harness 变式的 application construction 性能对比结果，分析 Requirement Grounding、Graph Orchestration、Blueprint 和 Realization 各阶段的独立贡献；

2. 验证 Graph Harness 的优势并非来自单一组件选择，而来自需求理解、关系编排和架构实现之间的协同作用；

3. 明确 application-level orchestration pipeline 中各关键模块对于构建高质量 MAS application 的作用。

## Q3：Orchestration Representation Comparison

### 实验目的

验证 Graph-based Orchestration 是否能够有效支持 MAS application construction。现有 application construction 方法通常依赖任务列表、线性 workflow 或隐式 LLM planning 组织 agent 和工具之间的关系，但真实应用中通常同时包含任务依赖、能力需求、工具调用、资源约束以及治理规则等复杂关系，单一序列或隐式表示难以完整描述应用结构。因此，本实验保持 Graph Harness 的整体 pipeline、LLM backbone 和组件生态一致，仅改变 application orchestration 阶段采用的结构化表示方式，比较不同表示范式对于 application blueprint 构建和最终 MAS application 执行效果的影响。

### 实验方案

固定：

- Application Requirement；
- Component Ecosystem；
- Requirement Grounding 模块；
- Application Realization 模块；

仅替换 Graph Harness 中 Application Orchestration 的内部表示方式，设计以下不同 orchestration paradigm：

| 方法                         | Orchestration Representation                                 | 主要特点                                                     |
| :--------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Flat Component Selection     | 使用组件列表表示可用 agent/tool，仅根据语义匹配选择组件      | 只能表示组件存在关系，缺少组件之间的结构依赖                 |
| Sequence-based Orchestration | 使用线性序列表示 application workflow                        | 能表示执行顺序，但难以表达并行、反馈和复杂依赖               |
| Tree-based Planning          | 使用任务分解树表示 application structure                     | 能表达任务层级关系，但难以描述跨层组件连接和资源依赖         |
| Workflow-based Orchestration | 使用预定义 workflow 模板组织任务执行                         | 能表达部分流程模式，但缺少动态能力组合和约束调整             |
| Agent Graph Orchestration    | 使用 agent communication graph 表示 agent 之间关系           | 能表达 agent 交互，但忽略 application-level capability 和 resource 关系 |
| Graph Harness (Ours)         | 使用 graph 显式表示 task、capability、component、resource 和 constraint 之间关系 | 支持多类型关系建模和动态 application orchestration           |

针对不同表示方式生成的 Application Blueprint，从以下方面进行评价：

| 评价指标                        | 实验设计                                                   | 验证内容                                               |
| :------------------------------ | :--------------------------------------------------------- | :----------------------------------------------------- |
| Capability Composition Accuracy | 比较 blueprint 中能力选择和组合是否满足需求                | 验证不同表示方式对于能力组织的影响                     |
| Dependency Consistency          | 检查任务、能力、工具之间的依赖关系是否正确                 | 验证复杂关系建模能力                                   |
| Workflow Validity               | 分析生成 workflow 是否存在错误顺序、不可执行路径或缺失环节 | 验证 application structure 合理性                      |
| Constraint Satisfaction         | 检查领域约束和治理规则是否正确作用于 application structure | 验证表示方式对于约束表达能力                           |
| Execution Performance           | 将不同 blueprint 实例化为 MAS application 并执行任务       | 验证 orchestration representation 对最终系统性能的影响 |

进一步设计具有不同结构复杂度的 application 场景：

| 场景                         | 实验设计                                         | 验证能力                   |
| :--------------------------- | :----------------------------------------------- | :------------------------- |
| Sequential Application       | 单路径任务流程，例如文献检索→分析→总结           | 验证基础 workflow 表达能力 |
| Multi-branch Application     | 多任务并行执行，例如数据收集、模拟和验证同时进行 | 验证复杂依赖关系表达能力   |
| Feedback-driven Application  | 存在验证反馈和循环优化，例如生成→评估→修正       | 验证动态关系建模能力       |
| Constraint-heavy Application | 存在安全、审核和权限限制，例如医疗和法律场景     | 验证约束关系表达能力       |

### 实验产出

1. 不同 orchestration representation 下 Application Blueprint 构建质量和最终 MAS application 性能对比结果；

2. 验证 Graph-based Orchestration 相比序列、树结构和隐式规划方式，能够更完整地表达 MAS application 中任务、能力、资源和约束之间的复杂关系；

3. 分析 graph 在 application construction 中的作用并证明其不仅是组件表示方式，而是支持动态应用编排的结构化组织机制。

## Q4：Intermediate Representation Analysis

### 实验目的

验证 Application Blueprint 是否能够作为 MAS application construction 过程中有效的中间表示。现有 MAS generation 方法通常直接从任务需求生成 agent 结构或 workflow，使需求理解、应用架构设计和系统实现高度耦合，导致需求约束容易丢失，同时难以对生成结果进行分析、修改和复用。因此，本实验通过比较不同 application construction intermediate representation，分析 Application Blueprint 在需求保持、能力组织、约束表达以及架构实现一致性方面的优势，验证 application-level intermediate representation 对 MAS application construction 的必要性。

### 实验方案

保持以下条件一致：

- Application Requirement；
- Harness Component Ecosystem；
- LLM backbone；
- Application Realization 模块；

仅改变 Module 2 输出的中间表示形式，设计以下不同 application construction pipeline：

| 方法                         | Intermediate Representation                                  | 主要特点                                             |
| :--------------------------- | :----------------------------------------------------------- | :--------------------------------------------------- |
| Direct Generation            | 无中间表示，直接从 Requirement 生成 MAS                      | 需求理解、架构设计和实现过程完全耦合                 |
| Task Representation          | 使用任务列表和任务依赖作为中间表示                           | 能表达应用需要完成的功能，但缺少能力、资源和约束信息 |
| Workflow Representation      | 使用 workflow graph 表示任务执行顺序                         | 能表达流程关系，但难以表示组件能力和治理约束         |
| Agent Graph Representation   | 使用 agent topology 作为中间表示                             | 能描述 agent 协作关系，但过早进入具体实现层          |
| Application Blueprint (Ours) | 使用 Function、Capability、Resource、Workflow 和 Constraint 统一表示应用架构 | 在应用需求和执行系统之间提供完整架构抽象             |

对于不同 intermediate representation 生成的 MAS application，从以下方面进行评价：

| 评价指标                           | 实验设计                                                     | 验证内容                                        |
| :--------------------------------- | :----------------------------------------------------------- | :---------------------------------------------- |
| Requirement Preservation           | 比较输入需求中的目标、功能和约束在中间表示中的保留程度       | 验证不同表示对于应用意图保持能力                |
| Capability Organization Quality    | 分析中间表示是否能够正确描述任务所需能力及其组合关系         | 验证 application-level capability modeling 能力 |
| Architecture-to-Execution Fidelity | 比较中间表示与最终 executable MAS 之间的一致性               | 验证中间表示对于系统实现的指导作用              |
| Constraint Representation          | 检查领域约束是否能够在中间表示中被显式表达并正确执行         | 验证复杂应用需求建模能力                        |
| Modification Cost                  | 对已有 application 增加新需求，比较不同表示修改所需范围和成本 | 验证中间表示的可编辑性和可维护性                |

进一步设计不同类型的 application modification 场景：

| 场景                   | 修改内容                                          | 验证能力                           |
| :--------------------- | :------------------------------------------------ | :--------------------------------- |
| Function Addition      | 增加新的 application function，例如增加验证模块   | 验证表示对于功能扩展的支持能力     |
| Capability Replacement | 替换底层实现组件，例如替换 simulation tool        | 验证表示对于实现变化的隔离能力     |
| Constraint Injection   | 增加新的运行约束，例如 human approval requirement | 验证表示对于治理规则变化的适应能力 |
| Workflow Adjustment    | 修改任务执行流程，例如增加 feedback loop          | 验证表示对于应用结构调整的支持能力 |

### 实验产出

1. 不同 intermediate representation 下 MAS application construction 性能对比结果，验证 Application Blueprint 在需求保持、能力组织和执行一致性方面的优势；

2. 分析 Application Blueprint 相比 task、workflow 和 agent-level representation 的优势，证明 MAS application construction 需要面向应用层的中间抽象；

3. 验证 Application Blueprint 不仅能够支持 MAS generation，还能够提高 application 的可解释性、可编辑性和动态调整能力。

## Q5：Large-scale Component Ecosystem Analysis

### 实验目的

验证 Graph Harness 在复杂组件生态中的 application construction 能力。真实 MAS application 构建通常面临大量可复用 agent、tool、domain module 和 external resource，而人工从庞大的组件空间中选择合适能力并设计协作关系难以扩展。因此，本实验通过逐步扩大 Harness 可用组件生态规模，分析 Graph Harness 在大规模候选组件环境下的组件选择、关系组织以及应用构建稳定性，验证其作为 application-level orchestration layer 的扩展能力。

### 实验方案

固定 application requirement 和任务目标，仅改变 Harness Component Ecosystem 的规模和复杂程度，构建不同规模的应用构建环境。

组件生态包含：

- Agent components；
- Tool resources；
- Domain modules；
- Workflow patterns。

通过增加组件数量以及引入功能相似但适用范围不同的干扰组件，模拟真实应用开发中的复杂组件生态。

| 场景                      | Component Ecosystem 规模  | 实验设计                                       | 验证内容                               |
| :------------------------ | :------------------------ | :--------------------------------------------- | :------------------------------------- |
| Small Ecosystem           | 少量核心组件（20-50）     | 提供完成任务所需的基础 agent、tool 和 resource | 验证基础 application construction 能力 |
| Medium Ecosystem          | 中等规模组件库（100-500） | 增加功能相近的候选组件和冗余模块               | 验证组件筛选和组合能力                 |
| Large Ecosystem           | 大规模组件生态（1000+）   | 引入大量不同领域、不同能力和不同约束条件的组件 | 验证复杂环境下的扩展能力               |
| Distractor-rich Ecosystem | 大规模 + 高相似干扰组件   | 添加名称相似但功能不匹配的 agent/tool          | 验证错误组件选择抵抗能力               |

针对不同组件规模下生成的 MAS application，从以下方面进行评价：

| 评价指标                      | 实验设计                                                  | 验证内容                          |
| :---------------------------- | :-------------------------------------------------------- | :-------------------------------- |
| Construction Success Rate     | 统计不同生态规模下成功生成满足需求 application 的比例     | 验证规模增加是否影响整体构建能力  |
| Capability Selection Accuracy | 比较选择组件与真实需求组件之间的匹配程度                  | 验证 Harness 的能力发现和筛选能力 |
| Architecture Validity         | 分析生成 Blueprint 中组件关系、依赖关系和执行流程是否正确 | 验证大规模组件环境下的组织能力    |
| Construction Cost             | 统计构建时间、LLM 调用次数、组件搜索数量和 token 消耗     | 分析规模增加带来的额外成本        |
| Performance Degradation       | 比较不同组件规模下最终 application 执行效果变化           | 验证系统扩展后的稳定性            |

### 实验产出

1. 不同组件生态规模下 Graph Harness 的 application construction 性能变化结果，验证其在复杂组件环境中的扩展能力；

2. 分析 Graph Harness 如何通过显式组件关系建模减少无效搜索和错误组合，验证 application-level orchestration 对大规模构建任务的价值；

3. 验证 Graph Harness 不仅能够在小规模组件库中完成 MAS 构建，同时能够适应真实应用场景中的大规模能力生态。

## Q6：Incremental Application Reconfiguration

### 实验目的

验证 Graph Harness 在动态应用环境中的增量重构能力。真实 MAS application 通常并非一次性构建完成，而是在长期运行过程中持续面对需求变化、组件失效以及约束调整等情况。传统 MAS generation 方法通常采用重新生成整个系统的方式处理变化，容易造成已有有效结构丢失和大量重复构建成本。因此，本实验模拟真实应用生命周期中的动态变化场景，比较 Graph Harness 与重新生成方法在 application modification、功能保持以及调整成本方面的差异，验证 Graph Harness 作为持续应用编排层的能力。

### 实验方案

首先基于初始 application requirement 构建 MAS application：

$$
A_0=H(R_0,E_0)
$$

随后人为引入应用变化：

$$
\Delta=(\Delta R,\Delta E,\Delta C)
$$

其中：

- $\Delta R$ 表示需求变化；
- $\Delta E$ 表示组件生态变化；
- $\Delta C$ 表示约束变化。

Graph Harness 需要根据已有 application structure 对受影响部分进行局部调整：

$$
A_1=Update(A_0,\Delta)
$$

并与以下方法进行比较：

| 方法                    | Application Update Strategy                                  | 验证内容                               |
| :---------------------- | :----------------------------------------------------------- | :------------------------------------- |
| Full Regeneration       | 检测变化后重新从需求生成完整 MAS application                 | 模拟传统 MAS generation 方法           |
| Prompt-based Revision   | 将变化需求输入 LLM，直接修改已有 MAS 描述                    | 验证隐式修改能力                       |
| Workflow-level Revision | 修改已有 workflow 节点和执行顺序                             | 验证流程级调整能力                     |
| Graph Harness (Ours)    | 基于 Application Blueprint 对受影响组件、关系和约束进行增量重构 | 验证 application-level adaptation 能力 |

设计以下动态变化场景：

| 场景                     | 实验设计                                                     | 验证能力                            |
| :----------------------- | :----------------------------------------------------------- | :---------------------------------- |
| Requirement Expansion    | 在已有 application 基础上增加新功能需求，例如增加 evidence verification 或 simulation validation | 验证新增功能的局部扩展能力          |
| Requirement Modification | 修改原有任务目标或输出要求，例如调整分析目标或增加新的评价标准 | 验证 application structure 调整能力 |
| Component Failure        | 移除已有 agent、tool 或 external resource，例如 simulation tool 不可用 | 验证组件替换和故障恢复能力          |
| Constraint Injection     | 新增安全、权限或人工审核要求，例如增加 human approval requirement | 验证约束驱动的结构调整能力          |

从以下方面评价不同方法的动态重构能力：

| 评价指标                | 实验设计                                          | 验证内容                       |
| :---------------------- | :------------------------------------------------ | :----------------------------- |
| Adaptation Success Rate | 统计变化后 application 是否满足新的需求和约束     | 验证动态调整有效性             |
| Functional Preservation | 比较修改前后未受影响功能是否保持正常              | 验证局部重构能力               |
| Modification Locality   | 统计实际修改组件和关系占整个 application 的比例   | 验证是否避免无必要的大规模重构 |
| Adaptation Cost         | 比较更新所需时间、token 消耗、调用次数和构建步骤  | 验证增量调整效率               |
| Recovery Performance    | 在组件失效情况下评价恢复后的 application 执行效果 | 验证环境变化适应能力           |

### 实验产出

1. 不同 application change scenario 下 Graph Harness 与重新生成方法的适应性能对比结果，验证其增量重构优势；

2. 分析 Graph Harness 如何利用 Application Blueprint 中显式的组件关系和约束关系，仅调整受影响部分而保持已有有效结构；

3. 验证 Graph Harness 不仅能够完成一次性 MAS application construction，还能够支持应用生命周期中的持续演化。

## Q7：Construction Efficiency Analysis

### 实验目的

评估 Graph Harness 在 MAS application construction 过程中的资源利用效率。现有 MAS 构建方法通常依赖 LLM 进行反复规划、试错和组件选择，随着应用复杂度增加，容易产生大量无效组件探索、重复 workflow 调整以及冗余生成成本。Graph Harness 通过 Application Requirement Model、Graph-based Orchestration 和 Application Blueprint 显式组织应用结构，有望减少不必要的搜索和重构过程。因此，本实验在相同 application construction 任务和组件生态条件下，对比不同方法在构建成本、规划效率和最终构建质量之间的关系，验证 Graph Harness 在应用级 MAS 构建中的效率优势。

### 实验方案

保持：

- Application Requirement；
- Component Ecosystem；
- Execution Environment；
- Target Application Quality；

一致，仅比较不同 construction paradigm 完成 application construction 所需的资源消耗。

比较以下方法：

| 方法                        | Construction Strategy                                        | 验证内容                           |
| :-------------------------- | :----------------------------------------------------------- | :--------------------------------- |
| Direct MAS Generation       | 直接根据需求生成 MAS application                             | 模拟无显式规划的构建方式           |
| Plan-based Construction     | 通过任务规划后生成 MAS                                       | 验证任务规划对于减少构建成本的作用 |
| Workflow-based Construction | 基于 workflow 模板组织 application                           | 验证固定流程方法的效率             |
| Component-based Assembly    | 检索组件后进行组合                                           | 验证组件复用方法的成本             |
| Graph Harness (Ours)        | 基于需求建模、graph orchestration 和 blueprint realization 完成 application construction | 验证结构化编排带来的效率提升       |

从以下方面评估不同方法的 construction efficiency：

| 评价指标                  | 实验设计                                          | 验证内容                         |
| :------------------------ | :------------------------------------------------ | :------------------------------- |
| Construction Time         | 统计从输入 requirement 到 executable MAS 的总耗时 | 验证整体构建效率                 |
| LLM Interaction Cost      | 统计 LLM 调用次数、token 消耗和推理轮数           | 验证 Harness 是否减少无效推理    |
| Component Search Cost     | 统计候选组件评估数量和筛选过程                    | 验证显式关系组织是否降低搜索成本 |
| Planning Iteration Number | 统计 application blueprint 或 workflow 修正次数   | 验证结构化规划是否减少试错过程   |
| Quality-Cost Trade-off    | 比较不同方法达到相同 application quality 所需成本 | 验证效率提升是否伴随性能保持     |

进一步分析不同 application complexity 下的效率变化：

| 场景                         | 实验设计                              | 验证能力                               |
| :--------------------------- | :------------------------------------ | :------------------------------------- |
| Simple Application           | 少量任务和组件的基础 application      | 验证基础构建成本                       |
| Multi-component Application  | 增加多个 agent、tool 和 resource 依赖 | 验证复杂组件组合效率                   |
| Constraint-heavy Application | 增加安全、审核和领域约束              | 验证复杂约束处理效率                   |
| Long-horizon Application     | 增加多阶段 workflow 和反馈过程        | 验证长期 application construction 效率 |

### 实验产出

1. 不同 MAS application construction 方法在时间、token 消耗、组件搜索成本和规划迭代次数方面的对比结果；

2. 分析 Graph Harness 如何通过显式 application structure modeling 减少无效搜索和重复生成，提高 construction efficiency；

3. 验证 Graph Harness 能够在保持 application construction quality 的同时，实现更优的质量—成本权衡。

## Q8：Construction Robustness Analysis

### 实验目的

评估 Graph Harness 在不确定应用构建环境下的稳定性。真实 MAS application construction 过程中，用户需求通常存在模糊性和不完整性，组件生态可能包含错误描述或冗余组件，同时应用约束可能存在冲突或动态变化。传统 MAS generation 方法通常依赖完整且准确的输入信息，一旦需求理解或组件匹配出现偏差，容易导致整体 application structure 错误。因此，本实验通过模拟不同类型的构建扰动，分析 Graph Harness 在需求不确定性、组件噪声和约束冲突条件下的性能退化情况，验证其作为 application-level orchestration layer 的鲁棒性。

### 实验方案

在保持基础 application requirement 和 component ecosystem 不变的情况下，人为引入不同类型的 construction perturbation，并比较不同方法在扰动环境下构建 MAS application 的稳定性。

比较方法：

| 方法                        | Construction Strategy                                        | 验证内容                                      |
| :-------------------------- | :----------------------------------------------------------- | :-------------------------------------------- |
| Direct MAS Generation       | 直接根据扰动后的需求生成 MAS                                 | 验证无结构化编排方法的稳定性                  |
| Workflow-based Construction | 基于任务 workflow 生成 MAS                                   | 验证固定流程方法对于输入变化的适应能力        |
| Component-based Assembly    | 基于组件匹配进行 application 构建                            | 验证组件检索方法对于组件噪声的抵抗能力        |
| Graph Harness (Ours)        | 基于 requirement grounding、graph orchestration 和 blueprint refinement 完成构建 | 验证结构化 application orchestration 的鲁棒性 |

设计以下扰动场景：

| 扰动类型                       | 实验设计                                                     | 验证内容                                      |
| :----------------------------- | :----------------------------------------------------------- | :-------------------------------------------- |
| Requirement Ambiguity          | 对 application requirement 进行模糊化处理，例如省略关键任务、使用非专业描述或存在语义歧义 | 验证 Harness 对不完整用户需求的理解和补全能力 |
| Requirement Conflict           | 在需求中加入相互冲突的目标，例如同时要求最高准确率和最低成本 | 验证 Harness 对需求冲突检测和权衡能力         |
| Component Metadata Noise       | 修改 agent/tool 的能力描述、输入输出信息或领域标签，引入错误组件信息 | 验证 Harness 对错误组件信息的抵抗能力         |
| Distractor Component Injection | 增加大量名称相似但功能不匹配的 agent/tool                    | 验证 Harness 的错误组件选择避免能力           |
| Resource Failure               | 随机移除部分 tool、database 或 external resource             | 验证 Harness 对资源不可用情况的恢复能力       |
| Constraint Noise               | 添加错误、冗余或部分冲突的应用约束                           | 验证 Harness 对复杂约束环境的处理能力         |

针对不同扰动条件，采用以下指标进行评价：

| 评价指标                      | 实验设计                                                     | 验证内容                              |
| :---------------------------- | :----------------------------------------------------------- | :------------------------------------ |
| Relative Performance Drop     | 比较扰动前后 application construction performance 的下降比例 | 验证系统对于输入扰动的敏感程度        |
| Requirement Recovery Accuracy | 在需求缺失或模糊情况下，统计正确恢复关键需求的比例           | 验证 Requirement Grounding 的鲁棒性   |
| Architecture Validity         | 分析扰动环境下生成 Blueprint 是否仍满足依赖和约束关系        | 验证 application orchestration 稳定性 |
| Constraint Violation Rate     | 统计生成 application 违反约束的比例                          | 验证安全和治理能力                    |
| Failure Recovery Rate         | 在组件失效情况下，统计成功恢复可执行 application 的比例      | 验证动态故障处理能力                  |

进一步分析不同扰动强度下系统性能变化：

| 扰动等级            | 实验设计                       | 验证能力               |
| :------------------ | :----------------------------- | :--------------------- |
| Low Perturbation    | 少量缺失或轻微错误信息         | 验证基础稳定性         |
| Medium Perturbation | 多个组件错误或部分需求缺失     | 验证复杂环境适应能力   |
| High Perturbation   | 大量噪声、冲突约束和资源不可用 | 验证极端情况下的鲁棒性 |

### 实验产出

1. 不同扰动类型和强度下 Graph Harness 与 baseline 方法的性能退化曲线，验证其面对不确定 application construction 环境时的稳定性；

2. 分析 Graph Harness 如何利用显式 application structure、component relationship 和 constraint modeling 降低错误需求和组件噪声带来的影响；

3. 验证 Graph Harness 不仅能够在理想组件生态中构建 MAS application，同时能够在真实复杂环境中保持可靠的 application orchestration 能力。

## Q9：Medical Application Case Study

### 实验目的

验证 Graph Harness 在高约束医疗领域中的 MAS application construction 能力。医疗应用通常涉及复杂知识检索、专业推理、风险评估以及人工审核等多种能力，同时需要满足证据可靠性、安全性和责任边界等严格约束。传统 MAS construction 方法通常关注任务完成，而难以同时组织领域知识、工具资源和治理流程。因此，本实验选择医疗辅助决策场景，分析 Graph Harness 如何根据医疗应用需求自动组织相关能力组件，构建满足医学实践要求的专用 MAS application。

### 实验方案

选择具有代表性的医疗 application construction 任务，输入领域需求和可用组件生态，由 Graph Harness 自动生成 Application Blueprint 和 Executable MAS Application。

实验场景包括：

| 场景                             | Application Requirement                  | 验证能力                             |
| :------------------------------- | :--------------------------------------- | :----------------------------------- |
| Clinical Decision Support        | 根据患者信息和医学证据辅助制定治疗建议   | 验证医学知识检索、推理和决策支持能力 |
| Differential Diagnosis Assistant | 根据症状、检查结果和病史生成鉴别诊断建议 | 验证多源信息整合和推理能力           |
| Medical Literature Review        | 自动检索、总结和分析医学研究证据         | 验证文献检索、证据评价和知识组织能力 |
| Treatment Safety Analysis        | 分析治疗方案风险和潜在禁忌               | 验证风险评估和安全约束处理能力       |

针对每个医疗应用需求，分析 Graph Harness 的构建过程：

| 分析内容               | 实验设计                                                     | 验证目标                 |
| :--------------------- | :----------------------------------------------------------- | :----------------------- |
| Requirement Grounding  | 展示医疗需求如何被转换为任务、能力需求和约束条件             | 验证领域需求理解能力     |
| Blueprint Construction | 分析生成的 Application Blueprint，包括功能、能力、资源和治理结构 | 验证医疗应用架构组织能力 |
| Component Organization | 分析选择的 agent、tool 和 knowledge resource                 | 验证领域组件组合能力     |
| Constraint Integration | 检查 evidence requirement、human review、safety constraint 是否被纳入 application structure | 验证高风险领域治理能力   |
| Execution Trace        | 展示生成 MAS application 的实际运行流程                      | 验证最终系统可执行性     |

同时，与通用 MAS construction 方法进行对比：

| 方法                        | 主要分析内容                                          |
| :-------------------------- | :---------------------------------------------------- |
| Direct MAS Generation       | 是否遗漏医疗专业流程和关键约束                        |
| Workflow-based Construction | 是否能够表达复杂医学能力依赖                          |
| Component-based Assembly    | 是否能够正确选择医疗相关组件                          |
| Graph Harness               | 是否能够根据医学需求形成完整 application architecture |

### 实验产出

1. 展示 Graph Harness 在医疗场景下从需求理解、应用编排到 MAS realization 的完整构建过程，验证其领域专用 application construction 能力；

2. 分析 Graph Harness 如何自动组织医学知识检索、临床推理、风险评估和人工审核等能力，证明其能够处理高约束领域需求；

3. 验证 Graph Harness 不仅能够生成通用 MAS，而能够根据具体领域实践构建满足专业要求的 domain-specific MAS application。

## Q10：Financial Application Case Study

### 实验目的

验证 Graph Harness 在金融领域复杂决策型 MAS application construction 中的能力。金融应用通常需要同时处理实时数据获取、市场分析、风险评估、策略生成以及合规审查等多个环节，不同功能模块之间存在复杂的数据依赖和业务约束。相比通用任务型 MAS，金融 application construction 不仅要求系统具备分析能力，还需要满足风险控制、信息可靠性和监管约束等要求。因此，本实验选择金融分析和决策支持场景，分析 Graph Harness 如何根据金融业务需求自动组织数据资源、分析能力、决策模块和风险治理流程，构建满足实际业务要求的专用 MAS application。

### 实验方案

选择具有代表性的金融 application construction 任务，输入金融业务需求以及可用 agent、tool 和 resource ecosystem，由 Graph Harness 自动生成 Application Blueprint 和 Executable MAS Application。

实验场景包括：

| 场景                           | Application Requirement                          | 验证能力                             |
| :----------------------------- | :----------------------------------------------- | :----------------------------------- |
| Investment Research Assistant  | 根据市场数据、企业信息和新闻资料生成投资分析报告 | 验证数据检索、信息分析和报告生成能力 |
| Risk Assessment System         | 根据企业财务信息和市场状态评估投资风险           | 验证风险建模和多因素分析能力         |
| Financial Compliance Assistant | 分析交易行为和业务流程是否满足监管要求           | 验证规则约束和合规检查能力           |
| Market Intelligence System     | 持续收集市场信息并生成趋势分析                   | 验证动态数据处理和长期监测能力       |

针对每个金融应用需求，分析 Graph Harness 的 application construction 过程：

| 分析内容                 | 实验设计                                                     | 验证目标                                   |
| :----------------------- | :----------------------------------------------------------- | :----------------------------------------- |
| Requirement Grounding    | 展示金融业务需求如何转换为任务、能力需求和约束条件           | 验证金融场景需求理解能力                   |
| Blueprint Construction   | 分析生成 Blueprint 中的数据处理、分析推理、风险控制和决策流程 | 验证金融 application architecture 组织能力 |
| Component Organization   | 分析选择的数据源、分析 agent、预测工具和规则模块             | 验证金融组件组合能力                       |
| Risk-aware Orchestration | 检查风险评估、合规检查和人工审核模块是否被正确加入 application | 验证复杂业务约束处理能力                   |
| Execution Trace          | 展示 MAS application 实际执行过程，包括数据获取、分析、风险评估和最终输出 | 验证端到端金融 application 可执行性        |

进一步，与不同 construction paradigm 进行对比分析：

| 方法                        | 主要分析内容                                                 |
| :-------------------------- | :----------------------------------------------------------- |
| Direct MAS Generation       | 是否能够正确理解金融业务流程和风险约束                       |
| Workflow-based Construction | 是否能够表达数据、分析和审核之间的复杂依赖                   |
| Component-based Assembly    | 是否能够选择适合金融场景的数据和分析组件                     |
| Graph Harness               | 是否能够形成包含数据流、分析流程和治理机制的完整金融 application |

### 实验产出

1. 展示 Graph Harness 在金融场景下从业务需求到 Application Blueprint 再到 Executable MAS 的完整构建过程，验证其面向商业决策场景的 application construction 能力；

2. 分析 Graph Harness 如何自动组织数据获取、市场分析、风险评估和合规审查等模块，证明其能够处理具有复杂业务约束的领域专用 MAS application；

3. 验证 Graph Harness 的价值不仅在于生成可执行 MAS，而在于根据金融领域需求构建满足业务流程和风险治理要求的完整应用系统。

## Q11：Scientific Discovery Application Case Study

### 实验目的

验证 Graph Harness 在开放式科学研究场景中的 MAS application construction 能力。科学研究任务通常具有目标抽象、流程非固定和工具依赖复杂等特点，系统不仅需要完成信息检索和知识总结，还需要支持假设生成、计算分析、实验设计以及结果验证等多阶段协同过程。相比传统任务型 MAS，科学发现 application 需要根据研究目标动态组织不同类型的知识资源、计算工具和推理能力。因此，本实验选择科学发现相关任务，分析 Graph Harness 如何根据研究需求自动构建包含知识获取、假设生成、计算验证和反馈优化的专用 MAS application。

### 实验方案

选择具有代表性的科学研究 application construction 任务，输入研究目标、可用科研工具和知识资源，由 Graph Harness 自动生成 Application Blueprint 和 Executable MAS Application。

实验场景包括：

| 场景                                  | Application Requirement                  | 验证能力                             |
| :------------------------------------ | :--------------------------------------- | :----------------------------------- |
| Literature-driven Discovery Assistant | 根据已有文献发现研究趋势和潜在研究方向   | 验证文献检索、知识组织和信息分析能力 |
| Scientific Hypothesis Generation      | 根据已有知识提出新的科学假设             | 验证知识推理和假设生成能力           |
| Material/Drug Discovery System        | 根据目标性质寻找候选材料或分子并进行筛选 | 验证检索、预测、模拟和验证能力       |
| Experiment Design Assistant           | 根据研究目标设计实验方案并评估可行性     | 验证规划、工具调用和实验验证能力     |

针对每个科学研究应用需求，分析 Graph Harness 的构建过程：

| 分析内容                    | 实验设计                                                     | 验证目标                            |
| :-------------------------- | :----------------------------------------------------------- | :---------------------------------- |
| Requirement Grounding       | 分析研究目标如何被转换为研究任务、能力需求和验证约束         | 验证开放式科研需求理解能力          |
| Blueprint Construction      | 展示生成 Blueprint 中知识获取、推理、计算和验证阶段之间的组织关系 | 验证科学 workflow 编排能力          |
| Capability Organization     | 分析选择的文献数据库、推理 agent、simulation tool 和 analysis module | 验证科研资源组合能力                |
| Discovery Loop Construction | 检查是否形成 hypothesis generation、evaluation 和 refinement 的反馈闭环 | 验证开放探索型 application 构建能力 |
| Execution Trace             | 展示 MAS application 执行过程，包括知识检索、假设生成、计算分析和结果验证 | 验证科研 application 可执行性       |

进一步，与不同 construction paradigm 进行对比：

| 方法                        | 主要分析内容                                           |
| :-------------------------- | :----------------------------------------------------- |
| Direct MAS Generation       | 是否能够从科研目标自动形成完整 discovery workflow      |
| Workflow-based Construction | 是否能够支持开放式假设生成和反馈验证过程               |
| Component-based Assembly    | 是否能够正确组合科研工具和知识资源                     |
| Graph Harness               | 是否能够根据研究目标动态组织知识、推理、计算和验证能力 |

### 实验产出

1. 展示 Graph Harness 在科学发现任务中从研究目标、Application Blueprint 到 Executable MAS 的完整构建过程，验证其面向开放探索型任务的 application construction 能力；

2. 分析 Graph Harness 如何组织文献检索、知识推理、假设生成、模拟计算和结果验证等科研能力，证明其能够构建复杂多阶段科学研究 MAS application；

3. 验证 Graph Harness 不仅适用于固定流程任务，还能够支持具有探索性和迭代性的领域专用 MAS application 构建。

## Q12：Legal Application Case Study

### 实验目的

验证 Graph Harness 在法律领域知识密集型 MAS application construction 中的能力。法律应用通常涉及事实分析、法律条文检索、案例匹配、证据组织以及专业审核等多个环节，不同任务之间存在复杂的信息依赖和责任边界。相比一般知识问答系统，法律 MAS application 不仅需要生成分析结果，还需要保证信息来源可靠、推理过程可追溯以及最终结论符合专业审查要求。因此，本实验选择法律服务相关场景，分析 Graph Harness 如何根据法律任务需求自动组织知识检索、事实分析、规则推理和审核模块，构建满足专业要求的领域专用 MAS application。

### 实验方案

选择具有代表性的法律 application construction 任务，输入法律业务需求以及可用法律知识资源、工具和 agent 组件，由 Graph Harness 自动生成 Application Blueprint 和 Executable MAS Application。

实验场景包括：

| 场景                        | Application Requirement                        | 验证能力                             |
| :-------------------------- | :--------------------------------------------- | :----------------------------------- |
| Legal Research Assistant    | 根据法律问题检索相关法规、案例并生成分析报告   | 验证法律知识检索、案例分析和总结能力 |
| Contract Review System      | 自动分析合同条款风险并识别潜在问题             | 验证文本理解、风险识别和规则匹配能力 |
| Case Analysis Assistant     | 根据案件事实匹配相关法律规则并辅助形成分析意见 | 验证事实抽取、法律推理和证据组织能力 |
| Compliance Review Assistant | 检查企业业务流程是否满足法律和监管要求         | 验证规则检查和合规分析能力           |

针对每个法律应用需求，分析 Graph Harness 的构建过程：

| 分析内容                            | 实验设计                                                     | 验证目标                                   |
| :---------------------------------- | :----------------------------------------------------------- | :----------------------------------------- |
| Requirement Grounding               | 分析法律业务需求如何转换为任务、能力需求和专业约束           | 验证法律场景需求理解能力                   |
| Blueprint Construction              | 展示生成 Blueprint 中信息检索、事实分析、规则匹配和审核流程之间的组织关系 | 验证法律 application architecture 构建能力 |
| Knowledge and Evidence Organization | 分析选择的法律数据库、案例检索工具、规则分析模块以及证据验证组件 | 验证法律知识资源组合能力                   |
| Professional Constraint Integration | 检查引用要求、来源可靠性和人工复核机制是否被纳入 application structure | 验证专业领域约束建模能力                   |
| Execution Trace                     | 展示 MAS application 从事实输入、法律检索、规则分析到最终意见生成的执行过程 | 验证法律 application 可执行性和可追溯性    |

进一步，与不同 construction paradigm 进行对比分析：

| 方法                        | 主要分析内容                                                 |
| :-------------------------- | :----------------------------------------------------------- |
| Direct MAS Generation       | 是否能够自动形成完整法律分析流程，以及是否遗漏证据和审核环节 |
| Workflow-based Construction | 是否能够表达法律检索、推理和复核之间的复杂依赖               |
| Component-based Assembly    | 是否能够正确选择法律数据库、分析工具和规则模块               |
| Graph Harness               | 是否能够构建包含事实分析、证据组织、规则推理和专业审核的完整法律 application |

### 实验产出

1. 展示 Graph Harness 在法律场景下从业务需求、Application Blueprint 到 Executable MAS 的完整构建过程，验证其面向知识密集型领域的 application construction 能力；

2. 分析 Graph Harness 如何组织法律检索、事实分析、规则推理、证据验证和专业审核等能力，证明其能够处理具有复杂知识依赖和责任约束的领域专用 MAS application；

3. 验证 Graph Harness 能够根据不同领域实践要求动态构建专用 application architecture，而非依赖固定 agent 组合或通用 workflow。