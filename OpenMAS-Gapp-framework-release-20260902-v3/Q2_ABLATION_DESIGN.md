# Q2 组件级消融实验设计

## 1. 进入 Q2 的判断

Q1 已满足进入 Q2 的协议门槛：五种方法均能产生可评估输出；Graph Harness 在约束密集结构上具有稳定优势；优势同时出现在 Blueprint、运行有效性和 Trace contract；结果覆盖 20 个 case、3 个 seed；构建成本可控。因此 Q2 可以开始，但当前证据仍应称为受控 pilot，而不是生产环境结论。Q2 的目的正是检验 Q1 的优势是否来自明确的 pipeline 模块，而不是 case 设计或组件检索偏置。

## 2. 因果问题和公平性

Q2 只改变一个模块，其余条件固定：同一 case、同一 Harness Graph、同一 DeepSeek adapter、同一输出 schema、同一 token/调用预算、同一 seed、同一 Minimal MAR runtime 和同一 contracts。每个结果都记录 `q=Q2`、`variant` 与 `removed_module`，避免把消融结果误认为 Q1 方法名。

因果隔离协议 v2 额外要求：原始用户需求不得包含参考 pipeline 的阶段描述；`w/o Requirement Grounding` 不得调用 Full 的 ARG prompt；`w/o Blueprint` 不得接收手工或编译生成的 artifact contract。违反任一条件的旧结果不得进入主表。

Graph resource isolation v3 对 multi-branch case 增加执行约束：HotpotQA 的两个检索分支分别绑定 supporting title 对应的整篇文档；MuSiQue 按 `paragraph_support_idx` 将第一跳分配给 branch A、其余支持段落分配给 branch B。路由只使用文档索引，不使用 decomposition answer 或最终答案。Full 通过 typed `resource_requirement --uses--> component_requirement` 绑定资源；`w/o Graph Orchestration` 只能使用统一、无分支标签的完整检索 context。缺失资源绑定时执行器 fail closed。

## 3. Full 与五个消融变体

| 变体 | 唯一改变 | 预期失效指纹 |
|---|---|---|
| Full Graph Harness | 完整 grounding→graph orchestration→Blueprint→realization | 作为配对比较基准，不预设每个数据集和指标都最高 |
| w/o Requirement Grounding | 原始需求直接检索组件，不形成结构化任务/能力/约束模型 | requirement satisfaction、constraint recall 下降 |
| w/o Graph Orchestration | 保留需求模型，但只做 flat component retrieval/linear composition | relation recall 和 capability organization 下降 |
| w/o Blueprint | orchestration 结果直接映射 MAS；`blueprint_present=false` | Blueprint fidelity 记为 0，执行一致性下降 |
| w/o Constraint-aware Orchestration | ARG 仍识别 constraints，CCG 不注入 Human Gate/Evidence Check | constraint orchestration 和 runtime Trace 下降，ARG constraint recall 不变 |
| w/o Realization | 保留 Blueprint，普通 Prompt 直接生成 Agent，不做组件绑定、接口契约和执行策略编译 | blueprint-to-execution fidelity 和 runtime validity 下降 |

实现位于 `openmas_bench/ablation.py`。`w/o Blueprint` 使用不向执行器暴露的 serialization carrier 保持统一 `ConstructionResult` 接口；评估器依据 `blueprint_present=false` 将 IR fidelity 置零，同时从 direct MEG 检查约束是否落实。`w/o Realization` 走独立的 Prompt-generation 路径，禁止调用正式 `realize_blueprint()`，不查询组件绑定、不生成 artifact contract，也不复制 typed Blueprint edges。

每个消融必须满足单模块干预契约：其他上游表示仍存在，且移除模块的信息不能经 metadata、carrier 或执行器视图泄漏回运行时。验证脚本检查的是这些结构签名，而不只是 `removed_module` 标签。

## 4. 指标

- Requirement Satisfaction：任务和约束覆盖（`requirement_coverage`）。
- Capability Organization Quality：可接受关系召回（`orchestration_relation_recall`）。
- Blueprint Fidelity：Blueprint 节点被执行图引用的比例；无 Blueprint 变体为 0。
- Constraint Detection：ARG 对约束的识别（`constraint_recall`）。
- Constraint Orchestration：Blueprint 或 direct MEG 对约束的落实（`constraint_orchestration_recall` / `constraint_satisfaction`）。
- Execution Performance：Minimal MAR 的 runtime validity、能力执行、约束执行、Trace contract 和禁止组件的综合分。

报告必须同时给 overall、validation split、四种 family、mean/std、fallback/retry/cost，以及 per-case/per-seed 行，不能只报告一个总均值。主结论使用同 case、同 seed 的 `Full - ablation` 配对差值，并报告 Full win / tie / ablation win。负差值是合法结果，表示该消融在对应样本上更好，不能删除、改分或强行解释成 Full 获胜。

## 5. 确定性协议检查

运行：

```powershell
python scripts/validate_q2_ablation.py
```

脚本使用 deterministic adapter，不消耗 API，运行 20×6 个 construction 和每 case 的全部 execution tasks，并写出 `q2_ablation_deterministic.json`。检查项包括：所有 construction valid、Full 只保留 case 原有 constraints、constraint ablation 保留 ARG 检测但清空 CCG 注入、graph ablation 不含 typed relations、Blueprint ablation 明确标记，以及 Realization ablation 确实绕过编译器、组件绑定和接口契约。

## 6. 当前协议结果和解释

确定性运行已覆盖 120 个 construction runs，所有变体均可序列化和执行。失效指纹符合方向：去掉 Requirement Grounding 后 requirement satisfaction 最低；去掉 Graph Orchestration 后 relation recall 为 0；去掉 Constraint-aware Orchestration 后 constraint satisfaction 下降；去掉 Realization 后 runtime validity 为 0 且 fidelity 下降；去掉 Blueprint 后 IR fidelity 按定义为 0。

该结果只是“实现语义正确”的 sanity check，不能作为真实模型效果。当前 deterministic adapter 会触发 fallback，这是预期的协议验证行为，正式 Q2 必须用与 Q1 相同的 DeepSeek adapter、相同重试预算和相同 3 seeds 运行。

## 7. 运行顺序

1. 先通过 deterministic signature checks。
2. 用 1 case×6 variants×1 seed 做 DeepSeek connectivity/JSON/费用检查。
3. 再运行正式 20×6×3 construction，并执行所有 execution tasks。
4. 报告 mean/std、validation split、family 分组、fallback rate、token/cost、runtime validity 和 Trace contract。

Q2 只有在各变体产生可解释、非单 case 的差异时，才可以声称模块贡献；否则应回到 case contracts、组件生态和 runtime 设计进行修正。
