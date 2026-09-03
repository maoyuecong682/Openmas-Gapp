# QA 数据集汇总表

## 通用知识与推理

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| MMLU | 百科/57学科 | 中-高 | 多学科学术知识（四选一） | Accuracy | 14,042 | MIT | https://huggingface.co/datasets/cais/mmlu |
| MMLU-Pro | 百科/14学科 | 高（10选项） | 多步推理、抗猜测、跨学科整合 | Accuracy | 12,032 | MIT | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro |
| ARC (Easy + Challenge) | 科学推理 | 易/难 | 小学-初中科学多步推理 | Accuracy | 3,548（两档合计） | CC-BY-SA-4.0 | https://huggingface.co/datasets/allenai/ai2_arc |
| HellaSwag | 常识推理 | 中（已近饱和） | 对抗性常识句子补全 | Accuracy | 10,003 | MIT | https://huggingface.co/datasets/allenai/hellaswag |
| BIG-Bench Hard (BBH) | 推理 | 高 | 逻辑演绎、算术、时间推理、语义歧义（23任务） | Accuracy / EM | 每任务数百~数千 | MIT | https://github.com/suzgunmirac/BIG-Bench-Hard |
| StrategyQA | 隐式多跳推理 | 高 | 隐式多跳 yes/no，带金标准子问题分解 | Accuracy | 2,780（train 2,290/dev 490） | [未核验] | https://huggingface.co/datasets/Chiahuali/StrategyQA |
| LogiQA / LogiQA 2.0 | 逻辑 | 中-高 | 演绎/归纳/溯因逻辑（中英双语，公务员考试题） | Accuracy | 8,678（2.0 更大） | 无明确许可 | https://huggingface.co/datasets/lmguan/logiqa |
| MMLU-Redux | 百科/57学科（清洗版） | 高 | MMLU 清洗版（修复282个错误+噪声），泛化/噪声控制/角色路由 | Accuracy | 3,213 | [未核验] | https://github.com/aryopg/mmlu-redux |
| CommonsenseQA | 常识推理 | 中 | 概念联想型常识推理，5选1 | Accuracy | 12,247 | [未核验] | https://huggingface.co/datasets/tau/commonsense_qa |

---

## 多跳阅读与检索推理

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| HotpotQA | 多跳/百科 | 中-高 | 2跳桥接型/比较型问答，句级支持事实监督 | F1 / EM | 7,405 | CC BY-SA 4.0 | https://huggingface.co/datasets/hotpotqa/hotpotqa |
| MuSiQue | 多跳/可控制 | 高（2/3/4跳） | 跳数可控，支持段落+不可答对比题，适合"检索-推理-验证"分工 | F1 / EM | 25K–27K | CC BY 4.0 | https://huggingface.co/datasets/StonyBrookNLP/musique |
| DROP | 阅读+数值推理 | 中-高 | 段落阅读后做离散运算（加减/排序/计数/日期） | F1 / EM | 9,535（val） | CC BY-SA 4.0 | https://huggingface.co/datasets/ucinlp/drop |

---

## 数学推理

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| GSM8K | 小学数学 | 易-中 | 2–8步算术文字题，适合 RL rollout 训练 | Accuracy（数值匹配） | 1,319 | MIT | https://huggingface.co/datasets/openai/gsm8k |
| MATH | 竞赛数学 | 高 | 7学科符号推理（Prealgebra/Algebra/数论/几何等），`\boxed{}` 提取判分 | Accuracy | 5,000 | MIT | https://huggingface.co/datasets/hendrycks/competition_math |
| MathQA | 数学+推理程序 | 中 | 5选1 MCQ + 操作程序标注，适合"推理→程序化验证"分工 | Accuracy | 4,485 | Apache-2.0 [未核验] | https://huggingface.co/datasets/MathQA/MathQA |
| MATH-500 | 竞赛数学（子集） | 高 | MATH 500题子集，推理 RL 论文标准评测集 | Accuracy | 500 | MIT | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 |
| AIME 2024/2025 | 竞赛数学 | 极高 | 美国数学邀请赛，答案 0–999 整数可严格判分 | Accuracy | 30/年 | MAA 版权（镜像许可不一） | https://huggingface.co/datasets/di-zhang-fdu/AIME2024 |
| TheoremQA | 定理应用 | 高 | 定理选择→应用→求解（数学/物理/EE/CS/金融跨学科） | Accuracy | 800 | CC BY-NC-SA 4.0 (非商业) | https://huggingface.co/datasets/wenhuchen/TheoremQA |
| MultiArith | 算术文字题 | 中 | 经典多步算术 baseline，被 Self-Consistency/EIB/GTD 等多篇 MAS 论文使用 | Accuracy | ~600 | [未核验] | https://huggingface.co/datasets/ChilleD/MultiArith |
| SVAMP | 数学鲁棒性 | 中-高 | 对抗式改写算术文字题，测算术泛化鲁棒性 | Accuracy | ~1,000 | [未核验] | https://github.com/arkilpatel/SVAMP |
| MAWPS | 算术文字题 | 中 | 多类算术题型（加减乘除+公式归纳），泛化能力基准 | Accuracy | 3,320 | [未核验] | https://github.com/sroy9/mawps |
| ASDiv-A | 小学数学多样性 | 中 | 题型多样性（12种语言模式+6类问题结构），语言多样性泛化 | Accuracy | 2,305 | [未核验] | https://github.com/chaochun/nlu-asdiv-dataset |
| AQuA | 数学选择题 | 高 | 含推理程序的5选1数学题，选项推理+解释生成 | MCQ Accuracy | ~100,000 | [未核验] | https://github.com/google-deepmind/AQuA |
| Game-of-24 | 算术组合搜索 | 中-高 | 分支搜索+剪枝+唯一表达式验证，ToT 经典任务，适合搜索型 agent | Success Rate | — | [未核验] | https://huggingface.co/datasets/test-time-compute/game-of-24 |
| Beyond-AIME | 竞赛数学扩展 | 极高 | 前沿模型不饱和的竞赛数学，Agent Q-Mix 使用 | Accuracy | — | [未核验] | https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME |
| HMMT2025 | 竞赛数学 | 极高 | 复杂组合+代数+几何推理 | Accuracy | — | [未核验] | https://huggingface.co/datasets/MathArena/hmmt_feb_2025 |

---

## 代码生成

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| HumanEval | 代码（函数级） | 中 | 函数补全（签名+docstring→实现），隐藏单元测试 | pass@k | 164 | MIT | https://huggingface.co/datasets/openai/openai_humaneval |
| MBPP | 代码（基础） | 易-中 | 简单 Python 编程（描述+3测试用例），适合 warm-up | Accuracy | 500（sanitized: 427） | CC-BY-4.0 | https://huggingface.co/datasets/google-research-datasets/mbpp |
| CodeContests | 代码（竞赛） | 高 | 竞赛编程（Codeforces等），含多解法+测试用例，带难度标签 | Accuracy（测试通过） | 165（HF版） | CC-BY-4.0 | https://huggingface.co/datasets/deepmind/code_contests |
| APPS | 代码（分档） | 易/中/难 | 编程题三档（intro/comp/interview），隐藏测试判分 | Accuracy | 10,000（全量） | MIT | https://huggingface.co/datasets/codeparrot/apps |
| BigCodeBench | 代码（第三方库） | 高 | 60+真实库（pandas/numpy等）函数级编程，agentic coding 最贴近 | Accuracy | 1,140 | Apache-2.0 | https://huggingface.co/datasets/bigcode/bigcodebench |
| HumanEval+ | 代码（强化测试） | 高 | HumanEval 每题扩至~80个测试用例（含边界），HieraMAS 使用 | pass@k | 164 | [未核验] | https://github.com/evalplus/evalplus |
| LiveCodeBench v6 | 代码（实时新题） | 高 | 近实时竞赛/LeetCode 新题，无污染 OOD 评测，Agent Q-Mix 使用 | pass@k | 持续更新 | [未核验] | https://github.com/livecodebench/livecodebench |

---

## 多智能体 / Agent 交互

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| GAIA | Agent/通用助手 | 高（L1–L3） | 多步推理+工具使用+多模态，人类92% vs GPT-4+插件15% | Accuracy / pass@1 | 466（300保密） | gated 需申请 | https://huggingface.co/datasets/gaia-benchmark/GAIA |
| AgentBench | Agent/8环境 | 异构 | OS/DB/KG/网页/具身多环境多轮交互，2025.10 FC版已集成 AgentRL | Success Rate（按环境） | 多轮约13k生成 | Apache-2.0 | https://github.com/THUDM/AgentBench |
| WebArena | Web Agent | 高 | 网页导航/点击/表单/搜索/购物长程任务，Docker 自托管 | Task Success Rate | 812 | Apache-2.0 | https://github.com/web-arena-x/webarena |
| τ³-bench | 工具-Agent-用户 | 高 | 零售/航空/银行域 API 调用+对话+规则遵循，含历史轨迹 | pass^k | retail/airline/banking | MIT | https://github.com/sierra-research/tau2-bench |
| SWE-bench Verified | 代码仓库修复 | 高 | GitHub issue→patch→自动化测试，多 agent 软件协作标准底座 | Resolved Rate | 500 | MIT（代码） | https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified |
| AgentBoard | Multi-turn Agent | 异构 | 9环境聚合评测（alfworld/webarena/babyai等），过程级 progress rate 指标 | Progress Rate, Success Rate | ~1,012 | GPL-2.0 | https://huggingface.co/datasets/hkust-nlp/agentboard |
| AgentCompany | 企业 Agent | 高 | 模拟软件公司（GitLab/RocketChat等），岗位级多 agent 协作 | Task Success Rate | 数百级 | MIT | https://github.com/TheAgentCompany/TheAgentCompany |
| **MultiAgentBench** · Research | 学术提案 | 高 | 文献研究+学术提案写作（创新性/安全性/可行性）+ 协作写作 | LLM-as-a-Judge, TS, CS | — | [未核验] | https://github.com/ulab-uiuc/MARBLE |
| **MultiAgentBench** · Minecraft | 游戏建造 | 高 | 多 agent 空间协作、环境交互、方块建造 | Block Hit Rate | — | [未核验] | https://github.com/ulab-uiuc/MARBLE |
| **MultiAgentBench** · Database | 数据库操作 | 中-高 | 多 agent 数据库查询/修改/验证 | Task Score / Pass Rate | — | [未核验] | https://github.com/ulab-uiuc/MARBLE |
| **MultiAgentBench** · Coding | 软件开发 | 高 | 多 agent 分工/调试/测试通过 | Pass Rate | — | [未核验] | https://github.com/ulab-uiuc/MARBLE |
| **MultiAgentBench** · Werewolf | 社交推理/对抗 | 高 | 欺骗推理、阵营胜利（对抗博弈） | Partial-Day Score + Victory Rate | — | [未核验] | https://github.com/ulab-uiuc/MARBLE |
| **MultiAgentBench** · Bargaining | 谈判博弈 | 高 | 利益冲突、策略沟通（对抗博弈） | Competition Score / Utility | — | [未核验] | https://github.com/ulab-uiuc/MARBLE |

---

## 图论协作任务（AGENTSNET）

> 来源: Grötschla et al., *AGENTSNET: Coordination and Collaborative Reasoning in Multi-Agent LLMs*, 2025 — agent 直接对应图节点，约束=边关系，与 Graph-driven MAS Pipeline 天然映射；图由脚本随机生成，规模与难度可控

| 任务 | 领域 | 难度 | 主要考察点 | 常用指标 | 下载链接 |
|---|---|---|---|---|---|
| Graph Coloring | 图论/局部约束满足 | 高 | 相邻节点颜色不冲突，全局最少色数，agent=节点/边=冲突约束 | Binary solved, soft score | https://github.com/floriangroetschla/AgentsNet |
| Consensus | 分布式一致性 | 高 | 全网 agent 协商达成一致答案，agent=节点/边=通信链接 | Binary solved | https://github.com/floriangroetschla/AgentsNet |
| Leader Election | 分布式选主 | 极高 | 全网选出唯一 leader，agent=节点/边=通信拓扑，graph evolution=角色切换 | Binary solved | https://github.com/floriangroetschla/AgentsNet |
| Maximum Matching | 图匹配 | 高 | 全局合法边选择最大化，agent=边/节点，reward=全局匹配数 | Binary solved, soft score | https://github.com/floriangroetschla/AgentsNet |
| Minimum Vertex Cover | 图覆盖 | 极高 | 覆盖所有边且近似最小，agent=节点/graph evolution=覆盖状态更新 | Binary solved, soft score | https://github.com/floriangroetschla/AgentsNet |

---

## 结构化多智能体基准（MASBench）

> 来源: Ke et al., *MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks*, 2026 — 把任务结构参数化为五轴可控变量，与 Graph-driven MAS pipeline 高度互补

| 轴 | 含义 | 难度来源 | 对 Graph-driven MAS 的意义 | 下载链接 |
|---|---|---|---|---|
| Depth | 依赖链深度 | 严格顺序依赖 | 对应 `depend` 边的长度与层数，深链任务未必适合并行 MAS | https://huggingface.co/datasets/Salesforce/MASBench |
| Horizon | 长程上下文跨度 | 信息保持时间长 | 考察长期规划与记忆，对 graph trajectory 的时间跨度提出要求 | https://huggingface.co/datasets/Salesforce/MASBench |
| Breadth | 分支宽度 | 子任务数量多 | 对应图的出度/分支因子，考察任务分解和汇总 | https://huggingface.co/datasets/Salesforce/MASBench |
| Parallel | 可并行度 | 多子任务可同时解 | 对应图的独立连通分量数，最适合展示 MAS 优势 | https://huggingface.co/datasets/Salesforce/MASBench |
| Robustness | 错误信息干扰 | 上游错误 note 注入 | 对应图节点/边被污染后的恢复能力，考察验证与抗污染 | https://huggingface.co/datasets/Salesforce/MASBench |

---

## 专业领域 · 医学

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| MedQA | 临床医学 | 高 | USMLE 执业考试风格，4选1，临床病例推理 | Accuracy | 1,273 | 无显式许可 | https://huggingface.co/datasets/openlifescienceai/medqa |
| MedMCQA | 医学 | 中 | 印度入学考试（AIIMS/NEET），21+学科，single/multi 题型 | Accuracy | ~4.2K | CC BY-SA 4.0 [未核验] | https://huggingface.co/datasets/openlifescienceai/medmcqa |
| PubMedQA | 生物医学 | 中-高 | 基于 PubMed 文献的三分类（yes/no/maybe），专业文献检索与推理 | Accuracy | 1,000（专家标注） | MIT [未核验] | https://huggingface.co/datasets/qiaojin/PubMedQA |
| MMLU 医学子集 | 医学 | 中 | 9个医学学科（解剖/临床/遗传/护理/毒理学等）4选1 | Accuracy | ~1,664 | MIT | https://huggingface.co/datasets/cais/mmlu |

---

## 专业领域 · 法律

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| CaseHOLD | 美国判例法 | 中-高 | 法律引用/要旨识别（5选1），53K+判例多选题 | Accuracy | ~53K | 无显式许可 | https://huggingface.co/datasets/casehold/casehold |
| LegalBench | 法律 | 高（多样） | 162种法律推理任务（分类/抽取/生成/蕴含），合同/证据/成文法解释 | 按任务各异 | 数万 | 任务级混合许可 | https://huggingface.co/datasets/nguha/legalbench |

---

## 专业领域 · 金融

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| FinQA | 金融/财务 | 中-高 | S&P 500 财报数值推理，**带可执行推理程序**，适合程序化 reward | Accuracy（数值匹配） | 1,147 | CC BY 4.0 | https://huggingface.co/datasets/ibm-research/finqa |
| FinanceBench | 金融/SEC | 高 | SEC 10-K/10-Q/8-K 财报开放式问答，带证据句 span | Accuracy / 证据匹配 | 150（HF 样本）/ 全量 10,231（GitHub） | CC BY-NC 4.0 (非商业) | https://huggingface.co/datasets/PatronusAI/financebench |

---

## 专业领域 · 科学

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| SciQ | 中学科学 | 易-中 | 中学物理/化学/生物/地球科学 4选1，每题附支持段落 | Accuracy | 1,000 | [未核验]（AI2 发布） | https://huggingface.co/datasets/allenai/sciq |
| SciBench | 大学科学 | 高 | 大学物理/化学/数学开放式解题，需计算+推理 | Accuracy（数值/表达式匹配） | 695 | 无显式许可 | https://huggingface.co/datasets/xw27/scibench |

---

## 百科与开放域

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| Natural Questions | 百科/开放域 | 中-高 | 谷歌真实搜索问题，长答案（段落）+短答案（span），基于维基百科 | F1 / EM | 7,842 | CC BY-SA 3.0 [未核验] | https://huggingface.co/datasets/google-research-datasets/natural_questions |
| TriviaQA | 百科 | 中 | 百科问答（维基+网络双域），需跨文档证据聚合 | F1 / EM | ~95K | [未核验] | https://huggingface.co/datasets/mandarjoshi/trivia_qa |
| WebQuestions | 百科/KG | 易-中 | Freebase 知识图谱实体问答（命名实体抽取） | Accuracy | 6,642 | unknown | https://huggingface.co/datasets/stanfordnlp/web_questions |

---

## 多语言

| 数据集 | 领域 | 难度 | 主要考察点 | 常用指标 | 规模（测试集） | 许可 | 下载链接 |
|---|---|---|---|---|---|---|---|
| XQuAD | 多语言抽取 | 中 | SQuAD 风格跨语言抽取（10–12语言），答案 span 对齐 | F1 / EM | 11,490 | CC BY-SA 4.0 [未核验] | https://huggingface.co/datasets/google/xquad |
| MMMLU | 多语言知识 | 中 | MMLU 翻译（14语言，含简体中文），57学科跨语言知识 | Accuracy | 各语言数万 | [未核验]（OpenAI 发布，访问受限） | https://huggingface.co/datasets/openai/mmmlu |

---

## 许可合规速查

| 级别 | 数据集 | 使用约束 |
|---|---|---|
| 🟢 可直接发布 | GSM8K、MATH、MMLU、MMLU-Pro、ARC、BBH、HumanEval、MBPP、CodeContests、APPS、BigCodeBench、FinQA、MuSiQue、HotpotQA、DROP、τ³-bench、AgentBench、WebArena、AgentCompany、SWE-bench、NQ、XQuAD | MIT / CC BY / Apache / CC BY-SA |
| 🟡 谨慎确认后使用 | MedQA、MedMCQA、PubMedQA、SciQ、SciBench、CaseHOLD、TriviaQA、WebQuestions、MMMLU、StrategyQA、MMLU-Redux、CommonsenseQA、MultiArith、SVAMP、MAWPS、ASDiv-A、AQuA、Game-of-24、HumanEval+、LiveCodeBench v6、MultiAgentBench全部场景、AGENTSNET全部任务、MASBench | 无显式许可或 [未核验]，建议邮件确认 |
| 🔴 不进发布版 | TheoremQA (CC BY-NC-SA)、FinanceBench (CC BY-NC)、LegalBench (混合许可)、LogiQA (无许可)、AIME (MAA版权)、Beyond-AIME (ByteDance)、HMMT2025 (MathArena)、GAIA (gated/保密) | 非商业限制或版权受限 |
