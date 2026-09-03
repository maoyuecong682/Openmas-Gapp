# OpenMAS-Gapp Q1-Q4 Frozen Protocol v1.0

This protocol freezes the minimal causal-chain experiment before adding scale,
noise, domain breadth, or production LLM adapters.

## Scientific object

```text
H: (natural-language requirement R, shared harness ecosystem G_H)
   -> requirement model M_R
   -> application blueprint B
   -> executable MAS application A
```

`M_R`, `G_H`, `B`, and `A` are separate typed objects. `MASSpec` is a legacy
pilot object and must not be used as the Q1-Q4 intermediate representation.

## Layer invariants

| Layer | Contains | Must not contain |
| --- | --- | --- |
| `M_R` | goal, tasks, task dependencies, capability requirements, constraints | concrete agents, tools, components, runtime configuration |
| `G_H` | reusable task patterns, capabilities, components, resources, controls and typed relations | case-specific selected architecture |
| `B` | tasks, abstract component/resource requirements, controls, orchestration relations | concrete agent/tool instances or runtime parameters |
| `A` | bound agent/tool/memory/control instances, invocation and execution relations | unbound abstract requirements |

Every executable node must point to the Blueprint node it realizes. MAR is
Blueprint-preserving: it may bind an abstract requirement, but may not silently
redesign the application.

## Frozen method interface

All Q1 methods implement:

```python
construct(request: ConstructionRequest) -> ConstructionResult
```

The request supplies the same raw requirement, Harness Graph, construction
budget, and metadata. The result always supplies `M_R`, `B`, `A`, and telemetry.
The default shared budget is 12 components, 24 edges, 32 planning steps, and one
model call. A later LLM adapter must use the same backbone, decoding settings,
prompt budget, component ecosystem, and runtime for all methods.

## Q1 minimal baselines

| Method | Controlled construction paradigm | Deliberately absent structure |
| --- | --- | --- |
| Direct MAS Generation | one-pass requirement-to-application generation | explicit planning and relation reasoning |
| Plan-based Construction | requirement-to-task-plan, then realization | constraint-aware component graph |
| Component-based Assembly | retrieve and combine reusable components | explicit orchestration relations |
| Workflow-based Construction | linear workflow template | feedback and non-local governance |
| Graph Harness | typed grounding, graph composition, constraint-aware Blueprint, realization | none of the proposed modules |

The deterministic implementations are interface proxies, not paper-quality LLM
results. Their outputs can validate schemas, metrics, budgets, and adapters only.

## Minimal cases

Eight expert-authored development cases cover two instances each of sequential,
multi-branch, feedback-driven, and constraint-heavy construction. A case contains
one requirement, one Harness ecosystem, one reference `M_R`, one contract-level
reference Blueprint, and one or more execution-task placeholders. The reference
Blueprint defines required behavior and relations; it does not assert a unique
valid MAS topology.

## Metrics and empty-set rules

- Requirement task F1
- Capability requirement F1
- Constraint recall
- Orchestration relation recall
- Blueprint coverage
- Blueprint-to-application realization fidelity
- Executable validity
- Construction telemetry: planning steps, model calls, inspected components

When a case has no required constraints, constraint recall is `1` only if the
method also predicts no constraints. This prevents both false penalties and
unsupported constraint invention. Smoke aggregates are explicitly marked
`formal_result=false` and must not be reported as final experimental evidence.

## Q2-Q4 extension boundary

- Q2 changes one Graph Harness module at a time while keeping this interface.
- Q3 swaps only GAO's internal orchestration representation.
- Q4 swaps only the intermediate representation between grounding and MAR.

No new dataset family or baseline should be added until these controlled variants
run under the same contract.
