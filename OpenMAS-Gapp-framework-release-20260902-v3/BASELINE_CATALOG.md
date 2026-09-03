# Self-Implemented Reference Baselines

All methods use the same Domain Package and emit the same `MASSpec`. They are not copied implementations of external papers.

| ID | Representation | Knowledge | Construction | Main diagnostic |
|---|---|---|---|---|
| `single_agent` | one-node MEG | all package capabilities | degenerate one agent | lower structural bound |
| `universal_fixed` | fixed linear graph | package catalog | Planner-Executor-Reviewer | fixed workflow vs construction |
| `domain_template` | domain template graph | domain profile | hand-authored stages | template reuse vs dynamic routing |
| `direct_prompt` | free-form proxy | requirement text | keyword/mock-LLM routing | explicit representation benefit |
| `json_spec` | flat JSON fields | requirement text | JSON-to-MAS adapter | JSON structure vs graph relations |
| `rag_example` | copied/adapted graph | nearest prior package | example retrieval | AGM vs untyped example retrieval |
| `flat_capability` | capability list | typed catalog, no edges | flat selection + linearization | graph relations vs list selection |
| `rule_compiler` | explicit rule graph | contracts | deterministic rules | rule compiler diagnostic |
| `search_composer` | candidate graph | contracts | constrained subset search | composition/search reference |

`rule_compiler` and `search_composer` are strong diagnostic references, not strawmen. `direct_prompt` uses a deterministic proxy in the offline pilot; a real LLM adapter can replace it later under the same IO contract.

