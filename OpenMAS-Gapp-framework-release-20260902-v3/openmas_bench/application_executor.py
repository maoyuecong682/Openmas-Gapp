"""Execute a constructed MAS application against one benchmark item.

The executor is deliberately blind to the ablation label and gold answer.  Its
only system description is the executable application produced by the selected
construction pipeline.  Each executable node receives predecessor artifacts
and emits a new artifact; the terminal node emits the benchmark prediction.
"""
from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .llm import LLMAdapter
from .domains import DomainContext, get_domain_plugin
from .domains.bbh import split_bbh_task_resources
from .domains import finqa_tool
from .schema import ConstructionCase, ConstructionResult


@dataclass
class NodeExecution:
    node_id: str
    implementation_ref: str
    role: str
    predecessors: list[str]
    artifact: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    retries: int
    json_repaired: bool
    artifact_truncated: bool = False
    artifact_original_chars: int = 0
    resource_keys: list[str] | None = None


@dataclass
class ApplicationTaskResult:
    prediction: Any
    output_field: str
    application_digest: str
    node_executions: list[NodeExecution]
    input_tokens: int
    output_tokens: int
    calls: int
    retries: int
    repairs: int
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ApplicationTaskExecutor:
    def __init__(self, adapter: LLMAdapter, data_root: Path):
        self.adapter = adapter
        self.data_root = data_root

    def execute(self, dataset, row: dict[str, Any], case: ConstructionCase,
                construction: ConstructionResult, seed: int) -> ApplicationTaskResult:
        construction.validate(case.request())
        app = construction.application
        order, predecessors = _execution_order(app.nodes, app.edges)
        blueprint_nodes = {node.id: node for node in construction.blueprint.nodes}
        harness_nodes = {node.id: node for node in case.harness.nodes}
        blueprint_visible = bool(construction.blueprint.metadata.get("blueprint_present", True))
        execution_blueprint_visible = blueprint_visible and construction.application.metadata.get("blueprint_preserving", True) is not False
        task_payload = _task_payload(dataset.dataset_id, row, self.data_root)
        application_plan = _visible_application_plan(construction, execution_blueprint_visible)
        _assert_information_boundary(construction, application_plan, execution_blueprint_visible)
        branch_resources = _branch_resources(dataset.dataset_id, row)
        output_field = _output_field(dataset.execution.metric_name)
        benchmark_output_contract = _benchmark_output_contract(
            dataset.execution.metric_name, dataset.dataset_id, row, self.data_root)
        benchmark_reasoning_contract = _benchmark_reasoning_contract(
            dataset.execution.metric_name, dataset.dataset_id, row, self.data_root)
        # This policy is invariant across variants. Module-specific behavior
        # comes from the constructed application and realization contracts,
        # never from revealing the variant label to the executor.
        evidence_policy = "Use upstream artifacts as the only evidence and do not invent unsupported facts."
        artifacts: dict[str, str] = {}
        finqa_tool_results: dict[str, dict[str, Any]] = {}
        app_nodes_by_id = {app_node.id: app_node for app_node in app.nodes}
        entrypoints = set(app.entrypoints)
        executions: list[NodeExecution] = []
        totals = {"input": 0, "output": 0, "calls": 0, "retries": 0, "repairs": 0}
        inherited_resources: dict[str, set[str]] = {}
        resource_audit = {"expected_branch_count": len(branch_resources),
                          "merge_nodes": 0, "merge_complete": 0,
                          "cross_branch_access_count": 0,
                          "resource_isolation_violation": 0}
        finqa_tool_eligible = (dataset.dataset_id == "FinQA"
                               and _finqa_structured_tool_eligible(construction))
        finqa_tool_audit: dict[str, Any] = {
            "eligible": finqa_tool_eligible,
            "invoked": False,
            "gold_used": False,
        }
        if finqa_tool_eligible:
            finqa_tool_audit["protocol_repairs"] = 0
        if dataset.dataset_id == "FinQA":
            _assert_finqa_tool_boundary(construction, finqa_tool_eligible)

        for index, node_id in enumerate(order):
            node = next(node for node in app.nodes if node.id == node_id)
            bp = blueprint_nodes.get(node.realizes_blueprint_node) if execution_blueprint_visible else None
            component = harness_nodes.get(node.implementation_ref)
            role = str(node.config.get("execution_instruction") or
                       (bp.description if bp is not None and execution_blueprint_visible else "") or
                       (component.description if component is not None else "General task worker"))
            artifact_contract = str(node.config.get("artifact_contract", "")).strip()
            incoming = [_bounded_artifact(artifacts[x], MAX_ARTIFACT_CHARS)
                        for x in predecessors[node_id] if x in artifacts]
            incoming = _fit_total(incoming, MAX_TOTAL_UPSTREAM_CHARS)
            terminal = index == len(order) - 1
            finqa_role = _finqa_node_role(node) if dataset.dataset_id == "FinQA" else ""
            inherited_tool_result = _inherit_finqa_tool_result(
                predecessors[node_id], finqa_tool_results)
            structured_execute = finqa_tool_eligible and finqa_role == "execute"
            structured_verify = (finqa_tool_eligible and finqa_role == "verify"
                                 and inherited_tool_result is not None)
            tool_copy_answer = (finqa_tool_eligible and finqa_role == "answer"
                                and inherited_tool_result is not None)
            required = ({"selected_evidence", "steps"}
                        if structured_execute else
                        ({output_field} if terminal else {"artifact"}))
            system = _node_system_prompt(
                node.kind, role, terminal, output_field, evidence_policy,
                artifact_contract, (benchmark_output_contract if terminal else
                                    benchmark_reasoning_contract))
            policy_hint = str(node.config.get("policy_hint", "")).strip()
            if policy_hint:
                system += " Construction policy: " + policy_hint
            if terminal and construction.application.metadata.get("variant") == "full_graph_harness":
                if node.config.get("rerank_candidates"):
                    system += (
                        " Terminal reranking policy: compare all candidate answers from upstream artifacts, "
                        "prefer the best verified candidate, and return only the final answer."
                    )
                if node.config.get("math_verify"):
                    system += (
                        " Numerical verification policy: recompute or normalize equivalent expressions, "
                        "then choose the candidate that exactly matches the verified value."
                    )
            if dataset.dataset_id in {"FinQA", "FinanceBench"}:
                if structured_execute:
                    system += " " + FINQA_STRUCTURED_TOOL_CONTRACT
                elif structured_verify:
                    system += (
                        " FinQA verifier boundary: inspect only the supplied TOOL_TRACE. "
                        "Check selected evidence, ordered steps, resolved operands, units, and tool status. "
                        "Do not recompute, replace, or propose another final_value."
                    )
                elif not tool_copy_answer:
                    if dataset.dataset_id == "FinQA":
                        system += (
                            " FinQA protocol: use the supplied FINQA_TABLE and text as evidence, preserve signs "
                            "and units, and carry one numeric candidate answer."
                        )
                    else:
                        system += (
                            " FinanceBench protocol: use the supplied filing evidence and question reasoning as evidence, "
                            "reconcile the relevant snippets, and carry one concise candidate answer."
                        )
            long_text_terminal = terminal and output_field in {"code", "patch"}
            # Only the entry node receives the raw benchmark prompt. Every
            # downstream node must work from predecessor artifacts; otherwise
            # it could bypass a removed orchestration/grounding module by
            # solving the original task independently.
            resource_keys = [str(key) for key in node.config.get("resource_bindings", [])]
            inherited = set(resource_keys)
            for predecessor in predecessors[node_id]:
                inherited.update(inherited_resources.get(predecessor, set()))
            inherited_resources[node_id] = inherited
            if (construction.application.metadata.get("variant") == "full_graph_harness"
                    and branch_resources and node.config.get("merge_stage")):
                resource_audit["merge_nodes"] += 1
                if not set(branch_resources).issubset(inherited):
                    resource_audit["resource_isolation_violation"] += 1
                    raise ValueError(
                        f"merge node {node.id} did not receive all isolated branches: "
                        f"expected={sorted(branch_resources)} received={sorted(inherited)}"
                    )
                resource_audit["merge_complete"] += 1
            resource_access = bool(node.config.get("resource_access"))
            visible_task = task_payload if node_id in entrypoints or resource_access else {
                "dataset": task_payload.get("dataset"),
                "output_contract": "Use upstream artifacts; do not solve an unseen task from scratch.",
            }
            if branch_resources and construction.application.metadata.get("variant") == "full_graph_harness":
                visible_task["branch_resources"] = branch_resources
            if resource_keys:
                missing_resources = [key for key in resource_keys if key not in branch_resources]
                if missing_resources:
                    raise ValueError(
                        f"missing isolated branch resources for {node.id}: {missing_resources}"
                    )
                visible_task = {
                    "dataset": task_payload.get("dataset"),
                    "question": task_payload.get("question"),
                    "context": "\n\n".join(branch_resources[key] for key in resource_keys
                                            if key in branch_resources),
                    "choices": task_payload.get("choices"),
                    "resource_access_policy": (
                        "This is an isolated branch resource. Use only this context "
                        "for evidence retrieval; do not assume access to other branches."
                    ),
                }
            elif resource_access and node_id not in entrypoints:
                visible_task = dict(task_payload)
                visible_task["resource_access_policy"] = (
                    "This node is connected to explicit Graph Harness resource nodes. "
                    "Use the original question/context/options only to verify and preserve task-critical facts; "
                    "combine them with upstream artifacts instead of discarding upstream work."
                )
            if structured_execute:
                # The compiled tool call receives the public question/table but
                # never the normalized gold answer or annotated FinQA program.
                visible_task = dict(task_payload)
                visible_task["resource_access_policy"] = (
                    "Read-only compiled FinQA tool input: select table cells or text spans only from this public evidence."
                )
            if structured_verify:
                visible_task = {
                    "dataset": "FinQA",
                    "verification_scope": "TOOL_TRACE only",
                }
                incoming = ["TOOL_TRACE: " + json.dumps(inherited_tool_result, ensure_ascii=False)]
            if terminal and construction.application.metadata.get("variant") == "full_graph_harness":
                visible_task["terminal_decision_context"] = _terminal_decision_context(
                    dataset.dataset_id, task_payload, output_field, [p for p in predecessors[node_id] if p in artifacts],
                    incoming,
                )
            user = json.dumps({
                "task": visible_task,
                "application_plan": _local_application_plan(
                    application_plan,
                    node.realizes_blueprint_node,
                    [app_nodes_by_id[p].realizes_blueprint_node for p in predecessors[node_id]
                     if p in app_nodes_by_id],
                ),
                "role": role,
                "artifact_contract": artifact_contract,
                "capabilities": list(node.capabilities),
                "upstream_artifacts": incoming,
                "output_contract": ({output_field: benchmark_output_contract or
                                                     "final benchmark prediction"}
                                    if terminal else {"artifact": "work product for downstream nodes"}),
            }, ensure_ascii=False)
            if dataset.dataset_id in {"FinQA", "FinanceBench"}:
                user_data = json.loads(user)
                if dataset.dataset_id == "FinQA" and structured_execute:
                    user_data["finqa_structured_tool_schema"] = FINQA_STRUCTURED_TOOL_SCHEMA
                elif dataset.dataset_id == "FinQA" and not finqa_tool_eligible:
                    # Generic ablations receive no compiler schema or tool state.
                    user_data["finqa_numeric_contract"] = {
                        "terminal_format": "single numeric value",
                        "do_not_use_gold_program": True,
                    }
                elif dataset.dataset_id == "FinanceBench":
                    user_data["financebench_contract"] = {
                        "preserve_evidence_snippets": True,
                        "reconcile_multiple_snippets": True,
                        "terminal_format": "concise evidence-backed answer",
                    }
                user = json.dumps(user_data, ensure_ascii=False)
            response = None
            if tool_copy_answer:
                artifact = str(inherited_tool_result["final_value"])
            else:
                response = (self.adapter.generate_text(system, user, seed + index)
                            if long_text_terminal else
                            self.adapter.generate_json(system, user, seed + index, required))
                if structured_execute:
                    try:
                        tool_result = _execute_finqa_tool_call(response.value, task_payload)
                    except ValueError as exc:
                        # One schema-level repair is allowed. It receives only
                        # the validation error and original gold-blind prompt.
                        totals["input"] += response.input_tokens
                        totals["output"] += response.output_tokens
                        totals["calls"] += 1
                        totals["retries"] += response.retry_count
                        totals["repairs"] += int(response.json_repaired)
                        repair_system = (
                            system + " Executor validation rejected the prior tool call: " + str(exc)
                            + " Return one corrected object satisfying the exact schema."
                        )
                        response = self.adapter.generate_json(
                            repair_system, user, seed + index, required)
                        tool_result = _execute_finqa_tool_call(response.value, task_payload)
                        finqa_tool_audit["protocol_repairs"] += 1
                    finqa_tool_results[node_id] = tool_result
                    inherited_tool_result = tool_result
                    artifact = "TOOL_TRACE: " + json.dumps(tool_result, ensure_ascii=False)
                    finqa_tool_audit.update({
                        "invoked": True,
                        "selected_evidence": tool_result["selected_evidence"],
                        "selected_evidence_count": len(tool_result["selected_evidence"]),
                        "step_count": len(tool_result["steps"]),
                        "steps": tool_result["steps"],
                        "computed_value": tool_result["final_value"],
                        "result_unit": tool_result["final_unit"],
                        "tool_status": tool_result["tool_status"],
                    })
                else:
                    value = (response.value.get("text") if long_text_terminal
                             else response.value.get(output_field if terminal else "artifact"))
                    artifact = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            if inherited_tool_result is not None:
                finqa_tool_results[node_id] = inherited_tool_result
            if terminal and output_field in {"code", "patch"} and isinstance(artifact, str):
                # Code/patch sandboxes consume the generated source directly.
                # LLMs may still wrap it in the intermediate artifact envelope;
                # unwrap only explicit source fields, never arbitrary JSON.
                try:
                    envelope = json.loads(artifact)
                except (TypeError, ValueError):
                    envelope = None
                if isinstance(envelope, dict):
                    for key in (output_field, "candidate_answer", "text"):
                        candidate = envelope.get(key)
                        if isinstance(candidate, str) and candidate.strip():
                            artifact = candidate
                            break
            if terminal and isinstance(artifact, str):
                domain_plugin = get_domain_plugin(dataset.dataset_id, dataset.execution.metric_name)
                if domain_plugin is not None:
                    artifact = domain_plugin.normalize_terminal(
                        artifact, DomainContext(dataset.dataset_id, dataset.execution.metric_name,
                                                row, self.data_root))
            original_chars = len(artifact)
            artifact_limit = LONG_ARTIFACT_CHARS if long_text_terminal else MAX_ARTIFACT_CHARS
            artifact_truncated = original_chars > artifact_limit
            if artifact_truncated:
                artifact = artifact[:artifact_limit] + "\n[artifact truncated by executor]"
            artifacts[node_id] = artifact
            executions.append(NodeExecution(
                node.id, node.implementation_ref, role, list(predecessors[node_id]), artifact,
                response.input_tokens if response is not None else 0,
                response.output_tokens if response is not None else 0,
                response.latency_ms if response is not None else 0.0,
                response.retry_count if response is not None else 0,
                response.json_repaired if response is not None else False,
                artifact_truncated, original_chars,
                resource_keys,
            ))
            if response is not None:
                totals["input"] += response.input_tokens
                totals["output"] += response.output_tokens
                totals["calls"] += 1
                totals["retries"] += response.retry_count
                totals["repairs"] += int(response.json_repaired)

        prediction = artifacts[order[-1]]
        digest_source = json.dumps(asdict(app), sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
        resource_audit["cross_branch_access_count"] = sum(
            len(set(execution.resource_keys or [])) > 1 for execution in executions
        )
        return ApplicationTaskResult(prediction, output_field, digest, executions,
                                     totals["input"], totals["output"], totals["calls"],
                                     totals["retries"], totals["repairs"],
                                     {"resource_audit": resource_audit,
                                      "finqa_tool_audit": finqa_tool_audit})


MAX_ARTIFACT_CHARS = 6000
MAX_TOTAL_UPSTREAM_CHARS = 16000
LONG_ARTIFACT_CHARS = 60000


def _bounded_artifact(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[upstream artifact truncated]"


def _fit_total(values: list[str], limit: int) -> list[str]:
    remaining = limit
    result = []
    for value in values:
        if remaining <= 0:
            break
        clipped = value[:remaining]
        result.append(clipped + ("\n[upstream total truncated]" if len(clipped) < len(value) else ""))
        remaining -= len(clipped)
    return result


def _node_system_prompt(kind: str, role: str, terminal: bool, output_field: str,
                        evidence_policy: str, artifact_contract: str = "",
                        benchmark_output_contract: str = "") -> str:
    instruction = ("You are one node in an already-constructed multi-agent application. "
                   "Perform only the assigned role, use upstream artifacts as evidence, and do not claim access to tools or tests not shown.")
    if kind == "control":
        instruction += " Audit the upstream work against the stated constraint and correct it when necessary."
    if terminal:
        formats = {
            "answer": "Return only the concise final answer in the answer field.",
            "code": "Return only the complete executable Python implementation text as the response body, without JSON, commentary, or markdown fences.",
            "patch": "Return only the complete unified diff as the response body, without JSON, commentary, or markdown fences.",
        }
        instruction += " " + formats[output_field]
    else:
        instruction += " Return your work product in the artifact field."
    if artifact_contract:
        instruction += " You must satisfy this bound artifact contract: " + artifact_contract
    if benchmark_output_contract:
        instruction += (" Benchmark scoring contract: " if terminal else
                        " Benchmark reasoning contract: ") + benchmark_output_contract
    return instruction + " " + evidence_policy + " Assigned role: " + role


def _output_field(metric_name: str) -> str:
    if metric_name == "unit_test_pass":
        return "code"
    if metric_name == "swebench_resolved":
        return "patch"
    return "answer"


def _benchmark_output_contract(metric_name: str, dataset_id: str = "",
                               row: dict[str, Any] | None = None,
                               data_root: Path | None = None) -> str:
    """Return the dataset's public prediction semantics, invariant by variant."""
    if metric_name == "pubmedqa_accuracy":
        return (
            "Return exactly one lowercase label: yes, no, or maybe. Classify the "
            "paper authors' conclusion with respect to the original question: yes "
            "when it supports the proposition, no when it rejects or reports no "
            "meaningful effect/difference, and maybe only when the authors' conclusion "
            "is genuinely inconclusive. Ordinary methodological caveats or requests "
            "for more research do not change an otherwise supported yes/no conclusion "
            "to maybe. Do not return an explanation."
        )
    plugin = get_domain_plugin(dataset_id, metric_name)
    if plugin is not None:
        return plugin.output_contract(DomainContext(
            dataset_id, metric_name, row or {}, data_root or Path(".")))
    return ""


def _benchmark_reasoning_contract(metric_name: str, dataset_id: str = "",
                                  row: dict[str, Any] | None = None,
                                  data_root: Path | None = None) -> str:
    """Dataset semantics for intermediate artifacts, without terminal formatting."""
    if metric_name == "pubmedqa_accuracy":
        return (
            "Preserve the original question verbatim. Infer the paper authors' "
            "conclusion with respect to that exact question and carry one current "
            "label (yes, no, or maybe) in CANDIDATE_ANSWER. Use yes when the authors "
            "support the proposition, no when they reject it or report no meaningful "
            "effect/difference, and maybe only when their conclusion is genuinely "
            "inconclusive. Ordinary methodological caveats or requests for further "
            "research must not replace an otherwise supported yes/no label."
        )
    plugin = get_domain_plugin(dataset_id, metric_name)
    if plugin is not None:
        return plugin.reasoning_contract(DomainContext(
            dataset_id, metric_name, row or {}, data_root or Path(".")))
    return ""


def _task_payload(dataset_id: str, row: dict[str, Any], data_root: Path) -> dict[str, Any]:
    raw = row.get("raw") or {}
    payload = {
        "dataset": dataset_id,
        "question": row.get("question") or raw.get("prompt") or raw.get("problem_statement") or "",
        "context": row.get("context") or "",
        "choices": row.get("choices"),
    }
    if dataset_id == "MBPP":
        tests = (row.get("raw") or {}).get("test_list") or []
        names = sorted(set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", "\n".join(map(str, tests)))))
        if names:
            payload["required_function_names"] = names
    if dataset_id == "SWE-bench":
        payload.update({
            "repository": raw.get("repo"),
            "base_commit": raw.get("base_commit"),
            "issue": raw.get("problem_statement", ""),
            "hints": raw.get("hints_text", ""),
            "repository_context": _swe_repository_context(raw, data_root),
        })
    plugin = get_domain_plugin(dataset_id)
    if plugin is not None:
        plugin.augment_task_payload(payload, DomainContext(dataset_id, "", row, data_root))
    return payload


def _branch_resources(dataset_id: str, row: dict[str, Any]) -> dict[str, str]:
    """Materialize branch evidence without exposing gold answers."""
    raw = row.get("raw") or {}
    plugin = get_domain_plugin(dataset_id)
    if plugin is not None:
        return plugin.branch_resources(DomainContext(dataset_id, "", row, Path(".")))
    if dataset_id == "HotpotQA":
        support = raw.get("supporting_facts") or {}
        titles = list(dict.fromkeys(str(title) for title in support.get("title", [])))
        context = raw.get("context") or {}
        documents = {str(title): sentences for title, sentences
                     in zip(context.get("title", []), context.get("sentences", []))}
        values = [f"[{title}] {' '.join(map(str, documents.get(title, [])))}"
                  for title in titles[:2]]
    elif dataset_id == "MuSiQue":
        paragraphs = {paragraph.get("idx"): paragraph for paragraph in raw.get("paragraphs", [])}
        indices = [hop.get("paragraph_support_idx")
                   for hop in raw.get("question_decomposition", [])
                   if isinstance(hop.get("paragraph_support_idx"), int)]
        docs = []
        for index in indices:
            paragraph = paragraphs.get(index)
            if paragraph is not None:
                docs.append(f"[{paragraph.get('title', '')}] {paragraph.get('paragraph_text', '')}")
        values = docs[:1] + (["\n\n".join(docs[1:])] if len(docs) > 1 else [])
    elif dataset_id == "StrategyQA":
        # StrategyQA rows do not ship explicit branch paragraphs, but they do
        # contain a gold-free fact bundle and a decomposition trace.  Split
        # those into two auditable branch resources so graph_harness can keep
        # its multi-branch resource contract without hallucinating evidence.
        facts = raw.get("facts") or []
        if isinstance(facts, list) and facts:
            midpoint = max(1, len(facts) // 2)
            first = " ".join(str(item) for item in facts[:midpoint]).strip()
            second = " ".join(str(item) for item in facts[midpoint:]).strip()
            values = [first, second] if second else [first]
        else:
            decomp = raw.get("decomposition") or []
            if isinstance(decomp, list) and decomp:
                midpoint = max(1, len(decomp) // 2)
                first = " ".join(str(item) for item in decomp[:midpoint]).strip()
                second = " ".join(str(item) for item in decomp[midpoint:]).strip()
                values = [first, second] if second else [first]
            else:
                description = str(raw.get("description") or raw.get("question") or "").strip()
                values = [description, description] if description else []
    elif dataset_id == "FinQA":
        context = raw.get("context") or {}
        values = []
        if isinstance(context, dict):
            pre_text = [str(item).strip() for item in context.get("pre_text", []) if str(item).strip()]
            post_text = [str(item).strip() for item in context.get("post_text", []) if str(item).strip()]
            table = context.get("table") or []
            if isinstance(table, list) and table:
                table_lines = []
                for row_index, row_values in enumerate(table[:4]):
                    if isinstance(row_values, list):
                        row_text = " | ".join(str(cell).strip() for cell in row_values if str(cell).strip())
                    else:
                        row_text = str(row_values).strip()
                    if row_text:
                        table_lines.append(f"row_{row_index}: {row_text}")
                if table_lines:
                    values.append("FINQA_TABLE:\n" + "\n".join(table_lines))
            if pre_text:
                values.append("FINQA_PRE_TEXT: " + " ".join(pre_text[:4]))
            if post_text:
                values.append("FINQA_POST_TEXT: " + " ".join(post_text[:4]))
        if not values:
            values = [str(raw.get("prompt") or raw.get("question") or "").strip()]
    elif dataset_id == "FinanceBench":
        evidence = raw.get("evidence") or []
        snippets = []
        if isinstance(evidence, list):
            for index, item in enumerate(evidence):
                if isinstance(item, dict):
                    text = str(item.get("evidence_text") or "").strip()
                    if text:
                        snippets.append(f"EVIDENCE_{index + 1}: {text}")
        # Qualified Q2 rows have at least two independent evidence records.
        # Route records, not metadata labels or arbitrary character shards.
        values = ["\n\n".join(snippets[::2]), "\n\n".join(snippets[1::2])] \
            if len(snippets) >= 2 else snippets
        if not values:
            values = [str(raw.get("question") or row.get("question") or "").strip()]
    elif dataset_id in {"BBH", "BBH-Full"}:
        task = str(raw.get("task") or "").strip()
        input_text = str(raw.get("input") or row.get("question") or "").strip()
        values = list(_split_bbh_task_resources(task, input_text)) if input_text else []
    elif dataset_id == "SciBench":
        # SciBench is a sequential derivation task. The problem and unit are
        # already in the invariant task payload; exposing synthetic branches
        # only to Full would violate the paired input boundary.
        values = []
    elif dataset_id == "DROP":
        passage = str(raw.get("passage") or "").strip()
        question = str(raw.get("question") or "").strip()
        values = []
        if question:
            values.append("DROP_QUESTION: " + question)
        if passage:
            values.append("DROP_PASSAGE: " + passage)
    elif dataset_id == "MATH-500":
        problem = str(raw.get("problem") or row.get("question") or "").strip()
        subject = str(raw.get("subject") or "").strip()
        level = raw.get("level")
        prefix = "MATH500_PROBLEM"
        if subject or level is not None:
            prefix += f" [{subject or 'unknown'}; level={level}]"
        values = [f"{prefix}: {problem}"] if problem else []
    elif dataset_id == "MedQA":
        data = raw.get("data") or {}
        question = str(data.get("Question") or row.get("question") or "").strip()
        options = data.get("Options") or {}
        values = []
        if question:
            values.append("MEDQA_QUESTION: " + question)
        if isinstance(options, dict) and options:
            rendered = " | ".join(f"{key}: {str(value).strip()}" for key, value in sorted(options.items()))
            values.append("MEDQA_OPTIONS: " + rendered)
    else:
        return {}
    return {f"branch_{index}": value for index, value in enumerate(values) if value}


def _split_bbh_task_resources(task: str, input_text: str) -> tuple[str, str]:
    """Compatibility export; new code should import from domains.bbh."""
    return split_bbh_task_resources(task, input_text)


def _visible_application_plan(construction: ConstructionResult, blueprint_visible: bool) -> dict[str, Any]:
    if blueprint_visible:
        return {
            "nodes": [{"id": node.id, "kind": node.kind, "description": node.description,
                       "capabilities": node.capability_refs}
                      for node in construction.blueprint.nodes],
            "edges": [{"source": edge.source, "target": edge.target, "relation": edge.relation}
                      for edge in construction.blueprint.edges],
            "constraints": list(construction.blueprint.constraint_refs),
        }
    app = construction.application
    return {
        "nodes": [{"id": node.id, "kind": node.kind,
                   "description": node.config.get("execution_instruction", ""),
                   "capabilities": node.capabilities} for node in app.nodes],
        "edges": [{"source": edge.source, "target": edge.target, "relation": edge.relation}
                  for edge in app.edges],
        "constraints": [],
    }


def _local_application_plan(plan: dict[str, Any], node_id: str,
                            predecessor_ids: list[str]) -> dict[str, Any]:
    """Keep the node's typed execution context small and actionable."""
    relevant = set(predecessor_ids) | {node_id}
    nodes = [node for node in plan.get("nodes", []) if node.get("id") in relevant]
    edges = [edge for edge in plan.get("edges", [])
             if edge.get("source") in relevant and edge.get("target") in relevant]
    return {"nodes": nodes, "edges": edges, "constraints": plan.get("constraints", [])}


def _terminal_decision_context(dataset_id: str, task_payload: dict[str, Any], output_field: str,
                               source_ids: list[str], upstream_artifacts: list[str]) -> dict[str, Any]:
    """Give the terminal Graph Harness node a compact candidate pool to rerank."""
    candidate_pool = []
    for source_id, artifact in zip(source_ids, upstream_artifacts):
        extracted = _extract_candidate_hints(artifact, dataset_id, output_field)
        candidate_pool.append({
            "source_node": source_id,
            "candidate_hints": extracted,
            "artifact_excerpt": artifact[:700],
        })
    return {
        "verification_mode": _verification_mode(dataset_id, output_field),
        "question": task_payload.get("question"),
        "choices": task_payload.get("choices"),
        "candidate_pool": candidate_pool,
    }


def _verification_mode(dataset_id: str, output_field: str) -> str:
    dataset_key = str(dataset_id).casefold()
    if dataset_key in {"math-500", "math500", "finqa", "gsm8k", "drop"} or output_field in {"code", "patch"}:
        return "numeric_verification"
    if dataset_key in {"medqa", "mmlu", "mmlu-pro", "arc", "logiqa", "sciq"}:
        return "option_rerank"
    if dataset_key in {"hotpotqa", "musique", "strategyqa", "pubmedqa"}:
        return "evidence_span_rerank"
    return "general_final_selection"


def _extract_candidate_hints(text: str, dataset_id: str, output_field: str) -> list[str]:
    hints = []
    lowered = text.casefold()
    patterns = [
        r"candidate_answer\s*[:=]\s*([^\n,}]+)",
        r"final_answer\s*[:=]\s*([^\n,}]+)",
        r"predicted_answer\s*[:=]\s*([^\n,}]+)",
        r"answer\s*[:=]\s*([^\n,}]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate = match.group(1).strip().strip('"\'')
            if candidate and candidate not in hints:
                hints.append(candidate[:120])
    if output_field == "answer":
        if dataset_id.casefold() in {"medqa", "mmlu", "mmlu-pro", "arc", "logiqa", "sciq"}:
            for option in re.findall(r"\b[A-D]\b", text.upper()):
                if option not in hints:
                    hints.append(option)
        if dataset_id.casefold() in {"pubmedqa"}:
            for label in ("yes", "no", "maybe"):
                if label in lowered and label not in hints:
                    hints.append(label)
    if dataset_id.casefold() in {"math-500", "math500", "finqa", "gsm8k", "drop"}:
        for number in re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text):
            if number not in hints:
                hints.append(number)
    return hints[:8]


FINQA_STRUCTURED_TOOL_SCHEMA = finqa_tool.STRUCTURED_TOOL_SCHEMA
FINQA_STRUCTURED_TOOL_CONTRACT = finqa_tool.STRUCTURED_TOOL_CONTRACT
_finqa_structured_tool_eligible = finqa_tool.structured_tool_eligible
_finqa_node_role = finqa_tool.node_role
_inherit_finqa_tool_result = finqa_tool.inherit_tool_result
_execute_finqa_tool_call = finqa_tool.execute_tool_call
_assert_finqa_tool_boundary = finqa_tool.assert_tool_boundary


def _assert_information_boundary(construction: ConstructionResult,
                                 application_plan: dict[str, Any],
                                 blueprint_visible: bool) -> None:
    """Fail closed if an ablation execution view reintroduces removed data."""
    variant = construction.application.metadata.get("variant")
    if variant == "w/o_blueprint" and blueprint_visible:
        raise ValueError("information boundary violation: w/o Blueprint exposed Blueprint view")
    if variant == "w/o_realization":
        if blueprint_visible:
            raise ValueError("information boundary violation: generic realization exposed Blueprint view")
        if any("abstract Blueprint" in str(node.get("description", ""))
               for node in application_plan.get("nodes", [])):
            raise ValueError("information boundary violation: generic realization contains Blueprint description")
    if variant == "w/o_graph_orchestration":
        if any(edge.get("relation") in {"feedback", "reviews", "constrained_by"}
               for edge in application_plan.get("edges", [])):
            raise ValueError("information boundary violation: w/o Graph exposed typed relation")
        if any(node.config.get("resource_bindings")
               for node in construction.application.nodes):
            raise ValueError("information boundary violation: w/o Graph exposed branch resource binding")
    if variant == "w/o_requirement_grounding":
        if "requirement_model" in application_plan or "goal" in application_plan:
            raise ValueError("information boundary violation: w/o Requirement exposed structured model")


def _swe_repository_context(raw: dict[str, Any], data_root: Path,
                            max_files: int = 6, max_chars: int = 24000) -> str:
    """Retrieve issue-relevant source without consulting the gold patch/tests."""
    commit = str(raw.get("base_commit", ""))
    archives = list((data_root / "q2_datasets" / "swebench_repos").glob(f"*/base-{commit}.tar.gz"))
    if not archives:
        return "Repository archive unavailable."
    issue = str(raw.get("problem_statement", ""))
    identifiers = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{4,}\b", issue))
    stop = {"consider", "following", "however", "output", "input", "model", "array", "true", "false"}
    identifiers = {x for x in identifiers if x.lower() not in stop}
    candidates: list[tuple[int, str, str]] = []
    try:
        with tarfile.open(archives[0], "r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.endswith((".py", ".js", ".ts", ".java", ".go", ".rs")):
                    continue
                stream = archive.extractfile(member)
                if stream is None or member.size > 300000:
                    continue
                text = stream.read().decode("utf-8", errors="ignore")
                score = sum(1 for token in identifiers if token in text or token.lower() in member.name.lower())
                if score:
                    candidates.append((score, member.name, text))
    except (OSError, tarfile.TarError):
        return "Repository archive could not be inspected."
    parts = []
    for _, name, text in sorted(candidates, key=lambda x: (-x[0], len(x[2]), x[1]))[:max_files]:
        remaining = max_chars - sum(len(x) for x in parts)
        if remaining <= 0:
            break
        parts.append(f"FILE: {name}\n{text[:remaining]}")
    return "\n\n".join(parts) if parts else "No issue-relevant source files were retrieved."


def _execution_order(nodes, edges) -> tuple[list[str], dict[str, list[str]]]:
    node_ids = {node.id for node in nodes}
    declaration_order = {node.id: index for index, node in enumerate(nodes)}
    graph: dict[str, list[str]] = defaultdict(list)
    predecessors: dict[str, list[str]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            continue
        predecessors[edge.target].append(edge.source)
        if edge.relation == "feedback":
            continue
        graph[edge.source].append(edge.target)
        indegree[edge.target] += 1
    queue = deque(sorted((x for x, degree in indegree.items() if degree == 0),
                         key=declaration_order.get))
    order = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for target in graph[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    order.extend(sorted(node_ids.difference(order), key=declaration_order.get))
    return order, {node_id: predecessors[node_id] for node_id in order}
