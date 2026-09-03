from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from typing import Iterable

from .schema import Capability, DomainPackage, MASEdge, MASNode, MASSpec


class Builder(ABC):
    name: str

    @abstractmethod
    def build(self, package: DomainPackage) -> MASSpec:
        raise NotImplementedError

    def spec(self, package: DomainPackage, nodes: list[MASNode], edges: list[MASEdge], selected: Iterable[str], **metadata) -> MASSpec:
        value = MASSpec(package.package_id, self.name, nodes, edges, list(dict.fromkeys(selected)), metadata)
        value.validate()
        return value


class SingleAgentBuilder(Builder):
    name = "single_agent"

    def build(self, package: DomainPackage) -> MASSpec:
        selected = [x.id for x in package.capabilities]
        node = MASNode("generalist", "agent", "Generalist", selected)
        return self.spec(package, [node], [], selected, construction="degenerate_single_node")


class UniversalFixedBuilder(Builder):
    name = "universal_fixed"

    def build(self, package: DomainPackage) -> MASSpec:
        selected = [x.id for x in package.capabilities]
        thirds = [selected[i::3] for i in range(3)]
        nodes = [
            MASNode("planner", "agent", "Planner", thirds[0]),
            MASNode("executor", "agent", "Executor", thirds[1]),
            MASNode("reviewer", "agent", "Reviewer", thirds[2]),
        ]
        edges = [MASEdge("planner", "executor"), MASEdge("executor", "reviewer")]
        return self.spec(package, nodes, edges, selected, construction="same_topology_for_all_domains")


DOMAIN_TEMPLATES = {
    "software_engineering": ["understand", "retrieve", "analyze", "modify", "test", "review"],
    "financial_analysis": ["retrieve", "extract", "calculate", "verify", "report"],
    "biomedical_evidence": ["retrieve", "screen", "extract", "synthesize", "verify", "report"],
}


class DomainTemplateBuilder(Builder):
    name = "domain_template"

    def build(self, package: DomainPackage) -> MASSpec:
        stages = DOMAIN_TEMPLATES.get(package.domain, ["plan", "execute", "review"])
        nodes = []
        assigned = []
        for stage in stages:
            caps = [x.id for x in package.capabilities if stage in _cap_text(x)]
            assigned.extend(caps)
            nodes.append(MASNode(stage, "agent", stage.title(), caps))
        edges = [MASEdge(a.id, b.id) for a, b in zip(nodes, nodes[1:])]
        return self.spec(package, nodes, edges, assigned, template=package.domain)


class DirectPromptBuilder(Builder):
    """A deterministic proxy for direct requirement-to-MAS prompting.

    It deliberately uses requirement text only and does not inspect typed contracts.
    A model adapter can replace ``_keyword_select`` without changing the benchmark IO.
    """

    name = "direct_prompt"

    def build(self, package: DomainPackage) -> MASSpec:
        text = " ".join(_flatten_requirement(package)).lower()
        selected = _keyword_select(text, package.capabilities)
        if not selected:
            selected = [x.id for x in package.capabilities[:2]]
        nodes = [MASNode(f"agent_{i+1}", "agent", _role(cap), [cap.id]) for i, cap in enumerate(selected)]
        edges = [MASEdge(a.id, b.id) for a, b in zip(nodes, nodes[1:])]
        return self.spec(package, nodes, edges, [x.id for x in selected], representation="free_form_text")


class JSONSpecBuilder(Builder):
    name = "json_spec"

    def build(self, package: DomainPackage) -> MASSpec:
        selected = _keyword_select(" ".join(_flatten_requirement(package)).lower(), package.capabilities, threshold=0.07)
        nodes = [MASNode(_safe_id(x.id), "agent", _role(x), [x.id], config={"json_fields": ["role", "input", "output"]}) for x in selected]
        edges = _linear_edges(nodes)
        return self.spec(package, nodes or [MASNode("worker", "agent", "Worker")], edges, [x.id for x in selected], representation="flat_json")


@dataclass
class Example:
    requirement: str
    spec: MASSpec


class RAGExampleBuilder(Builder):
    name = "rag_example"

    def __init__(self, examples: list[Example] | None = None):
        self.examples = examples or []

    def build(self, package: DomainPackage) -> MASSpec:
        query = " ".join(_flatten_requirement(package)).lower()
        if not self.examples:
            base = DomainTemplateBuilder().build(package)
            return MASSpec(package.package_id, self.name, base.nodes, base.edges, base.selected_capabilities, {"retrieved": None})
        example = max(self.examples, key=lambda x: SequenceMatcher(None, query, x.requirement.lower()).ratio())
        available = {x.id for x in package.capabilities}
        selected = [x for x in example.spec.selected_capabilities if x in available]
        nodes = [MASNode(x.id, x.kind, x.role, [c for c in x.capabilities if c in available], list(x.tools), dict(x.config)) for x in example.spec.nodes]
        return self.spec(package, nodes, list(example.spec.edges), selected, retrieved=example.spec.package_id)


class FlatCapabilityBuilder(Builder):
    name = "flat_capability"

    def build(self, package: DomainPackage) -> MASSpec:
        text = " ".join(_flatten_requirement(package)).lower()
        selected = _keyword_select(text, package.capabilities, threshold=0.04)
        nodes = [MASNode(_safe_id(x.id), "agent", _role(x), [x.id]) for x in selected]
        return self.spec(package, nodes or [MASNode("worker", "agent", "Worker")], _linear_edges(nodes), [x.id for x in selected], relations_used=False)


class RuleCompilerBuilder(Builder):
    name = "rule_compiler"

    def build(self, package: DomainPackage) -> MASSpec:
        required = {x.target for x in package.contracts if x.kind == "capability_required"}
        selected = [x for x in package.capabilities if x.id in required]
        nodes = [MASNode(_safe_id(x.id), "agent", _role(x), [x.id]) for x in selected]
        by_cap = {x.capabilities[0]: x.id for x in nodes if x.capabilities}
        edges = []
        for contract in package.contracts:
            if contract.kind == "order_required" and "<" in contract.target:
                left, right = contract.target.split("<", 1)
                if left in by_cap and right in by_cap:
                    edges.append(MASEdge(by_cap[left], by_cap[right]))
        if not edges:
            edges = _linear_edges(nodes)
        return self.spec(package, nodes or [MASNode("worker", "agent", "Worker")], edges, [x.id for x in selected], rules=len(package.contracts))


class SearchComposerBuilder(Builder):
    name = "search_composer"

    def build(self, package: DomainPackage) -> MASSpec:
        caps = package.capabilities
        required = {x.target for x in package.contracts if x.kind == "capability_required"}
        best = None
        best_score = (-1, -10**9)
        for size in range(1, min(len(caps), 10) + 1):
            for candidate in combinations(caps, size):
                ids = {x.id for x in candidate}
                coverage = len(required & ids)
                score = (coverage, -size)
                if score > best_score:
                    best, best_score = candidate, score
                if coverage == len(required):
                    break
            if best_score[0] == len(required):
                break
        selected = list(best or caps[:1])
        nodes = [MASNode(_safe_id(x.id), "agent", _role(x), [x.id]) for x in selected]
        edges = _topological_edges(nodes, package)
        return self.spec(package, nodes, edges, [x.id for x in selected], objective={"coverage": best_score[0], "size": len(selected)})


BUILDERS = {
    cls.name: cls for cls in [SingleAgentBuilder, UniversalFixedBuilder, DomainTemplateBuilder,
                              DirectPromptBuilder, JSONSpecBuilder, RAGExampleBuilder, FlatCapabilityBuilder,
                              RuleCompilerBuilder, SearchComposerBuilder]
}


def get_builder(name: str) -> Builder:
    try:
        return BUILDERS[name]()
    except KeyError as exc:
        raise KeyError(f"unknown baseline {name}; choose from {sorted(BUILDERS)}") from exc


def _flatten_requirement(package: DomainPackage) -> list[str]:
    values = []
    for value in package.requirement.values():
        if isinstance(value, list):
            values.extend(str(x) for x in value)
        else:
            values.append(str(value))
    return values


def _keyword_select(text: str, caps: list[Capability], threshold: float = 0.1) -> list[Capability]:
    scored = []
    words = {x.strip(".,:;()[]").lower() for x in text.split() if len(x) > 2}
    for cap in caps:
        cap_words = {x.strip(".,:;()[]").lower() for x in _cap_text(cap).split() if len(x) > 2}
        score = len(words & cap_words) / max(1, len(cap_words))
        if score >= threshold or any(tag.lower() in text for tag in cap.tags):
            scored.append((score, cap))
    return [x[1] for x in sorted(scored, key=lambda x: (-x[0], x[1].id))]


def _cap_text(cap: Capability) -> str:
    return " ".join([cap.id, cap.kind, cap.description, *cap.tags]).lower()


def _role(cap: Capability) -> str:
    return " ".join(x.capitalize() for x in cap.id.replace("-", "_").split("_")) + " Agent"


def _safe_id(value: str) -> str:
    return value.replace("-", "_").replace(".", "_")


def _linear_edges(nodes: list[MASNode]) -> list[MASEdge]:
    return [MASEdge(a.id, b.id) for a, b in zip(nodes, nodes[1:])]


def _topological_edges(nodes: list[MASNode], package: DomainPackage) -> list[MASEdge]:
    by_cap = {x.capabilities[0]: x.id for x in nodes if x.capabilities}
    edges = []
    for contract in package.contracts:
        if contract.kind == "order_required" and "<" in contract.target:
            left, right = contract.target.split("<", 1)
            if left in by_cap and right in by_cap:
                edges.append(MASEdge(by_cap[left], by_cap[right]))
    return edges or _linear_edges(nodes)
