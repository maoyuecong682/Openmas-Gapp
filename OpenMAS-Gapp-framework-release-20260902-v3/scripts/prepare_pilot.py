from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path
from typing import Any

from openmas_bench.io import write_json, write_jsonl
from openmas_bench.schema import Capability, Contract, DomainPackage, ExecutionTask, Mutation


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
CLEAN = ROOT / "cleaned"
PACKAGES = ROOT / "packages"
SOURCES = {
    "finqa": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "pubmedqa": "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
    "swebench_verified": "https://datasets-server.huggingface.co/rows?dataset=princeton-nlp%2FSWE-bench_Verified&config=default&split=test&offset=0&length=100",
}


def download_sources() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in SOURCES.items():
        target = RAW / f"{name}.json"
        if not target.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "OpenMAS-Gapp/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
                handle.write(response.read())
        manifest[name] = {"url": url, "file": str(target.relative_to(ROOT)), "sha256": sha256(target), "bytes": target.stat().st_size}
    write_json(RAW / "sources.manifest.json", manifest)


def clean_sources(limit: int = 30) -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    _clean_finqa(RAW / "finqa.json", CLEAN / "finance_tasks.jsonl", limit)
    _clean_pubmedqa(RAW / "pubmedqa.json", CLEAN / "biomedical_tasks.jsonl", limit)
    _clean_swebench(RAW / "swebench_verified.json", CLEAN / "software_tasks.jsonl", limit)


def _clean_finqa(source: Path, target: Path, limit: int) -> None:
    rows = json.loads(source.read_text(encoding="utf-8"))
    values = []
    for i, row in enumerate(rows[:limit]):
        qa = row.get("qa", {})
        context = {"pre_text": row.get("pre_text", []), "post_text": row.get("post_text", []), "table": row.get("table", [])}
        values.append({"task_id": f"finqa_{i:04d}", "prompt": qa.get("question", ""), "answer": qa.get("answer"), "context": context, "metadata": {"program": qa.get("program"), "gold_inds": qa.get("gold_inds"), "source": "FinQA"}})
    write_jsonl(target, values)


def _clean_pubmedqa(source: Path, target: Path, limit: int) -> None:
    rows = json.loads(source.read_text(encoding="utf-8"))
    values = []
    for i, (pmid, row) in enumerate(list(rows.items())[:limit]):
        contexts = row.get("CONTEXTS", [])
        values.append({"task_id": f"pubmedqa_{i:04d}", "prompt": row.get("QUESTION", ""), "answer": row.get("final_decision") or row.get("LONG_ANSWER"), "context": {"contexts": contexts, "labels": row.get("LABELS", [])}, "metadata": {"pmid": pmid, "source": "PubMedQA"}})
    write_jsonl(target, values)


def _clean_swebench(source: Path, target: Path, limit: int) -> None:
    raw = json.loads(source.read_text(encoding="utf-8"))
    rows = [x.get("row", x) for x in raw.get("rows", raw if isinstance(raw, list) else [])]
    values = []
    for i, row in enumerate(rows[:limit]):
        values.append({"task_id": row.get("instance_id", f"swebench_{i:04d}"), "prompt": row.get("problem_statement", ""), "answer": row.get("patch", ""), "context": {"repo": row.get("repo"), "base_commit": row.get("base_commit"), "test_patch": row.get("test_patch"), "hints": row.get("hints_text")}, "metadata": {"source": "SWE-bench Verified", "version": row.get("version")}})
    write_jsonl(target, values)


def build_packages(seed: int = 17, tasks_per_package: int = 5) -> None:
    random.seed(seed)
    PACKAGES.mkdir(parents=True, exist_ok=True)
    specs = [
        ("software", "software_engineering", "Repository Maintenance Assistant", CLEAN / "software_tasks.jsonl", software_caps(), software_contracts()),
        ("finance", "financial_analysis", "Financial Report Analysis Assistant", CLEAN / "finance_tasks.jsonl", finance_caps(), finance_contracts()),
        ("biomedical", "biomedical_evidence", "Biomedical Evidence Assistant", CLEAN / "biomedical_tasks.jsonl", biomedical_caps(), biomedical_contracts()),
    ]
    summary = []
    for prefix, domain, application, task_file, caps, contracts in specs:
        tasks = read_jsonl(task_file)
        for offset in range(0, len(tasks), tasks_per_package):
            chunk = tasks[offset:offset + tasks_per_package]
            if not chunk:
                continue
            package_id = f"{prefix}_{offset // tasks_per_package:03d}"
            profile = package_profile(domain, offset // tasks_per_package)
            package = DomainPackage(
                package_id=package_id,
                domain=domain,
                application=application,
                source={"dataset": chunk[0]["metadata"].get("source"), "raw_tasks": [x["task_id"] for x in chunk], "construction_version": "pilot-0.1"},
                requirement=profile["requirement"],
                capabilities=caps,
                contracts=contracts,
                execution_tasks=[ExecutionTask(x["task_id"], x["prompt"], x.get("answer"), x.get("context"), x.get("metadata", {})) for x in chunk],
                mutations=profile["mutations"],
                split="dev" if offset // tasks_per_package < 4 else "test",
                metadata={"source_provenance_required": True, "gold_graph": False, "notes": "Contracts define acceptable behavior; no unique MAS topology is required."},
            )
            package.validate()
            write_json(PACKAGES / f"{package_id}.json", package.to_dict())
            summary.append({"package_id": package_id, "domain": domain, "tasks": len(chunk), "split": package.split})
    write_json(PACKAGES / "index.json", summary)


def package_profile(domain: str, index: int) -> dict[str, Any]:
    profiles = {
        "software_engineering": [
            ("bug repair", "understand repository issue, locate the fault, propose a tested patch", "reviewer approval for changed code"),
            ("regression repair", "identify regression, reproduce it, patch it, and run relevant tests", "tests must pass before final report"),
            ("maintenance planning", "inspect a repository issue and produce an evidence-backed change plan", "all claims must cite repository artifacts"),
        ],
        "financial_analysis": [
            ("numerical audit", "extract figures from a financial table, compute the answer, and show the evidence", "calculations must be independently checked"),
            ("filing analysis", "retrieve relevant report evidence and explain a financial answer", "final claims must link to table or text evidence"),
            ("risk analysis", "compare financial facts and produce a cautious analytical response", "uncertain conclusions require risk review"),
        ],
        "biomedical_evidence": [
            ("evidence retrieval", "find and synthesize biomedical evidence for a focused question", "claims must be traceable to retrieved evidence"),
            ("evidence adjudication", "compare biomedical findings and report uncertainty or disagreement", "contradictory evidence must be surfaced"),
            ("literature briefing", "produce a concise evidence briefing from relevant biomedical passages", "no unsupported clinical recommendation is allowed"),
        ],
    }
    name, process, governance = profiles[domain][index % len(profiles[domain])]
    return {"requirement": {"goal": name, "process": [process], "resources": ["approved_domain_sources", "evidence_store"], "governance": [governance], "output": ["structured_answer", "evidence_refs", "uncertainty"]}, "mutations": [Mutation("add_human_gate", "add", "requirement.governance", "human approval required for high-risk output", ["human_gate", "approval_trace"]), Mutation("source_restriction", "replace", "requirement.resources", ["approved_domain_sources"], ["no_unapproved_source"])]}


def software_caps() -> list[Capability]:
    return [Capability("issue_understanding", "reasoning", "parse issue and acceptance criteria", tags=["issue", "bug"]), Capability("repository_search", "tool", "locate files, symbols, and history", tags=["repository", "code"]), Capability("code_analysis", "reasoning", "localize likely fault and dependencies", tags=["bug", "analyze"]), Capability("patch_generation", "generation", "produce a minimal code patch", tags=["patch", "fix"]), Capability("test_execution", "tool", "run targeted and regression tests", tags=["test"]), Capability("regression_review", "governance", "review patch evidence and regressions", tags=["review"]), Capability("human_approval", "governance", "escalate risky changes to a human", risk="high")]


def finance_caps() -> list[Capability]:
    return [Capability("report_retrieval", "tool", "retrieve approved financial report passages", tags=["report", "evidence"]), Capability("table_parsing", "tool", "parse tables and row/column references", tags=["table", "extract"]), Capability("numerical_reasoning", "reasoning", "derive arithmetic answer from cited values", tags=["calculate", "number"]), Capability("evidence_linking", "governance", "link claims to report evidence", tags=["evidence", "cite"]), Capability("calculation_verification", "verification", "independently check calculations", tags=["verify", "calculate"]), Capability("report_generation", "generation", "write structured analytical response", tags=["answer", "report"]), Capability("risk_review", "governance", "flag uncertainty and unsupported recommendation", risk="high")]


def biomedical_caps() -> list[Capability]:
    return [Capability("literature_retrieval", "tool", "retrieve approved biomedical passages", tags=["evidence", "literature"]), Capability("relevance_screening", "reasoning", "screen passages against the question", tags=["relevant", "question"]), Capability("evidence_extraction", "reasoning", "extract findings, population, and limitations", tags=["evidence", "finding"]), Capability("evidence_synthesis", "reasoning", "synthesize consistent and conflicting findings", tags=["synthesis", "answer"]), Capability("contradiction_check", "verification", "detect disagreement and uncertainty", tags=["uncertain", "conflict"]), Capability("citation_verification", "governance", "verify claim-to-source references", tags=["cite", "source"]), Capability("expert_review", "governance", "escalate high-risk interpretation", risk="high")]


def software_contracts() -> list[Contract]:
    return _contracts(["issue_understanding", "repository_search", "code_analysis", "patch_generation", "test_execution", "regression_review"], [("repository_search", "code_analysis"), ("code_analysis", "patch_generation"), ("patch_generation", "test_execution"), ("test_execution", "regression_review")], ["test_execution", "regression_review"])


def finance_contracts() -> list[Contract]:
    return _contracts(["report_retrieval", "table_parsing", "numerical_reasoning", "evidence_linking", "calculation_verification", "report_generation"], [("report_retrieval", "table_parsing"), ("table_parsing", "numerical_reasoning"), ("numerical_reasoning", "calculation_verification"), ("calculation_verification", "report_generation")], ["evidence_linking", "calculation_verification"])


def biomedical_contracts() -> list[Contract]:
    return _contracts(["literature_retrieval", "relevance_screening", "evidence_extraction", "evidence_synthesis", "contradiction_check", "citation_verification"], [("literature_retrieval", "relevance_screening"), ("relevance_screening", "evidence_extraction"), ("evidence_extraction", "evidence_synthesis"), ("evidence_synthesis", "citation_verification")], ["citation_verification", "contradiction_check"])


def _contracts(required: list[str], orders: list[tuple[str, str]], runtime: list[str]) -> list[Contract]:
    values = [Contract(f"required_{x}", "capability_required", x, "selected_capability", f"capability {x} must be represented") for x in required]
    values += [Contract(f"order_{a}_{b}", "order_required", f"{a}<{b}", "reachable", f"{a} must precede {b}") for a, b in orders]
    values += [Contract(f"runtime_{x}", "runtime_required", x, "event_present", f"runtime evidence for {x} must be present") for x in runtime]
    return values


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--packages", action="store_true")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    if args.download:
        download_sources()
    if args.clean:
        clean_sources(args.limit)
    if args.packages:
        build_packages()
