from __future__ import annotations

from .schema import Q3Case, Q3Task


def build_q3_suite() -> list[Q3Case]:
    return [
        Q3Case("q3_seq_library_review", "library_review", "sequential", "software", "Review a library change request, inspect evidence, compare alternatives, verify the path, and publish a summary.", _tasks("inspect", "compare", "verify", "publish")),
        Q3Case("q3_branch_market_scan", "market_scan", "multi_branch", "finance", "Collect market signals in parallel, reconcile them, cross-check conflicts, and issue a conclusion.", _tasks("collect_a", "collect_b", "reconcile", "cross_check", "report")),
        Q3Case("q3_feedback_patch_loop", "patch_loop", "feedback_driven", "software", "Generate a patch, run validation, diagnose failures, revise the patch, and stop only when the loop closes.", _tasks("generate", "validate", "diagnose", "revise", "finalize")),
        Q3Case("q3_constraint_clinical", "clinical_review", "constraint_heavy", "biomedical", "Retrieve evidence, assess risk, enforce approval, and release only after governance checks pass.", _tasks("retrieve", "assess", "review", "release")),
        Q3Case("q3_seq_policy_digest", "policy_digest", "sequential", "legal", "Read policy material, extract findings, validate citations, and draft a digest.", _tasks("read", "extract", "validate", "draft")),
        Q3Case("q3_branch_due_diligence", "due_diligence", "multi_branch", "finance", "Analyze statements and risks in parallel, merge them, verify consistency, and produce a memo.", _tasks("statements", "risks", "merge", "verify", "memo")),
        Q3Case("q3_feedback_science_screen", "science_screen", "feedback_driven", "science", "Screen candidates, evaluate them, revise with feedback, and report only accepted candidates.", _tasks("screen", "evaluate", "feedback", "revise", "report")),
        Q3Case("q3_constraint_legal", "legal_review", "constraint_heavy", "legal", "Organize facts, retrieve authority, require professional review, and publish a constrained answer.", _tasks("facts", "authority", "review", "publish")),
    ]


def _tasks(*labels: str) -> list[Q3Task]:
    return [Q3Task(f"t{i+1}", label) for i, label in enumerate(labels)]

