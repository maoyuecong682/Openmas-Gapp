"""Dataset adapters for the revised cross-dataset Q2 table.

This module separates four concerns for every dataset:
1. requirement template: turns a row into an application construction request;
2. component ecosystem: reusable stages for that application;
3. execution adapter: parses/normalizes a model answer when available;
4. primary metric: the one metric shown in the cross-dataset table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RequirementTemplate:
    dataset_id: str
    domain: str
    family: str
    text: str
    stages: tuple[tuple[str, str], ...]
    constraints: tuple[tuple[str, str, str, str], ...] = ()

    def render(self, row: dict[str, Any]) -> str:
        question = row.get("question") or row.get("prompt") or ""
        return f"{self.text}\nDataset item: {question}"


class ExecutionAdapter:
    metric_name = "task_success"

    def normalize_prediction(self, value: Any) -> Any:
        return value

    def score(self, prediction: Any, gold: Any) -> float | None:
        if prediction is None or gold is None:
            return None
        return float(str(self.normalize_prediction(prediction)).strip().lower() == str(self.normalize_prediction(gold)).strip().lower())


class MultipleChoiceAdapter(ExecutionAdapter):
    metric_name = "accuracy"

    def normalize_prediction(self, value: Any) -> str:
        text = str(value).strip().lower()
        for token in ("answer:", "option", "choice"):
            text = text.replace(token, "")
        text = text.strip()
        # Keep option content intact.  The case builder canonicalizes integer
        # gold labels to their choice text, while predictions may be a letter,
        # an index, or the full choice content.
        if text.startswith("(") and ")" in text:
            text = text[1:text.find(")")].strip() or text
        return text.splitlines()[0].strip() if text else text

    def score(self, prediction: Any, gold: Any) -> float | None:
        import re
        if prediction is None or gold is None:
            return None
        p = self.normalize_prediction(prediction)
        raw_gold = str(gold).strip().lower()
        if raw_gold.startswith("choice:"):
            payload = raw_gold.split(":", 1)[1]
            letter, _, text = payload.partition("|")
            g = letter.strip()
            gold_text = self.normalize_prediction(text)
            if p == gold_text:
                return 1.0
            # Accept standard answer forms without treating an arbitrary first
            # character as an option label.
            label_match = re.search(
                r"(?:^|\b)(?:answer\s*(?:is|:)?\s*|option\s*|choice\s*|\()?([a-j])(?:\)|[.:\-]|\b)",
                str(prediction).strip().casefold(),
            )
            if label_match:
                return float(label_match.group(1) == g)
            # Some providers return a short explanation followed by the exact
            # option text. Require the full normalized text to avoid substring
            # matches such as "ion" inside "generation".
            normalized_verbose = " ".join(re.findall(r"[\w]+", str(prediction).casefold()))
            normalized_gold_text = " ".join(re.findall(r"[\w]+", gold_text.casefold()))
            if normalized_gold_text and re.search(
                    rf"(?<!\w){re.escape(normalized_gold_text)}(?!\w)", normalized_verbose):
                return 1.0
        else:
            g = self.normalize_prediction(gold)
        # Normalized cases may carry both the canonical letter and option
        # text. This keeps letter-only and text answers equivalent without
        # allowing arbitrary first-character matches.
        if p == g:
            return 1.0
        # Accept a bare option letter only when the canonical gold is also a
        # bare letter. Do not compare first characters of arbitrary answer
        # text (e.g. "a" and "algebra"), which creates false positives.
        if len(p) == 1 and p in "abcdefghij" and len(g) == 1 and g in "abcdefghij":
            return float(p == g)
        return 0.0


class MathAnswerAdapter(ExecutionAdapter):
    metric_name = "math_answer_accuracy"

    def normalize_prediction(self, value: Any) -> str:
        import re
        text = str(value).strip()
        boxed = re.findall(r"\\boxed\{(.+?)\}", text)
        text = boxed[-1] if boxed else text
        # Canonicalize common equivalent LaTeX forms without requiring a CAS.
        text = text.replace("\\left", "").replace("\\right", "")
        text = text.replace("\\!", "").replace("\\,", "")
        text = text.replace("\\pi", "pi").replace("π", "pi")
        text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
        text = re.sub(r"\\(?:dfrac|tfrac)\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
        text = re.sub(r"\s+", "", text).lower()
        # The preceding fraction rewrite intentionally avoids ambiguous
        # nested braces; remove the harmless parentheses it introduces around
        # atomic numerator/denominator expressions.
        text = re.sub(r"\(([^(),]+)\)/\(([^(),]+)\)", r"\1/\2", text)
        # Strip one redundant pair of outer delimiters.
        while len(text) >= 2 and text[0] == "{" and text[-1] == "}":
            text = text[1:-1]
        return text

    def score(self, prediction: Any, gold: Any) -> float | None:
        if prediction is None or gold is None:
            return None
        return float(self.normalize_prediction(prediction) == self.normalize_prediction(gold))


class NumericAdapter(ExecutionAdapter):
    metric_name = "numeric_accuracy"

    def normalize_prediction(self, value: Any) -> str:
        import re
        text = str(value).replace(",", "")
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return nums[-1] if nums else text.strip().lower()


class FinanceBenchAdapter(ExecutionAdapter):
    metric_name = "financebench_accuracy"

    def normalize_prediction(self, value: Any) -> str:
        import json
        import re
        text = str(value).strip()
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            for key in ("answer", "candidate_answer", "final_answer", "value", "text"):
                candidate = payload.get(key)
                if candidate is not None:
                    text = str(candidate).strip()
                    break
        text = re.sub(r"^(?:final\s+answer|candidate_answer|answer|value)\s*[:=]\s*", "", text, flags=re.I)
        text = text.replace("$", "").replace(",", "").strip()
        return " ".join(text.split())

    def score(self, prediction: Any, gold: Any) -> float | None:
        import re
        if prediction is None or gold is None or not str(gold).strip():
            return None
        p = self.normalize_prediction(prediction)
        g = self.normalize_prediction(gold)

        def _numeric(text: str) -> float | None:
            cleaned = text.replace("%", "")
            match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
            if not match:
                return None
            try:
                return float(match.group(0))
            except ValueError:
                return None

        p_num = _numeric(p)
        g_num = _numeric(g)
        if p_num is not None and g_num is not None:
            tolerance = max(0.01, abs(g_num) * 0.005)
            return float(abs(p_num - g_num) <= tolerance)
        p_norm = re.sub(r"[^a-z0-9%./+-]", " ", p.casefold())
        g_norm = re.sub(r"[^a-z0-9%./+-]", " ", g.casefold())
        return float(" ".join(p_norm.split()) == " ".join(g_norm.split()))


class BBHAdapter(ExecutionAdapter):
    metric_name = "bbh_accuracy"

    def normalize_prediction(self, value: Any) -> str:
        import json
        import re
        text = str(value).strip()
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            for key in ("answer", "candidate_answer", "final_answer", "value", "text"):
                candidate = payload.get(key)
                if candidate is not None:
                    text = str(candidate).strip()
                    break
        text = re.sub(r"^(?:final\s+answer|answer|result|output)\s*[:=]\s*", "", text, flags=re.I)
        text = text.replace("True", "true").replace("False", "false")
        text = re.sub(r"^[\s\"'`([{]+|[\s\"'`)\]}]+$", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.casefold()

    @staticmethod
    def _option_label(value: Any) -> str | None:
        """Extract an explicit BBH multiple-choice label."""
        import re
        match = re.match(r"^\s*\(?([a-jA-J])\)?(?:\s|[.):\-]|$)", str(value))
        return match.group(1).casefold() if match else None

    def score(self, prediction: Any, gold: Any) -> float | None:
        if prediction is None or gold is None:
            return None
        gold_label = self._option_label(gold)
        normalized_gold = self.normalize_prediction(gold)
        if gold_label and normalized_gold == gold_label:
            return float(self._option_label(prediction) == gold_label)
        return float(self.normalize_prediction(prediction) == normalized_gold)


class SciBenchAdapter(NumericAdapter):
    metric_name = "scibench_numeric_accuracy"

    def normalize_prediction(self, value: Any) -> str:
        number, unit = self._value_and_unit(value)
        return f"{number} {unit}".strip() if number is not None else str(value).strip().casefold()

    def score(self, prediction: Any, gold: Any) -> float | None:
        import math
        if prediction is None or gold is None:
            return None
        predicted, predicted_unit = self._value_and_unit(prediction)
        expected, expected_unit = self._value_and_unit(gold)
        if predicted is None or expected is None:
            return 0.0
        numeric_match = math.isclose(predicted, expected, rel_tol=5e-3, abs_tol=1e-8)
        unit_match = not expected_unit or predicted_unit == expected_unit
        return float(numeric_match and unit_match)

    @staticmethod
    def _value_and_unit(value: Any) -> tuple[float | None, str]:
        import json
        import re
        explicit_unit = ""
        if isinstance(value, dict):
            explicit_unit = str(value.get("unit") or "")
            value = value.get("value", value.get("answer", ""))
        else:
            text = str(value).strip()
            try:
                payload = json.loads(text)
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, dict):
                explicit_unit = str(payload.get("unit") or "")
                value = payload.get("value", payload.get("answer", ""))
        text = str(value).strip().replace(",", "")
        text = re.sub(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*[×x]\s*10\s*\^\s*\{?([+-]?\d+)\}?",
                      lambda match: f"{match.group(1)}e{match.group(2)}", text)
        match = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text)
        if not match:
            return None, SciBenchAdapter._normalize_unit(explicit_unit)
        try:
            number = float(match.group(0))
        except ValueError:
            return None, SciBenchAdapter._normalize_unit(explicit_unit)
        suffix = text[match.end():].strip(" .,:;=()[]")
        unit = explicit_unit or suffix
        return number, SciBenchAdapter._normalize_unit(unit)

    @staticmethod
    def _normalize_unit(value: str) -> str:
        import re
        text = str(value).strip().casefold()
        text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", text)
        text = text.replace("\\cdot", "*").replace("\\,", "")
        text = text.replace("$", "").replace("~", "").replace("{", "").replace("}", "")
        text = re.sub(r"\s+", "", text)
        return text


class FinQAAdapter(NumericAdapter):
    """Numeric adapter for the audited FinQA pilot.

    The row retains FinQA's table and program in ``raw``.  This adapter only
    scores the final numeric answer; a full release should additionally audit
    execution of the annotated program.
    """
    metric_name = "finqa_numeric_accuracy"

    def normalize_prediction(self, value: Any) -> str:
        import json, re
        text = str(value).strip()
        # Accept an explicit answer envelope but ignore explanatory numbers.
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            for key in ("answer", "candidate_answer", "final_answer", "value"):
                if payload.get(key) is not None:
                    text = str(payload[key]).strip()
                    break
        labeled = re.findall(r"(?:candidate_answer|final_answer|answer)\s*[:=]\s*([-+]?\d[\d,]*(?:\.\d+)?%?)", text, re.I)
        if labeled:
            text = labeled[-1]
        return super().normalize_prediction(text)

    def score(self, prediction: Any, gold: Any) -> float | None:
        """Accept FinQA's documented rounding convention for numeric programs."""
        if prediction is None or gold is None or not str(gold).strip():
            return None
        try:
            predicted = float(self.normalize_prediction(prediction).replace("%", ""))
            expected = float(self.normalize_prediction(gold).replace("%", ""))
        except ValueError:
            return 0.0
        tolerance = max(0.01, 0.5 if abs(expected) >= 100 else abs(expected) * 0.005)
        return float(abs(predicted - expected) <= tolerance)


class PubMedQAAdapter(ExecutionAdapter):
    metric_name = "pubmedqa_accuracy"

    def normalize_prediction(self, value: Any) -> str:
        import re
        text = str(value).strip().lower()
        text = re.sub(r"^(?:final\s+answer|answer|label)\s*:\s*", "", text)
        explicit = re.match(r"^(yes|no|maybe)\b", text)
        if explicit:
            return explicit.group(1)
        return text.splitlines()[0].strip()


class YesNoAdapter(ExecutionAdapter):
    metric_name = "yes_no_accuracy"

    def normalize_prediction(self, value: Any) -> str:
        import re
        text = str(value).strip().lower()
        if text.startswith("choice:"):
            _, _, payload = text.partition(":")
            _, _, text = payload.partition("|")
            text = text or payload
        text = text.strip()
        text = re.sub(r"^[\s\"'`([{]+|[\s\"'`，。,.!?:;)\]}]+$", "", text)
        for token in ("true", "yes", "1", "是"):
            if text == token or text.startswith(token + " "):
                return "yes"
        for token in ("false", "no", "0", "否"):
            if text == token or text.startswith(token + " "):
                return "no"
        first = text.splitlines()[0].strip()
        first = re.sub(r"^[\s\"'`([{]+|[\s\"'`，。,.!?:;)\]}]+$", "", first)
        return first


class DropAdapter(ExecutionAdapter):
    """DROP accepts numbers, dates, and entity spans; it is not numeric-only."""
    metric_name = "drop_em_f1"

    def normalize_prediction(self, value: Any) -> str:
        import re
        text = str(value).strip().lower()
        text = re.sub(r"[^a-z0-9%./ -]", " ", text)
        return " ".join(text.split())

    def score(self, prediction: Any, gold: Any) -> float | None:
        if prediction is None or gold is None:
            return None
        p, g = self.normalize_prediction(prediction), self.normalize_prediction(gold)
        if p == g:
            return 1.0
        from collections import Counter
        pt, gt = p.split(), g.split()
        overlap = sum((Counter(pt) & Counter(gt)).values())
        precision = overlap / len(pt) if pt else 0.0
        recall = overlap / len(gt) if gt else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


class TextF1Adapter(ExecutionAdapter):
    metric_name = "f1"

    def normalize_prediction(self, value: Any) -> str:
        """HotpotQA-style normalization: lowercase, punctuation/articles, whitespace."""
        import re
        import string
        text = str(value).casefold()
        text = "".join(character for character in text if character not in string.punctuation)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return " ".join(text.split())

    def score(self, prediction: Any, gold: Any) -> float | None:
        if prediction is None or gold is None:
            return None
        from collections import Counter
        p = self.normalize_prediction(prediction).split()
        g = self.normalize_prediction(gold).split()
        if not p or not g: return 0.0
        overlap = sum((Counter(p) & Counter(g)).values())
        precision = overlap / len(p); recall = overlap / len(g)
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


class CodeTestAdapter(ExecutionAdapter):
    metric_name = "unit_test_pass"

    def score(self, prediction: Any, gold: Any) -> float | None:
        # Actual code evaluation is performed by the sandbox executor. A text
        # answer alone is not awarded correctness.
        return None


class PatchTraceAdapter(ExecutionAdapter):
    metric_name = "swebench_resolved"

    def score(self, prediction: Any, gold: Any) -> float | None:
        return None


@dataclass(frozen=True)
class DatasetAdapter:
    dataset_id: str
    source_file: str
    split: str
    license: str
    template: RequirementTemplate
    execution: ExecutionAdapter

    @property
    def remote_spec(self) -> dict[str, Any] | None:
        """Return the remote row source, when this adapter supports one.

        Remote sources are deliberately metadata only.  The row fetcher owns
        transport and normalization so importing the benchmark never creates a
        Hugging Face datasets cache.
        """
        specs = {
            "MedQA": {
                "dataset": "openlifescienceai/medqa",
                "config": "default",
                "split": "test",
                "kind": "medqa",
            },
            "MedMCQA": {
                "dataset": "openlifescienceai/medmcqa",
                "config": "default",
                "split": "validation",
                "kind": "medmcqa",
            },
            "PubMedQA": {
                "dataset": "qiaojin/PubMedQA",
                "config": "pqa_labeled",
                "split": "train",
                "kind": "pubmedqa",
            },
            "MMLU-Medical": {
                "dataset": "cais/mmlu",
                "configs": (
                    "anatomy",
                    "clinical_knowledge",
                    "college_biology",
                    "college_medicine",
                    "medical_genetics",
                    "professional_medicine",
                    "human_aging",
                    "nutrition",
                    "virology",
                ),
                "split": "test",
                "kind": "mmlu_medical",
            },
        }
        spec = specs.get(self.dataset_id)
        return dict(spec) if spec else None

    @property
    def task_profile(self) -> dict[str, Any]:
        """Answer-independent prior used to specialize harness construction."""
        profiles = {
            "GSM8K": {"task_family": "numeric_reasoning", "requires_numeric_tool": True,
                      "requires_multi_branch": False, "requires_evidence_merge": False,
                      "requires_constraint_gate": False},
            "MATH-500": {"task_family": "symbolic_reasoning", "requires_numeric_tool": True,
                          "requires_multi_branch": False, "requires_evidence_merge": False,
                          "requires_constraint_gate": False},
            "MMLU": {"task_family": "multiple_choice", "requires_numeric_tool": False,
                      "requires_multi_branch": False, "requires_evidence_merge": False,
                      "requires_constraint_gate": False},
            "MMLU-Pro": {"task_family": "multiple_choice_hard", "requires_numeric_tool": False,
                         "requires_multi_branch": False, "requires_evidence_merge": False,
                         "requires_constraint_gate": False},
            "ARC": {"task_family": "science_reasoning", "requires_numeric_tool": False,
                    "requires_multi_branch": False, "requires_evidence_merge": False,
                    "requires_constraint_gate": False},
            "BBH": {"task_family": "hard_reasoning", "requires_numeric_tool": False,
                    "requires_multi_branch": True, "requires_evidence_merge": True,
                    "requires_constraint_gate": False},
            "BBH-Full": {"task_family": "hard_reasoning", "requires_numeric_tool": False,
                         "requires_multi_branch": True, "requires_evidence_merge": True,
                         "requires_constraint_gate": False},
            "FinanceBench": {"task_family": "financial_evidence", "requires_numeric_tool": True,
                             "requires_multi_branch": True, "requires_evidence_merge": True,
                             "requires_constraint_gate": False},
            "SciBench": {"task_family": "scientific_numeric", "requires_numeric_tool": True,
                         "requires_multi_branch": False, "requires_evidence_merge": False,
                         "requires_constraint_gate": False},
            "LogiQA": {"task_family": "logical_reasoning", "requires_numeric_tool": False,
                       "requires_multi_branch": False, "requires_evidence_merge": False,
                       "requires_constraint_gate": False},
            "HotpotQA": {"task_family": "multi_hop_qa", "requires_numeric_tool": False,
                          "requires_multi_branch": True, "requires_evidence_merge": True,
                          "requires_constraint_gate": False},
            "DROP": {"task_family": "reading_math", "requires_numeric_tool": True,
                      "requires_multi_branch": False, "requires_evidence_merge": False,
                      "requires_constraint_gate": False},
            "PubMedQA": {"task_family": "biomedical_evidence", "requires_numeric_tool": False,
                         "requires_multi_branch": True, "requires_evidence_merge": True,
                         "requires_constraint_gate": True},
            "MedMCQA": {"task_family": "medical_multiple_choice", "requires_numeric_tool": False,
                         "requires_multi_branch": False, "requires_evidence_merge": False,
                         "requires_constraint_gate": True},
            "MMLU-Medical": {"task_family": "medical_multiple_choice", "requires_numeric_tool": False,
                              "requires_multi_branch": False, "requires_evidence_merge": False,
                              "requires_constraint_gate": True},
            "FinQA": {"task_family": "financial_program", "requires_numeric_tool": True,
                       "requires_multi_branch": False, "requires_evidence_merge": False,
                       "requires_constraint_gate": False},
            "MuSiQue": {"task_family": "multi_hop_qa", "requires_numeric_tool": False,
                        "requires_multi_branch": True, "requires_evidence_merge": True,
                        "requires_constraint_gate": False},
        }
        profile = dict(profiles.get(self.dataset_id, {}))
        profile.setdefault("task_family", self.template.family)
        profile.setdefault("requires_numeric_tool", False)
        profile.setdefault("requires_multi_branch", self.template.family == "multi_branch")
        profile.setdefault("requires_evidence_merge", self.template.family == "multi_branch")
        profile.setdefault("requires_constraint_gate", bool(self.template.constraints))
        profile["dataset"] = self.dataset_id
        return profile

    def build_requirement(self, row: dict[str, Any]) -> str:
        return self.template.render(row)

    def primary_score(self, prediction: Any, gold: Any, runtime_valid: float, trace_rate: float) -> float | None:
        # Runtime and trace are diagnostic columns, not multiplicative gates.
        # Gating made a correct answer score zero for unrelated contract noise.
        return self.execution.score(prediction, gold)


def _template(dataset_id, domain, family, text, stages, constraints=()):
    return RequirementTemplate(dataset_id, domain, family, text, tuple(stages), tuple(constraints))


DATASET_ADAPTERS: dict[str, DatasetAdapter] = {
    "gsm8k": DatasetAdapter("GSM8K", "q2_datasets/normalized/gsm8k.jsonl", "test", "MIT", _template("GSM8K", "math", "sequential", "Solve the multi-step arithmetic problem, show intermediate calculations, independently verify the result, and return the final numeric answer.", [("parse", "Parse the arithmetic problem"), ("plan", "Plan the calculation"), ("calculate", "Execute the arithmetic calculation"), ("verify", "Verify the calculation"), ("answer", "Return the numeric answer")]), NumericAdapter()),
    "math500": DatasetAdapter("MATH-500", "q2_datasets/normalized/math500.jsonl", "test", "MIT", _template("MATH-500", "math", "sequential", "Solve the competition mathematics problem, select the applicable theorem or strategy, derive the solution, verify it, and return the final answer.", [("parse", "Parse the mathematical problem"), ("strategy", "Select a solution strategy"), ("solve", "Derive the solution"), ("verify", "Verify the derivation"), ("answer", "Return the final answer")]), MathAnswerAdapter()),
    "mmlu": DatasetAdapter("MMLU", "q2_datasets/normalized/mmlu.jsonl", "test", "MIT", _template("MMLU", "multidomain", "sequential", "Answer the multidomain academic question by identifying the relevant knowledge, reasoning over the choices, checking the conclusion, and returning one option.", [("identify", "Identify the domain and relevant knowledge"), ("reason", "Reason over the candidate choices"), ("check", "Check the selected option"), ("answer", "Return one option")]), MultipleChoiceAdapter()),
    "mmlu_pro": DatasetAdapter("MMLU-Pro", "q2_datasets/normalized/mmlu_pro.jsonl", "test", "MIT", _template("MMLU-Pro", "multidomain", "sequential", "Answer the difficult multidomain question by identifying the domain, comparing all options, checking the conclusion, and returning one option.", [("identify", "Identify the domain and relevant knowledge"), ("reason", "Reason over all candidate choices"), ("check", "Check the selected option"), ("answer", "Return one option")]), MultipleChoiceAdapter()),
    "arc": DatasetAdapter("ARC", "q2_datasets/normalized/arc.jsonl", "test", "Apache-2.0", _template("ARC", "science", "sequential", "Answer the science reasoning question using the provided choices, verify the evidence, and return one option.", [("interpret", "Interpret the science question"), ("compare", "Compare candidate choices"), ("verify", "Verify the selected choice"), ("answer", "Return one option")]), MultipleChoiceAdapter()),
    "humaneval": DatasetAdapter("HumanEval", "q2_datasets/normalized/humaneval.jsonl", "test", "MIT", _template("HumanEval", "software", "feedback_driven", "Implement the requested function, run the hidden-style tests, diagnose failures, revise the implementation, and return a tested function.", [("understand", "Understand the function specification"), ("implement", "Implement the function"), ("test", "Run unit tests"), ("revise", "Revise after failures"), ("report", "Return the tested implementation")]), CodeTestAdapter()),
    "mbpp": DatasetAdapter("MBPP", "q2_datasets/normalized/mbpp.jsonl", "test", "CC-BY-4.0", _template("MBPP", "software", "feedback_driven", "Implement the described Python program, execute the provided tests, repair failures, and return a passing implementation.", [("understand", "Understand the programming task"), ("implement", "Implement the program"), ("test", "Run the provided tests"), ("repair", "Repair failing behavior"), ("report", "Return the implementation")]), CodeTestAdapter()),
    "bbh": DatasetAdapter("BBH", "q2_datasets/normalized/bbh.jsonl", "test", "MIT", _template("BBH", "reasoning", "sequential", "Solve the BIG-Bench Hard reasoning task, inspect all relevant clues, verify the conclusion, and return the exact answer.", [("parse", "Parse the reasoning task"), ("reason", "Reason over the clues"), ("verify", "Verify the conclusion"), ("answer", "Return the exact answer")]), ExecutionAdapter()),
    "bbh_full": DatasetAdapter("BBH-Full", "q2_datasets/normalized/bbh_full.jsonl", "test", "MIT", _template("BBH-Full", "reasoning", "multi_branch", "Solve the BIG-Bench Hard reasoning task by splitting the clues into two reasoning branches, merging them, verifying the conclusion, and returning the exact answer.", [("parse", "Parse the reasoning task"), ("reason_a", "Reason over the first clue branch"), ("reason_b", "Reason over the second clue branch"), ("merge", "Merge the reasoning branches"), ("verify", "Verify the conclusion"), ("answer", "Return the exact answer")]), BBHAdapter()),
    "financebench": DatasetAdapter("FinanceBench", "q2_datasets/normalized/financebench.jsonl", "train", "CC BY-NC 4.0", _template("FinanceBench", "finance", "multi_branch", "Answer the finance question by retrieving evidence from the filing, reconciling the evidence, checking the calculation, and returning a concise answer.", [("retrieve_a", "Retrieve first evidence branch"), ("retrieve_b", "Retrieve second evidence branch"), ("synthesize", "Synthesize evidence across branches"), ("verify", "Verify the finance result"), ("answer", "Return a concise answer")]), FinanceBenchAdapter()),
    "scibench": DatasetAdapter("SciBench", "q2_datasets/normalized/scibench.jsonl", "train", "unknown", _template("SciBench", "science", "sequential", "Solve the science problem by interpreting the prompt, deriving the result, checking the calculation, and returning the final answer.", [("parse", "Parse the science problem"), ("derive", "Derive the result"), ("verify", "Verify the result"), ("answer", "Return the final answer")]), SciBenchAdapter()),
    "logiqa": DatasetAdapter("LogiQA", "q2_datasets/normalized/logiqa.jsonl", "test", "Apache-2.0", _template("LogiQA", "logic", "sequential", "Solve the logical reasoning question by analyzing the passage, comparing the choices, verifying the inference, and returning one option.", [("interpret", "Interpret the logical passage"), ("reason", "Reason over the argument"), ("verify", "Verify the inference"), ("answer", "Return one option")]), MultipleChoiceAdapter()),
    "hotpotqa": DatasetAdapter("HotpotQA", "q2_datasets/normalized/hotpotqa.jsonl", "validation", "CC-BY-SA-4.0", _template("HotpotQA", "open_domain", "multi_branch", "Answer the multi-hop question by retrieving supporting evidence from both hops, synthesizing the evidence, verifying citations, and returning a concise answer.", [("retrieve_a", "Retrieve first-hop evidence"), ("retrieve_b", "Retrieve second-hop evidence"), ("synthesize", "Synthesize both evidence streams"), ("verify", "Verify supporting facts"), ("answer", "Return the answer")]), TextF1Adapter()),
    "drop": DatasetAdapter("DROP", "q2_datasets/normalized/drop.jsonl", "validation", "CC-BY-SA-4.0", _template("DROP", "reading_math", "sequential", "Answer the reading-comprehension question by extracting relevant numbers or dates, performing the required discrete operation, verifying the result, and returning the answer.", [("retrieve", "Retrieve relevant passage evidence"), ("extract", "Extract numbers or dates"), ("compute", "Perform the required operation"), ("verify", "Verify the result"), ("answer", "Return the answer")]), DropAdapter()),
    "medqa": DatasetAdapter("MedQA", "q9_datasets/normalized/medqa.jsonl", "test", "unverified", _template("MedQA", "medicine", "constraint_heavy", "Answer the clinical question using evidence retrieval, differential reasoning, safety checking, and mandatory professional review before returning one option.", [("retrieve", "Retrieve clinical evidence"), ("reason", "Reason over the clinical case"), ("safety", "Check safety and contraindications"), ("review", "Obtain professional review"), ("answer", "Return one option")], [("human_review", "human_approval", "answer", "required")]), MultipleChoiceAdapter()),
    "medmcqa": DatasetAdapter("MedMCQA", "q9_datasets/normalized/medmcqa.jsonl", "validation", "CC-BY-SA-4.0; verify source terms", _template("MedMCQA", "medicine", "constraint_heavy", "Answer the medical multiple-choice question using evidence retrieval, differential reasoning, safety checking, and mandatory professional review before returning one option.", [("retrieve", "Retrieve medical evidence"), ("reason", "Reason over the clinical question"), ("safety", "Check safety and contraindications"), ("review", "Obtain professional review"), ("answer", "Return one option")], [("human_review", "human_approval", "answer", "required")]), MultipleChoiceAdapter()),
    "pubmedqa": DatasetAdapter("PubMedQA", "q9_datasets/normalized/pubmedqa.jsonl", "pilot", "MIT; verify source terms", _template("PubMedQA", "biomedical", "constraint_heavy", "Answer the biomedical yes/no/maybe question by retrieving the relevant evidence, distinguishing findings from background, checking uncertainty, and returning exactly yes, no, or maybe.", [("retrieve", "Retrieve relevant biomedical evidence"), ("interpret", "Interpret the study findings"), ("uncertainty", "Check uncertainty and alternative explanations"), ("review", "Review the evidence chain"), ("answer", "Return yes, no, or maybe")], [("evidence_review", "evidence_approval", "answer", "required")]), PubMedQAAdapter()),
    "mmlu_medical": DatasetAdapter("MMLU-Medical", "q9_datasets/normalized/mmlu_medical.jsonl", "test", "MIT", _template("MMLU-Medical", "medicine", "constraint_heavy", "Answer the medical knowledge question using evidence retrieval, differential reasoning, safety checking, and mandatory professional review before returning one option.", [("retrieve", "Retrieve medical evidence"), ("reason", "Reason over the medical choices"), ("safety", "Check safety and contraindications"), ("review", "Obtain professional review"), ("answer", "Return one option")], [("human_review", "human_approval", "answer", "required")]), MultipleChoiceAdapter()),
    "finqa": DatasetAdapter("FinQA", "q2_datasets/normalized/finqa.jsonl", "pilot", "CC-BY-4.0", _template("FinQA", "finance", "sequential", "Answer the financial question by locating the relevant table values, selecting the arithmetic program, executing the calculation, verifying units, and returning the numeric result.", [("retrieve", "Retrieve relevant report and table evidence"), ("select", "Select the relevant values and operation"), ("execute", "Execute the arithmetic program"), ("verify", "Verify units and calculation"), ("answer", "Return the numeric result")]), FinQAAdapter()),
    "musique": DatasetAdapter("MuSiQue", "q2_datasets/normalized/musique.jsonl", "pilot", "CC-BY-4.0", _template("MuSiQue", "open_domain", "multi_branch", "Answer the controlled multi-hop question by retrieving linked evidence, following the annotated reasoning chain, merging the branches, verifying support, and returning the answer.", [("retrieve_a", "Retrieve first evidence branch"), ("retrieve_b", "Retrieve linked second evidence branch"), ("merge", "Merge the multi-hop evidence"), ("verify", "Verify the reasoning chain"), ("answer", "Return the answer")]), TextF1Adapter()),
    "strategyqa": DatasetAdapter("StrategyQA", "q2_datasets/normalized/strategyqa.jsonl", "pilot", "verify", _template("StrategyQA", "commonsense", "multi_branch", "Answer the implicit multi-hop question by grounding the requirement, decomposing it into subquestions, checking the supporting facts, and returning yes or no.", [("ground", "Ground the question and target"), ("decompose", "Decompose into subquestions"), ("reason", "Reason over the supporting facts"), ("verify", "Verify the conclusion"), ("answer", "Return yes or no")]), YesNoAdapter()),
    "sciq": DatasetAdapter("SciQ", "q2_datasets/normalized/sciq.jsonl", "test", "unverified", _template("SciQ", "science", "sequential", "Answer the science question using the supporting passage, compare the four candidate choices, verify the evidence, and return one choice.", [("retrieve", "Retrieve supporting science evidence"), ("interpret", "Interpret the evidence"), ("compare", "Compare candidate choices"), ("verify", "Verify the selected choice"), ("answer", "Return one choice")]), MultipleChoiceAdapter()),
    "swebench_verified": DatasetAdapter("SWE-bench", "q2_datasets/normalized/swebench_verified.jsonl", "test", "verify", _template("SWE-bench", "software", "feedback_driven", "Repair the repository issue, run targeted tests, diagnose failures, revise the patch, obtain review, and emit only a validated patch.", [("inspect", "Inspect the repository issue"), ("patch", "Construct a patch"), ("test", "Run targeted tests"), ("revise", "Revise after test feedback"), ("review", "Review the patch"), ("emit", "Emit the validated patch")]), PatchTraceAdapter()),
}


def all_adapters() -> list[DatasetAdapter]:
    return list(DATASET_ADAPTERS.values())
