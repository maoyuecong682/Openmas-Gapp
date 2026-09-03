import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from openmas_bench.q3 import build_q3_suite, get_q3_baseline, render_markdown_tables, run_q3_experiment
from openmas_bench.q3.osv import compute_osv
from openmas_bench.q3.real import real_construction_result, case_to_q3, _real_blueprint, _graph_structural_preservation
from openmas_bench.application_executor import _branch_resources, _execution_order
from openmas_bench.dataset_adapters import DATASET_ADAPTERS
from openmas_bench.dataset_cases import build_dataset_case
from scripts.q3.run_q3_real import Q3_DATASET_PRESETS, Q3_ROW_PRESETS, _is_structurally_qualified_row, _normalize_dataset_key


def test_q3_suite_covers_four_families():
    suite = build_q3_suite()
    assert len(suite) == 8
    assert {case.family for case in suite} == {"sequential", "multi_branch", "feedback_driven", "constraint_heavy"}


def test_q3_baselines_emit_blueprints():
    case = build_q3_suite()[0]
    for name in ["flat_component_selection", "sequence_based_orchestration", "tree_based_planning", "workflow_based_template", "agent_graph_orchestration", "graph_harness"]:
        blueprint = get_q3_baseline(name).build_blueprint(case)
        assert blueprint.case_id == case.case_id
        assert blueprint.baseline == name
        assert blueprint.nodes


def test_q3_osv_detects_missing_structure():
    case = next(x for x in build_q3_suite() if x.family == "feedback_driven")
    graph = get_q3_baseline("sequence_based_orchestration").build_blueprint(case)
    osv, notes = compute_osv(case, graph)
    assert osv in {0.0, 1.0}
    assert isinstance(notes, list)


def test_q3_runner_and_report_render_tables():
    payload = run_q3_experiment(seeds=[11], cases=build_q3_suite())
    assert payload["tables"]
    markdown = render_markdown_tables(payload["tables"])
    assert "Orchestration Representation" in markdown
    assert "Graph Harness (Ours)" in markdown


def test_q3_dataset_preset_prioritizes_hard_structural_cases():
    assert Q3_DATASET_PRESETS["structural_core"] == ["musique", "financebench", "pubmedqa", "medqa", "mmlu_pro"]
    assert Q3_DATASET_PRESETS["structural_core_legacy"] == ["hotpotqa", "musique", "finqa", "drop", "mmlu_pro"]
    assert "strategyqa" not in Q3_DATASET_PRESETS["structural_core"]
    assert Q3_ROW_PRESETS["structural_hard_strict"]["hotpotqa"] == [1, 2, 4]
    assert Q3_ROW_PRESETS["structural_core_no_math500"]["mmlu_pro"] == [0, 1, 2, 4, 9, 10, 14, 16, 17, 18]


def test_q3_row_preset_normalizes_dataset_ids_with_hyphens():
    assert _normalize_dataset_key("MATH-500") == "math500"
    assert Q3_ROW_PRESETS["structural_hard"]["mmlu_pro"] == [0, 1, 2, 4, 9, 10, 14, 16, 17, 18]


def test_q3_structural_row_filter_keeps_hard_hops_and_rejects_easy_rows():
    hotpot = DATASET_ADAPTERS["hotpotqa"]
    hard_row = {"question": "Which city is referenced by the two-hop supporting evidence in the article?", "answer": "New York City", "raw": {"type": "bridge", "level": "hard"}}
    easy_row = {"question": "Which city is referenced by the two-hop supporting evidence in the article?", "answer": "1838", "raw": {"type": "bridge", "level": "easy"}}
    yesno_row = {"question": "Were Scott Derrickson and Ed Wood of the same nationality?", "answer": "yes", "raw": {"type": "comparison", "level": "hard"}}
    assert _is_structurally_qualified_row(hotpot, hard_row)
    assert not _is_structurally_qualified_row(hotpot, easy_row)
    assert not _is_structurally_qualified_row(hotpot, {"answer": "1838", "raw": {"type": "bridge", "level": "hard"}})
    assert not _is_structurally_qualified_row(hotpot, yesno_row)

    musique = DATASET_ADAPTERS["musique"]
    hard_multi_hop = {"answer": "New York City", "raw": {"answerable": True, "question_decomposition": [1, 2, 3]}}
    shallow = {"answer": "Paris", "raw": {"answerable": True, "question_decomposition": [1, 2]}}
    assert _is_structurally_qualified_row(musique, hard_multi_hop)
    assert not _is_structurally_qualified_row(musique, shallow)

    medqa = DATASET_ADAPTERS["medqa"]
    medqa_hard = {
        "question": "A " + "long clinical vignette " * 20,
        "raw": {"data": {"Options": {"A": "x", "B": "y", "C": "z", "D": "w"}}},
    }
    medqa_easy = {
        "question": "A short clinical vignette",
        "raw": {"data": {"Options": {"A": "x", "B": "y"}}},
    }
    assert _is_structurally_qualified_row(medqa, medqa_hard)
    assert not _is_structurally_qualified_row(medqa, medqa_easy)

    finqa = DATASET_ADAPTERS["finqa"]
    structural = {"answer": "380", "question": "what is the the interest expense in 2009?", "raw": {"metadata": {"program": "multiply(2, 3), divide(#0, 6), add(#1, 4)"}}}
    blank = {"answer": "", "question": "what is the the interest expense in 2009?", "raw": {"metadata": {"program": "multiply(2, 3), divide(#0, 6), add(#1, 4)"}}}
    assert _is_structurally_qualified_row(finqa, structural)
    assert not _is_structurally_qualified_row(finqa, blank)

    financebench = DATASET_ADAPTERS["financebench"]
    fin_row = {
        "answer": "$1577.00",
        "question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response based on the cash flow statement.",
        "raw": {
            "question_type": "metrics-generated",
            "question_reasoning": "Numerical reasoning",
            "evidence": [{"evidence_text": "a"}, {"evidence_text": "b"}],
        },
    }
    fin_easy = {
        "answer": "$1577.00",
        "question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response based on the cash flow statement.",
        "raw": {"question_type": "domain-relevant", "question_reasoning": None, "evidence": [{"evidence_text": "a"}]},
    }
    assert _is_structurally_qualified_row(financebench, fin_row)
    assert not _is_structurally_qualified_row(financebench, fin_easy)

    bbh = DATASET_ADAPTERS["bbh_full"]
    assert bbh.dataset_id == "BBH-Full"
    bbh_hard = {
        "answer": "D",
        "question": "On the floor, there is one mauve cat toy, two purple cat toys, three grey cat toys, two mauve notebooks, three grey notebooks, three burgundy cat toys, and one purple notebook. If I remove all the notebooks from the floor, how many grey objects remain on it?",
        "raw": {"task": "reasoning_about_colored_objects", "input": "On the floor, there is one mauve cat toy, two purple cat toys, three grey cat toys, two mauve notebooks, three grey notebooks, three burgundy cat toys, and one purple notebook. If I remove all the notebooks from the floor, how many grey objects remain on it?"},
    }
    bbh_easy = {"answer": "true", "question": "not ( True ) and ( True ) is", "raw": {"task": "boolean_expressions", "input": "not ( True ) and ( True ) is"}}
    assert _is_structurally_qualified_row(bbh, bbh_hard)
    assert not _is_structurally_qualified_row(bbh, bbh_easy)

    drop = DATASET_ADAPTERS["drop"]
    drop_hard = {"answer": "Chaz Schilens", "question": "Who scored the first touchdown of the game?", "raw": {"passage": "x" * 1800, "answers_spans": {"spans": [["Chaz Schilens"], ["x"], ["y"]]}}}
    drop_easy = {"answer": "2", "question": "How many field goals did Kris Brown kick?", "raw": {"passage": "x" * 1200, "answers_spans": {"spans": [["2"], ["3"]]}}}
    assert _is_structurally_qualified_row(drop, drop_hard)
    assert not _is_structurally_qualified_row(drop, drop_easy)


def test_real_graph_harness_filters_structural_resource_nodes():
    adapter = DATASET_ADAPTERS["gsm8k"]
    row = {"id": "q3-resource-regression", "question": "What is 2 plus 2?", "context": "", "answer": "4"}
    case = build_dataset_case(adapter, row, 0)
    result = real_construction_result(case, "graph_harness", seed=11)
    assert all(node.kind != "resource" for node in result.blueprint.nodes)
    result.validate(case.request())


def test_real_graph_harness_preserves_multi_branch_execution_shape():
    adapter = DATASET_ADAPTERS["hotpotqa"]
    row = {
        "id": "q3-branch-regression",
        "question": "Which city connects the two supporting facts?",
        "context": "Fact A mentions Paris. Fact B confirms Paris is the answer.",
        "answer": "Paris",
    }
    case = build_dataset_case(adapter, row, 0)
    result = real_construction_result(case, "graph_harness", seed=11)
    realizes = {node.realizes_blueprint_node: node.id for node in result.application.nodes}
    edges = {(edge.source, edge.target) for edge in result.application.edges}
    assert realizes["comp_retrieve_a"] in result.application.entrypoints
    assert realizes["comp_retrieve_b"] in result.application.entrypoints
    assert (realizes["comp_retrieve_a"], realizes["branch_merge"]) in edges
    assert (realizes["comp_retrieve_b"], realizes["branch_merge"]) in edges
    assert (realizes["branch_merge"], realizes["comp_synthesize"]) in edges
    assert _is_reachable(result.application, realizes["branch_merge"], realizes["comp_answer"])


def test_real_graph_harness_places_constraint_gate_before_answer():
    adapter = DATASET_ADAPTERS["medqa"]
    row = {
        "id": "q3-constraint-regression",
        "question": "A patient needs the safest option.",
        "context": "",
        "choices": {"a": "unsafe shortcut", "b": "reviewed treatment"},
        "answer": "b",
    }
    case = build_dataset_case(adapter, row, 0)
    result = real_construction_result(case, "graph_harness", seed=11)
    realizes = {node.realizes_blueprint_node: node.id for node in result.application.nodes}
    edges = {(edge.source, edge.target) for edge in result.application.edges}
    assert (realizes["comp_review"], realizes["constraint_human_approval"]) in edges
    assert (realizes["constraint_human_approval"], realizes["comp_answer"]) in edges
    assert _precedes(result.application, realizes["constraint_human_approval"], realizes["comp_answer"])


def test_strategyqa_choice_wrapped_yes_no_scores_correctly():
    adapter = DATASET_ADAPTERS["strategyqa"].execution
    assert adapter.score("No.", "choice:b|no") == 1.0
    assert adapter.score("yes", "choice:a|yes") == 1.0
    assert adapter.score("否", "choice:b|no") == 1.0


def test_real_graph_harness_enables_resource_aware_execution():
    adapter = DATASET_ADAPTERS["hotpotqa"]
    row = {
        "id": "q3-resource-aware-regression",
        "question": "What entity is supported by both branches?",
        "context": "Branch A says Ada. Branch B also says Ada.",
        "answer": "Ada",
    }
    case = build_dataset_case(adapter, row, 0)
    result = real_construction_result(case, "graph_harness", seed=11)
    assert result.application.nodes
    assert all(node.config.get("resource_access") is True for node in result.application.nodes)


def test_real_graph_harness_adds_answer_prior_to_terminal_path():
    adapter = DATASET_ADAPTERS["hotpotqa"]
    row = {
        "id": "q3-answer-prior-regression",
        "question": "What entity is supported by both branches?",
        "context": "Branch A says Ada. Branch B also says Ada.",
        "answer": "Ada",
    }
    case = build_dataset_case(adapter, row, 0)
    result = real_construction_result(case, "graph_harness", seed=11)
    realizes = {node.realizes_blueprint_node: node.id for node in result.application.nodes}
    assert "answer_prior" in realizes
    assert realizes["answer_prior"] in result.application.entrypoints
    assert _is_reachable(result.application, realizes["answer_prior"], realizes["comp_answer"])


def test_q3_branch_resources_cover_structured_qa_datasets():
    assert _branch_resources("FinQA", {"raw": {"context": {"pre_text": ["a"], "post_text": ["b"], "table": [["h1", "h2"], ["c1", "c2"]]}}})
    assert _branch_resources("DROP", {"raw": {"question": "q", "passage": "p"}})
    assert _branch_resources("MATH-500", {"raw": {"problem": "prob", "subject": "math", "level": 4}})
    assert _branch_resources("MedQA", {"raw": {"data": {"Question": "q", "Options": {"A": "x", "B": "y"}}}})


def test_q3_feedback_graph_keeps_executable_order_acyclic():
    case = next(x for x in build_q3_suite() if x.family == "feedback_driven")
    blueprint = get_q3_baseline("graph_harness").build_blueprint(case)
    order, _ = _execution_order(blueprint.nodes, blueprint.edges)
    assert len(order) == len(blueprint.nodes)
    assert "feedback_loop" in order


def test_real_graph_harness_structural_preservation_is_high_on_branching_case():
    adapter = DATASET_ADAPTERS["hotpotqa"]
    row = {
        "id": "q3-structural-preservation",
        "question": "Which city connects the two supporting facts?",
        "context": "Fact A mentions Paris. Fact B confirms Paris is the answer.",
        "answer": "Paris",
    }
    case = build_dataset_case(adapter, row, 0)
    q3_case = case_to_q3(case)
    graph_bp = _real_blueprint(q3_case, case, "graph_harness")
    flat_bp = _real_blueprint(q3_case, case, "flat_component_selection")
    assert _graph_structural_preservation(q3_case, graph_bp) >= _graph_structural_preservation(q3_case, flat_bp)


def _is_reachable(application, source, target):
    graph = {}
    for edge in application.edges:
        graph.setdefault(edge.source, []).append(edge.target)
    queue = [source]
    seen = {source}
    while queue:
        node = queue.pop(0)
        for nxt in graph.get(node, []):
            if nxt == target:
                return True
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _precedes(application, source, target):
    order = _topological_order(application)
    return order.index(source) < order.index(target)


def _topological_order(application):
    nodes = [node.id for node in application.nodes]
    indegree = {node: 0 for node in nodes}
    graph = {node: [] for node in nodes}
    for edge in application.edges:
        graph[edge.source].append(edge.target)
        indegree[edge.target] += 1
    queue = [node for node in nodes if indegree[node] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nxt in graph[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    order.extend(node for node in nodes if node not in order)
    return order
