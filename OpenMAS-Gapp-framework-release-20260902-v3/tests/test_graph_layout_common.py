from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from graph_layout_common import ordered_layer_positions


def test_ordered_layout_keeps_feedback_chain_left_to_right():
    positions = ordered_layer_positions(
        ["a", "b", "c"],
        [
            {"source": "a", "target": "b", "relation": "precedes"},
            {"source": "b", "target": "c", "relation": "precedes"},
            {"source": "c", "target": "b", "relation": "feedback"},
        ],
        left=0,
        top=0,
        width=420,
        height=260,
        padding_x=20,
        padding_y=20,
    )
    assert positions["a"][0] < positions["b"][0] < positions["c"][0]


def test_ordered_layout_stacks_same_level_nodes_vertically():
    positions = ordered_layer_positions(
        ["a", "b", "c"],
        [
            {"source": "a", "target": "c", "relation": "precedes"},
            {"source": "b", "target": "c", "relation": "precedes"},
        ],
        left=0,
        top=0,
        width=420,
        height=260,
        padding_x=20,
        padding_y=20,
    )
    assert math.isclose(positions["a"][0], positions["b"][0])
    assert positions["a"][1] != positions["b"][1]
    assert positions["a"][0] < positions["c"][0]
