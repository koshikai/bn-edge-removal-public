"""S. pombe model against the original paper (not via biodivine).

Davidich & Bornholdt (2008) PLoS ONE 3(2): e1672.  The threshold network below is
transcribed from Table 1 (links), Figure 1 (self-degradation loops) and Table 2 (the
cell-cycle sequence); see docs/experiments/2026-10-05_s_pombe_source_check.md and
docs/experiments/2026-10-05_s_pombe_source_conformance.md.
"""

from __future__ import annotations

from itertools import product

from bn_edge_removal.models.network_spec import load_network_spec
from bn_edge_removal.models.s_pombe import SPombeModel

_SPEC = load_network_spec("s_pombe")
_INDEX = {label: i for i, label in enumerate(_SPEC.node_labels)}
_ALL_STATES = [list(bits) for bits in product([0, 1], repeat=_SPEC.n_nodes)]
_NO_CONTROL = [0] * _SPEC.m_edges

# Table 1: (parent, daughter, a_ij), in the row order of the table.
_TABLE1 = [
    ("Start", "SK", +1),
    ("SK", "Ste9", -1),
    ("SK", "Rum1", -1),
    ("Cdc2_Cdc13", "Cdc25", +1),
    ("Wee1_Mik1", "Cdc2_Cdc13_star", -1),
    ("Rum1", "Cdc2_Cdc13", -1),
    ("Rum1", "Cdc2_Cdc13_star", -1),
    ("Cdc2_Cdc13", "Rum1", -1),
    ("Ste9", "Cdc2_Cdc13", -1),
    ("Ste9", "Cdc2_Cdc13_star", -1),
    ("Cdc2_Cdc13_star", "Slp1", +1),
    ("Slp1", "Cdc2_Cdc13", -1),
    ("Slp1", "Cdc2_Cdc13_star", -1),
    ("Slp1", "PP", +1),
    ("PP", "Ste9", +1),
    ("PP", "Rum1", +1),
    ("PP", "Wee1_Mik1", +1),
    ("Cdc25", "Cdc2_Cdc13_star", +1),
    ("Cdc2_Cdc13", "Ste9", -1),
    ("PP", "Cdc25", -1),
    ("Cdc2_Cdc13", "Wee1_Mik1", -1),
    ("Cdc2_Cdc13_star", "Rum1", -1),
    ("Cdc2_Cdc13_star", "Ste9", -1),
]
# Figure 1, yellow loops: a_ii = -1.
_SELF_DEGRADING = ("Start", "SK", "Slp1", "PP")
# The text prints theta = 1 / -1; only values strictly between 0 and +-1 reproduce
# Table 2 and Table 3 of the paper (see the doc), so the half values are used here.
_THETA = {"Cdc2_Cdc13_star": 0.5, "Cdc2_Cdc13": -0.5}
# Table 2, columns as printed.
_TABLE2_COLUMNS = (
    "Start",
    "SK",
    "Cdc2_Cdc13",
    "Ste9",
    "Rum1",
    "Slp1",
    "Cdc2_Cdc13_star",
    "Wee1_Mik1",
    "Cdc25",
    "PP",
)
_TABLE2_ROWS = (
    "1001100100",
    "0101100100",
    "0000000100",
    "0010000100",
    "0010000010",
    "0010001010",
    "0010011010",
    "0000010011",
    "0001100101",
    "0001100100",
)


def _paper_next_state(x: list[int]) -> list[int]:
    nxt = []
    for label, i in _INDEX.items():
        total = sum(
            sign * x[_INDEX[parent]]
            for parent, daughter, sign in _TABLE1
            if daughter == label
        )
        if label in _SELF_DEGRADING:
            total -= x[i]
        theta = _THETA.get(label, 0.0)
        nxt.append(1 if total > theta else 0 if total < theta else x[i])
    return nxt


def _table2_states() -> list[list[int]]:
    states = []
    for row in _TABLE2_ROWS:
        state = [0] * _SPEC.n_nodes
        for label, bit in zip(_TABLE2_COLUMNS, row, strict=True):
            state[_INDEX[label]] = int(bit)
        states.append(state)
    return states


def test_paper_threshold_model_reproduces_its_own_table2():
    states = _table2_states()

    for current, following in zip(states, states[1:], strict=False):
        assert _paper_next_state(current) == following
    assert _paper_next_state(states[-1]) == states[-1]


def test_every_node_matches_the_paper_for_every_state():
    model = SPombeModel()

    for x in _ALL_STATES:
        assert model.next_state(x, _NO_CONTROL) == _paper_next_state(x)


def test_start_decays_as_in_the_paper():
    model = SPombeModel()

    for x in _ALL_STATES:
        assert model.next_state(x, _NO_CONTROL)[0] == 0
        assert _paper_next_state(x)[0] == 0


def test_table2_sequence_is_followed_from_step1():
    model = SPombeModel()
    states = _table2_states()

    assert states[0][_INDEX["Start"]] == 1
    assert states[1][_INDEX["Start"]] == 0
    for current, following in zip(states, states[1:], strict=False):
        assert model.next_state(current, _NO_CONTROL) == following
    assert states[-1] == model.goal_states()[0]


def test_attractors_and_basins_match_table3():
    """Table 3: 12 fixed points (G1 basin 762) and one 3-state cycle (basin 208)."""
    model = SPombeModel()
    basins: dict[frozenset[tuple[int, ...]], int] = {}
    for x in _ALL_STATES:
        path = [tuple(x)]
        while (nxt := tuple(model.next_state(list(path[-1]), _NO_CONTROL))) not in path:
            path.append(nxt)
        attractor = frozenset(path[path.index(nxt) :])
        basins[attractor] = basins.get(attractor, 0) + 1

    fixed = sorted(size for attractor, size in basins.items() if len(attractor) == 1)
    cycles = [(len(a), size) for a, size in basins.items() if len(a) > 1]
    assert fixed == [2, 2, 2, 2, 2, 2, 2, 2, 2, 18, 18, 762]
    assert cycles == [(3, 208)]
    assert basins[frozenset([tuple(model.goal_states()[0])])] == 762


def test_non_self_edges_of_the_spec_are_exactly_table1():
    spec_edges = {
        (
            _SPEC.node_labels[edge.source - 1],
            _SPEC.node_labels[edge.target - 1],
            +1 if edge.sign == "activation" else -1,
        )
        for edge in _SPEC.all_edges
        if edge.source != edge.target
    }

    assert spec_edges == set(_TABLE1)


def test_self_edges_of_the_spec_are_the_hold_on_tie_nodes():
    """Start's loop is a degradation with no other input: no dependency, no edge."""
    self_edges = {
        _SPEC.node_labels[edge.source - 1]: edge.sign
        for edge in _SPEC.all_edges
        if edge.source == edge.target
    }

    assert self_edges == dict.fromkeys(
        ("Ste9", "Rum1", "Cdc25", "Wee1_Mik1"), "activation"
    )
