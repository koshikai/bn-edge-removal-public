"""Checks for the imported networks: GINsim from drqn (yeast n18, tumor n20) and
BBM from transfer (emt_switch n12, tlgl_reduced n18, hcc_reduced n19,
integrated_mammalian n20).

The specs have no removable edges yet; the checks cover the dynamics only:

- ``all_edges`` is exactly the set of functional dependencies of
  ``update_equations`` (source, target, sign), over all states;
- the goal states are a union of attractors of the uncontrolled synchronous
  dynamics (one attractor for the GINsim networks);
- the number of attractors equals the drqn audit
  (``docs/experiments/2026-10-05_ginsim_target_audit.json`` at drqn ``aff23fb``:
  1 for yeast, 13 for tumor); for the BBM networks, the count printed by
  ``scripts/import_bbm_from_transfer.py --report``;
- for the BBM networks, every goal state satisfies the phenotype target;
- ``AllEdgeModel`` follows ``update_equations`` and cancels each dependency when
  its control bit is set (checked on a fixed subset of states).

The whole-state checks are vectorised with numpy because the networks have
2**18 and 2**20 states.
"""

from __future__ import annotations

import re
from functools import cache

import numpy as np
import pytest

from bn_edge_removal.analysis.edge_selection import AllEdgeModel
from bn_edge_removal.models.network_spec import NetworkSpec, load_network_spec

_ATTRACTORS = {
    "yeast_irons2009_n18": 1,
    "tumor_cohen2015_n20": 13,
    "emt_switch_n12": 3,
    "tlgl_reduced_n18": 3,
    "hcc_reduced_n19": 21,
    "integrated_mammalian_n20": 3,
}
_NAMES = list(_ATTRACTORS)
_GINSIM = ("yeast_irons2009_n18", "tumor_cohen2015_n20")
# Phenotype targets of transfer bbm_metadata.yaml at f438ede: (label, value).
_PHENOTYPES = {
    "emt_switch_n12": ("ZEB1", 1),
    "tlgl_reduced_n18": ("Apoptosis_", 1),
    "hcc_reduced_n19": ("ZEB1", 1),
    "integrated_mammalian_n20": ("pRB", 1),
}


def _expressions(spec: NetworkSpec) -> list[str]:
    exprs: list[str] = []
    for equation in spec.update_equations:
        match = re.fullmatch(r"x(\d+)' = (.+)", equation)
        assert match is not None, equation
        exprs.append(match.group(2))
    return exprs


def _to_numpy(expression: str) -> str:
    expression = re.sub(r"\bnot\b", "~", expression)
    expression = re.sub(r"\band\b", "&", expression)
    return re.sub(r"\bor\b", "|", expression)


def _eval(expression: str, states: np.ndarray) -> np.ndarray:
    env = {f"x{i + 1}": states[:, i] for i in range(states.shape[1])}
    value = eval(_to_numpy(expression), {"__builtins__": {}}, env)
    return np.broadcast_to(np.asarray(value, dtype=bool), (states.shape[0],))


@cache
def _all_states(n: int) -> np.ndarray:
    ids = np.arange(2**n, dtype=np.int64)
    return ((ids[:, None] >> (n - 1 - np.arange(n))) & 1).astype(bool)


def _ids(states: np.ndarray) -> np.ndarray:
    n = states.shape[1]
    return (states.astype(np.int64) << (n - 1 - np.arange(n))).sum(axis=1)


@cache
def _successors(name: str) -> np.ndarray:
    spec = load_network_spec(name)
    states = _all_states(spec.n_nodes)
    nxt = np.stack([_eval(e, states) for e in _expressions(spec)], axis=1)
    return _ids(nxt)


@pytest.mark.parametrize("name", _NAMES)
def test_all_edges_are_exactly_the_functional_dependencies(name: str) -> None:
    spec = load_network_spec(name)
    states = _all_states(spec.n_nodes)
    found: set[tuple[int, int, str]] = set()
    for target, expression in enumerate(_expressions(spec), start=1):
        for source in range(1, spec.n_nodes + 1):
            low = states.copy()
            low[:, source - 1] = False
            high = states.copy()
            high[:, source - 1] = True
            f_low, f_high = _eval(expression, low), _eval(expression, high)
            up = bool((~f_low & f_high).any())
            down = bool((f_low & ~f_high).any())
            assert not (up and down), f"non-monotone edge {source}->{target}"
            if up or down:
                found.add((source, target, "activation" if up else "inhibition"))
    listed = {(e.source, e.target, e.sign) for e in spec.all_edges}
    assert listed == found


@pytest.mark.parametrize("name", _NAMES)
def test_goal_states_are_attractors(name: str) -> None:
    spec = load_network_spec(name)
    succ = _successors(name)
    goal = set(_ids(np.array(spec.goal_states, dtype=bool)).tolist())
    remaining = set(goal)
    count = 0
    while remaining:
        start = remaining.pop()
        x = int(succ[start])
        while x != start:
            assert x in goal, "goal set is not a union of cycles"
            remaining.discard(x)
            x = int(succ[x])
        count += 1
    if name in _GINSIM:
        assert count == 1


@pytest.mark.parametrize("name", list(_PHENOTYPES))
def test_goal_states_satisfy_phenotype(name: str) -> None:
    spec = load_network_spec(name)
    label, value = _PHENOTYPES[name]
    column = list(spec.node_labels).index(label)
    assert all(state[column] == value for state in spec.goal_states)


@pytest.mark.parametrize("name", _NAMES)
def test_number_of_attractors_matches_drqn_audit(name: str) -> None:
    succ = _successors(name)
    # States on a cycle are exactly those reached again after len(succ) steps
    # from themselves; iterate the map until the image stops shrinking.
    image = np.unique(succ)
    while True:
        nxt = np.unique(succ[image])
        if len(nxt) == len(image):
            break
        image = nxt
    cyclic = set(image.tolist())
    count = 0
    while cyclic:
        start = cyclic.pop()
        x = int(succ[start])
        while x != start:
            cyclic.discard(x)
            x = int(succ[x])
        count += 1
    assert count == _ATTRACTORS[name]


@pytest.mark.parametrize("name", _NAMES)
def test_all_edge_model_follows_equations_and_cancels_edges(name: str) -> None:
    spec = load_network_spec(name)
    model = AllEdgeModel(spec)
    states = _all_states(spec.n_nodes)[::4099]
    succ = _successors(name)
    for row in states:
        x = row.astype(int).tolist()
        expected = model.id_to_state(int(succ[model.state_to_id(x)]))
        assert model.next_state(x, [0] * model.m_edges) == expected
    for edge in model.all_edges:
        u = [0] * model.m_edges
        u[edge.index] = 1
        for row in states[:16]:
            x = row.astype(int).tolist()
            flipped = x.copy()
            flipped[edge.source - 1] ^= 1
            target = edge.target - 1
            assert (
                model.next_state(x, u)[target] == model.next_state(flipped, u)[target]
            )
