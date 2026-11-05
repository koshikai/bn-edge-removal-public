"""Checks of ``phase_dp`` against enumeration of whole input sequences.

The specification is checked as a predicate on a slice of the input sequence:
``before`` on ``u_0 .. u_e`` and ``after`` on ``u_e .. u_{T-1}``, where ``e`` is
the entry step (the whole sequence when the episode starts in the goal, and
nothing before it).  Only time-invariant specifications are used, so a slice can
be checked by the same predicate.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np
import pytest

from bn_edge_removal.analysis.edge_selection import AllEdgeModel
from bn_edge_removal.analysis.exact_monotone import VectorNetwork
from bn_edge_removal.analysis.input_automata import (
    InputAutomaton,
    Objective,
    automaton_dp,
    budget,
    min_duration,
    monotone,
    recovery,
    unconstrained,
)
from bn_edge_removal.analysis.phase_spec import Phase, phase_dp
from bn_edge_removal.models.network_spec import load_network_spec

type Seq = tuple[int, ...]
type Case = tuple[str, tuple[int, ...], tuple[int, int], tuple[int, int] | None]

_DURATION = 2
_BUDGET = 2
_PHASES = 3


def _bit(u: int, k: int) -> int:
    return (u >> k) & 1


def _min_duration_ok(s: Seq, m: int, d: int) -> bool:
    for k in range(m):
        t = 0
        while t < len(s):
            if _bit(s[t], k):
                start = t
                while t < len(s) and _bit(s[t], k):
                    t += 1
                if t != len(s) and t - start < d:
                    return False
            else:
                t += 1
    return True


def _specs(m: int) -> dict[str, tuple[InputAutomaton, Callable[[Seq], bool]]]:
    return {
        "none": (unconstrained(m), lambda s: True),
        "monotone": (
            monotone(m),
            lambda s: all(s[t] & ~s[t + 1] == 0 for t in range(len(s) - 1)),
        ),
        "recovery1": (
            recovery(m, 1),
            lambda s: all(
                not (_bit(s[t], k) and _bit(s[t + 1], k))
                for k in range(m)
                for t in range(len(s) - 1)
            ),
        ),
        "budget2": (
            budget(m, _BUDGET),
            lambda s: sum(bin(u).count("1") for u in s) <= _BUDGET,
        ),
        "min_duration2": (
            min_duration(m, _DURATION),
            lambda s: _min_duration_ok(s, m, _DURATION),
        ),
    }


def _goal(net: VectorNetwork, goal_node: tuple[int, int] | None) -> np.ndarray:
    if goal_node is None:
        return net.goal_mask()
    node, value = goal_node
    ids = np.arange(net.n_states)
    return ((ids >> (net.n_nodes - 1 - node)) & 1) == value


def _brute(
    case: Case, goal: np.ndarray, ok: Callable[[Seq], bool], phase: Phase
) -> list[int | None]:
    name, edge_set, (reach, steps), _ = case
    spec = load_network_spec(name)
    model = AllEdgeModel(spec)
    m = len(edge_set)
    n = spec.n_nodes

    def state_id(x: tuple[int, ...]) -> int:
        return int("".join(map(str, x)), 2)

    def full(u: int) -> list[int]:
        vector = [0] * model.m_edges
        for k in range(m):
            vector[edge_set[k]] = _bit(u, k)
        return vector

    seqs = list(product(range(2**m), repeat=steps))
    best: list[int | None] = []
    for sid in range(2**n):
        x0 = tuple(model.id_to_state(sid))
        value: int | None = None
        for s in seqs:
            path = [x0]
            for u in s:
                path.append(tuple(model.next_state(list(path[-1]), full(u))))
            inside = [bool(goal[state_id(p)]) for p in path]
            first = next((t for t, v in enumerate(inside) if v), None)
            if first is None or first > reach or not all(inside[first:]):
                continue
            if phase == "all":
                part_ok = ok(s)
            elif phase == "before":
                part_ok = ok(s[:first])
            else:
                part_ok = ok(s[max(first - 1, 0) :])
            if not part_ok:
                continue
            c = sum(bin(u).count("1") for u in s)
            value = c if value is None else min(value, c)
        best.append(value)
    return best


_CASES: list[Case] = [
    ("cortical", (2, 5), (3, 5), None),
    ("cortical", (2, 5), (3, 5), (0, 0)),
    ("wnt5a", (0, 4), (3, 5), (0, 1)),
    ("wnt5a", (0, 4), (3, 5), (5, 1)),
    ("wnt5a", (0, 4), (3, 5), (0, 0)),
]


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("spec_name", list(_specs(2)))
@pytest.mark.parametrize("phase", ["all", "before", "after"])
def test_phase_dp_matches_enumeration(
    case: Case,
    spec_name: str,
    phase: Phase,
) -> None:
    model, edge_set, horizon, goal_node = case
    automaton, ok = _specs(len(edge_set))[spec_name]
    net = VectorNetwork(load_network_spec(model))
    goal = _goal(net, goal_node)
    want = _brute(case, goal, ok, phase)
    result = phase_dp(net, edge_set, Objective(goal, horizon), automaton, phase)
    got = [
        int(c) if f else None
        for f, c in zip(result.feasible, result.min_cost, strict=True)
    ]
    assert got == want


@pytest.mark.parametrize("model", ["cortical", "wnt5a", "cell_cycle10"])
@pytest.mark.parametrize("spec_name", list(_specs(3)))
def test_phase_all_equals_automaton_dp(model: str, spec_name: str) -> None:
    net = VectorNetwork(load_network_spec(model))
    edge_set = tuple(range(min(3, len(net.edges))))
    automaton, _ = _specs(len(edge_set))[spec_name]
    for goal_node in (None, (0, 1), (1, 0)):
        goal = _goal(net, goal_node)
        want = automaton_dp(net, edge_set, Objective(goal, (4, 7)), automaton)
        got = phase_dp(net, edge_set, Objective(goal, (4, 7)), automaton, "all")
        assert np.array_equal(got.feasible, want.feasible)
        assert np.array_equal(got.min_cost, want.min_cost)


def test_phases_are_not_trivial() -> None:
    """The enumeration cases tell the three phases apart.

    In these small cases ``monotone`` gives the same count in every phase, so the
    separation is shown with ``recovery1`` and ``budget2``.
    """
    net = VectorNetwork(load_network_spec("wnt5a"))
    goal = _goal(net, (0, 0))
    objective = Objective(goal, (3, 5))
    none = int(phase_dp(net, (0, 4), objective, unconstrained(2), "all").feasible.sum())
    for spec_name in ("recovery1", "budget2"):
        automaton, _ = _specs(2)[spec_name]
        counts = {
            phase: int(
                phase_dp(net, (0, 4), objective, automaton, phase).feasible.sum()
            )
            for phase in ("all", "before", "after")
        }
        assert len(set(counts.values())) == _PHASES, spec_name
        assert counts["all"] < min(counts["before"], counts["after"]) < none
