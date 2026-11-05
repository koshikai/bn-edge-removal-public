"""Checks of the input automata against direct predicates on whole input sequences.

For small networks every input sequence ``u_0 .. u_{T-1}`` is enumerated, each
specification is checked as a predicate on the whole sequence (only obligations
inside ``0 .. T - 1``), and the best ``first_entry`` cost is compared with
``automaton_dp``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

import numpy as np
import pytest

from bn_edge_removal.analysis.edge_selection import AllEdgeModel
from bn_edge_removal.analysis.exact_constrained import InputSpec, constrained_dp
from bn_edge_removal.analysis.exact_monotone import VectorNetwork, monotone_dp
from bn_edge_removal.analysis.input_automata import (
    CostKind,
    InputAutomaton,
    Objective,
    automaton_dp,
    budget,
    concurrency,
    max_duration,
    min_duration,
    monotone,
    recovery,
    treatment_end,
    unconstrained,
)
from bn_edge_removal.models.network_spec import load_network_spec

type Seq = tuple[int, ...]


def _bit(u: int, k: int) -> int:
    return (u >> k) & 1


def _runs(seq: Seq, k: int) -> list[tuple[int, int]]:
    """(start, length) of the maximal runs of edge ``k`` being removed."""
    runs: list[tuple[int, int]] = []
    t = 0
    while t < len(seq):
        if _bit(seq[t], k):
            s = t
            while t < len(seq) and _bit(seq[t], k):
                t += 1
            runs.append((s, t - s))
        else:
            t += 1
    return runs


def _predicates(
    m: int, steps: int
) -> dict[str, tuple[InputAutomaton, Callable[[Seq], bool]]]:
    edges = range(m)
    return {
        "none": (unconstrained(m), lambda s: True),
        "monotone": (
            monotone(m),
            lambda s: all(s[t] & ~s[t + 1] == 0 for t in range(len(s) - 1)),
        ),
        "recovery2": (
            recovery(m, 2),
            lambda s: all(
                not (_bit(s[t], k) and _bit(s[t + i], k))
                for k in edges
                for t in range(len(s))
                for i in (1, 2)
                if t + i < len(s)
            ),
        ),
        "concurrency1": (
            concurrency(m, 1),
            lambda s: all(bin(u).count("1") <= 1 for u in s),
        ),
        "budget3": (budget(m, 3), lambda s: sum(bin(u).count("1") for u in s) <= 3),
        "min_duration3": (
            min_duration(m, 3),
            lambda s: all(
                start + length == len(s) or length >= 3
                for k in edges
                for start, length in _runs(s, k)
            ),
        ),
        "max_duration2": (
            max_duration(m, 2),
            lambda s: all(length <= 2 for k in edges for _, length in _runs(s, k)),
        ),
        "treatment_end": (
            treatment_end(m, steps - 2),
            lambda s: all(u == 0 for u in s[steps - 2 :]),
        ),
    }


@dataclass(frozen=True)
class _Variant:
    """A goal ``node = value`` instead of the model goal, and the hold option."""

    hold_without_removal: bool = False
    goal_node: tuple[int, int] | None = None
    hold_removal_steps: int | None = None
    hold_frozen: bool = False


def _brute(
    name: str,
    edge_set: tuple[int, ...],
    horizon: tuple[int, int],
    ok: Callable[[Seq], bool],
    variant: _Variant | None = None,
) -> tuple[list[int | None], list[int | None]]:
    reach, steps = horizon
    spec = load_network_spec(name)
    model = AllEdgeModel(spec)
    variant = variant or _Variant()
    goal = {tuple(s) for s in spec.goal_states}
    if variant.goal_node is not None:
        node, value = variant.goal_node
        goal = {
            tuple(model.id_to_state(i))
            for i in range(2**spec.n_nodes)
            if model.id_to_state(i)[node] == value
        }
    m = len(edge_set)
    seqs = [s for s in product(range(2**m), repeat=steps) if ok(s)]

    def full(u: int) -> list[int]:
        vector = [0] * model.m_edges
        for k in range(m):
            vector[edge_set[k]] = _bit(u, k)
        return vector

    per_step: list[int | None] = []
    onset: list[int | None] = []
    for state_id in range(2**spec.n_nodes):
        best_p: int | None = None
        best_o: int | None = None
        x0 = tuple(model.id_to_state(state_id))
        for s in seqs:
            path = [x0]
            for u in s:
                path.append(tuple(model.next_state(list(path[-1]), full(u))))
            inside = [p in goal for p in path]
            first = next((t for t, v in enumerate(inside) if v), None)
            if first is None or first > reach or not all(inside[first:]):
                continue
            if variant.hold_without_removal and any(s[first:]):
                continue
            j = variant.hold_removal_steps
            if j is not None and any(s[first + j :]):
                continue
            entry_input = s[first - 1] if first > 0 else 0
            if variant.hold_frozen and any(u != entry_input for u in s[first:]):
                continue
            c_p = sum(bin(u).count("1") for u in s)
            c_o = sum(
                bin(u & ~(s[t - 1] if t else 0)).count("1") for t, u in enumerate(s)
            )
            best_p = c_p if best_p is None else min(best_p, c_p)
            best_o = c_o if best_o is None else min(best_o, c_o)
        per_step.append(best_p)
        onset.append(best_o)
    return per_step, onset


_CASES = [("cortical", (2, 5), (3, 5)), ("wnt5a", (0, 4), (3, 5))]
_NAMES = list(_predicates(2, 5))


@pytest.mark.parametrize(("model", "edge_set", "horizon"), _CASES)
@pytest.mark.parametrize("spec_name", _NAMES)
def test_automaton_dp_matches_enumeration(
    model: str, edge_set: tuple[int, ...], horizon: tuple[int, int], spec_name: str
) -> None:
    automaton, ok = _predicates(len(edge_set), horizon[1])[spec_name]
    net = VectorNetwork(load_network_spec(model))
    goal = net.goal_mask()
    want_p, want_o = _brute(model, edge_set, horizon, ok)
    pairs: list[tuple[CostKind, list[int | None]]] = [
        ("per_step", want_p),
        ("onset", want_o),
    ]
    for cost, want in pairs:
        result = automaton_dp(net, edge_set, Objective(goal, horizon, cost), automaton)
        got = [
            int(c) if f else None
            for f, c in zip(result.feasible, result.min_cost, strict=True)
        ]
        assert got == want, cost


@pytest.mark.parametrize("model", ["cortical", "wnt5a", "cell_cycle10"])
def test_matches_existing_dps(model: str) -> None:
    net = VectorNetwork(load_network_spec(model))
    goal = net.goal_mask()
    edge_set = (0, 3, 7)
    for automaton, reference in [
        (
            unconstrained(3),
            constrained_dp(net, edge_set, goal, (5, 8), InputSpec("none")),
        ),
        (monotone(3), monotone_dp(net, edge_set, goal, 5, 8)),
        (
            recovery(3, 2),
            constrained_dp(net, edge_set, goal, (5, 8), InputSpec("recovery", 2)),
        ),
    ]:
        got = automaton_dp(net, edge_set, Objective(goal, (5, 8)), automaton)
        assert np.array_equal(got.feasible, reference.feasible)
        assert np.array_equal(got.min_cost, reference.min_cost)


# Goals "node = value" that the network does not keep by itself.
_HOLD_CASES = [
    ("cortical", (2, 5), (3, 5), (0, 0)),
    ("wnt5a", (0, 4), (3, 5), (0, 1)),
    ("wnt5a", (0, 4), (3, 5), (5, 1)),
]


@pytest.mark.parametrize("case", _HOLD_CASES)
@pytest.mark.parametrize("spec_name", ["none", "monotone", "recovery2"])
@pytest.mark.parametrize("hold_without_removal", [False, True])
def test_node_goal_matches_enumeration(
    case: tuple[str, tuple[int, ...], tuple[int, int], tuple[int, int]],
    spec_name: str,
    hold_without_removal: bool,
) -> None:
    model, edge_set, horizon, goal_node = case
    automaton, ok = _predicates(len(edge_set), horizon[1])[spec_name]
    net = VectorNetwork(load_network_spec(model))
    ids = np.arange(net.n_states)
    node, value = goal_node
    goal = ((ids >> (net.n_nodes - 1 - node)) & 1) == value
    want_p, want_o = _brute(
        model, edge_set, horizon, ok, _Variant(hold_without_removal, goal_node)
    )
    pairs: list[tuple[CostKind, list[int | None]]] = [
        ("per_step", want_p),
        ("onset", want_o),
    ]
    for cost, want in pairs:
        objective = Objective(goal, horizon, cost, hold_without_removal)
        result = automaton_dp(net, edge_set, objective, automaton)
        got = [
            int(c) if f else None
            for f, c in zip(result.feasible, result.min_cost, strict=True)
        ]
        assert got == want, cost


# wnt5a, goal "node 0 = 1", edges (0, 4), (H, T) = (3, 5): feasible states.
_WNT5A_FREE = 128
_WNT5A_HELD = 18


def test_hold_without_removal_loses_states() -> None:
    net = VectorNetwork(load_network_spec("wnt5a"))
    ids = np.arange(net.n_states)
    goal = ((ids >> (net.n_nodes - 1)) & 1) == 1
    free = automaton_dp(net, (0, 4), Objective(goal, (3, 5)), unconstrained(2))
    held = automaton_dp(
        net,
        (0, 4),
        Objective(goal, (3, 5), hold_without_removal=True),
        unconstrained(2),
    )
    assert int(free.feasible.sum()) == _WNT5A_FREE
    assert int(held.feasible.sum()) == _WNT5A_HELD


@pytest.mark.parametrize("case", _HOLD_CASES)
@pytest.mark.parametrize("spec_name", ["none", "monotone", "budget3"])
@pytest.mark.parametrize(
    "hold", [("steps", 0), ("steps", 1), ("steps", 2), ("frozen", 0)]
)
def test_hold_window_and_frozen_match_enumeration(
    case: tuple[str, tuple[int, ...], tuple[int, int], tuple[int, int]],
    spec_name: str,
    hold: tuple[str, int],
) -> None:
    model, edge_set, horizon, goal_node = case
    automaton, ok = _predicates(len(edge_set), horizon[1])[spec_name]
    net = VectorNetwork(load_network_spec(model))
    ids = np.arange(net.n_states)
    node, value = goal_node
    goal = ((ids >> (net.n_nodes - 1 - node)) & 1) == value
    steps = hold[1] if hold[0] == "steps" else None
    frozen = hold[0] == "frozen"
    variant = _Variant(
        goal_node=goal_node, hold_removal_steps=steps, hold_frozen=frozen
    )
    want_p, want_o = _brute(model, edge_set, horizon, ok, variant)
    pairs: list[tuple[CostKind, list[int | None]]] = [
        ("per_step", want_p),
        ("onset", want_o),
    ]
    for cost in pairs:
        objective = Objective(
            goal, horizon, cost[0], hold_removal_steps=steps, hold_frozen=frozen
        )
        result = automaton_dp(net, edge_set, objective, automaton)
        got = [
            int(c) if f else None
            for f, c in zip(result.feasible, result.min_cost, strict=True)
        ]
        assert got == cost[1], cost[0]


def test_hold_window_and_frozen_are_not_trivial() -> None:
    net = VectorNetwork(load_network_spec("wnt5a"))
    ids = np.arange(net.n_states)
    goal = ((ids >> (net.n_nodes - 1)) & 1) == 1
    counts = {}
    for name, objective in {
        "none": Objective(goal, (3, 5)),
        "steps1": Objective(goal, (3, 5), hold_removal_steps=1),
        "frozen": Objective(goal, (3, 5), hold_frozen=True),
    }.items():
        result = automaton_dp(net, (0, 4), objective, unconstrained(2))
        counts[name] = int(result.feasible.sum())
    assert counts["none"] > counts["steps1"] > _WNT5A_HELD
    assert counts["none"] > counts["frozen"]
