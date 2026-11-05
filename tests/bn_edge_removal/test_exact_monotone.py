"""Checks of the vectorised exact analysis against plain Python references."""

from __future__ import annotations

from functools import cache
from itertools import combinations

import numpy as np
import pytest

from bn_edge_removal.analysis.edge_selection import AllEdgeModel
from bn_edge_removal.analysis.exact_monotone import (
    INF,
    SelectionProblem,
    greedy_dp_select,
    monotone_feasible,
    VectorNetwork,
    greedy_static_select,
    monotone_dp,
    random_sets,
    static_success,
)
from bn_edge_removal.models.network_spec import load_network_spec

_CASES = [
    ("cortical", (0, 3, 7), 5, 10),
    ("cortical", (2, 5), 3, 6),
    ("wnt5a", (0, 4, 9), 7, 10),
    ("wnt5a", (1, 2), 0, 4),
    ("cell_cycle10", (5, 12, 20), 7, 10),
]


def _reference(
    name: str, edge_set: tuple[int, ...], reach: int, steps: int
) -> list[int | None]:
    """Minimum cost per initial state by plain recursion (None if infeasible)."""
    spec = load_network_spec(name)
    model = AllEdgeModel(spec)
    goal = {tuple(state) for state in spec.goal_states}
    m = len(edge_set)

    def full(u: int) -> list[int]:
        vector = [0] * model.m_edges
        for k in range(m):
            if (u >> k) & 1:
                vector[edge_set[k]] = 1
        return vector

    @cache
    def best(t: int, x: tuple[int, ...], h: int, holding: bool) -> int | None:
        if t == steps:
            return 0 if holding else None
        options: list[int] = []
        for u in range(2**m):
            if u & h != h:
                continue
            nxt = tuple(model.next_state(list(x), full(u)))
            if holding:
                if nxt not in goal:
                    continue
                rest = best(t + 1, nxt, u, True)
            elif nxt in goal:
                rest = best(t + 1, nxt, u, True)
            elif t + 1 < reach:
                rest = best(t + 1, nxt, u, False)
            else:
                rest = None
            if rest is not None:
                options.append(rest + bin(u).count("1"))
        return min(options) if options else None

    results: list[int | None] = []
    for state_id in range(2**spec.n_nodes):
        x = tuple(model.id_to_state(state_id))
        if x in goal:
            results.append(best(0, x, 0, True))
        elif reach == 0:
            results.append(None)
        else:
            results.append(best(0, x, 0, False))
    return results


@pytest.mark.parametrize(("name", "edge_set", "reach", "steps"), _CASES)
def test_monotone_dp_matches_reference(
    name: str, edge_set: tuple[int, ...], reach: int, steps: int
) -> None:
    net = VectorNetwork(load_network_spec(name))
    result = monotone_dp(net, edge_set, net.goal_mask(), reach, steps)
    expected = _reference(name, edge_set, reach, steps)
    got = [
        int(c) if f else None
        for f, c in zip(result.feasible, result.min_cost, strict=True)
    ]
    assert got == expected
    assert np.all(result.min_cost[~result.feasible] == INF)


@pytest.mark.parametrize("name", ["cortical", "wnt5a", "cell_cycle10", "s_pombe"])
def test_successors_match_all_edge_model(name: str) -> None:
    spec = load_network_spec(name)
    net = VectorNetwork(spec)
    model = AllEdgeModel(spec)
    for removed in [(), (0,), (1, 3), tuple(range(0, len(net.edges), 4))]:
        succ = net.successors(frozenset(removed))
        u = [1 if i in removed else 0 for i in range(model.m_edges)]
        for state_id in range(0, net.n_states, 7):
            x = model.id_to_state(state_id)
            assert model.state_to_id(model.next_state(x, u)) == int(succ[state_id])


@pytest.mark.parametrize(
    ("name", "reach", "steps"), [("cortical", 5, 10), ("wnt5a", 7, 10), ("wnt5a", 2, 3)]
)
def test_static_success_matches_trajectories(name: str, reach: int, steps: int) -> None:
    net = VectorNetwork(load_network_spec(name))
    goal = net.goal_mask()
    succ = net.successors(frozenset({0}))
    ok = static_success(succ, goal, reach, steps)
    for state_id in range(net.n_states):
        path = [state_id]
        for _ in range(steps):
            path.append(int(succ[path[-1]]))
        inside = [bool(goal[s]) for s in path]
        first = next((t for t, v in enumerate(inside) if v), None)
        expected = first is not None and first <= reach and all(inside[first:])
        assert bool(ok[state_id]) == expected


def test_static_success_is_monotone_dp_with_one_fixed_mask() -> None:
    net = VectorNetwork(load_network_spec("cortical"))
    goal = net.goal_mask()
    for edges in combinations(range(len(net.edges)), 2):
        static = static_success(net.successors(frozenset(edges)), goal, 5, 10)
        dp = monotone_dp(net, edges, goal, 5, 10).feasible
        assert np.all(static <= dp)


def test_greedy_counts_increase_and_respect_size() -> None:
    net = VectorNetwork(load_network_spec("cell_cycle10"))
    goal = net.goal_mask()
    domain = np.ones(net.n_states, dtype=bool)
    candidates = tuple(range(len(net.edges)))
    selected, counts = greedy_static_select(
        SelectionProblem(net, goal, domain, 7, 10), candidates, 3
    )
    assert len(selected) == len(counts) - 1 <= 3
    assert all(b > a for a, b in zip(counts, counts[1:], strict=False))
    assert counts[0] == int(static_success(net.successors(), goal, 7, 10).sum())


def test_random_sets_are_distinct_and_reproducible() -> None:
    first = random_sets(tuple(range(10)), 3, 5, seed=7)
    assert first == random_sets(tuple(range(10)), 3, 5, seed=7)
    assert len(set(first)) == 5
    assert all(len(s) == 3 and len(set(s)) == 3 for s in first)
    assert len(random_sets((1, 2, 3), 2, 10, seed=0)) == 3


def test_input_self_loops_of_tumor() -> None:
    net = VectorNetwork(load_network_spec("tumor_cohen2015_n20"))
    loops = {(net.edges[i].source, net.edges[i].target) for i in net.input_self_loops()}
    assert loops == {(1, 1), (2, 2)}


@pytest.mark.parametrize(("name", "edge_set", "reach", "steps"), _CASES)
def test_monotone_feasible_matches_dp(
    name: str, edge_set: tuple[int, ...], reach: int, steps: int
) -> None:
    net = VectorNetwork(load_network_spec(name))
    goal = net.goal_mask()
    full = monotone_dp(net, edge_set, goal, reach, steps).feasible
    assert np.array_equal(monotone_feasible(net, edge_set, goal, reach, steps), full)


def test_greedy_dp_counts_increase_and_stop() -> None:
    net = VectorNetwork(load_network_spec("cell_cycle10"))
    goal = net.goal_mask()
    domain = np.ones(net.n_states, dtype=bool)
    problem = SelectionProblem(net, goal, domain, 7, 10)
    selected, counts = greedy_dp_select(problem, tuple(range(len(net.edges))), 4)
    assert len(selected) == len(counts) - 1 <= 4
    assert all(b > a for a, b in zip(counts, counts[1:], strict=False))
    final = monotone_feasible(net, selected, goal, 7, 10)
    assert counts[-1] == int(final.sum())


def test_table_edge_model_matches_all_edge_model() -> None:
    from bn_edge_removal.analysis.exact_monotone import TableEdgeModel

    spec = load_network_spec("wnt5a")
    net = VectorNetwork(spec)
    generic = AllEdgeModel(spec)
    edge_set = (0, 4, 9)
    model = TableEdgeModel(net, edge_set)
    assert (model.n_nodes, model.m_edges) == (spec.n_nodes, 3)
    for state_id in range(net.n_states):
        x = model.id_to_state(state_id)
        assert model.state_to_id(x) == state_id
        assert model.is_goal(x) == (tuple(x) in {tuple(g) for g in spec.goal_states})
        for u in range(8):
            bits = [(u >> k) & 1 for k in range(3)]
            full = [0] * generic.m_edges
            for k, edge in enumerate(edge_set):
                full[edge] = bits[k]
            assert model.next_state(x, bits) == generic.next_state(x, full)
