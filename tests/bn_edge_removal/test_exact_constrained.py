"""Checks of ``constrained_dp`` against a plain recursion and ``monotone_dp``."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from bn_edge_removal.analysis.edge_selection import AllEdgeModel
from bn_edge_removal.analysis.exact_constrained import (
    Constraint,
    InputSpec,
    constrained_dp,
    flag_transitions,
)
from bn_edge_removal.analysis.exact_monotone import VectorNetwork, monotone_dp
from bn_edge_removal.control.stl_recovery import check_violation, update_flags
from bn_edge_removal.models.network_spec import load_network_spec

_CASES = [
    ("cortical", (0, 3, 7), 5, 10),
    ("wnt5a", (0, 4, 9), 7, 10),
    ("wnt5a", (1, 2), 2, 5),
    ("cell_cycle10", (5, 12, 20), 7, 10),
]


def _reference(
    name: str,
    edge_set: tuple[int, ...],
    horizon: tuple[int, int],
    constraint: Constraint,
    tau: int,
) -> list[int | None]:
    """Minimum cost per initial state by plain recursion (None if infeasible)."""
    reach, steps = horizon
    spec = load_network_spec(name)
    model = AllEdgeModel(spec)
    goal = {tuple(state) for state in spec.goal_states}
    m = len(edge_set)

    def full(bits: list[int]) -> list[int]:
        vector = [0] * model.m_edges
        for k, bit in enumerate(bits):
            vector[edge_set[k]] = bit
        return vector

    def moves(flag: tuple[int, ...]) -> list[tuple[list[int], tuple[int, ...]]]:
        out: list[tuple[list[int], tuple[int, ...]]] = []
        for u in range(2**m):
            bits = [(u >> k) & 1 for k in range(m)]
            if constraint == "none":
                out.append((bits, flag))
            elif constraint == "monotone":
                if all(b or not f for b, f in zip(bits, flag, strict=True)):
                    out.append(
                        (
                            bits,
                            tuple(int(b or f) for b, f in zip(bits, flag, strict=True)),
                        )
                    )
            elif not check_violation(list(flag), bits):
                out.append((bits, tuple(update_flags(list(flag), bits, tau))))
        return out

    @cache
    def best(
        t: int, x: tuple[int, ...], flag: tuple[int, ...], holding: bool
    ) -> int | None:
        if t == steps:
            return 0 if holding else None
        options: list[int] = []
        for bits, nxt_flag in moves(flag):
            nxt = tuple(model.next_state(list(x), full(bits)))
            if holding and nxt not in goal:
                continue
            if nxt in goal:
                rest = best(t + 1, nxt, nxt_flag, True)
            elif t + 1 < reach:
                rest = best(t + 1, nxt, nxt_flag, False)
            else:
                rest = None
            if rest is not None:
                options.append(rest + sum(bits))
        return min(options) if options else None

    start: tuple[int, ...] = () if constraint == "none" else (0,) * m
    results: list[int | None] = []
    for state_id in range(2**spec.n_nodes):
        x = tuple(model.id_to_state(state_id))
        if x in goal:
            results.append(best(0, x, start, True))
        elif reach == 0:
            results.append(None)
        else:
            results.append(best(0, x, start, False))
    return results


@pytest.mark.parametrize(("name", "edge_set", "reach", "steps"), _CASES)
@pytest.mark.parametrize(
    ("constraint", "tau"),
    [("none", 2), ("monotone", 2), ("recovery", 1), ("recovery", 2), ("recovery", 3)],
)
def test_constrained_dp_matches_reference(
    name: str,
    edge_set: tuple[int, ...],
    reach: int,
    steps: int,
    constraint: Constraint,
    tau: int,
) -> None:
    net = VectorNetwork(load_network_spec(name))
    result = constrained_dp(
        net, edge_set, net.goal_mask(), (reach, steps), InputSpec(constraint, tau)
    )
    expected = _reference(name, edge_set, (reach, steps), constraint, tau)
    got = [
        int(c) if f else None
        for f, c in zip(result.feasible, result.min_cost, strict=True)
    ]
    assert got == expected


@pytest.mark.parametrize(("name", "edge_set", "reach", "steps"), _CASES)
def test_monotone_matches_monotone_dp(
    name: str, edge_set: tuple[int, ...], reach: int, steps: int
) -> None:
    net = VectorNetwork(load_network_spec(name))
    goal = net.goal_mask()
    a = constrained_dp(net, edge_set, goal, (reach, steps), InputSpec("monotone"))
    b = monotone_dp(net, edge_set, goal, reach, steps)
    assert np.array_equal(a.feasible, b.feasible)
    assert np.array_equal(a.min_cost, b.min_cost)


@pytest.mark.parametrize(("name", "edge_set", "reach", "steps"), _CASES)
def test_constraints_only_remove_options(
    name: str, edge_set: tuple[int, ...], reach: int, steps: int
) -> None:
    net = VectorNetwork(load_network_spec(name))
    goal = net.goal_mask()
    free = constrained_dp(net, edge_set, goal, (reach, steps), InputSpec("none"))
    for spec in [
        InputSpec("monotone"),
        InputSpec("recovery", 1),
        InputSpec("recovery", 3),
    ]:
        sub = constrained_dp(net, edge_set, goal, (reach, steps), spec)
        assert np.all(sub.feasible <= free.feasible)
        assert np.all(sub.min_cost[sub.feasible] >= free.min_cost[sub.feasible])


def test_flag_space_sizes() -> None:
    assert len(flag_transitions(3, "none")[0]) == 1
    assert len(flag_transitions(3, "monotone")[0]) == 8
    assert len(flag_transitions(3, "recovery", 2)[0]) == 27


def _reference_onset(
    name: str,
    edge_set: tuple[int, ...],
    horizon: tuple[int, int],
    spec: InputSpec,
) -> list[int | None]:
    """Minimum number of removal onsets per initial state by plain recursion."""
    reach, steps = horizon
    model_spec = load_network_spec(name)
    model = AllEdgeModel(model_spec)
    goal = {tuple(state) for state in model_spec.goal_states}
    m = len(edge_set)

    def full(bits: tuple[int, ...]) -> list[int]:
        vector = [0] * model.m_edges
        for k, bit in enumerate(bits):
            vector[edge_set[k]] = bit
        return vector

    def allowed(
        bits: tuple[int, ...], removed: tuple[int, ...], cool: tuple[int, ...]
    ) -> bool:
        if spec.kind == "monotone":
            return all(b or not r for b, r in zip(bits, removed, strict=True))
        if spec.kind == "recovery":
            return not check_violation(list(cool), list(bits))
        return True

    @cache
    def best(
        t: int,
        x: tuple[int, ...],
        prev: tuple[int, ...],
        removed: tuple[int, ...],
        cool: tuple[int, ...],
        holding: bool,
    ) -> int | None:
        if t == steps:
            return 0 if holding else None
        options: list[int] = []
        for u in range(2**m):
            bits = tuple((u >> k) & 1 for k in range(m))
            if not allowed(bits, removed, cool):
                continue
            nxt = tuple(model.next_state(list(x), full(bits)))
            if holding and nxt not in goal:
                continue
            new_removed = tuple(int(a or b) for a, b in zip(bits, removed, strict=True))
            new_cool = tuple(update_flags(list(cool), list(bits), spec.tau))
            if nxt in goal:
                rest = best(t + 1, nxt, bits, new_removed, new_cool, True)
            elif t + 1 < reach:
                rest = best(t + 1, nxt, bits, new_removed, new_cool, False)
            else:
                rest = None
            if rest is not None:
                onsets = sum(b and not p for b, p in zip(bits, prev, strict=True))
                options.append(rest + onsets)
        return min(options) if options else None

    zero = (0,) * m
    results: list[int | None] = []
    for state_id in range(2**model_spec.n_nodes):
        x = tuple(model.id_to_state(state_id))
        if x in goal:
            results.append(best(0, x, zero, zero, zero, True))
        elif reach == 0:
            results.append(None)
        else:
            results.append(best(0, x, zero, zero, zero, False))
    return results


@pytest.mark.parametrize(("name", "edge_set", "reach", "steps"), _CASES)
@pytest.mark.parametrize(
    "spec",
    [
        InputSpec("none", cost="onset"),
        InputSpec("monotone", cost="onset"),
        InputSpec("recovery", 2, cost="onset"),
    ],
)
def test_onset_cost_matches_reference(
    name: str, edge_set: tuple[int, ...], reach: int, steps: int, spec: InputSpec
) -> None:
    net = VectorNetwork(load_network_spec(name))
    result = constrained_dp(net, edge_set, net.goal_mask(), (reach, steps), spec)
    expected = _reference_onset(name, edge_set, (reach, steps), spec)
    got = [
        int(c) if f else None
        for f, c in zip(result.feasible, result.min_cost, strict=True)
    ]
    assert got == expected


@pytest.mark.parametrize(("name", "edge_set", "reach", "steps"), _CASES)
def test_recovery_onset_equals_per_step(
    name: str, edge_set: tuple[int, ...], reach: int, steps: int
) -> None:
    net = VectorNetwork(load_network_spec(name))
    goal = net.goal_mask()
    a = constrained_dp(net, edge_set, goal, (reach, steps), InputSpec("recovery", 2))
    b = constrained_dp(
        net, edge_set, goal, (reach, steps), InputSpec("recovery", 2, cost="onset")
    )
    assert np.array_equal(a.min_cost, b.min_cost)
