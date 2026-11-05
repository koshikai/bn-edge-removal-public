"""Price of temporal input specifications for a goal set (exact, no learning).

Given a network, a goal mask and a domain of initial states:

* ``select_edges`` chooses up to ``max_size`` removable edges greedily, each step
  adding the edge that maximises the number of feasible initial states under a
  given specification;
* ``price_fixed`` evaluates a fixed removable set under several specifications
  and costs, and compares each with the unconstrained case on the same states.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from bn_edge_removal.analysis.exact_monotone import VectorNetwork
from bn_edge_removal.analysis.input_automata import (
    CostKind,
    InputAutomaton,
    Objective,
    automaton_dp,
)

type Builder = Callable[[int], InputAutomaton]


@dataclass(frozen=True)
class PriceSetup:
    """Network, goal, initial-state domain, candidates and horizon.

    Args:
        net: Vectorised network.
        goal: Boolean goal mask over all states.
        domain: Boolean mask of the initial states that are counted.
        candidates: Zero-based ``all_edges`` indices that may be selected.
        horizon: ``(H, T)``.
    """

    net: VectorNetwork
    goal: np.ndarray
    domain: np.ndarray
    candidates: tuple[int, ...]
    horizon: tuple[int, int]


def feasible_count(setup: PriceSetup, edge_set: tuple[int, ...], build: Builder) -> int:
    """Number of domain states from which ``first_entry`` is feasible."""
    result = automaton_dp(
        setup.net, edge_set, Objective(setup.goal, setup.horizon), build(len(edge_set))
    )
    return int((result.feasible & setup.domain).sum())


def select_edges(
    setup: PriceSetup, build: Builder, max_size: int
) -> tuple[tuple[int, ...], list[int]]:
    """Greedy selection by the feasible count under ``build``.

    Ties go to the lowest index.  Stops at ``max_size``, when no candidate
    increases the count, or when every domain state is feasible.

    Returns:
        The selected edges and the count after each step (first: empty set).
    """
    selected: tuple[int, ...] = ()
    counts = [feasible_count(setup, selected, build)]
    total = int(setup.domain.sum())
    while len(selected) < max_size and counts[-1] < total:
        best: tuple[int, int] | None = None
        for edge in setup.candidates:
            if edge in selected:
                continue
            value = feasible_count(setup, (*selected, edge), build)
            if best is None or value > best[1]:
                best = (edge, value)
        if best is None or best[1] <= counts[-1]:
            break
        selected = (*selected, best[0])
        counts.append(best[1])
    return selected, counts


def price_fixed(
    setup: PriceSetup,
    edge_set: tuple[int, ...],
    specs: dict[str, Builder],
    costs: tuple[CostKind, ...] = ("per_step", "onset"),
) -> list[dict[str, object]]:
    """Feasible counts and minimum costs of ``edge_set`` under each spec.

    ``specs`` must contain ``"none"``; every other spec is compared with it on
    the states feasible in both (``lost_vs_none``, ``mean_extra_cost_vs_none``,
    ``states_with_extra_cost``).
    """
    if "none" not in specs:
        raise ValueError("specs must contain 'none'")
    m = len(edge_set)
    rows: list[dict[str, object]] = []
    for cost in costs:
        objective = Objective(setup.goal, setup.horizon, cost)
        free = automaton_dp(setup.net, edge_set, objective, specs["none"](m))
        free_ok = free.feasible & setup.domain
        for name, build in specs.items():
            result = (
                free
                if name == "none"
                else automaton_dp(setup.net, edge_set, objective, build(m))
            )
            ok = result.feasible & setup.domain
            costs_ok = result.min_cost[ok]
            row: dict[str, object] = {
                "spec": name,
                "cost": cost,
                "feasible": int(ok.sum()),
                "mean_min_cost": float(costs_ok.mean()) if costs_ok.size else None,
            }
            if name != "none":
                both = ok & free_ok
                row["lost_vs_none"] = int((free_ok & ~ok).sum())
                row["mean_extra_cost_vs_none"] = (
                    float((result.min_cost[both] - free.min_cost[both]).mean())
                    if both.any()
                    else None
                )
                row["states_with_extra_cost"] = int(
                    (both & (result.min_cost > free.min_cost)).sum()
                )
            rows.append(row)
    return rows
