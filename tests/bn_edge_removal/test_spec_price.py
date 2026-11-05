"""Checks of the spec-price helpers."""

from __future__ import annotations

import numpy as np

from bn_edge_removal.analysis.exact_monotone import VectorNetwork
from bn_edge_removal.analysis.input_automata import (
    Objective,
    automaton_dp,
    monotone,
    recovery,
    unconstrained,
)
from bn_edge_removal.analysis.spec_price import PriceSetup, price_fixed, select_edges
from bn_edge_removal.models.network_spec import load_network_spec


def _setup() -> PriceSetup:
    net = VectorNetwork(load_network_spec("s_pombe"))
    goal = net.goal_mask()
    domain = np.ones(net.n_states, dtype=bool)
    return PriceSetup(net, goal, domain, tuple(range(len(net.edges))), (7, 10))


def test_select_edges_counts_increase() -> None:
    setup = _setup()
    selected, counts = select_edges(setup, unconstrained, 3)
    assert len(selected) == len(counts) - 1 <= 3
    assert all(b > a for a, b in zip(counts, counts[1:], strict=False))


def test_price_fixed_matches_direct_dp() -> None:
    setup = _setup()
    edges = (0, 5, 9)
    specs = {
        "none": unconstrained,
        "monotone": monotone,
        "recovery": lambda m: recovery(m, 2),
    }
    rows = price_fixed(setup, edges, specs)
    for row in rows:
        build = specs[str(row["spec"])]
        cost = "onset" if row["cost"] == "onset" else "per_step"
        direct = automaton_dp(
            setup.net, edges, Objective(setup.goal, (7, 10), cost), build(3)
        )
        assert row["feasible"] == int(direct.feasible.sum())
        if row["spec"] != "none":
            free = automaton_dp(
                setup.net, edges, Objective(setup.goal, (7, 10), cost), unconstrained(3)
            )
            assert row["lost_vs_none"] == int((free.feasible & ~direct.feasible).sum())
