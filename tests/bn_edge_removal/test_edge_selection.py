"""Checks for the edge-selection analysis module."""

from __future__ import annotations

import pytest

from bn_edge_removal.analysis.edge_selection import (
    AllEdgeModel,
    SubsetEdgeModel,
    WindowConfig,
    current_edge_indices,
    cycle_sign,
    edges_from_spec,
    enumerate_simple_cycles,
    greedy_select,
    minimum_hitting_set,
    verify_all_edge_model,
    window_success_count,
)
from bn_edge_removal.models.cell_cycle10 import CellCycle10Model
from bn_edge_removal.models.cortical import CorticalModel
from bn_edge_removal.models.network_spec import load_network_spec
from bn_edge_removal.models.s_pombe import SPombeModel
from bn_edge_removal.models.wnt5a import Wnt5aModel

_MODELS = {
    "cortical": CorticalModel(),
    "wnt5a": Wnt5aModel(),
    "cell_cycle10": CellCycle10Model(),
    "s_pombe": SPombeModel(),
}

_CONFIGS = {
    "cortical": WindowConfig(5, 10, "monotone"),
    "wnt5a": WindowConfig(7, 10, "recovery", recovery_tau=2),
    "cell_cycle10": WindowConfig(7, 10, "monotone"),
    "s_pombe": WindowConfig(7, 10, "monotone"),
}


@pytest.mark.parametrize("name", list(_MODELS))
def test_generic_simulator_matches_base_model(name: str) -> None:
    spec = load_network_spec(name)
    verify_all_edge_model(_MODELS[name], spec)


@pytest.mark.parametrize("name", list(_MODELS))
def test_minimum_hitting_sets_cover_cycles(name: str) -> None:
    spec = load_network_spec(name)
    edges = edges_from_spec(spec)
    cycles = enumerate_simple_cycles(edges, spec.n_nodes)
    positive = [cycle for cycle in cycles if cycle_sign(cycle, edges) == "positive"]

    fas = minimum_hitting_set([frozenset(cycle) for cycle in cycles])
    fes = minimum_hitting_set([frozenset(cycle) for cycle in positive])

    assert fas is not None
    assert fes is not None
    assert all(set(fas) & set(cycle) for cycle in cycles)
    assert all(set(fes) & set(cycle) for cycle in positive)


@pytest.mark.parametrize(
    ("name", "expected_size"),
    [("cortical", 1), ("wnt5a", 1), ("cell_cycle10", 2)],
)
def test_greedy_select_reaches_all_states(name: str, expected_size: int) -> None:
    spec = load_network_spec(name)
    generic = AllEdgeModel(spec)
    current = current_edge_indices(spec, edges_from_spec(spec))
    selected, result = greedy_select(generic, _CONFIGS[name], len(current))

    assert len(selected) == expected_size
    assert result.success_count == result.total_states


@pytest.mark.parametrize(
    ("name", "labels"),
    [
        ("cortical", ["5-|2"]),
        ("wnt5a", ["2-|7"]),
        ("cell_cycle10", ["5-|6", "10->10"]),
    ],
)
def test_known_sufficient_sets(name: str, labels: list[str]) -> None:
    spec = load_network_spec(name)
    edges = edges_from_spec(spec)
    indices = tuple(
        next(edge.index for edge in edges if edge.label == label) for label in labels
    )
    generic = AllEdgeModel(spec)
    result = window_success_count(
        SubsetEdgeModel(generic, indices),
        _CONFIGS[name],
    )

    assert result.success_count == result.total_states
