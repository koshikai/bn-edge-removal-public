"""Spec <-> code checks for every packaged network.

- every removable edge is cancelled by its control bit, and that bit touches nothing
  but the update of the edge's target;
- ``all_edges`` is exactly the set of dependencies of the implemented update rules
  (source, target and sign) without intervention;
- ``update_equations`` reproduce ``next_state`` for every state and control input;
- every goal state is a fixed point without intervention.
"""

from __future__ import annotations

import re
from itertools import product

import pytest

from bn_edge_removal.models.cell_cycle10 import CellCycle10Model
from bn_edge_removal.models.cortical import CorticalModel
from bn_edge_removal.control.env import (
    EdgeRemovalModel,
    unmaintainable_goal_states,
)
from bn_edge_removal.models.network_spec import (
    RemovableEdge,
    load_network_spec,
)
from bn_edge_removal.models.s_pombe import SPombeModel
from bn_edge_removal.models.wnt5a import Wnt5aModel

_MODELS: dict[str, EdgeRemovalModel] = {
    "cortical": CorticalModel(),
    "wnt5a": Wnt5aModel(),
    "cell_cycle10": CellCycle10Model(),
    "s_pombe": SPombeModel(),
}

_EDGE_CASES = [
    pytest.param(name, edge, id=f"{name}-u{edge.index}-{edge.source}to{edge.target}")
    for name in _MODELS
    for edge in load_network_spec(name).removable_edges
]


def _source_influences_target(
    model: EdgeRemovalModel, edge: RemovableEdge, u: list[int]
) -> bool:
    """Whether flipping x_source changes x_target' for some state under input u."""
    for bits in product([0, 1], repeat=model.n_nodes):
        x = list(bits)
        flipped = x.copy()
        flipped[edge.source - 1] ^= 1
        target = edge.target - 1
        if model.next_state(x, u)[target] != model.next_state(flipped, u)[target]:
            return True
    return False


@pytest.mark.parametrize(("name", "edge"), _EDGE_CASES)
def test_removable_edge_exists_without_intervention(
    name: str, edge: RemovableEdge
) -> None:
    model = _MODELS[name]
    assert _source_influences_target(model, edge, [0] * model.m_edges)


@pytest.mark.parametrize(("name", "edge"), _EDGE_CASES)
def test_removing_edge_cancels_source_influence(name: str, edge: RemovableEdge) -> None:
    model = _MODELS[name]
    others = [i for i in range(model.m_edges) if i != edge.index - 1]
    for other_bits in product([0, 1], repeat=len(others)):
        u = [0] * model.m_edges
        u[edge.index - 1] = 1
        for i, bit in zip(others, other_bits, strict=True):
            u[i] = bit
        assert not _source_influences_target(model, edge, u), f"u={u}"


@pytest.mark.parametrize("name", list(_MODELS))
def test_model_dimensions_match_spec(name: str) -> None:
    spec = load_network_spec(name)
    model = _MODELS[name]
    assert model.n_nodes == spec.n_nodes
    assert model.m_edges == spec.m_edges == len(spec.removable_edges)


def _dependencies(model: EdgeRemovalModel, u: list[int]) -> dict[tuple[int, int], str]:
    """Implemented dependencies under input u: (source, target) -> sign (1-based)."""
    signs: dict[tuple[int, int], set[str]] = {}
    for bits in product([0, 1], repeat=model.n_nodes):
        low = list(bits)
        next_low = model.next_state(low, u)
        for source in range(model.n_nodes):
            if low[source]:
                continue
            high = low.copy()
            high[source] = 1
            next_high = model.next_state(high, u)
            for target in range(model.n_nodes):
                if next_high[target] != next_low[target]:
                    sign = "activation" if next_high[target] else "inhibition"
                    signs.setdefault((source + 1, target + 1), set()).add(sign)
    return {edge: "/".join(sorted(found)) for edge, found in signs.items()}


@pytest.mark.parametrize("name", list(_MODELS))
def test_all_edges_are_exactly_the_implemented_dependencies(name: str) -> None:
    spec = load_network_spec(name)
    model = _MODELS[name]
    declared = {(edge.source, edge.target): edge.sign for edge in spec.all_edges}
    assert len(declared) == len(spec.all_edges), "duplicate edge in all_edges"

    implemented = _dependencies(model, [0] * model.m_edges)

    assert sorted(set(declared) - set(implemented)) == [], "declared but not in code"
    assert sorted(set(implemented) - set(declared)) == [], "in code but not declared"
    assert implemented == declared


@pytest.mark.parametrize(("name", "edge"), _EDGE_CASES)
def test_control_bit_only_changes_its_target_update(
    name: str, edge: RemovableEdge
) -> None:
    model = _MODELS[name]
    others = [i for i in range(model.n_nodes) if i != edge.target - 1]
    for bits in product([0, 1], repeat=model.n_nodes):
        x = list(bits)
        for u_bits in product([0, 1], repeat=model.m_edges):
            u = list(u_bits)
            if u[edge.index - 1]:
                continue
            removed = u.copy()
            removed[edge.index - 1] = 1
            kept_next = model.next_state(x, u)
            removed_next = model.next_state(x, removed)
            assert [kept_next[i] for i in others] == [removed_next[i] for i in others]


def _compile_update_equations(name: str) -> list[tuple[int, str]]:
    """Turn ``"x3' = <expr>"`` into (target index, Python expression over x / u)."""
    compiled: list[tuple[int, str]] = []
    for equation in load_network_spec(name).update_equations:
        match = re.fullmatch(r"x0*(\d+)' = (.+)", equation)
        assert match is not None, equation
        expression = re.sub(
            r"\b([xu])0*(\d+)\b",
            lambda m: f"{m.group(1)}[{int(m.group(2)) - 1}]",
            match.group(2),
        )
        compiled.append((int(match.group(1)) - 1, expression))
    return compiled


@pytest.mark.parametrize("name", list(_MODELS))
def test_update_equations_reproduce_next_state(name: str) -> None:
    model = _MODELS[name]
    equations = _compile_update_equations(name)
    assert [target for target, _expr in equations] == list(range(model.n_nodes))

    for bits in product([0, 1], repeat=model.n_nodes):
        x = list(bits)
        for u_bits in product([0, 1], repeat=model.m_edges):
            u = list(u_bits)
            expected = [
                int(bool(eval(expr, {"__builtins__": {}}, {"x": x, "u": u})))
                for _target, expr in equations
            ]
            assert model.next_state(x, u) == expected, f"x={x} u={u}"


@pytest.mark.parametrize("name", list(_MODELS))
def test_goal_states_are_fixed_points_without_intervention(name: str) -> None:
    model = _MODELS[name]
    for goal in model.goal_states():
        assert model.next_state(goal, [0] * model.m_edges) == goal
    assert unmaintainable_goal_states(model) == []
