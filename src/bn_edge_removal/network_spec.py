"""Load Boolean-network metadata from YAML specs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any

import yaml


@dataclass(frozen=True)
class RemovableEdge:
    index: int
    source: int
    target: int
    sign: str


@dataclass(frozen=True)
class NetworkEdge:
    source: int
    target: int
    sign: str
    removable: bool
    removable_index: int | None


@dataclass(frozen=True)
class NetworkSpec:
    name: str
    n_nodes: int
    m_edges: int
    goal_states: tuple[tuple[int, ...], ...]
    removable_edges: tuple[RemovableEdge, ...]
    all_edges: tuple[NetworkEdge, ...]
    update_equations: tuple[str, ...]


def _as_binary_state(values: Any, n_nodes: int, label: str) -> tuple[int, ...]:
    if not isinstance(values, list) or len(values) != n_nodes:
        raise ValueError(f"{label} must be a list of length {n_nodes}")
    parsed: list[int] = []
    for idx, value in enumerate(values):
        if value not in (0, 1):
            raise ValueError(f"{label}[{idx}] must be 0 or 1")
        parsed.append(1 if value == 1 else 0)
    return tuple(parsed)


def _parse_removable_edges(values: Any, n_nodes: int) -> tuple[RemovableEdge, ...]:
    if not isinstance(values, list):
        raise ValueError("removable_edges must be a list")

    edges: list[RemovableEdge] = []
    for entry in values:
        if not isinstance(entry, dict):
            raise ValueError("each removable edge must be a mapping")
        try:
            index = int(entry["index"])
            source = int(entry["source"])
            target = int(entry["target"])
            sign = _parse_sign(entry["sign"])
        except KeyError as exc:
            raise ValueError("removable edge is missing required keys") from exc

        if source < 1 or source > n_nodes or target < 1 or target > n_nodes:
            raise ValueError("edge source/target must be in [1, n_nodes]")

        edges.append(
            RemovableEdge(
                index=index,
                source=source,
                target=target,
                sign=sign,
            )
        )

    return tuple(edges)


def _parse_sign(value: Any) -> str:
    sign = str(value)
    if sign not in {"activation", "inhibition"}:
        raise ValueError("edge sign must be 'activation' or 'inhibition'")
    return sign


def _parse_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _validate_removable_indices(indices: list[int], m_edges: int) -> None:
    if len(indices) != m_edges:
        raise ValueError(
            f"m_edges={m_edges} does not match removable edges={len(indices)}"
        )
    expected_indices = list(range(1, m_edges + 1))
    if sorted(indices) != expected_indices:
        raise ValueError(
            "removable edge indices must be contiguous starting from 1 "
            f"(expected {expected_indices}, got {sorted(indices)})"
        )


def _parse_all_edges(
    values: Any, n_nodes: int, m_edges: int
) -> tuple[NetworkEdge, ...]:
    if not isinstance(values, list):
        raise ValueError("all_edges must be a list")

    edges: list[NetworkEdge] = []
    removable_indices: list[int] = []
    for entry in values:
        if not isinstance(entry, dict):
            raise ValueError("each edge in all_edges must be a mapping")
        try:
            source = int(entry["source"])
            target = int(entry["target"])
            sign = _parse_sign(entry["sign"])
            removable = _parse_bool(entry["removable"], label="removable")
        except KeyError as exc:
            raise ValueError("all_edges entry is missing required keys") from exc

        if source < 1 or source > n_nodes or target < 1 or target > n_nodes:
            raise ValueError("edge source/target must be in [1, n_nodes]")

        removable_index_raw = entry.get("removable_index")
        removable_index: int | None = None
        if removable:
            if removable_index_raw is None:
                raise ValueError(
                    "removable edge in all_edges must have removable_index"
                )
            removable_index = int(removable_index_raw)
            removable_indices.append(removable_index)
        elif removable_index_raw is not None:
            raise ValueError("fixed edge in all_edges must not have removable_index")

        edges.append(
            NetworkEdge(
                source=source,
                target=target,
                sign=sign,
                removable=removable,
                removable_index=removable_index,
            )
        )

    _validate_removable_indices(removable_indices, m_edges)
    return tuple(edges)


def _parse_update_equations(values: Any, n_nodes: int) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError("update_equations must be a list")
    if len(values) != n_nodes:
        raise ValueError(
            f"update_equations must have length {n_nodes} (got {len(values)})"
        )
    equations: list[str] = []
    for idx, value in enumerate(values):
        if not isinstance(value, str):
            raise ValueError(f"update_equations[{idx}] must be a string")
        equations.append(value.strip())
    return tuple(equations)


def _all_edges_from_removable(
    removable_edges: tuple[RemovableEdge, ...],
) -> tuple[NetworkEdge, ...]:
    return tuple(
        NetworkEdge(
            source=edge.source,
            target=edge.target,
            sign=edge.sign,
            removable=True,
            removable_index=edge.index,
        )
        for edge in removable_edges
    )


def _validate_removable_consistency(
    removable_edges: tuple[RemovableEdge, ...], all_edges: tuple[NetworkEdge, ...]
) -> None:
    all_removable = sorted(
        (
            edge.removable_index,
            edge.source,
            edge.target,
            edge.sign,
        )
        for edge in all_edges
        if edge.removable
    )
    explicit_removable = sorted(
        (edge.index, edge.source, edge.target, edge.sign) for edge in removable_edges
    )
    if all_removable != explicit_removable:
        raise ValueError(
            "removable_edges and all_edges are inconsistent in index/source/target/sign"
        )


@cache
def load_network_spec(name: str) -> NetworkSpec:
    """Load and validate one network spec from package YAML files."""
    resource = files("bn_edge_removal.network_specs").joinpath(
        f"{name}.yaml"
    )
    if not resource.is_file():
        raise FileNotFoundError(f"network spec not found: {name}")

    data = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"network spec {name} must be a YAML mapping")

    if "n_nodes" not in data or "m_edges" not in data:
        raise ValueError(f"network spec {name} is missing n_nodes or m_edges")

    n_nodes = int(data["n_nodes"])
    m_edges = int(data["m_edges"])

    goal_states_raw = data.get("goal_states", [])
    if not isinstance(goal_states_raw, list):
        raise ValueError("goal_states must be a list")
    goal_states = tuple(
        _as_binary_state(state, n_nodes, "goal_state") for state in goal_states_raw
    )

    removable_edges = _parse_removable_edges(data.get("removable_edges", []), n_nodes)
    _validate_removable_indices([edge.index for edge in removable_edges], m_edges)

    if "all_edges" in data:
        all_edges = _parse_all_edges(data["all_edges"], n_nodes, m_edges)
    else:
        all_edges = _all_edges_from_removable(removable_edges)

    update_equations = _parse_update_equations(
        data.get("update_equations", []), n_nodes
    )

    _validate_removable_consistency(removable_edges, all_edges)

    return NetworkSpec(
        name=str(data.get("name", name)),
        n_nodes=n_nodes,
        m_edges=m_edges,
        goal_states=goal_states,
        removable_edges=removable_edges,
        all_edges=all_edges,
        update_equations=update_equations,
    )
