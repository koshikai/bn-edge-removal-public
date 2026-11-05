"""Selection criteria for removable edges in Boolean networks.

The module provides:

* a generic simulator in which every dependency listed in a network spec is
  removable;
* structural criteria (simple cycles, minimum feedback arc/edge sets, and
  canalizing edges);
* exact window-success checks under the monotone / recovery input constraints.

The generic simulator is intended for analysis only.  Removing an edge
``source -> target`` substitutes the source variable by ``0`` in the target
update function, which is the standard edge-deletion semantics for monotone
dependencies.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from functools import cache
from types import CodeType
from typing import Literal

from bn_edge_removal.models.base import BooleanNetworkModel
from bn_edge_removal.models.encoding import int_to_bits
from bn_edge_removal.models.network_spec import NetworkSpec

type ConstraintType = Literal["monotone", "recovery"]
type BinaryState = tuple[int, ...]
type FlagState = tuple[int, ...]


@dataclass(frozen=True)
class Edge:
    """One dependency in a network spec.

    Args:
        index: Zero-based index in ``NetworkSpec.all_edges``.
        source: One-based source node.
        target: One-based target node.
        sign: ``"activation"`` or ``"inhibition"``.
    """

    index: int
    source: int
    target: int
    sign: str

    @property
    def label(self) -> str:
        """Human-readable edge label such as ``2-|1``."""
        arrow = "->" if self.sign == "activation" else "-|"
        return f"{self.source}{arrow}{self.target}"


@dataclass(frozen=True)
class WindowConfig:
    """Success condition and input constraint of one evaluation.

    Args:
        reach_horizon: Latest step at which the goal suffix may start.
        max_steps: Episode length.
        constraint_type: ``"monotone"`` or ``"recovery"``.
        recovery_tau: Cooldown length for the recovery constraint.
    """

    reach_horizon: int
    max_steps: int
    constraint_type: ConstraintType
    recovery_tau: int = 2

    def __post_init__(self) -> None:
        if self.reach_horizon < 0:
            raise ValueError("reach_horizon must be >= 0")
        if self.max_steps < self.reach_horizon:
            raise ValueError("max_steps must be >= reach_horizon")
        if self.constraint_type not in {"monotone", "recovery"}:
            raise ValueError("constraint_type must be 'monotone' or 'recovery'")
        if self.constraint_type == "recovery" and self.recovery_tau < 1:
            raise ValueError("recovery_tau must be >= 1")


@dataclass(frozen=True)
class WindowResult:
    """Number of initial states with a successful window trajectory."""

    success_count: int
    total_states: int

    @property
    def success_rate(self) -> float:
        """Fraction of successful initial states."""
        return self.success_count / self.total_states if self.total_states else 0.0


def edges_from_spec(spec: NetworkSpec) -> tuple[Edge, ...]:
    """Return all edges of ``spec`` in ``all_edges`` order."""
    return tuple(
        Edge(i, edge.source, edge.target, edge.sign)
        for i, edge in enumerate(spec.all_edges)
    )


def current_edge_indices(spec: NetworkSpec, edges: tuple[Edge, ...]) -> tuple[int, ...]:
    """Return the zero-based all-edge indices of ``spec.removable_edges``."""
    by_key = {(edge.source, edge.target): edge.index for edge in edges}
    return tuple(by_key[(edge.source, edge.target)] for edge in spec.removable_edges)


def label_set(edges: tuple[Edge, ...], indices: tuple[int, ...]) -> str:
    """Format a set of edge indices as ``[2-|1, 5->3]``."""
    return "[" + ", ".join(edges[i].label for i in indices) + "]"


def normalize_expression(expression: str) -> str:
    """Normalize ``x01`` to ``x1`` and neutralise existing ``u`` inputs."""
    expression = re.sub(r"\bu0*(\d+)\b", "0", expression)
    return re.sub(r"\bx0*(\d+)\b", lambda m: f"x{int(m.group(1))}", expression)


def compile_equations(spec: NetworkSpec) -> tuple[str, ...]:
    """Return normalized update expressions with all controls set to zero."""
    compiled: list[str] = []
    for equation in spec.update_equations:
        match = re.match(r"x0*(\d+)'\s*=\s*(.+)", equation, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"bad equation: {equation!r}")
        compiled.append(normalize_expression(match.group(2)))
    return tuple(compiled)


def _eval_expression(
    code: CodeType,
    *,
    x: list[int],
    u: list[int] | None = None,
) -> int:
    variables = {f"x{i}": x[i - 1] for i in range(1, len(x) + 1)}
    if u is not None:
        variables.update({f"u{i}": u[i - 1] for i in range(1, len(u) + 1)})
    return int(bool(eval(code, {"__builtins__": {}}, variables)))


class AllEdgeModel:
    """Simulator in which every edge in ``spec.all_edges`` is removable.

    Removing edge ``(j, i)`` replaces every occurrence of ``x_j`` in ``f_i`` by
    ``(x_j and not u_k)``.  This sets the source to zero, which removes both an
    activating and an inhibitory contribution.
    """

    def __init__(self, spec: NetworkSpec) -> None:
        self.spec = spec
        self.n_nodes = spec.n_nodes
        self.all_edges = edges_from_spec(spec)
        self.m_edges = len(self.all_edges)
        self._code: list[CodeType] = []

        base = compile_equations(spec)
        for target, expression in enumerate(base, start=1):
            incoming = {
                edge.source: edge.index + 1
                for edge in self.all_edges
                if edge.target == target
            }

            def replace(
                match: re.Match[str],
                incoming: dict[int, int] = incoming,
            ) -> str:
                source = int(match.group(1))
                control = incoming.get(source)
                if control is None:
                    return match.group(0)
                return f"(x{source} and not u{control})"

            transformed = re.sub(r"\bx(\d+)\b", replace, expression)
            self._code.append(compile(transformed, f"<f{target}>", "eval"))

    def next_state(self, x: list[int], u: list[int]) -> list[int]:
        """Return the successor of ``x`` under the full input vector ``u``."""
        if len(x) != self.n_nodes:
            raise ValueError(f"x must have length {self.n_nodes}")
        if len(u) != self.m_edges:
            raise ValueError(f"u must have length {self.m_edges}")
        return [_eval_expression(code, x=x, u=u) for code in self._code]

    def goal_states(self) -> list[list[int]]:
        """Goal states of the underlying network."""
        return [list(state) for state in self.spec.goal_states]

    def is_goal(self, x: list[int]) -> bool:
        """Whether ``x`` is a goal state."""
        return tuple(x) in {tuple(state) for state in self.spec.goal_states}

    def state_to_id(self, x: list[int]) -> int:
        """Encode a state as an integer (first node is most significant)."""
        value = 0
        for bit in x:
            value = (value << 1) | (1 if bit else 0)
        return value

    def id_to_state(self, value: int) -> list[int]:
        """Decode a state produced by :meth:`state_to_id`."""
        return int_to_bits(value, self.n_nodes)


class SubsetEdgeModel:
    """Expose a subset of ``AllEdgeModel`` edges as the control coordinates.

    Args:
        generic: Simulator with all edges removable.
        edge_indices: Zero-based all-edge indices that may be removed.
    """

    def __init__(
        self,
        generic: AllEdgeModel,
        edge_indices: tuple[int, ...],
    ) -> None:
        self.generic = generic
        self.edge_indices = tuple(edge_indices)
        self.n_nodes = generic.n_nodes
        self.m_edges = len(self.edge_indices)

    def next_state(self, x: list[int], u: list[int]) -> list[int]:
        """Return the successor under subset-local input ``u``."""
        if len(u) != self.m_edges:
            raise ValueError(f"u must have length {self.m_edges}")
        full = [0] * self.generic.m_edges
        for bit, edge_index in enumerate(self.edge_indices):
            if u[bit]:
                full[edge_index] = 1
        return self.generic.next_state(x, full)

    def goal_states(self) -> list[list[int]]:
        """Goal states of the underlying network."""
        return self.generic.goal_states()

    def is_goal(self, x: list[int]) -> bool:
        """Whether ``x`` is a goal state."""
        return self.generic.is_goal(x)

    def state_to_id(self, x: list[int]) -> int:
        """Encode a state as an integer."""
        return self.generic.state_to_id(x)

    def id_to_state(self, value: int) -> list[int]:
        """Decode a state produced by :meth:`state_to_id`."""
        return self.generic.id_to_state(value)


def verify_all_edge_model(base: BooleanNetworkModel, spec: NetworkSpec) -> None:
    """Check that the generic simulator reproduces the base model.

    The comparison covers all states, ``u=0``, and every input combination of
    the current removable edges.

    Raises:
        AssertionError: If any state or input differs.
    """
    generic = AllEdgeModel(spec)
    edges = generic.all_edges
    current = current_edge_indices(spec, edges)

    for state_id in range(2**spec.n_nodes):
        x = int_to_bits(state_id, spec.n_nodes)
        if base.next_state(x, [0] * base.m_edges) != generic.next_state(
            x, [0] * generic.m_edges
        ):
            raise AssertionError(f"{spec.name}: mismatch at u=0, state={x}")

    for mask in range(2 ** len(current)):
        base_u = [(mask >> bit) & 1 for bit in range(len(current))]
        generic_u = [0] * generic.m_edges
        for base_index, bit in enumerate(base_u):
            if bit:
                generic_u[current[base_index]] = 1
        for state_id in range(2**spec.n_nodes):
            x = int_to_bits(state_id, spec.n_nodes)
            expected = base.next_state(x, base_u)
            actual = generic.next_state(x, generic_u)
            if expected != actual:
                raise AssertionError(
                    f"{spec.name}: mismatch at u={base_u}, state={x}: "
                    f"{expected} != {actual}"
                )


def dependency_signs(spec: NetworkSpec) -> dict[tuple[int, int], str]:
    """Return the truth-table sign of every declared dependency."""
    equations = compile_equations(spec)
    n_nodes = spec.n_nodes
    result: dict[tuple[int, int], str] = {}
    for target, expression in enumerate(equations, start=1):
        code: CodeType = compile(expression, f"<f{target}>", "eval")
        incoming = {edge.source for edge in spec.all_edges if edge.target == target}
        for source in sorted(incoming):
            pairs: list[tuple[int, int]] = []
            others = [i for i in range(n_nodes) if i != source - 1]
            for bits in itertools.product([0, 1], repeat=n_nodes - 1):
                x = [0] * n_nodes
                for i, bit in zip(others, bits, strict=True):
                    x[i] = bit
                x[source - 1] = 0
                low = _eval_expression(code, x=x)
                x[source - 1] = 1
                high = _eval_expression(code, x=x)
                pairs.append((low, high))
            if all(low == high for low, high in pairs):
                result[(source, target)] = "none"
            elif all(low <= high for low, high in pairs):
                result[(source, target)] = "activation"
            elif all(low >= high for low, high in pairs):
                result[(source, target)] = "inhibition"
            else:
                result[(source, target)] = "mixed"
    return result


def verify_edge_signs(spec: NetworkSpec) -> None:
    """Check declared edge signs against the update functions.

    Raises:
        AssertionError: If a declared sign is redundant or mixed.
    """
    signs = dependency_signs(spec)
    for edge in spec.all_edges:
        actual = signs[(edge.source, edge.target)]
        if actual != edge.sign:
            raise AssertionError(
                f"{spec.name}: edge {edge.source}->{edge.target} declared "
                f"{edge.sign}, truth table says {actual}"
            )


def canalizing_edges(spec: NetworkSpec) -> dict[int, bool]:
    """Return edge index -> whether its source canalizes the target function.

    A source is canalizing when one of its two values forces the target output
    regardless of the other inputs.  This is the broad candidate condition
    behind Method 2.1 of Murrugarra & Dimitrova (2015), without the
    transition-specific side conditions.
    """
    equations = compile_equations(spec)
    n_nodes = spec.n_nodes
    result: dict[int, bool] = {}
    for target, expression in enumerate(equations, start=1):
        code: CodeType = compile(expression, f"<f{target}>", "eval")
        incoming = [edge for edge in spec.all_edges if edge.target == target]
        for edge in incoming:
            result[edge_index(spec, edge.source, edge.target)] = _is_canalizing(
                code, edge.source, n_nodes
            )
    return result


def edge_index(spec: NetworkSpec, source: int, target: int) -> int:
    """Return the all-edge index of ``source -> target``."""
    for index, edge in enumerate(spec.all_edges):
        if edge.source == source and edge.target == target:
            return index
    raise KeyError(f"edge {source}->{target} not found in {spec.name}")


def _is_canalizing(code: CodeType, source: int, n_nodes: int) -> bool:
    others = [i for i in range(n_nodes) if i != source - 1]
    for value in (0, 1):
        outputs: set[int] = set()
        for bits in itertools.product([0, 1], repeat=n_nodes - 1):
            x = [0] * n_nodes
            for i, bit in zip(others, bits, strict=True):
                x[i] = bit
            x[source - 1] = value
            outputs.add(_eval_expression(code, x=x))
            if len(outputs) > 1:
                break
        if len(outputs) == 1:
            return True
    return False


def enumerate_simple_cycles(
    edges: tuple[Edge, ...],
    n_nodes: int,
) -> list[tuple[int, ...]]:
    """Return every simple directed cycle once as a tuple of edge indices."""
    outgoing: dict[int, list[tuple[int, int]]] = {i: [] for i in range(1, n_nodes + 1)}
    for edge in edges:
        outgoing[edge.source].append((edge.target, edge.index))

    cycles: list[tuple[int, ...]] = []
    for start in range(1, n_nodes + 1):

        def dfs(
            node: int,
            path: tuple[int, ...],
            visited: set[int],
            start: int = start,
        ) -> None:
            for nxt, edge_index_ in outgoing[node]:
                if nxt == start:
                    cycles.append((*path, edge_index_))
                elif nxt > start and nxt not in visited:
                    dfs(nxt, (*path, edge_index_), visited | {nxt})

        dfs(start, (), {start})

    unique: dict[frozenset[int], tuple[int, ...]] = {}
    for cycle in cycles:
        unique.setdefault(frozenset(cycle), cycle)
    return list(unique.values())


def cycle_sign(cycle: tuple[int, ...], edges: tuple[Edge, ...]) -> str:
    """Return ``"positive"`` or ``"negative"`` for a cycle."""
    inhibitions = sum(1 for index in cycle if edges[index].sign == "inhibition")
    return "positive" if inhibitions % 2 == 0 else "negative"


def minimum_hitting_set(sets: list[frozenset[int]]) -> tuple[int, ...] | None:
    """Return an exact minimum hitting set by branch and bound."""
    remaining_sets = [item for item in sets if item]
    if not remaining_sets:
        return ()
    remaining_sets.sort(key=len)
    best: tuple[int, ...] | None = None

    def disjoint_lower_bound(remaining: list[frozenset[int]]) -> int:
        used: set[int] = set()
        count = 0
        for item in remaining:
            if not (item & used):
                used |= item
                count += 1
        return count

    def search(chosen: tuple[int, ...], remaining: list[frozenset[int]]) -> None:
        nonlocal best
        if not remaining:
            if best is None or len(chosen) < len(best):
                best = chosen
            return
        if best is not None and len(chosen) + disjoint_lower_bound(remaining) >= len(
            best
        ):
            return
        target = min(remaining, key=len)
        for element in sorted(target):
            search(
                (*chosen, element),
                [item for item in remaining if element not in item],
            )

    search((), remaining_sets)
    return best


def _transition(
    model: SubsetEdgeModel,
    config: WindowConfig,
    x: BinaryState,
    flags: FlagState,
    action: int,
) -> tuple[BinaryState, FlagState]:
    u = tuple(int_to_bits(action, model.m_edges))
    next_x = tuple(model.next_state(list(x), list(u)))
    if config.constraint_type == "monotone":
        next_flags = tuple(int(flags[i] or u[i]) for i in range(model.m_edges))
    else:
        next_flags = tuple(
            config.recovery_tau if u[i] else max(0, flags[i] - 1)
            for i in range(model.m_edges)
        )
    return next_x, next_flags


def window_success_count(
    model: SubsetEdgeModel,
    config: WindowConfig,
) -> WindowResult:
    """Count initial states from which a goal suffix is reachable.

    Success means: there is a trajectory of length ``max_steps`` and a
    ``k <= reach_horizon`` such that every state at ``t = k, ..., max_steps``
    is a goal state.  Input constraints are enforced strictly.
    """
    goal_set = {tuple(state) for state in model.goal_states()}
    zero_flags: FlagState = tuple(0 for _ in range(model.m_edges))

    @cache
    def allowed_actions(flags: FlagState) -> tuple[int, ...]:
        actions: list[int] = []
        for action in range(2**model.m_edges):
            u = int_to_bits(action, model.m_edges)
            if config.constraint_type == "monotone":
                valid = all(
                    not (flags[i] == 1 and u[i] == 0) for i in range(model.m_edges)
                )
            else:
                valid = all(
                    not (flags[i] > 0 and u[i] == 1) for i in range(model.m_edges)
                )
            if valid:
                actions.append(action)
        return tuple(actions)

    @cache
    def can_stay(t: int, x: BinaryState, flags: FlagState) -> bool:
        if x not in goal_set:
            return False
        if t == config.max_steps:
            return True
        for action in allowed_actions(flags):
            next_x, next_flags = _transition(model, config, x, flags, action)
            if can_stay(t + 1, next_x, next_flags):
                return True
        return False

    @cache
    def can_win(t: int, x: BinaryState, flags: FlagState) -> bool:
        if can_stay(t, x, flags):
            return True
        if t >= config.reach_horizon:
            return False
        for action in allowed_actions(flags):
            next_x, next_flags = _transition(model, config, x, flags, action)
            if can_win(t + 1, next_x, next_flags):
                return True
        return False

    success = 0
    total = 2**model.n_nodes
    for state_id in range(total):
        initial = tuple(model.id_to_state(state_id))
        if can_win(0, initial, zero_flags):
            success += 1
    return WindowResult(success_count=success, total_states=total)


def single_edge_results(
    model: AllEdgeModel,
    config: WindowConfig,
) -> dict[int, WindowResult]:
    """Evaluate no control and each single-edge control set."""
    baseline = window_success_count(SubsetEdgeModel(model, ()), config)
    results: dict[int, WindowResult] = {}
    for edge in model.all_edges:
        result = window_success_count(
            SubsetEdgeModel(model, (edge.index,)),
            config,
        )
        results[edge.index] = WindowResult(
            success_count=result.success_count - baseline.success_count,
            total_states=result.total_states,
        )
    return results


def top_marginal_set(
    model: AllEdgeModel,
    config: WindowConfig,
    size: int,
) -> tuple[int, ...]:
    """Return the ``size`` edges with the largest single-edge improvement."""
    gains = single_edge_results(model, config)
    ranked = sorted(gains, key=lambda index: (-gains[index].success_count, index))
    return tuple(ranked[:size])


def greedy_select(
    model: AllEdgeModel,
    config: WindowConfig,
    target_size: int,
) -> tuple[tuple[int, ...], WindowResult]:
    """Forward greedy selection maximising window success.

    Returns:
        The selected edge indices and their window result.  Selection stops
        early when every initial state succeeds.
    """
    selected: tuple[int, ...] = ()
    best = window_success_count(SubsetEdgeModel(model, selected), config)
    remaining = {edge.index for edge in model.all_edges}
    while len(selected) < target_size and remaining:
        best_edge: int | None = None
        best_result: WindowResult | None = None
        for edge_index_ in sorted(remaining):
            candidate = (*selected, edge_index_)
            result = window_success_count(SubsetEdgeModel(model, candidate), config)
            if best_result is None or result.success_count > best_result.success_count:
                best_edge = edge_index_
                best_result = result
        if best_edge is None or best_result is None:
            break
        selected = (*selected, best_edge)
        remaining.remove(best_edge)
        best = best_result
        if best.success_count == best.total_states:
            break
    return selected, best
