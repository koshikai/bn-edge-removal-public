"""Exact, vectorised analysis of edge-removal control for networks up to ~20 nodes.

The module works on all ``2**n`` states at once with numpy:

* ``VectorNetwork`` evaluates the update rules of a network spec with any set of
  ``all_edges`` removed (the source is replaced by 0 in the target's rule, as in
  ``AllEdgeModel``);
* ``static_success`` checks the ``first_entry`` goal condition for a removal set
  held from ``t = 0``;
* ``monotone_dp`` solves, for a removable set ``E``, the finite-horizon problem
  under the monotone input constraint (a removed edge stays removed): which
  initial states can satisfy ``first_entry`` and with what minimum number of
  removed edges summed over the episode, ``J = sum_t |u_t|``;
* ``greedy_static_select`` and ``random_sets`` choose removable sets.

``first_entry`` (as in ``control.env``): the first goal visit ``t_reach`` satisfies
``t_reach <= H`` and ``x_t`` is in the goal set for every ``t`` in ``[t_reach, T]``.
States are encoded with the first node as the most significant bit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from bn_edge_removal.analysis.edge_selection import (
    Edge,
    compile_equations,
    edges_from_spec,
)
from bn_edge_removal.models.network_spec import NetworkSpec

INF = np.iinfo(np.int16).max
"""Cost value of an infeasible (state, time) pair in ``monotone_dp``."""


def _to_numpy(expression: str) -> str:
    expression = re.sub(r"\bnot\b", "~", expression)
    expression = re.sub(r"\band\b", "&", expression)
    expression = re.sub(r"\bor\b", "|", expression)
    expression = re.sub(r"\bTrue\b", "_T", expression)
    expression = re.sub(r"\bFalse\b", "_F", expression)
    return re.sub(
        r"(?<![\w.])([01])(?![\w.])",
        lambda m: "_T" if m.group(1) == "1" else "_F",
        expression,
    )


@dataclass(frozen=True)
class MonotoneResult:
    """Result of ``monotone_dp`` over all initial states.

    Args:
        feasible: Whether some monotone input sequence satisfies ``first_entry``.
        min_cost: Minimum ``J`` for feasible states, ``INF`` otherwise.
    """

    feasible: np.ndarray
    min_cost: np.ndarray


class VectorNetwork:
    """Vectorised update rules of a network spec with removable ``all_edges``.

    Args:
        spec: Network spec with ``update_equations`` and ``all_edges``.
    """

    def __init__(self, spec: NetworkSpec) -> None:
        self.spec = spec
        self.n_nodes = spec.n_nodes
        self.edges: tuple[Edge, ...] = edges_from_spec(spec)
        self._codes = [
            compile(_to_numpy(expr), f"<f{i + 1}>", "eval")
            for i, expr in enumerate(compile_equations(spec))
        ]
        ids = np.arange(2**self.n_nodes, dtype=np.int64)
        shifts = self.n_nodes - 1 - np.arange(self.n_nodes)
        self._bits = ((ids[:, None] >> shifts) & 1).astype(bool)
        self._weights = (np.int64(1) << shifts).astype(np.int64)
        self._zeros = np.zeros(2**self.n_nodes, dtype=bool)
        self._cache: dict[frozenset[int], np.ndarray] = {}

    @property
    def n_states(self) -> int:
        """Number of states ``2**n``."""
        return 2**self.n_nodes

    def successors(self, removed: frozenset[int] = frozenset()) -> np.ndarray:
        """Successor state ids of every state with the given edges removed.

        Args:
            removed: Zero-based ``all_edges`` indices to remove.

        Returns:
            ``int64`` array of length ``2**n``.
        """
        if removed in self._cache:
            return self._cache[removed]
        cut: dict[int, set[int]] = {}
        for index in removed:
            edge = self.edges[index]
            cut.setdefault(edge.target, set()).add(edge.source)
        base = {f"x{i + 1}": self._bits[:, i] for i in range(self.n_nodes)}
        base["_T"], base["_F"] = np.True_, np.False_
        nxt = np.empty_like(self._bits)
        for target, code in enumerate(self._codes, start=1):
            env = dict(base)
            for source in cut.get(target, ()):
                env[f"x{source}"] = self._zeros
            value = eval(code, {"__builtins__": {}}, env)
            nxt[:, target - 1] = np.broadcast_to(
                np.asarray(value, dtype=bool), (self.n_states,)
            )
        result = nxt.astype(np.int64) @ self._weights
        self._cache[removed] = result
        return result

    def goal_mask(self) -> np.ndarray:
        """Boolean mask of the goal states of the spec."""
        mask = np.zeros(self.n_states, dtype=bool)
        for state in self.spec.goal_states:
            mask[int(np.dot(np.asarray(state, dtype=np.int64), self._weights))] = True
        return mask

    def input_self_loops(self) -> tuple[int, ...]:
        """Edges ``i -> i`` of nodes whose only dependency is themselves."""
        incoming: dict[int, list[Edge]] = {}
        for edge in self.edges:
            incoming.setdefault(edge.target, []).append(edge)
        return tuple(
            edges[0].index
            for target, edges in incoming.items()
            if len(edges) == 1 and edges[0].source == target
        )


def static_success(
    successors: np.ndarray,
    goal: np.ndarray,
    reach_horizon: int,
    max_steps: int,
) -> np.ndarray:
    """``first_entry`` success of every initial state under a fixed map.

    Args:
        successors: Successor ids for the removal set held from ``t = 0``.
        goal: Boolean goal mask.
        reach_horizon: ``H``.
        max_steps: ``T``.
    """
    current = np.arange(len(successors), dtype=np.int64)
    reached = goal.copy()
    failed = np.zeros(len(successors), dtype=bool)
    if reach_horizon == 0:
        failed |= ~reached
    for t in range(1, max_steps + 1):
        current = successors[current]
        inside = goal[current]
        failed |= reached & ~inside
        reached |= inside
        if t >= reach_horizon:
            failed |= ~reached
    return reached & ~failed


def monotone_dp(
    net: VectorNetwork,
    edge_set: tuple[int, ...],
    goal: np.ndarray,
    reach_horizon: int,
    max_steps: int,
) -> MonotoneResult:
    """Exact ``first_entry`` feasibility and minimum cost under monotone inputs.

    The control ``u_t`` is a subset of ``edge_set``; the monotone constraint
    requires ``u_t`` to contain every edge removed before ``t``.  The cost of an
    episode is ``J = sum_{t < T} |u_t|``.

    Args:
        net: Vectorised network.
        edge_set: Zero-based ``all_edges`` indices that may be removed.
        goal: Boolean goal mask.
        reach_horizon: ``H``.
        max_steps: ``T``.
    """
    m = len(edge_set)
    masks = range(2**m)
    succ = [
        net.successors(frozenset(edge_set[k] for k in range(m) if (u >> k) & 1))
        for u in masks
    ]
    cost = [bin(u).count("1") for u in masks]
    supersets = [[u for u in masks if u & h == h] for h in masks]
    n_states = net.n_states
    # value[h][s]: s = 0 pending (no goal visit yet), s = 1 holding.
    value = [
        [
            np.full(n_states, INF, dtype=np.int32),
            np.where(goal, 0, INF).astype(np.int32),
        ]
        for _ in masks
    ]
    for t in range(max_steps - 1, -1, -1):
        new = [
            [np.full(n_states, INF, dtype=np.int32) for _ in range(2)] for _ in masks
        ]
        for h in masks:
            for u in supersets[h]:
                nxt = succ[u]
                in_goal = goal[nxt]
                hold_next = value[u][1][nxt]
                pend_next = value[u][0][nxt]
                # From holding: must stay in the goal.
                cand_hold = np.where(in_goal, hold_next, INF) + cost[u]
                np.minimum(new[h][1], cand_hold, out=new[h][1])
                # From pending: enter the goal now, or keep pending if t + 1 < H.
                if t + 1 < reach_horizon:
                    cand_pend = np.where(in_goal, hold_next, pend_next) + cost[u]
                else:
                    cand_pend = np.where(in_goal, hold_next, INF) + cost[u]
                np.minimum(new[h][0], cand_pend, out=new[h][0])
        for h in masks:
            for s in range(2):
                np.minimum(new[h][s], INF, out=new[h][s])
        value = new
    if reach_horizon == 0:
        start = np.where(goal, value[0][1], INF)
    else:
        start = np.where(goal, value[0][1], value[0][0])
    feasible = start < INF
    return MonotoneResult(feasible=feasible, min_cost=np.where(feasible, start, INF))


def monotone_feasible(
    net: VectorNetwork,
    edge_set: tuple[int, ...],
    goal: np.ndarray,
    reach_horizon: int,
    max_steps: int,
) -> np.ndarray:
    """Feasibility part of ``monotone_dp`` only (boolean, faster)."""
    m = len(edge_set)
    masks = range(2**m)
    succ = [
        net.successors(frozenset(edge_set[k] for k in range(m) if (u >> k) & 1))
        for u in masks
    ]
    supersets = [[u for u in masks if u & h == h] for h in masks]
    n_states = net.n_states
    pend = [np.zeros(n_states, dtype=bool) for _ in masks]
    hold = [goal.copy() for _ in masks]
    for t in range(max_steps - 1, -1, -1):
        new_pend = [np.zeros(n_states, dtype=bool) for _ in masks]
        new_hold = [np.zeros(n_states, dtype=bool) for _ in masks]
        for h in masks:
            for u in supersets[h]:
                nxt = succ[u]
                enter = goal[nxt] & hold[u][nxt]
                new_hold[h] |= enter
                if t + 1 < reach_horizon:
                    new_pend[h] |= enter | (~goal[nxt] & pend[u][nxt])
                else:
                    new_pend[h] |= enter
        pend, hold = new_pend, new_hold
    if reach_horizon == 0:
        return goal & hold[0]
    return np.where(goal, hold[0], pend[0])


def greedy_dp_select(
    problem: SelectionProblem,
    candidates: tuple[int, ...],
    max_size: int,
) -> tuple[tuple[int, ...], list[int]]:
    """Forward greedy selection by the exact monotone feasibility count.

    Each step adds the candidate that maximises the number of initial states in
    ``problem.domain`` from which ``first_entry`` is feasible under the monotone
    constraint with removal times chosen freely (``monotone_feasible``).  Ties
    go to the lowest index.  Selection stops at ``max_size``, when no candidate
    increases the count, or when every state in the domain is feasible.

    Returns:
        The selected edges and the feasible count after each step (the first
        entry is the empty set).
    """
    net, goal, domain = problem.net, problem.goal, problem.domain

    def count(edge_set: tuple[int, ...]) -> int:
        ok = monotone_feasible(
            net, edge_set, goal, problem.reach_horizon, problem.max_steps
        )
        return int((ok & domain).sum())

    selected: tuple[int, ...] = ()
    counts = [count(selected)]
    total = int(domain.sum())
    while len(selected) < max_size and counts[-1] < total:
        best: tuple[int, int] | None = None
        for edge in sorted(set(candidates) - set(selected)):
            value = count((*selected, edge))
            if best is None or value > best[1]:
                best = (edge, value)
        if best is None or best[1] <= counts[-1]:
            break
        selected = (*selected, best[0])
        counts.append(best[1])
    return selected, counts


@dataclass(frozen=True)
class SelectionProblem:
    """Network, goal and horizon of one edge-selection run.

    Args:
        net: Vectorised network.
        goal: Boolean goal mask.
        domain: Boolean mask of the initial states that are counted.
        reach_horizon: ``H``.
        max_steps: ``T``.
    """

    net: VectorNetwork
    goal: np.ndarray
    domain: np.ndarray
    reach_horizon: int
    max_steps: int


def greedy_static_select(
    problem: SelectionProblem,
    candidates: tuple[int, ...],
    max_size: int,
) -> tuple[tuple[int, ...], list[int]]:
    """Forward greedy selection by the static success set.

    The static success set of ``M`` is the set of initial states (inside
    ``problem.domain``) that satisfy ``first_entry`` when some subset of ``M`` is
    removed from ``t = 0`` and kept.  Each step adds the candidate with the
    largest set; ties go to the lowest index.  Selection stops at ``max_size`` or
    when no candidate enlarges the set.

    Returns:
        The selected edges and the static success count after each step
        (the first entry is the empty set).
    """
    net, goal, domain = problem.net, problem.goal, problem.domain
    reach_horizon, max_steps = problem.reach_horizon, problem.max_steps
    cache: dict[frozenset[int], np.ndarray] = {}

    def static(removed: frozenset[int]) -> np.ndarray:
        if removed not in cache:
            ok = static_success(net.successors(removed), goal, reach_horizon, max_steps)
            cache[removed] = ok & domain
        return cache[removed]

    selected: tuple[int, ...] = ()
    union = static(frozenset())
    counts = [int(union.sum())]
    while len(selected) < max_size:
        subsets = [
            frozenset(c)
            for r in range(len(selected) + 1)
            for c in combinations(selected, r)
        ]
        best: tuple[int, int, np.ndarray] | None = None
        for edge in sorted(set(candidates) - set(selected)):
            grown = union.copy()
            for sub in subsets:
                grown |= static(sub | {edge})
            count = int(grown.sum())
            if best is None or count > best[1]:
                best = (edge, count, grown)
        if best is None or best[1] <= counts[-1]:
            break
        selected = (*selected, best[0])
        union = best[2]
        counts.append(best[1])
    return selected, counts


def random_sets(
    candidates: tuple[int, ...],
    size: int,
    number: int,
    seed: int,
) -> list[tuple[int, ...]]:
    """``number`` distinct random subsets of ``candidates`` of the given size."""
    rng = np.random.default_rng(seed)
    pool = sorted(candidates)
    found: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    limit = 1
    for k in range(size):
        limit = limit * (len(pool) - k) // (k + 1)
    while len(found) < min(number, limit):
        pick = tuple(sorted(int(x) for x in rng.choice(pool, size=size, replace=False)))
        if pick not in seen:
            seen.add(pick)
            found.append(pick)
    return found


class TableEdgeModel:
    """Edge-removal model backed by precomputed successor tables.

    Exposes the ``EdgeRemovalModel`` interface of ``control.env`` for a chosen
    removable set, so that the learning code can run on networks with
    ``2**18`` to ``2**20`` states.  Control bit ``k`` removes ``edge_set[k]``.

    Args:
        net: Vectorised network.
        edge_set: Zero-based ``all_edges`` indices that are removable.
    """

    def __init__(self, net: VectorNetwork, edge_set: tuple[int, ...]) -> None:
        self._n = net.n_nodes
        self._m = len(edge_set)
        self.edge_set = tuple(edge_set)
        self.labels = tuple(net.edges[i].label for i in edge_set)
        self._tables = [
            net.successors(
                frozenset(edge_set[k] for k in range(self._m) if (u >> k) & 1)
            ).tolist()
            for u in range(2**self._m)
        ]
        self._goal_ids = frozenset(int(i) for i in np.flatnonzero(net.goal_mask()))
        self._goals = [list(state) for state in net.spec.goal_states]

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self._n

    @property
    def m_edges(self) -> int:
        """Number of removable edges."""
        return self._m

    def state_to_id(self, x: list[int]) -> int:
        """Encode a state (first node is the most significant bit)."""
        value = 0
        for bit in x:
            value = (value << 1) | (1 if bit else 0)
        return value

    def id_to_state(self, value: int) -> list[int]:
        """Decode a state id."""
        return [(value >> (self._n - 1 - i)) & 1 for i in range(self._n)]

    def next_state(self, x: list[int], u: list[int]) -> list[int]:
        """Successor of ``x`` with the edges selected by ``u`` removed."""
        if len(u) != self._m:
            raise ValueError(f"u must have length {self._m}")
        mask = sum(1 << k for k, bit in enumerate(u) if bit)
        return self.id_to_state(self._tables[mask][self.state_to_id(x)])

    def goal_states(self) -> list[list[int]]:
        """Goal states of the network."""
        return [list(state) for state in self._goals]

    def is_goal(self, x: list[int]) -> bool:
        """Whether ``x`` is a goal state."""
        return self.state_to_id(x) in self._goal_ids
