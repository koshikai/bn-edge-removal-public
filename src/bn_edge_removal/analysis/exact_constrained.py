"""Exact ``first_entry`` analysis under a choice of input constraint.

Generalises ``exact_monotone.monotone_dp`` to three input constraints on the
removable set ``E`` (``|E| = m``), with the same flag semantics as
``control.stl_monotone`` / ``control.stl_recovery`` (zemi2, Table 1):

* ``"none"``: any subset of ``E`` may be removed at every step;
* ``"monotone"``: an edge removed once stays removed (flag ``h_k = 1``);
* ``"recovery"``: after removing edge ``k`` at ``t``, it may not be removed at
  ``t + 1 .. t + tau`` (flag ``h_k`` = remaining recovery steps).

Two costs are available: ``"per_step"``, ``J = sum_{t < T} |u_t|`` (an edge
held removed is paid at every step), and ``"onset"``, the number of removals that
start at ``t`` (edges in ``u_t`` that were not removed at ``t - 1``), which counts
an irreversible edit once.  All ``2**n`` states are handled at once with numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np

from bn_edge_removal.analysis.exact_monotone import INF, MonotoneResult, VectorNetwork

type Constraint = Literal["none", "monotone", "recovery"]
type CostKind = Literal["per_step", "onset"]
CONSTRAINTS: tuple[Constraint, ...] = ("none", "monotone", "recovery")


@dataclass(frozen=True)
class InputSpec:
    """Input constraint of ``constrained_dp``.

    Args:
        kind: ``"none"``, ``"monotone"`` or ``"recovery"``.
        tau: Recovery length (used only for ``"recovery"``).
        cost: ``"per_step"`` or ``"onset"`` (see the module docstring).
    """

    kind: Constraint
    tau: int = 2
    cost: CostKind = "per_step"

    @property
    def label(self) -> str:
        """Short name such as ``recovery_tau2`` or ``monotone_onset``."""
        base = f"recovery_tau{self.tau}" if self.kind == "recovery" else self.kind
        return base if self.cost == "per_step" else f"{base}_onset"


def flag_transitions(
    m: int, constraint: Constraint, tau: int = 2, *, track_previous: bool = False
) -> tuple[
    list[tuple[int, ...]], dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]]
]:
    """Flag states and, for each flag, the allowed inputs with the next flag.

    Inputs are bit masks over the ``m`` removable edges (bit ``k`` = edge ``k``).
    With ``track_previous``, the ``"none"`` flag is the previous input (needed for
    the ``"onset"`` cost); the other constraints already determine it.

    Returns:
        The flag states (the first one is the initial flag) and a mapping from a
        flag to ``[(u, next_flag), ...]``.
    """
    if constraint == "none" and not track_previous:
        flags: list[tuple[int, ...]] = [()]
        return flags, {(): [(u, ()) for u in range(2**m)]}
    if constraint == "none":
        flags = [tuple((h >> k) & 1 for k in range(m)) for h in range(2**m)]
        return flags, {
            flag: [(u, tuple((u >> k) & 1 for k in range(m))) for u in range(2**m)]
            for flag in flags
        }
    if constraint == "monotone":
        flags = [tuple((h >> k) & 1 for k in range(m)) for h in range(2**m)]
    elif constraint == "recovery":
        if tau < 1:
            raise ValueError("tau must be >= 1")
        flags = [tuple(f) for f in product(range(tau + 1), repeat=m)]
    else:
        raise ValueError(f"unknown constraint {constraint!r}")
    moves: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = {}
    for flag in flags:
        allowed: list[tuple[int, tuple[int, ...]]] = []
        for u in range(2**m):
            bits = [(u >> k) & 1 for k in range(m)]
            if constraint == "monotone":
                if any(flag[k] and not bits[k] for k in range(m)):
                    continue
                nxt = tuple(int(flag[k] or bits[k]) for k in range(m))
            else:
                if any(flag[k] > 0 and bits[k] for k in range(m)):
                    continue
                nxt = tuple(tau if bits[k] else max(0, flag[k] - 1) for k in range(m))
            allowed.append((u, nxt))
        moves[flag] = allowed
    return flags, moves


def constrained_dp(
    net: VectorNetwork,
    edge_set: tuple[int, ...],
    goal: np.ndarray,
    horizon: tuple[int, int],
    spec: InputSpec,
) -> MonotoneResult:
    """Exact ``first_entry`` feasibility and minimum cost under ``constraint``.

    Args:
        net: Vectorised network.
        edge_set: Zero-based ``all_edges`` indices that may be removed.
        goal: Boolean goal mask.
        horizon: ``(H, T)``.
        spec: Input constraint.
    """
    reach_horizon, max_steps = horizon
    m = len(edge_set)
    flags, moves = flag_transitions(
        m, spec.kind, spec.tau, track_previous=spec.cost == "onset"
    )
    index = {flag: i for i, flag in enumerate(flags)}

    def previous(flag: tuple[int, ...]) -> int:
        """Mask of the edges removed at the previous step."""
        if spec.kind == "recovery":
            return sum(1 << k for k, h in enumerate(flag) if h == spec.tau)
        return sum(1 << k for k, h in enumerate(flag) if h)

    def step_cost(flag: tuple[int, ...], u: int) -> int:
        if spec.cost == "per_step":
            return bin(u).count("1")
        return bin(u & ~previous(flag)).count("1")

    succ = [
        net.successors(frozenset(edge_set[k] for k in range(m) if (u >> k) & 1))
        for u in range(2**m)
    ]
    n_states = net.n_states
    # value[f][s]: s = 0 pending (no goal visit yet), s = 1 holding.
    value = [
        [
            np.full(n_states, INF, dtype=np.int32),
            np.where(goal, 0, INF).astype(np.int32),
        ]
        for _ in flags
    ]
    for t in range(max_steps - 1, -1, -1):
        new = [
            [np.full(n_states, INF, dtype=np.int32) for _ in range(2)] for _ in flags
        ]
        for f, flag in enumerate(flags):
            for u, nxt_flag in moves[flag]:
                g = index[nxt_flag]
                c = step_cost(flag, u)
                nxt = succ[u]
                in_goal = goal[nxt]
                hold_next = value[g][1][nxt]
                cand_hold = np.where(in_goal, hold_next, INF) + c
                np.minimum(new[f][1], cand_hold, out=new[f][1])
                if t + 1 < reach_horizon:
                    cand_pend = np.where(in_goal, hold_next, value[g][0][nxt]) + c
                else:
                    cand_pend = np.where(in_goal, hold_next, INF) + c
                np.minimum(new[f][0], cand_pend, out=new[f][0])
        for row in new:
            for arr in row:
                np.minimum(arr, INF, out=arr)
        value = new
    start_flag = index[flags[0]]
    if reach_horizon == 0:
        start = np.where(goal, value[start_flag][1], INF)
    else:
        start = np.where(goal, value[start_flag][1], value[start_flag][0])
    feasible = start < INF
    return MonotoneResult(feasible=feasible, min_cost=np.where(feasible, start, INF))
