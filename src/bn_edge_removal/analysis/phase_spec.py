"""An input specification applied only before or only after the first entry.

``automaton_dp`` applies the specification to the whole episode.  Here it is
applied to one phase of a ``first_entry`` episode:

* ``"all"``: every step ``0 .. T - 1`` (same as ``automaton_dp``);
* ``"before"``: the steps up to and including the entry step (the step whose
  successor is the first goal state); later steps are free;
* ``"after"``: the steps from the entry step on; the automaton starts in its
  initial flag at the entry step, and earlier steps are free.  An episode that
  starts in the goal is constrained from step 0.

The entry step belongs to both phases, so ``"all"`` is the conjunction of the
two phases for a time-invariant specification such as ``monotone``.  With
``monotone``, the loss under ``"before"`` counts initial states that need a
removal to be undone before the goal is reached; the loss under ``"after"``
counts those that need a removal present at the entry to be undone later.

Only obligations inside ``0 .. T - 1`` are checked, as in ``automaton_dp``.
The automaton receives the absolute time ``t``.  Cost is ``per_step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from bn_edge_removal.analysis.exact_monotone import INF, MonotoneResult, VectorNetwork
from bn_edge_removal.analysis.input_automata import Flag, InputAutomaton, Objective

type Phase = Literal["all", "before", "after"]
type Key = Flag | None
"""A flag of the constrained phase, or ``None`` for the free phase."""


def _flag_layers(automaton: InputAutomaton, max_steps: int) -> list[list[Flag]]:
    """Flags that may occur at each ``t``: the initial flag and its successors.

    The initial flag is added at every ``t`` because the ``"after"`` phase starts
    the automaton at the entry step.
    """
    layers: list[list[Flag]] = [[automaton.initial]]
    for t in range(max_steps):
        seen: dict[Flag, None] = {automaton.initial: None}
        for flag in layers[-1]:
            for u in range(2**automaton.m):
                nxt = automaton.step(flag, u, t)
                if nxt is not None:
                    seen.setdefault(nxt, None)
        layers.append(list(seen))
    return layers


@dataclass(frozen=True)
class _Layer:
    """One backward step of ``phase_dp``."""

    automaton: InputAutomaton
    phase: Phase
    goal: np.ndarray
    reach_horizon: int
    succ: list[np.ndarray]
    blocked: np.ndarray

    def _entered(
        self, hold_next: dict[Key, np.ndarray], nxt_flag: Key, u: int, t: int
    ) -> np.ndarray:
        """Hold value after entering the goal with input ``u`` at ``t``."""
        if self.phase == "all":
            return hold_next[nxt_flag]
        if self.phase == "before":
            return hold_next[None]
        entry_flag = self.automaton.step(self.automaton.initial, u, t)
        return self.blocked if entry_flag is None else hold_next[entry_flag]

    def values(
        self,
        t: int,
        flag: Key,
        hold_next: dict[Key, np.ndarray],
        pend_next: dict[Key, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hold and pending values at ``t`` for ``flag``."""
        best_h = self.blocked.copy()
        best_p = self.blocked.copy()
        for u, nxt in enumerate(self.succ):
            nxt_flag: Key = None
            if flag is not None:
                nxt_flag = self.automaton.step(flag, u, t)
                if nxt_flag is None:
                    continue
            c = bin(u).count("1")
            in_goal = self.goal[nxt]
            cand = np.where(in_goal, hold_next[nxt_flag][nxt], INF) + c
            np.minimum(best_h, cand, out=best_h)
            entered = self._entered(hold_next, nxt_flag, u, t)[nxt]
            later = pend_next[nxt_flag][nxt] if t + 1 < self.reach_horizon else INF
            np.minimum(best_p, np.where(in_goal, entered, later) + c, out=best_p)
        np.minimum(best_h, INF, out=best_h)
        np.minimum(best_p, INF, out=best_p)
        return best_h, best_p


def phase_dp(
    net: VectorNetwork,
    edge_set: tuple[int, ...],
    objective: Objective,
    automaton: InputAutomaton,
    phase: Phase,
) -> MonotoneResult:
    """Exact ``first_entry`` feasibility and minimum ``per_step`` cost.

    Args:
        net: Vectorised network.
        edge_set: Zero-based ``all_edges`` indices; bit ``k`` of ``u`` is edge ``k``.
        objective: Goal and ``(H, T)``; cost ``per_step`` and no hold options.
        automaton: Input specification with ``m == len(edge_set)``.
        phase: ``"all"``, ``"before"`` or ``"after"``.
    """
    m = len(edge_set)
    if automaton.m != m:
        raise ValueError("automaton.m must equal len(edge_set)")
    if phase not in ("all", "before", "after"):
        raise ValueError(f"unknown phase {phase!r}")
    if objective.cost != "per_step" or objective.window() is not None:
        raise ValueError("phase_dp supports per_step cost without hold options")
    if objective.hold_frozen:
        raise ValueError("phase_dp does not support hold_frozen")
    reach_horizon, max_steps = objective.horizon
    goal = objective.goal
    blocked = np.full(net.n_states, INF, dtype=np.int32)
    layer = _Layer(
        automaton,
        phase,
        goal,
        reach_horizon,
        [
            net.successors(frozenset(edge_set[k] for k in range(m) if (u >> k) & 1))
            for u in range(2**m)
        ],
        blocked,
    )
    flags = _flag_layers(automaton, max_steps)
    terminal = np.where(goal, 0, INF).astype(np.int32)
    keys: list[Key] = [None, *flags[max_steps]]
    hold_next: dict[Key, np.ndarray] = dict.fromkeys(keys, terminal)
    pend_next: dict[Key, np.ndarray] = dict.fromkeys(keys, blocked)
    for t in range(max_steps - 1, -1, -1):
        hold_now: dict[Key, np.ndarray] = {}
        pend_now: dict[Key, np.ndarray] = {}
        for flag in [None, *flags[t]]:
            hold_now[flag], pend_now[flag] = layer.values(t, flag, hold_next, pend_next)
        hold_next, pend_next = hold_now, pend_now

    hold_key = automaton.initial if phase in ("all", "after") else None
    pend_key = automaton.initial if phase in ("all", "before") else None
    start_pend = blocked if reach_horizon == 0 else pend_next[pend_key]
    start = np.where(goal, hold_next[hold_key], start_pend)
    feasible = start < INF
    return MonotoneResult(feasible=feasible, min_cost=np.where(feasible, start, INF))


__all__ = ["Phase", "phase_dp"]
