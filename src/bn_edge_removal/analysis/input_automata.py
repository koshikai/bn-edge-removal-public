"""Temporal input specifications as finite automata, and an exact DP over them.

An input specification over ``m`` removable edges is a deterministic automaton
whose flag records the progress of the specification (the "flag state" of the
graduation thesis and zemi2).  ``InputAutomaton.step(flag, u, t)`` returns the
next flag, or ``None`` when removing the edges of the bit mask ``u`` at time ``t``
violates the specification.  The first flag is the initial one.

Builders (``p_k`` means "edge ``k`` is removed at ``t``"):

* ``unconstrained``: no restriction.
* ``monotone``: ``G(p_k -> X p_k)``; a removed edge stays removed.
* ``recovery(tau)``: ``G(p_k -> X^1..X^tau not p_k)``; a holiday of ``tau`` steps.
* ``concurrency(k)``: at most ``k`` edges removed at the same step.
* ``budget(b)``: at most ``b`` edge-steps of removal in the episode.
* ``min_duration(d)``: a removal that starts lasts at least ``d`` steps.
* ``max_duration(length)``: an edge is removed for at most ``length`` consecutive
  steps.
* ``treatment_end(t_off)``: nothing is removed at ``t >= t_off``.
* ``previous_input``: records the previous input (for the ``onset`` cost).

``product`` combines automata (all must allow the input).  As in zemi2, a finite
episode checks only the obligations that fall inside ``0 .. T - 1``: an obligation
that would continue after ``T - 1`` is not enforced.

``automaton_dp`` returns, for every initial state, whether ``first_entry`` is
feasible and the minimum cost: ``"per_step"`` (``sum_t |u_t|``) or ``"onset"``
(removals that start at ``t``).

Options on the steps after the first entry (the "hold" phase):

* ``Objective.hold_removal_steps = j``: removal is allowed only in the first
  ``j`` hold steps; ``hold_without_removal`` is ``j = 0`` (the goal must be kept
  by the network alone).  The loss against the default measures how much keeping
  the goal needs continued removal.
* ``Objective.hold_frozen``: every hold step repeats the input of the entry step
  (zero when the episode starts in the goal).  The loss measures how much keeping
  the goal needs a change (e.g. a stop) of the removal after the entry.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from bn_edge_removal.analysis.exact_monotone import INF, MonotoneResult, VectorNetwork

type Flag = Hashable
type StepFn = Callable[[Flag, int, int], Flag | None]
type CostKind = Literal["per_step", "onset"]


@dataclass(frozen=True)
class InputAutomaton:
    """A temporal input specification over ``m`` removable edges.

    Args:
        name: Short label.
        m: Number of removable edges.
        initial: Initial flag.
        step: ``step(flag, u, t)`` -> next flag, or ``None`` if ``u`` is forbidden.
    """

    name: str
    m: int
    initial: Flag
    step: StepFn


def _bits(u: int, m: int) -> list[int]:
    return [(u >> k) & 1 for k in range(m)]


def unconstrained(m: int) -> InputAutomaton:
    """No restriction on the input."""
    return InputAutomaton("none", m, 0, lambda flag, u, t: 0)


def monotone(m: int) -> InputAutomaton:
    """A removed edge stays removed (flag: mask of edges removed so far)."""

    def step(flag: Flag, u: int, t: int) -> Flag | None:
        assert isinstance(flag, int)
        return None if flag & ~u else flag | u

    return InputAutomaton("monotone", m, 0, step)


def recovery(m: int, tau: int) -> InputAutomaton:
    """After removing an edge, it may not be removed for ``tau`` steps."""
    if tau < 1:
        raise ValueError("tau must be >= 1")

    def step(flag: Flag, u: int, t: int) -> Flag | None:
        assert isinstance(flag, tuple)
        bits = _bits(u, m)
        if any(h > 0 and b for h, b in zip(flag, bits, strict=True)):
            return None
        return tuple(
            tau if b else max(0, h - 1) for h, b in zip(flag, bits, strict=True)
        )

    return InputAutomaton(f"recovery_tau{tau}", m, (0,) * m, step)


def concurrency(m: int, k: int) -> InputAutomaton:
    """At most ``k`` edges removed at the same step."""
    return InputAutomaton(
        f"concurrency_k{k}",
        m,
        0,
        lambda flag, u, t: 0 if bin(u).count("1") <= k else None,
    )


def budget(m: int, b: int) -> InputAutomaton:
    """At most ``b`` edge-steps of removal in the episode (flag: amount used)."""

    def step(flag: Flag, u: int, t: int) -> Flag | None:
        assert isinstance(flag, int)
        used = flag + bin(u).count("1")
        return used if used <= b else None

    return InputAutomaton(f"budget_b{b}", m, 0, step)


def min_duration(m: int, d: int) -> InputAutomaton:
    """A removal that starts lasts at least ``d`` consecutive steps.

    Per-edge flag: ``-1`` not removed at the previous step; ``r >= 0`` removed at
    the previous step and ``r`` more steps of removal are still required.
    """
    if d < 1:
        raise ValueError("d must be >= 1")

    def step(flag: Flag, u: int, t: int) -> Flag | None:
        assert isinstance(flag, tuple)
        nxt: list[int] = []
        for r, b in zip(flag, _bits(u, m), strict=True):
            if r > 0 and not b:
                return None
            if not b:
                nxt.append(-1)
            elif r == -1:
                nxt.append(d - 1)
            else:
                nxt.append(max(0, r - 1))
        return tuple(nxt)

    return InputAutomaton(f"min_duration_d{d}", m, (-1,) * m, step)


def max_duration(m: int, length: int) -> InputAutomaton:
    """An edge is removed for at most ``length`` consecutive steps."""
    if length < 1:
        raise ValueError("length must be >= 1")

    def step(flag: Flag, u: int, t: int) -> Flag | None:
        assert isinstance(flag, tuple)
        nxt: list[int] = []
        for c, b in zip(flag, _bits(u, m), strict=True):
            if b and c >= length:
                return None
            nxt.append(c + 1 if b else 0)
        return tuple(nxt)

    return InputAutomaton(f"max_duration_l{length}", m, (0,) * m, step)


def treatment_end(m: int, t_off: int) -> InputAutomaton:
    """No removal at ``t >= t_off`` (the goal must then hold without control)."""
    return InputAutomaton(
        f"treatment_end_t{t_off}",
        m,
        0,
        lambda flag, u, t: 0 if t < t_off or u == 0 else None,
    )


def previous_input(m: int) -> InputAutomaton:
    """Records the previous input (flag: the last mask)."""
    return InputAutomaton("previous", m, 0, lambda flag, u, t: u)


def product(*automata: InputAutomaton) -> InputAutomaton:
    """Conjunction of specifications over the same edges."""
    m = automata[0].m
    if any(a.m != m for a in automata):
        raise ValueError("automata must have the same m")

    def step(flag: Flag, u: int, t: int) -> Flag | None:
        assert isinstance(flag, tuple)
        nxt: list[Flag] = []
        for automaton, part in zip(automata, flag, strict=True):
            value = automaton.step(part, u, t)
            if value is None:
                return None
            nxt.append(value)
        return tuple(nxt)

    name = "+".join(a.name for a in automata)
    return InputAutomaton(name, m, tuple(a.initial for a in automata), step)


def reachable_flags(automaton: InputAutomaton, max_steps: int) -> list[list[Flag]]:
    """Flags reachable at each ``t = 0 .. T`` from the initial flag."""
    layers: list[list[Flag]] = [[automaton.initial]]
    for t in range(max_steps):
        seen: dict[Flag, None] = {}
        for flag in layers[-1]:
            for u in range(2**automaton.m):
                nxt = automaton.step(flag, u, t)
                if nxt is not None:
                    seen.setdefault(nxt, None)
        layers.append(list(seen))
    return layers


@dataclass(frozen=True)
class Objective:
    """Goal, horizon and cost of ``automaton_dp``.

    Args:
        goal: Boolean goal mask.
        horizon: ``(H, T)``.
        cost: ``"per_step"`` or ``"onset"``.
        hold_without_removal: If true, nothing is removed after the first entry
            (same as ``hold_removal_steps = 0``).
        hold_removal_steps: If set, removal is allowed only in this many steps
            after the first entry.
        hold_frozen: If true, the input after the first entry repeats the input
            of the entry step.
    """

    goal: np.ndarray
    horizon: tuple[int, int]
    cost: CostKind = "per_step"
    hold_without_removal: bool = False
    hold_removal_steps: int | None = None
    hold_frozen: bool = False

    def window(self) -> int | None:
        """Number of hold steps with removal allowed (``None``: no limit)."""
        if self.hold_without_removal:
            return 0
        return self.hold_removal_steps


@dataclass(frozen=True)
class _Rules:
    """Cost and hold-phase rules of ``automaton_dp``.

    The hold index ``k`` counts the hold steps already taken, capped at the
    window (always 0 without a window).
    """

    cost: CostKind
    window: int | None
    frozen: bool

    def n_hold(self) -> int:
        return 1 if self.window is None else self.window + 1

    @staticmethod
    def previous(flag: Flag) -> int:
        assert isinstance(flag, tuple)
        prev = flag[1]
        assert isinstance(prev, int)
        return prev

    def step_cost(self, flag: Flag, u: int) -> int:
        if self.cost == "per_step":
            return bin(u).count("1")
        return bin(u & ~self.previous(flag)).count("1")

    def allowed(self, k: int, u: int, flag: Flag) -> bool:
        if self.window is not None and k >= self.window and u != 0:
            return False
        return not self.frozen or u == self.previous(flag)

    def next_index(self, k: int) -> int:
        return 0 if self.window is None else min(k + 1, self.window)


def automaton_dp(
    net: VectorNetwork,
    edge_set: tuple[int, ...],
    objective: Objective,
    automaton: InputAutomaton,
) -> MonotoneResult:
    """Exact ``first_entry`` feasibility and minimum cost under ``automaton``.

    Args:
        net: Vectorised network.
        edge_set: Zero-based ``all_edges`` indices; bit ``k`` of ``u`` is edge ``k``.
        objective: Goal, ``(H, T)`` and cost.
        automaton: Input specification with ``m == len(edge_set)``.
    """
    goal, cost = objective.goal, objective.cost
    reach_horizon, max_steps = objective.horizon
    m = len(edge_set)
    if automaton.m != m:
        raise ValueError("automaton.m must equal len(edge_set)")
    window = objective.window()
    frozen = objective.hold_frozen
    if window is not None and (window < 0 or frozen):
        raise ValueError("hold window must be >= 0 and not combined with frozen")
    need_prev = cost == "onset" or frozen
    spec = product(automaton, previous_input(m)) if need_prev else automaton
    layers = reachable_flags(spec, max_steps)
    succ = [
        net.successors(frozenset(edge_set[k] for k in range(m) if (u >> k) & 1))
        for u in range(2**m)
    ]
    rules = _Rules(cost, window, frozen)
    n_hold = rules.n_hold()

    n_states = net.n_states
    terminal = np.where(goal, 0, INF).astype(np.int32)
    pending_inf = np.full(n_states, INF, dtype=np.int32)
    value: dict[Flag, tuple[np.ndarray, list[np.ndarray]]] = {
        flag: (pending_inf, [terminal] * n_hold) for flag in layers[max_steps]
    }
    for t in range(max_steps - 1, -1, -1):
        new: dict[Flag, tuple[np.ndarray, list[np.ndarray]]] = {}
        for flag in layers[t]:
            pend = np.full(n_states, INF, dtype=np.int32)
            holds = [np.full(n_states, INF, dtype=np.int32) for _ in range(n_hold)]
            for u in range(2**m):
                nxt_flag = spec.step(flag, u, t)
                if nxt_flag is None:
                    continue
                v_pend, v_holds = value[nxt_flag]
                c = rules.step_cost(flag, u)
                nxt = succ[u]
                in_goal = goal[nxt]
                for k in range(n_hold):
                    if not rules.allowed(k, u, flag):
                        continue
                    cand = np.where(in_goal, v_holds[rules.next_index(k)][nxt], INF) + c
                    np.minimum(holds[k], cand, out=holds[k])
                entered = v_holds[0][nxt]
                if t + 1 < reach_horizon:
                    cand = np.where(in_goal, entered, v_pend[nxt]) + c
                else:
                    cand = np.where(in_goal, entered, INF) + c
                np.minimum(pend, cand, out=pend)
            np.minimum(pend, INF, out=pend)
            for hold in holds:
                np.minimum(hold, INF, out=hold)
            new[flag] = (pend, holds)
        value = new
    pend0, holds0 = value[spec.initial]
    start = np.where(goal, holds0[0], INF if reach_horizon == 0 else pend0)
    feasible = start < INF
    return MonotoneResult(feasible=feasible, min_cost=np.where(feasible, start, INF))


def flag_space_size(automaton: InputAutomaton, max_steps: int) -> int:
    """Largest number of reachable flags at one time step."""
    return max(len(layer) for layer in reachable_flags(automaton, max_steps))


__all__ = [
    "InputAutomaton",
    "Objective",
    "automaton_dp",
    "budget",
    "concurrency",
    "flag_space_size",
    "max_duration",
    "min_duration",
    "monotone",
    "previous_input",
    "product",
    "reachable_flags",
    "recovery",
    "treatment_end",
    "unconstrained",
]
