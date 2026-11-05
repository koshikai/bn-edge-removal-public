"""Evaluation utilities for trained policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from bn_edge_removal.control.action_selection import (
    select_greedy_action_with_sparse_tiebreak,
)
from bn_edge_removal.control.env import EdgeRemovalEnv
from bn_edge_removal.learning.q_learning import SparseQTable
from bn_edge_removal.models.encoding import int_to_bits

NO_GOAL_REACHED_TIME = -1.0
"""Value of ``avg_first_goal_time`` when no initial state ever reaches a goal."""


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of one greedy evaluation sweep over initial states.

    ``metrics`` keys (one episode of ``max_steps`` steps per initial state):

    - ``success_rate``: fraction of initial states whose trajectory satisfies
      ``trajectory_satisfies_goal`` *and* never violates the input constraint.
    - ``violation_rate``: fraction of initial states with at least one violation.
    - ``num_initial_states``: number of evaluated initial states.
    - ``num_goal_reached``: number of initial states whose trajectory visits a goal
      state at some ``t`` in ``0..max_steps`` (the denominator of the two averages
      below; success is not required).
    - ``avg_first_goal_step``: mean over those states of the first ``t >= 1`` with
      ``x(t)`` in the goal set, i.e. the number of transitions until a goal state is
      *observed after a step*.  ``t = 0`` is not counted, so an initial state that
      is already a goal counts as 1 when it stays in the goal set (and as its
      return time otherwise).  This is the definition behind every recorded value
      and is kept unchanged.  Its minimum is 1, so the value ``0.0`` only ever means
      "no initial state reached a goal" (``num_goal_reached == 0``).
    - ``avg_first_goal_time``: mean over the same states of the hitting time, the
      first ``t >= 0`` with ``x(t)`` in the goal set (0 for an initial goal state).
      ``NO_GOAL_REACHED_TIME`` (-1.0) when ``num_goal_reached == 0``, because 0.0
      is a legitimate hitting time.
    """

    metrics: dict[str, float]
    trajectories: list[dict[str, Any]]
    actions: list[dict[str, Any]]


def first_goal_step_and_time(goal_flags: list[bool]) -> tuple[int | None, int | None]:
    """Return ``(first t >= 1, first t >= 0)`` at which a trajectory is in the goal.

    Args:
        goal_flags: ``goal_flags[t]`` tells whether ``x(t)`` is a goal state.

    Returns:
        The legacy first-goal step and the hitting time; ``None`` when not reached.
    """
    step = next((t for t, flag in enumerate(goal_flags) if flag and t >= 1), None)
    time = next((t for t, flag in enumerate(goal_flags) if flag), None)
    return step, time


@dataclass
class MetricsAccumulator:
    """Accumulates per-episode outcomes into the ``EvaluationResult`` metrics."""

    total: int = 0
    success_count: int = 0
    violation_count: int = 0
    goal_reached_count: int = 0
    first_goal_step_count: int = 0
    total_first_goal_step: float = 0.0
    total_first_goal_time: float = 0.0

    def add_episode(
        self, goal_flags: list[bool], *, success: bool, violation: bool
    ) -> None:
        self.total += 1
        if success and not violation:
            self.success_count += 1
        if violation:
            self.violation_count += 1
        step, time = first_goal_step_and_time(goal_flags)
        if time is not None:
            self.goal_reached_count += 1
            self.total_first_goal_time += float(time)
        if step is not None:
            self.first_goal_step_count += 1
            self.total_first_goal_step += float(step)

    def metrics(self) -> dict[str, float]:
        total = self.total
        return {
            "success_rate": self.success_count / total if total else 0.0,
            "violation_rate": self.violation_count / total if total else 0.0,
            "avg_first_goal_step": (
                self.total_first_goal_step / self.first_goal_step_count
                if self.first_goal_step_count
                else 0.0
            ),
            "num_initial_states": float(total),
            "num_goal_reached": float(self.goal_reached_count),
            "avg_first_goal_time": (
                self.total_first_goal_time / self.goal_reached_count
                if self.goal_reached_count
                else NO_GOAL_REACHED_TIME
            ),
        }


def _evaluate(
    env: EdgeRemovalEnv,
    q_values_of: Callable[[int], np.ndarray],
    initial_states: list[list[int]] | None,
) -> EvaluationResult:
    """Roll out the greedy policy from every initial state.

    Args:
        q_values_of: Q-row of a state id (zeros for states never visited).
    """
    model = env.model
    if initial_states is None:
        initial_states = [
            int_to_bits(i, model.n_nodes) for i in range(2**model.n_nodes)
        ]

    trajectories: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    tally = MetricsAccumulator()

    for init_state in initial_states:
        state_id = env.reset(initial_state=init_state)
        init_id = model.state_to_id(init_state)

        for step in range(env.horizon.max_steps):
            allowed = env.allowed_actions()
            q_values = q_values_of(state_id)
            action = select_greedy_action_with_sparse_tiebreak(q_values, allowed)

            next_state_id, _reward, done, _info = env.step(action)
            actions.append({"init_state": init_id, "t": step, "action": action})

            state_id = next_state_id
            if done:
                break

        for t, state in enumerate(env.trajectory):
            trajectories.append(
                {
                    "init_state": init_id,
                    "t": t,
                    "state_id": model.state_to_id(state),
                }
            )

        tally.add_episode(
            [model.is_goal(state) for state in env.trajectory],
            success=env.success,
            violation=env.violation_any,
        )

    return EvaluationResult(
        metrics=tally.metrics(), trajectories=trajectories, actions=actions
    )


def evaluate_policy(
    env: EdgeRemovalEnv,
    q_table: np.ndarray,
    initial_states: list[list[int]] | None = None,
) -> EvaluationResult:
    """Evaluate the greedy policy of a dense Q-table."""
    return _evaluate(env, lambda state_id: q_table[state_id], initial_states)


def evaluate_policy_sparse(
    env: EdgeRemovalEnv,
    q_table: SparseQTable,
    initial_states: list[list[int]] | None = None,
) -> EvaluationResult:
    """Evaluate the greedy policy of a sparse Q-table."""
    zero_q = np.zeros(env.num_actions, dtype=float)
    return _evaluate(
        env, lambda state_id: q_table.get(state_id, zero_q), initial_states
    )
