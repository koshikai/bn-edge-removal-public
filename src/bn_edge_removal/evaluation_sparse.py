"""Evaluation utilities for sparse Q-tables."""

from __future__ import annotations

from typing import Any

import numpy as np

from bn_edge_removal.action_selection import (
    select_greedy_action_with_sparse_tiebreak,
)
from bn_edge_removal.encoding import int_to_bits
from bn_edge_removal.env import EdgeRemovalEnv
from bn_edge_removal.evaluation import EvaluationResult
from bn_edge_removal.q_learning_sparse import SparseQTable


def evaluate_policy_sparse(
    env: EdgeRemovalEnv,
    q_table: SparseQTable,
    initial_states: list[list[int]] | None = None,
) -> EvaluationResult:
    model = env.model
    if initial_states is None:
        initial_states = [
            int_to_bits(i, model.n_nodes) for i in range(2**model.n_nodes)
        ]

    trajectories: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    success_count = 0
    violation_count = 0
    total_first_goal = 0.0
    first_goal_count = 0

    zero_q = np.zeros(env.num_actions, dtype=float)

    for init_state in initial_states:
        state_id = env.reset(initial_state=init_state)
        init_id = model.state_to_id(init_state)
        goal_first_step: int | None = None

        for step in range(env.horizon.max_steps):
            allowed = env.allowed_actions()
            preferred = env.goal_preserving_actions(allowed)
            q_values = q_table.get(state_id, zero_q)
            action = select_greedy_action_with_sparse_tiebreak(q_values, preferred)

            next_state_id, _reward, done, _info = env.step(action)
            actions.append({"init_state": init_id, "t": step, "action": action})

            if goal_first_step is None and model.is_goal(env.trajectory[-1]):
                goal_first_step = step + 1

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

        if env.success and not env.violation_any:
            success_count += 1
        if env.violation_any:
            violation_count += 1
        if goal_first_step is not None:
            total_first_goal += float(goal_first_step)
            first_goal_count += 1

    total = len(initial_states)
    success_rate = success_count / total if total else 0.0
    violation_rate = violation_count / total if total else 0.0
    avg_first_goal = total_first_goal / first_goal_count if first_goal_count else 0.0

    metrics = {
        "success_rate": success_rate,
        "violation_rate": violation_rate,
        "avg_first_goal_step": avg_first_goal,
        "num_initial_states": float(total),
    }
    return EvaluationResult(metrics=metrics, trajectories=trajectories, actions=actions)
