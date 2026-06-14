"""Sparse tabular Q-learning for large augmented state spaces."""

from __future__ import annotations

import numpy as np

from bn_edge_removal.action_selection import (
    select_greedy_action_with_sparse_tiebreak,
)
from bn_edge_removal.env import EdgeRemovalEnv
from bn_edge_removal.q_learning import (
    EpsilonSchedule,
    QLearningConfig,
)

type SparseQTable = dict[int, np.ndarray]


def _epsilon_greedy_action(
    q_values: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    allowed: list[int],
) -> int:
    if rng.random() < epsilon:
        return int(rng.choice(allowed))
    return select_greedy_action_with_sparse_tiebreak(q_values, allowed)


def _get_or_create_row(
    q_table: SparseQTable, state_id: int, num_actions: int
) -> np.ndarray:
    row = q_table.get(state_id)
    if row is None:
        row = np.zeros(num_actions, dtype=float)
        q_table[state_id] = row
    return row


def train_q_learning_sparse(
    env: EdgeRemovalEnv,
    config: QLearningConfig,
    rng: np.random.Generator | None = None,
    initial_states: list[list[int]] | None = None,
    initial_state_strategy: str = "random",
) -> tuple[SparseQTable, list[dict[str, float]]]:
    rng = rng or np.random.default_rng()
    q_table: SparseQTable = {}
    history: list[dict[str, float]] = []

    if initial_states is not None and len(initial_states) == 0:
        raise ValueError("initial_states must be non-empty when provided")

    for episode in range(config.episodes):
        if initial_states is None:
            state_id = env.reset()
        else:
            if initial_state_strategy == "cycle":
                init_state = initial_states[episode % len(initial_states)]
            elif initial_state_strategy == "random":
                init_state = initial_states[int(rng.integers(0, len(initial_states)))]
            else:
                raise ValueError("initial_state_strategy must be 'random' or 'cycle'")
            state_id = env.reset(initial_state=init_state)

        episode_reward = 0.0
        epsilon = config.epsilon.value(episode, config.episodes)

        for _ in range(env.horizon.max_steps):
            q_row = _get_or_create_row(q_table, state_id, env.num_actions)
            allowed = env.allowed_actions()
            preferred = env.goal_preserving_actions(allowed)
            action = _epsilon_greedy_action(q_row, epsilon, rng, preferred)

            next_state_id, reward, done, _info = env.step(action)
            episode_reward += reward

            next_allowed = env.allowed_actions()
            next_preferred = env.goal_preserving_actions(next_allowed)
            next_row = q_table.get(next_state_id)
            next_max = (
                0.0 if next_row is None else float(np.max(next_row[next_preferred]))
            )

            td_target = reward + config.gamma * next_max
            q_row[action] += config.alpha * (td_target - q_row[action])

            state_id = next_state_id
            if done:
                break

        history.append(
            {
                "episode": float(episode),
                "reward": episode_reward,
                "success": 1.0 if env.success else 0.0,
                "violation": 1.0 if env.violation_any else 0.0,
                "epsilon": epsilon,
            }
        )

    return q_table, history


__all__ = [
    "EpsilonSchedule",
    "QLearningConfig",
    "SparseQTable",
    "train_q_learning_sparse",
]
