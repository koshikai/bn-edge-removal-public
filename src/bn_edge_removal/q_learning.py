"""Tabular Q-learning for edge removal control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bn_edge_removal.action_selection import (
    select_greedy_action_with_sparse_tiebreak,
)
from bn_edge_removal.env import EdgeRemovalEnv


@dataclass(frozen=True)
class EpsilonSchedule:
    start: float = 1.0
    end: float = 0.05

    def value(self, episode: int, total_episodes: int) -> float:
        if total_episodes <= 1:
            return self.end
        ratio = min(max(episode / (total_episodes - 1), 0.0), 1.0)
        return self.start + ratio * (self.end - self.start)


@dataclass(frozen=True)
class QLearningConfig:
    alpha: float = 0.5
    gamma: float = 0.99
    episodes: int = 15000
    epsilon: EpsilonSchedule = EpsilonSchedule()


def _epsilon_greedy_action(
    q_values: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    allowed: list[int],
) -> int:
    if rng.random() < epsilon:
        return int(rng.choice(allowed))
    return select_greedy_action_with_sparse_tiebreak(q_values, allowed)


def train_q_learning(
    env: EdgeRemovalEnv,
    config: QLearningConfig,
    rng: np.random.Generator | None = None,
    initial_states: list[list[int]] | None = None,
    initial_state_strategy: str = "random",
) -> tuple[np.ndarray, list[dict[str, float]]]:
    rng = rng or np.random.default_rng()
    q_table = np.zeros((env.num_states, env.num_actions), dtype=float)
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
            allowed = env.allowed_actions()
            preferred = env.goal_preserving_actions(allowed)
            action = _epsilon_greedy_action(q_table[state_id], epsilon, rng, preferred)
            next_state_id, reward, done, _info = env.step(action)
            episode_reward += reward

            next_allowed = env.allowed_actions()
            next_preferred = env.goal_preserving_actions(next_allowed)
            next_max = float(np.max(q_table[next_state_id][next_preferred]))
            td_target = reward + config.gamma * next_max
            q_table[state_id, action] += config.alpha * (
                td_target - q_table[state_id, action]
            )

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
