"""Tabular Q-learning for edge removal control.

The Q-table is either dense (``np.ndarray`` of shape ``(num_states, num_actions)``)
or sparse (``dict`` from visited state id to its row, for large augmented state
spaces).  Both variants run the same training loop and consume the random number
generator identically; an unvisited sparse row behaves as a row of zeros.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from bn_edge_removal.control.action_selection import (
    select_greedy_action_with_sparse_tiebreak,
)
from bn_edge_removal.control.env import EdgeRemovalEnv

type SparseQTable = dict[int, np.ndarray]


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
    # Backup horizon. ``1`` is the standard one-step update; larger values use
    # the n-step return (and the full-episode return when ``n_step`` is at least
    # ``env.horizon.max_steps``).
    n_step: int = 1

    def __post_init__(self) -> None:
        if self.n_step < 1:
            raise ValueError("n_step must be >= 1")


def _epsilon_greedy_action(
    q_values: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    allowed: list[int],
) -> int:
    if rng.random() < epsilon:
        return int(rng.choice(allowed))
    return select_greedy_action_with_sparse_tiebreak(q_values, allowed)


def _train(  # noqa: PLR0913  (shared loop of the two public trainers)
    env: EdgeRemovalEnv,
    config: QLearningConfig,
    rng: np.random.Generator,
    *,
    initial_states: list[list[int]] | None,
    initial_state_strategy: str,
    row: Callable[[int], np.ndarray],
    peek: Callable[[int], np.ndarray | None],
) -> list[dict[str, float]]:
    """Run the Q-learning loop, updating rows in place.

    Args:
        row: Writable Q-row of a state (created on demand for sparse tables).
        peek: Q-row of a state, or ``None`` when it has never been visited.

    Returns:
        Per-episode history.
    """
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
            q_row = row(state_id)
            allowed = env.allowed_actions()
            action = _epsilon_greedy_action(q_row, epsilon, rng, allowed)
            next_state_id, reward, done, _info = env.step(action)
            episode_reward += reward

            # Terminal transitions do not bootstrap: the episode ends at max_steps.
            next_row = None if done else peek(next_state_id)
            if next_row is None:
                next_max = 0.0
            else:
                next_allowed = env.allowed_actions()
                next_max = float(np.max(next_row[next_allowed]))
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
    return history


def _train_nstep(  # noqa: PLR0912, PLR0913  (n-step loop of the public trainers)
    env: EdgeRemovalEnv,
    config: QLearningConfig,
    rng: np.random.Generator,
    *,
    initial_states: list[list[int]] | None,
    initial_state_strategy: str,
    row: Callable[[int], np.ndarray],
    peek: Callable[[int], np.ndarray | None],
) -> list[dict[str, float]]:
    """Run n-step Q-learning, updating rows in place after each episode.

    Bootstrapping uses the Q-table as it stood at the start of the episode.
    The episode's targets are then applied after the episode.  If a
    state-action pair repeats within an episode, its updates are applied in
    episode order.
    """
    history: list[dict[str, float]] = []

    if initial_states is not None and len(initial_states) == 0:
        raise ValueError("initial_states must be non-empty when provided")

    n = config.n_step
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

        epsilon = config.epsilon.value(episode, config.episodes)
        states = [state_id]
        allowed = [env.allowed_actions()]
        actions: list[int] = []
        rewards: list[float] = []

        for _ in range(env.horizon.max_steps):
            q_row = row(state_id)
            action = _epsilon_greedy_action(q_row, epsilon, rng, allowed[-1])
            next_state_id, reward, done, _info = env.step(action)
            actions.append(action)
            rewards.append(reward)
            state_id = next_state_id
            states.append(state_id)
            allowed.append(env.allowed_actions())
            if done:
                break

        targets: list[tuple[int, int, float]] = []
        n_steps = len(actions)
        for t in range(n_steps):
            n_return = 0.0
            for k in range(t, min(t + n, n_steps)):
                n_return += (config.gamma ** (k - t)) * rewards[k]
            if t + n < n_steps:
                bootstrap_row = peek(states[t + n])
                if bootstrap_row is not None:
                    bootstrap_allowed = allowed[t + n]
                    if bootstrap_allowed:
                        n_return += (config.gamma**n) * float(
                            np.max(bootstrap_row[bootstrap_allowed])
                        )
            targets.append((states[t], actions[t], n_return))

        for state_t, action_t, n_return in targets:
            row_t = row(state_t)
            row_t[action_t] += config.alpha * (n_return - row_t[action_t])

        history.append(
            {
                "episode": float(episode),
                "reward": float(sum(rewards)),
                "success": 1.0 if env.success else 0.0,
                "violation": 1.0 if env.violation_any else 0.0,
                "epsilon": epsilon,
            }
        )
    return history


def train_q_learning(
    env: EdgeRemovalEnv,
    config: QLearningConfig,
    rng: np.random.Generator | None = None,
    initial_states: list[list[int]] | None = None,
    initial_state_strategy: str = "random",
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Train with a dense Q-table of shape ``(num_states, num_actions)``."""
    rng = rng or np.random.default_rng()
    q_table = np.zeros((env.num_states, env.num_actions), dtype=float)

    def row(state_id: int) -> np.ndarray:
        return q_table[state_id]

    trainer = _train if config.n_step == 1 else _train_nstep
    history = trainer(
        env,
        config,
        rng,
        initial_states=initial_states,
        initial_state_strategy=initial_state_strategy,
        row=row,
        peek=row,
    )
    return q_table, history


def train_q_learning_sparse(
    env: EdgeRemovalEnv,
    config: QLearningConfig,
    rng: np.random.Generator | None = None,
    initial_states: list[list[int]] | None = None,
    initial_state_strategy: str = "random",
) -> tuple[SparseQTable, list[dict[str, float]]]:
    """Train with a sparse Q-table holding rows of visited states only."""
    rng = rng or np.random.default_rng()
    q_table: SparseQTable = {}

    def row(state_id: int) -> np.ndarray:
        q_row = q_table.get(state_id)
        if q_row is None:
            q_row = np.zeros(env.num_actions, dtype=float)
            q_table[state_id] = q_row
        return q_row

    trainer = _train if config.n_step == 1 else _train_nstep
    history = trainer(
        env,
        config,
        rng,
        initial_states=initial_states,
        initial_state_strategy=initial_state_strategy,
        row=row,
        peek=q_table.get,
    )
    return q_table, history
