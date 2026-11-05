"""Tests for dense and sparse tabular Q-learning."""

from __future__ import annotations

import numpy as np
import pytest

from bn_edge_removal.control.env import (
    EdgeRemovalEnv,
    HorizonConfig,
    RewardConfig,
)
from bn_edge_removal.learning.evaluation import (
    evaluate_policy,
    evaluate_policy_sparse,
)
from bn_edge_removal.learning.q_learning import (
    EpsilonSchedule,
    QLearningConfig,
    train_q_learning,
    train_q_learning_sparse,
)
from bn_edge_removal.models.cortical import CorticalModel
from bn_edge_removal.models.wnt5a import Wnt5aModel


class StayModel:
    """One-node model whose state never changes."""

    n_nodes = 1
    m_edges = 1

    def next_state(self, x: list[int], u: list[int]) -> list[int]:
        _ = u
        return list(x)

    @staticmethod
    def goal_states() -> list[list[int]]:
        return [[1]]

    def is_goal(self, x: list[int]) -> bool:
        return x == [1]

    @staticmethod
    def state_to_id(x: list[int]) -> int:
        return int(x[0])

    @staticmethod
    def id_to_state(value: int) -> list[int]:
        return [int(value)]


def _one_step_env() -> EdgeRemovalEnv:
    return EdgeRemovalEnv(
        model=StayModel(),
        reward=RewardConfig(
            step=1.0, terminal=0.0, edge_cost=1.0, violation_penalty=0.0
        ),
        horizon=HorizonConfig(reach_horizon=0, max_steps=1),
    )


_GREEDY = QLearningConfig(
    alpha=1.0, gamma=1.0, episodes=5, epsilon=EpsilonSchedule(start=0.0, end=0.0)
)


def test_q_learning_config_rejects_nonpositive_n_step() -> None:
    with pytest.raises(ValueError, match="n_step"):
        QLearningConfig(n_step=0)


@pytest.mark.parametrize("train", [train_q_learning, train_q_learning_sparse])
def test_n_step_return_uses_the_full_episode_at_the_terminal(train) -> None:
    """With n_step == T, every action is scored by its full-episode return."""
    env = EdgeRemovalEnv(
        model=StayModel(),
        reward=RewardConfig(
            step=1.0, terminal=0.0, edge_cost=0.0, violation_penalty=0.0
        ),
        horizon=HorizonConfig(reach_horizon=0, max_steps=2),
        time_in_state=True,
    )
    config = QLearningConfig(
        alpha=1.0,
        gamma=0.5,
        episodes=1,
        epsilon=EpsilonSchedule(start=0.0, end=0.0),
        n_step=2,
    )
    table, history = train(
        env, config, rng=np.random.default_rng(0), initial_states=[[1]]
    )

    state_id_t0 = env.reset(initial_state=[1])
    env.step(0)
    state_id_t1 = env.state_id()
    q0 = (
        table[state_id_t0, 0]
        if isinstance(table, np.ndarray)
        else table[state_id_t0][0]
    )
    q1 = (
        table[state_id_t1, 0]
        if isinstance(table, np.ndarray)
        else table[state_id_t1][0]
    )

    # t=0: 1 + 0.5*1 = 1.5; t=1: 1 (terminal transition, no bootstrap).
    assert q0 == pytest.approx(1.5)
    assert q1 == pytest.approx(1.0)
    assert history[0]["reward"] == pytest.approx(2.0)


def test_td_update_does_not_bootstrap_from_terminal_state() -> None:
    """With max_steps=1 every transition is terminal, so Q must equal the reward.

    The no-intervention action keeps the extended state unchanged, so a TD target
    that bootstraps on the terminal transition would grow by gamma * Q each episode.
    """
    env = _one_step_env()
    q_table, history = train_q_learning(
        env, _GREEDY, rng=np.random.default_rng(0), initial_states=[[1]]
    )
    state_id = env.extended_state_id([1], [0])

    assert q_table[state_id, 0] == 1.0
    assert [h["reward"] for h in history] == [1.0] * 5


def test_sparse_td_update_does_not_bootstrap_from_terminal_state() -> None:
    env = _one_step_env()
    q_table, _history = train_q_learning_sparse(
        env, _GREEDY, rng=np.random.default_rng(0), initial_states=[[1]]
    )
    state_id = env.extended_state_id([1], [0])

    assert q_table[state_id][0] == 1.0


def test_td_update_bootstraps_on_non_terminal_transitions() -> None:
    env = EdgeRemovalEnv(
        model=StayModel(),
        reward=RewardConfig(
            step=1.0, terminal=0.0, edge_cost=1.0, violation_penalty=0.0
        ),
        horizon=HorizonConfig(reach_horizon=0, max_steps=2),
    )
    config = QLearningConfig(
        alpha=1.0, gamma=0.5, episodes=2, epsilon=EpsilonSchedule(start=0.0, end=0.0)
    )
    q_table, _history = train_q_learning(
        env, config, rng=np.random.default_rng(0), initial_states=[[1]]
    )
    state_id = env.extended_state_id([1], [0])

    # Episode 1: t=0 -> 1 + 0.5*0 = 1; t=1 (terminal) -> 1.
    # Episode 2: t=0 -> 1 + 0.5*1 = 1.5; t=1 (terminal) -> 1.
    assert q_table[state_id, 0] == 1.0
    assert _history[1]["reward"] == 2.0


def _cortical_env(seed: int) -> tuple[EdgeRemovalEnv, np.random.Generator]:
    rng = np.random.default_rng(seed)
    env = EdgeRemovalEnv(
        model=CorticalModel(),
        reward=RewardConfig(
            step=2.0, terminal=80.0, edge_cost=0.7, violation_penalty=200.0
        ),
        horizon=HorizonConfig(reach_horizon=5, max_steps=10),
        rng=rng,
    )
    return env, rng


def _wnt5a_env(seed: int) -> tuple[EdgeRemovalEnv, np.random.Generator]:
    rng = np.random.default_rng(seed)
    env = EdgeRemovalEnv(
        model=Wnt5aModel(),
        reward=RewardConfig(
            step=2.0, terminal=80.0, edge_cost=0.1, violation_penalty=200.0
        ),
        horizon=HorizonConfig(reach_horizon=7, max_steps=10),
        action_masking=True,
        constraint_type="recovery",
        recovery_tau=2,
        rng=rng,
    )
    return env, rng


_SMALL = QLearningConfig(
    alpha=0.5, gamma=0.99, episodes=300, epsilon=EpsilonSchedule(start=1.0, end=0.01)
)


@pytest.mark.parametrize("strategy", ["random", "cycle"])
def test_train_q_learning_is_deterministic_for_same_seed(strategy: str) -> None:
    model = CorticalModel()
    states = [model.id_to_state(i) for i in range(2**model.n_nodes)]
    results = []
    for _ in range(2):
        env, rng = _cortical_env(seed=11)
        q_table, history = train_q_learning(
            env, _SMALL, rng=rng, initial_states=states, initial_state_strategy=strategy
        )
        results.append((q_table, history, evaluate_policy(env, q_table)))

    (q1, h1, e1), (q2, h2, e2) = results
    assert np.array_equal(q1, q2)
    assert h1 == h2
    assert e1.metrics == e2.metrics
    assert e1.actions == e2.actions
    assert len(h1) == _SMALL.episodes
    assert set(h1[0]) == {"episode", "reward", "success", "violation", "epsilon"}


def test_train_q_learning_differs_across_seeds() -> None:
    env1, rng1 = _cortical_env(seed=1)
    env2, rng2 = _cortical_env(seed=2)
    q1, _ = train_q_learning(env1, _SMALL, rng=rng1)
    q2, _ = train_q_learning(env2, _SMALL, rng=rng2)
    assert not np.array_equal(q1, q2)


def test_train_q_learning_sparse_is_deterministic_for_same_seed() -> None:
    states = Wnt5aModel().controllable_initial_states()
    results = []
    for _ in range(2):
        env, rng = _wnt5a_env(seed=11)
        q_table, history = train_q_learning_sparse(
            env, _SMALL, rng=rng, initial_states=states, initial_state_strategy="cycle"
        )
        results.append((q_table, history, evaluate_policy_sparse(env, q_table, states)))

    (q1, h1, e1), (q2, h2, e2) = results
    assert q1.keys() == q2.keys()
    assert all(np.array_equal(q1[k], q2[k]) for k in q1)
    assert h1 == h2
    assert e1.metrics == e2.metrics
    assert e1.actions == e2.actions
    # Masked training never violates the recovery constraint.
    assert all(h["violation"] == 0.0 for h in h1)


def test_dense_and_sparse_training_agree() -> None:
    """Both implementations follow the same update rule and RNG stream."""
    model = CorticalModel()
    states = [model.id_to_state(i) for i in range(2**model.n_nodes)]
    env_d, rng_d = _cortical_env(seed=3)
    env_s, rng_s = _cortical_env(seed=3)
    dense, hist_d = train_q_learning(
        env_d, _SMALL, rng=rng_d, initial_states=states, initial_state_strategy="cycle"
    )
    sparse, hist_s = train_q_learning_sparse(
        env_s, _SMALL, rng=rng_s, initial_states=states, initial_state_strategy="cycle"
    )

    assert hist_d == hist_s
    for state_id, row in sparse.items():
        assert np.array_equal(dense[state_id], row)
    untouched = np.ones(dense.shape[0], dtype=bool)
    untouched[list(sparse)] = False
    assert not dense[untouched].any()


def test_dense_and_sparse_n_step_training_agree() -> None:
    """The n-step trainer keeps the same RNG stream and backups in both tables."""
    model = CorticalModel()
    states = [model.id_to_state(i) for i in range(2**model.n_nodes)]
    config = QLearningConfig(
        alpha=_SMALL.alpha,
        gamma=_SMALL.gamma,
        episodes=_SMALL.episodes,
        epsilon=_SMALL.epsilon,
        n_step=3,
    )
    env_d, rng_d = _cortical_env(seed=3)
    env_s, rng_s = _cortical_env(seed=3)
    dense, hist_d = train_q_learning(
        env_d, config, rng=rng_d, initial_states=states, initial_state_strategy="cycle"
    )
    sparse, hist_s = train_q_learning_sparse(
        env_s, config, rng=rng_s, initial_states=states, initial_state_strategy="cycle"
    )

    assert hist_d == hist_s
    for state_id, row in sparse.items():
        assert np.array_equal(dense[state_id], row)
    untouched = np.ones(dense.shape[0], dtype=bool)
    untouched[list(sparse)] = False
    assert not dense[untouched].any()


@pytest.mark.parametrize("train", [train_q_learning, train_q_learning_sparse])
def test_training_rejects_bad_initial_state_arguments(train) -> None:
    env, rng = _cortical_env(seed=0)
    with pytest.raises(ValueError, match="non-empty"):
        train(env, _SMALL, rng=rng, initial_states=[])
    with pytest.raises(ValueError, match="initial_state_strategy"):
        train(
            env,
            _SMALL,
            rng=rng,
            initial_states=[[0, 0, 0, 0, 0]],
            initial_state_strategy="bogus",
        )


@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_training_and_evaluation_query_only_executed_transitions(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even at a goal, no candidate action is simulated before selection."""
    model = StayModel()
    calls = 0
    original = model.next_state

    def counted_transition(x: list[int], u: list[int]) -> list[int]:
        nonlocal calls
        calls += 1
        return original(x, u)

    monkeypatch.setattr(model, "next_state", counted_transition)
    env = EdgeRemovalEnv(
        model=model,
        reward=RewardConfig(),
        horizon=HorizonConfig(reach_horizon=0, max_steps=3),
        action_masking=True,
        constraint_type="recovery",
    )
    assert calls == 0
    config = QLearningConfig(episodes=4)
    expected_training_calls = config.episodes * env.horizon.max_steps
    if kind == "dense":
        dense, _ = train_q_learning(env, config, initial_states=[[1]])
        assert calls == expected_training_calls
        evaluate_policy(env, dense, [[1]])
    else:
        sparse, _ = train_q_learning_sparse(env, config, initial_states=[[1]])
        assert calls == expected_training_calls
        evaluate_policy_sparse(env, sparse, [[1]])
    assert calls == expected_training_calls + env.horizon.max_steps
