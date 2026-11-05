"""Tests pinning the metric definitions of evaluate_policy(_sparse)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from bn_edge_removal.models.cortical import CorticalModel
from bn_edge_removal.control.env import (
    EdgeRemovalEnv,
    HorizonConfig,
    RewardConfig,
)
from bn_edge_removal.learning.evaluation import (
    NO_GOAL_REACHED_TIME,
    EvaluationResult,
    evaluate_policy,
)
from bn_edge_removal.learning.evaluation import (
    evaluate_policy_sparse,
)
from bn_edge_removal.models.wnt5a import Wnt5aModel


class ChainModel:
    """00 -> 01 -> 11 -> 11 (goal); 10 -> 10 (never reaches the goal)."""

    n_nodes = 2
    m_edges = 1

    def next_state(self, x: list[int], u: list[int]) -> list[int]:
        _ = u
        return {(0, 0): [0, 1], (0, 1): [1, 1], (1, 1): [1, 1], (1, 0): [1, 0]}[
            (x[0], x[1])
        ]

    @staticmethod
    def goal_states() -> list[list[int]]:
        return [[1, 1]]

    def is_goal(self, x: list[int]) -> bool:
        return x == [1, 1]

    @staticmethod
    def state_to_id(x: list[int]) -> int:
        return x[0] * 2 + x[1]

    @staticmethod
    def id_to_state(value: int) -> list[int]:
        return [value // 2, value % 2]


def _chain_env() -> EdgeRemovalEnv:
    return EdgeRemovalEnv(
        model=ChainModel(),
        reward=RewardConfig(),
        horizon=HorizonConfig(reach_horizon=2, max_steps=3),
    )


def _evaluate(
    kind: str, env: EdgeRemovalEnv, initial_states: list[list[int]] | None
) -> EvaluationResult:
    if kind == "dense":
        q_table = np.zeros((env.num_states, env.num_actions))
        return evaluate_policy(env, q_table, initial_states)
    empty: dict[int, Any] = {}
    return evaluate_policy_sparse(env, empty, initial_states)


@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_first_goal_metrics_definitions(kind: str) -> None:
    result = _evaluate(kind, _chain_env(), None)
    metrics = result.metrics

    assert metrics["num_initial_states"] == 4.0
    assert metrics["success_rate"] == 0.75
    assert metrics["violation_rate"] == 0.0
    assert metrics["num_goal_reached"] == 3.0
    # Legacy definition: first t >= 1 with x(t) in goal (t=0 is not counted).
    #   11 -> 1 (already at goal), 01 -> 1, 00 -> 2.
    assert metrics["avg_first_goal_step"] == pytest.approx(4 / 3)
    # Hitting time: first t >= 0 with x(t) in goal.  11 -> 0, 01 -> 1, 00 -> 2.
    assert metrics["avg_first_goal_time"] == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_initial_goal_state_counts_as_step_one_and_time_zero(kind: str) -> None:
    metrics = _evaluate(kind, _chain_env(), [[1, 1]]).metrics

    assert metrics["success_rate"] == 1.0
    assert metrics["avg_first_goal_step"] == 1.0
    assert metrics["avg_first_goal_time"] == 0.0
    assert metrics["num_goal_reached"] == 1.0


@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_never_reaching_goal_uses_sentinels(kind: str) -> None:
    metrics = _evaluate(kind, _chain_env(), [[1, 0]]).metrics

    assert metrics["success_rate"] == 0.0
    assert metrics["num_goal_reached"] == 0.0
    # 0.0 cannot be a real avg_first_goal_step (its minimum is 1).
    assert metrics["avg_first_goal_step"] == 0.0
    # 0.0 is a real hitting time, so the sentinel must differ from it.
    assert metrics["avg_first_goal_time"] == NO_GOAL_REACHED_TIME
    assert NO_GOAL_REACHED_TIME < 0.0


@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_trajectory_and_action_records(kind: str) -> None:
    result = _evaluate(kind, _chain_env(), [[0, 0]])

    assert result.trajectories == [
        {"init_state": 0, "t": 0, "state_id": 0},
        {"init_state": 0, "t": 1, "state_id": 1},
        {"init_state": 0, "t": 2, "state_id": 3},
        {"init_state": 0, "t": 3, "state_id": 3},
    ]
    assert result.actions == [
        {"init_state": 0, "t": 0, "action": 0},
        {"init_state": 0, "t": 1, "action": 0},
        {"init_state": 0, "t": 2, "action": 0},
    ]


def test_success_requires_no_violation() -> None:
    """A trajectory that satisfies the goal but violates the constraint fails."""

    class GoalModel(ChainModel):
        m_edges = 1

        def next_state(self, x: list[int], u: list[int]) -> list[int]:
            _ = (x, u)
            return [1, 1]

    env = EdgeRemovalEnv(
        model=GoalModel(),
        reward=RewardConfig(),
        horizon=HorizonConfig(reach_horizon=2, max_steps=3),
        constraint_type="monotone",
    )
    q_table = np.zeros((env.num_states, env.num_actions))
    # Prefer "remove" only while the edge is still present: remove at t=0, then
    # release it at t=1 (monotone violation).
    for x_id in range(4):
        q_table[env.extended_state_id(env.model.id_to_state(x_id), [0]), 1] = 1.0
    metrics = evaluate_policy(env, q_table, [[1, 1]]).metrics

    assert metrics["violation_rate"] == 1.0
    assert metrics["success_rate"] == 0.0
    assert metrics["num_goal_reached"] == 1.0


def test_cortical_no_intervention_baseline() -> None:
    """Q=0 (never intervene) on the adopted cortical horizon: 10/32 succeed."""
    env = EdgeRemovalEnv(
        model=CorticalModel(),
        reward=RewardConfig(),
        horizon=HorizonConfig(reach_horizon=5, max_steps=10),
    )
    result = evaluate_policy(env, np.zeros((env.num_states, env.num_actions)))

    assert result.metrics["success_rate"] == 10 / 32
    assert result.metrics["num_goal_reached"] == 10.0
    assert result.metrics["avg_first_goal_step"] == pytest.approx(1.8)
    assert result.metrics["avg_first_goal_time"] == pytest.approx(1.6)
    assert all(row["action"] == 0 for row in result.actions)


def test_wnt5a_no_intervention_baseline() -> None:
    model = Wnt5aModel()
    env = EdgeRemovalEnv(
        model=model,
        reward=RewardConfig(),
        horizon=HorizonConfig(reach_horizon=7, max_steps=10),
        action_masking=True,
        constraint_type="recovery",
        recovery_tau=2,
    )
    result = evaluate_policy_sparse(env, {}, model.controllable_initial_states())

    assert result.metrics["num_initial_states"] == 96.0
    assert result.metrics["success_rate"] == 76 / 96


@pytest.mark.parametrize(
    "model", [CorticalModel(), Wnt5aModel()], ids=["cortical", "wnt5a"]
)
def test_goal_states_are_fixed_points_without_intervention(model) -> None:
    for goal in model.goal_states():
        assert model.next_state(goal, [0] * model.m_edges) == goal
