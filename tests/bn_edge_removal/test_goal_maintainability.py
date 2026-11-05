"""Optional model-based diagnostics, isolated from model-free execution."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pytest

from bn_edge_removal.models.cell_cycle10 import CellCycle10Model
from bn_edge_removal.models.cortical import CorticalModel
from bn_edge_removal.models.encoding import bits_to_int, int_to_bits
from bn_edge_removal.control.env import (
    EdgeRemovalEnv,
    EdgeRemovalModel,
    HorizonConfig,
    RewardConfig,
    unmaintainable_goal_states,
)
from bn_edge_removal.models.wnt5a import Wnt5aModel


@dataclass(frozen=True)
class _TogglingGoalModel:
    """x1 flips every step whatever u is, so the goal [1, 0] is never kept."""

    n_nodes: int = 2
    m_edges: int = 1

    def next_state(self, x: list[int], u: list[int]) -> list[int]:
        return [1 - x[0], x[1] if u[0] else 0]

    @staticmethod
    def goal_states() -> list[list[int]]:
        return [[1, 0]]

    def is_goal(self, x: list[int]) -> bool:
        return x in self.goal_states()

    def state_to_id(self, x: list[int]) -> int:
        return bits_to_int(x)

    def id_to_state(self, value: int) -> list[int]:
        return int_to_bits(value, self.n_nodes)


def _env(model: EdgeRemovalModel) -> EdgeRemovalEnv:
    return EdgeRemovalEnv(
        model=model,
        reward=RewardConfig(),
        horizon=HorizonConfig(reach_horizon=2, max_steps=4),
    )


def test_unmaintainable_goal_states_lists_goals_no_action_keeps():
    assert unmaintainable_goal_states(_TogglingGoalModel()) == [[1, 0]]


def test_env_initialization_does_not_predict_successors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_prediction(
        self: _TogglingGoalModel, x: list[int], u: list[int]
    ) -> list[int]:
        raise AssertionError("Initialization must not query the transition model")

    monkeypatch.setattr(_TogglingGoalModel, "next_state", forbid_prediction)
    _env(_TogglingGoalModel())


def test_goal_preserving_actions_fall_back_to_all_candidates():
    env = _env(_TogglingGoalModel())
    env.reset(initial_state=[1, 0])

    assert env.goal_preserving_actions() == [0, 1]
    assert env.goal_preserving_actions([1]) == [1]


@pytest.mark.parametrize("model", [CorticalModel(), Wnt5aModel(), CellCycle10Model()])
def test_models_used_by_run_scripts_do_not_warn(model: EdgeRemovalModel):
    assert unmaintainable_goal_states(model) == []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _env(model)
