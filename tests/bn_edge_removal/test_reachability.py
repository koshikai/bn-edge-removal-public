from __future__ import annotations

import pytest

from bn_edge_removal.models.cell_cycle10 import CellCycle10Model
from bn_edge_removal.models.cortical import CorticalModel
from bn_edge_removal.models.encoding import int_to_bits
from bn_edge_removal.control.env import EdgeRemovalModel
from bn_edge_removal.analysis.reachability import (
    ReachabilityConfig,
    WitnessTrajectory,
    verify_all_initial_states,
)
from bn_edge_removal.control.stl_monotone import (
    check_violation as check_violation_monotone,
)
from bn_edge_removal.control.stl_monotone import (
    update_flags as update_flags_monotone,
)
from bn_edge_removal.control.stl_recovery import (
    check_violation as check_violation_recovery,
)
from bn_edge_removal.control.stl_recovery import (
    update_flags as update_flags_recovery,
)
from bn_edge_removal.models.wnt5a import Wnt5aModel

EXPECTED_CORTICAL_TOTAL = 32
EXPECTED_WNT5A_TOTAL = 128
EXPECTED_WNT5A_STRICT_SUCCESS_MAX_STEPS_2 = 112
EXPECTED_GOAL_STATE_COUNT = 2


def _assert_witness_consistent(
    model: EdgeRemovalModel,
    config: ReachabilityConfig,
    initial_state: list[int],
    witness: WitnessTrajectory,
) -> None:
    assert witness.states[0] == initial_state
    assert len(witness.states) == len(witness.actions) + 1
    assert 0 <= witness.goal_step <= len(witness.actions)

    state = list(initial_state)
    flags = [0] * model.m_edges

    if witness.goal_step == 0:
        assert model.is_goal(state)
        assert witness.actions == []
        return

    for step, action_id in enumerate(witness.actions, start=1):
        action = int_to_bits(action_id, model.m_edges)
        if config.enforce_constraints:
            if config.constraint_type == "monotone":
                assert check_violation_monotone(flags, action) is False
                flags = update_flags_monotone(flags, action)
            else:
                assert check_violation_recovery(flags, action) is False
                flags = update_flags_recovery(flags, action, config.recovery_tau)

        state = model.next_state(state, action)
        assert state == witness.states[step]
        if step < witness.goal_step:
            assert model.is_goal(state) is False

    assert model.is_goal(witness.states[witness.goal_step]) is True


def test_cortical_strict_reachability_max_steps_3():
    config = ReachabilityConfig(
        max_steps=3,
        constraint_type="monotone",
        enforce_constraints=True,
    )
    result = verify_all_initial_states(CorticalModel(), config)

    assert result.model_name == "cortical"
    assert result.mode == "strict"
    assert result.success_count == EXPECTED_CORTICAL_TOTAL
    assert result.total_initial_states == EXPECTED_CORTICAL_TOTAL
    assert result.global_reachable is True


def test_wnt5a_strict_reachability_max_steps_2():
    config = ReachabilityConfig(
        max_steps=2,
        constraint_type="recovery",
        recovery_tau=2,
        enforce_constraints=True,
    )
    result = verify_all_initial_states(Wnt5aModel(), config)

    assert result.model_name == "wnt5a"
    assert result.mode == "strict"
    assert result.success_count == EXPECTED_WNT5A_STRICT_SUCCESS_MAX_STEPS_2
    assert result.total_initial_states == EXPECTED_WNT5A_TOTAL
    assert result.global_reachable is False
    assert any(item.reachable is False for item in result.per_state_results)


def test_wnt5a_ignore_reachability_max_steps_2():
    config = ReachabilityConfig(
        max_steps=2,
        constraint_type="recovery",
        recovery_tau=2,
        enforce_constraints=False,
    )
    result = verify_all_initial_states(Wnt5aModel(), config)

    assert result.model_name == "wnt5a"
    assert result.mode == "ignore"
    assert result.success_count == EXPECTED_WNT5A_TOTAL
    assert result.total_initial_states == EXPECTED_WNT5A_TOTAL
    assert result.global_reachable is True


@pytest.mark.parametrize(
    ("model", "constraint_type"),
    [
        (CorticalModel(), "monotone"),
        (Wnt5aModel(), "recovery"),
    ],
)
def test_max_steps_zero_only_goal_states_reachable(
    model: EdgeRemovalModel,
    constraint_type: str,
):
    config = ReachabilityConfig(
        max_steps=0,
        constraint_type=constraint_type,
        recovery_tau=2,
        enforce_constraints=True,
    )
    result = verify_all_initial_states(model, config)

    reached_ids = {
        item.initial_state_id for item in result.per_state_results if item.reachable
    }
    goal_ids = {model.state_to_id(state) for state in model.goal_states()}

    assert result.success_count == EXPECTED_GOAL_STATE_COUNT
    assert reached_ids == goal_ids


def test_witness_replay_and_strict_non_violation():
    model = Wnt5aModel()
    config = ReachabilityConfig(
        max_steps=2,
        constraint_type="recovery",
        recovery_tau=2,
        enforce_constraints=True,
    )
    result = verify_all_initial_states(model, config)

    for item in result.per_state_results:
        if not item.reachable:
            assert item.witness is None
            continue
        assert item.witness is not None
        _assert_witness_consistent(model, config, item.initial_state, item.witness)


def test_cell_cycle10_model_name_and_counts():
    model = CellCycle10Model()
    config = ReachabilityConfig(
        max_steps=7,
        constraint_type="monotone",
        enforce_constraints=True,
    )
    result = verify_all_initial_states(model, config)

    assert result.model_name == "cell_cycle10"
    assert result.success_count == 1024
    assert result.total_initial_states == 1024
    assert result.global_reachable is True
