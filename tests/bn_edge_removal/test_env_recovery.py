from bn_edge_removal.models.encoding import bits_to_int, int_to_bits
from bn_edge_removal.control.env import (
    EdgeRemovalEnv,
    HorizonConfig,
    RewardConfig,
)
from bn_edge_removal.models.wnt5a import Wnt5aModel


def test_env_recovery_detects_repeated_intervention():
    env = EdgeRemovalEnv(
        model=Wnt5aModel(),
        reward=RewardConfig(),
        horizon=HorizonConfig(reach_horizon=1, max_steps=2),
        constraint_type="recovery",
        recovery_tau=2,
    )
    env.reset(initial_state=[0, 0, 0, 0, 0, 0, 0])

    action = bits_to_int([1, 0, 0, 0, 0, 0, 0, 0])

    _state, _reward, _done, info1 = env.step(action)
    assert info1["violation"] is False

    _state, _reward, _done, info2 = env.step(action)
    assert info2["violation"] is True


def test_goal_preserving_actions_filter_in_goal_state():
    env = EdgeRemovalEnv(
        model=Wnt5aModel(),
        reward=RewardConfig(),
        horizon=HorizonConfig(reach_horizon=7, max_steps=10),
        constraint_type="recovery",
        recovery_tau=2,
        action_masking=True,
    )
    goal_state = [0, 1, 0, 1, 1, 1, 1]
    env.reset(initial_state=goal_state)

    allowed = env.allowed_actions()
    preserving = env.goal_preserving_actions(allowed)

    assert len(preserving) > 0
    assert set(preserving).issubset(set(allowed))
    assert len(preserving) < len(allowed)

    for action in preserving:
        u = int_to_bits(action, env.m_edges)
        next_state = env.model.next_state(goal_state, u)
        assert env.model.is_goal(next_state)


def test_goal_preserving_actions_no_filter_outside_goal():
    env = EdgeRemovalEnv(
        model=Wnt5aModel(),
        reward=RewardConfig(),
        horizon=HorizonConfig(reach_horizon=7, max_steps=10),
        constraint_type="recovery",
        recovery_tau=2,
        action_masking=True,
    )
    env.reset(initial_state=[0, 0, 0, 0, 0, 0, 0])

    allowed = env.allowed_actions()
    preserving = env.goal_preserving_actions(allowed)

    assert preserving == allowed
