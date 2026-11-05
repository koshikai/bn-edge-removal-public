"""Invariants of the STL-style input constraints under action masking."""

from __future__ import annotations

import pytest

from bn_edge_removal.models.cortical import CorticalModel
from bn_edge_removal.models.encoding import bits_to_int, int_to_bits
from bn_edge_removal.control.env import (
    EdgeRemovalEnv,
    HorizonConfig,
    RewardConfig,
)
from bn_edge_removal.models.wnt5a import Wnt5aModel


def _masked_env(constraint_type: str) -> EdgeRemovalEnv:
    if constraint_type == "monotone":
        return EdgeRemovalEnv(
            model=CorticalModel(),
            reward=RewardConfig(),
            horizon=HorizonConfig(reach_horizon=5, max_steps=10),
            action_masking=True,
            constraint_type="monotone",
        )
    return EdgeRemovalEnv(
        model=Wnt5aModel(),
        reward=RewardConfig(),
        horizon=HorizonConfig(reach_horizon=7, max_steps=10),
        action_masking=True,
        constraint_type="recovery",
        recovery_tau=2,
    )


@pytest.mark.parametrize("constraint_type", ["monotone", "recovery"])
def test_masked_allowed_actions_never_violate(constraint_type: str) -> None:
    """Exhaust every reachable flag vector and every allowed action."""
    env = _masked_env(constraint_type)
    start = tuple([0] * env.m_edges)
    seen = {start}
    frontier = [start]
    checked = 0
    while frontier:
        h = frontier.pop()
        env.reset(initial_state=[0] * env.n_nodes)
        env.h = list(h)
        for action in env.allowed_actions():
            env.t = 0
            env.h = list(h)
            _state, _reward, _done, info = env.step(action)
            checked += 1
            assert info["violation"] is False, f"h={h} action={action}"
            next_h = tuple(env.h)
            if next_h not in seen:
                seen.add(next_h)
                frontier.append(next_h)
        assert env.violation_any is False

    expected_flag_states = 2**env.m_edges if constraint_type == "monotone" else None
    if expected_flag_states is not None:
        assert len(seen) == expected_flag_states
    assert checked > len(seen)


def test_unmasked_env_allows_violating_actions() -> None:
    env = _masked_env("monotone")
    env.action_masking = False
    env.reset(initial_state=[0] * env.n_nodes)
    env.step(env.num_actions - 1)
    assert env.allowed_actions() == list(range(env.num_actions))
    _state, _reward, _done, info = env.step(0)
    assert info["violation"] is True


def test_cached_admissible_actions_are_isolated_from_caller_mutation() -> None:
    env = _masked_env("monotone")
    env.h = [1] * env.m_edges
    expected = [env.num_actions - 1]
    actions = env.allowed_actions()
    assert actions == expected
    actions.clear()
    assert env.allowed_actions() == expected
    env.h = [0] * env.m_edges
    assert env.allowed_actions() == list(range(env.num_actions))
    env.constraint_type = "recovery"
    env.h = [1] * env.m_edges
    assert env.allowed_actions() == [0]


def test_recovery_tau2_shortest_reintervention_is_t_plus_3() -> None:
    env = _masked_env("recovery")
    env.reset(initial_state=[0] * env.n_nodes)
    edge_action = bits_to_int([1] + [0] * (env.m_edges - 1))

    def edge_allowed() -> bool:
        return any(int_to_bits(a, env.m_edges)[0] == 1 for a in env.allowed_actions())

    assert edge_allowed()  # t
    env.step(edge_action)
    assert not edge_allowed()  # t+1
    env.step(0)
    assert not edge_allowed()  # t+2
    env.step(0)
    assert edge_allowed()  # t+3
    _state, _reward, _done, info = env.step(edge_action)
    assert info["violation"] is False
