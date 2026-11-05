"""Success conditions (``goal_spec``) and the goal monitor of ``EdgeRemovalEnv``."""

from __future__ import annotations

import itertools

import pytest

from bn_edge_removal.control.env import (
    GOAL_STATUS_SIZE,
    EdgeRemovalEnv,
    HorizonConfig,
    RewardConfig,
    trajectory_satisfies_goal,
)

REACH_HORIZON = 2
MAX_STEPS = 4


class SwitchModel:
    """One node whose next value is the input: ``u = 1`` moves to the goal ``[1]``."""

    n_nodes = 1
    m_edges = 1

    def next_state(self, x: list[int], u: list[int]) -> list[int]:
        _ = x
        return [u[0]]

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


def make_env(goal_spec: str = "window", *, time_in_state: bool = False) -> EdgeRemovalEnv:
    return EdgeRemovalEnv(
        model=SwitchModel(),
        reward=RewardConfig(step=0.0, terminal=10.0, edge_cost=0.0, violation_penalty=0.0),
        horizon=HorizonConfig(reach_horizon=REACH_HORIZON, max_steps=MAX_STEPS),
        goal_spec=goal_spec,
        time_in_state=time_in_state,
    )


def rollout(env: EdgeRemovalEnv, x0: int, inputs: tuple[int, ...]) -> list[bool]:
    """Run one episode; ``inputs[t]`` sets ``x_{t+1}``.  Returns the goal flags."""
    env.reset(initial_state=[x0])
    for action in inputs:
        env.step(action)
    return [state == [1] for state in env.trajectory]


def first_entry_holds(flags: list[bool]) -> bool:
    t_reach = next((t for t, flag in enumerate(flags) if flag), None)
    return t_reach is not None and t_reach <= REACH_HORIZON and all(flags[t_reach:])


ALL_EPISODES = list(
    itertools.product((0, 1), itertools.product((0, 1), repeat=MAX_STEPS))
)


def test_default_state_id_is_unchanged() -> None:
    env = make_env()

    state_id = env.reset(initial_state=[1])
    next_id, _, _, _ = env.step(0)

    assert state_id == env.extended_state_id([1], [0])
    assert next_id == env.extended_state_id(env.x, env.h)
    assert env.num_states == 2 * env.flag_state_size


@pytest.mark.parametrize(("x0", "inputs"), ALL_EPISODES)
def test_window_matches_trajectory_satisfies_goal(
    x0: int, inputs: tuple[int, ...]
) -> None:
    for time_in_state in (False, True):
        env = make_env("window", time_in_state=time_in_state)
        rollout(env, x0, inputs)

        expected = trajectory_satisfies_goal(
            env.trajectory, [[1]], REACH_HORIZON, MAX_STEPS
        )
        assert env.success == expected
        # The window monitor agrees with the trajectory check.
        assert (env.goal_status == 1) == expected


@pytest.mark.parametrize(("x0", "inputs"), ALL_EPISODES)
def test_first_entry_requires_arrival_by_h_and_no_exit(
    x0: int, inputs: tuple[int, ...]
) -> None:
    env = make_env("first_entry")
    flags = rollout(env, x0, inputs)

    assert env.success == first_entry_holds(flags)
    if env.success:
        assert trajectory_satisfies_goal(env.trajectory, [[1]], REACH_HORIZON, MAX_STEPS)


def test_first_entry_rejects_leaving_and_reentering_before_h() -> None:
    env = make_env("first_entry")
    # x = 1, 0, 1, 1, 1: in the goal on [H, T], but it left after t = 0.
    rollout(env, 1, (0, 1, 1, 1))

    assert not env.success
    window = make_env("window")
    rollout(window, 1, (0, 1, 1, 1))
    assert window.success


def test_first_entry_rejects_arrival_after_h() -> None:
    env = make_env("first_entry")
    # x = 0, 0, 0, 1, 1: first visit at t = 3 > H.
    rollout(env, 0, (0, 0, 1, 1))

    assert not env.success


def test_first_entry_pays_the_terminal_bonus_on_success() -> None:
    env = make_env("first_entry")
    env.reset(initial_state=[0])
    rewards = [env.step(1)[1] for _ in range(MAX_STEPS)]

    assert env.success
    assert rewards[-1] == 10.0
    assert sum(rewards[:-1]) == 0.0


def test_time_in_state_ids_are_distinct_and_in_range() -> None:
    env = make_env("first_entry")
    assert env.time_in_state

    seen: dict[tuple[int, int, int, int], int] = {}
    for x0, inputs in ALL_EPISODES:
        state_id = env.reset(initial_state=[x0])
        seen[(env.x[0], env.h[0], env.t, env.goal_status)] = state_id
        for action in inputs:
            state_id = env.step(action)[0]
            seen[(env.x[0], env.h[0], env.t, env.goal_status)] = state_id

    assert len(set(seen.values())) == len(seen)
    assert all(0 <= state_id < env.num_states for state_id in seen.values())
    assert env.num_states == 2 * env.flag_state_size * (MAX_STEPS + 1) * GOAL_STATUS_SIZE


def test_unknown_goal_spec_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_env("reach_stay")
