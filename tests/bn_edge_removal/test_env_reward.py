from __future__ import annotations

from bn_edge_removal.control.env import (
    EdgeRemovalEnv,
    HorizonConfig,
    RewardConfig,
)
from bn_edge_removal.models.encoding import bits_to_int


class DummyModel:
    n_nodes = 1

    def __init__(self, m_edges: int) -> None:
        self.m_edges = m_edges

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


def test_new_edge_cost_penalizes_only_first_intervention_in_monotone() -> None:
    env = EdgeRemovalEnv(
        model=DummyModel(m_edges=3),
        reward=RewardConfig(
            step=0.0,
            terminal=0.0,
            edge_cost=1.0,
            new_edge_cost=2.0,
            violation_penalty=100.0,
        ),
        horizon=HorizonConfig(reach_horizon=1, max_steps=2),
        constraint_type="monotone",
    )
    env.reset(initial_state=[0])

    action = bits_to_int([1, 0, 1])  # 2 intervened edges

    _state, reward1, _done, info1 = env.step(action)
    _state, reward2, _done, info2 = env.step(action)

    assert info1["violation"] is False
    assert info2["violation"] is False
    assert reward1 == -6.0  # edge_cost*2 + new_edge_cost*2
    assert reward2 == -2.0  # only edge_cost remains


def test_new_edge_cost_reapplies_after_recovery_cooldown() -> None:
    env = EdgeRemovalEnv(
        model=DummyModel(m_edges=1),
        reward=RewardConfig(
            step=0.0,
            terminal=0.0,
            edge_cost=0.0,
            new_edge_cost=3.0,
            violation_penalty=100.0,
        ),
        horizon=HorizonConfig(reach_horizon=1, max_steps=3),
        constraint_type="recovery",
        recovery_tau=1,
    )
    env.reset(initial_state=[0])

    action_on = bits_to_int([1])
    action_off = bits_to_int([0])

    _state, reward1, _done, info1 = env.step(action_on)
    _state, reward2, _done, info2 = env.step(action_off)
    _state, reward3, _done, info3 = env.step(action_on)

    assert info1["violation"] is False
    assert info2["violation"] is False
    assert info3["violation"] is False
    assert reward1 == -3.0
    assert reward2 == 0.0
    assert reward3 == -3.0


class FlipToGoalModel(DummyModel):
    """x=0 moves to the goal x=1 in one step; x=1 stays at the goal."""

    def next_state(self, x: list[int], u: list[int]) -> list[int]:
        _ = (x, u)
        return [1]


def test_stabilization_reward_is_attributed_to_pre_transition_state() -> None:
    """r_stab(x_t) pays for the state the action is taken in, not x_{t+1}.

    Over an episode the reward covers x(0)..x(max_steps-1); the final state
    x(max_steps) only contributes through the terminal bonus.
    """
    env = EdgeRemovalEnv(
        model=FlipToGoalModel(m_edges=1),
        reward=RewardConfig(
            step=2.0, terminal=0.0, edge_cost=0.0, violation_penalty=0.0
        ),
        horizon=HorizonConfig(reach_horizon=1, max_steps=3),
    )
    env.reset(initial_state=[0])

    rewards = [env.step(0)[1] for _ in range(3)]

    assert env.trajectory == [[0], [1], [1], [1]]
    assert rewards == [0.0, 2.0, 2.0]  # x(0) is not a goal; x(3) is never paid
    assert env.success is True


def test_terminal_bonus_is_paid_once_on_the_last_step() -> None:
    env = EdgeRemovalEnv(
        model=FlipToGoalModel(m_edges=1),
        reward=RewardConfig(
            step=0.0, terminal=7.0, edge_cost=0.0, violation_penalty=0.0
        ),
        horizon=HorizonConfig(reach_horizon=1, max_steps=3),
    )
    env.reset(initial_state=[0])

    steps = [env.step(0) for _ in range(3)]

    assert [reward for _s, reward, _d, _i in steps] == [0.0, 0.0, 7.0]
    assert [done for _s, _r, done, _i in steps] == [False, False, True]
    assert steps[-1][3]["success"] is True


def test_new_edge_cost_charges_every_non_violating_intervention_in_recovery() -> None:
    """Under recovery, h_i == 0 means "not cooling down", not "never removed".

    Every intervention that respects the cooldown therefore pays new_edge_cost,
    while an intervention during the cooldown (a violation) does not.
    """
    env = EdgeRemovalEnv(
        model=DummyModel(m_edges=1),
        reward=RewardConfig(
            step=0.0,
            terminal=0.0,
            edge_cost=0.0,
            new_edge_cost=3.0,
            violation_penalty=100.0,
        ),
        horizon=HorizonConfig(reach_horizon=1, max_steps=6),
        constraint_type="recovery",
        recovery_tau=2,
    )
    env.reset(initial_state=[0])
    on, off = bits_to_int([1]), bits_to_int([0])

    rewards = [env.step(a)[1] for a in (on, on, off, off, on, off)]

    #          onset  violation      cooldown   onset again
    assert rewards == [-3.0, -100.0, 0.0, 0.0, -3.0, 0.0]


def test_goal_onset_cost_does_not_charge_maintained_removals_in_monotone() -> None:
    """At the goal, only a *new* removal pays; keeping an edge removed is free.

    This is the difference from ``edge_cost`` and the reason the onset version
    does not collapse under the monotone constraint.
    """
    env = EdgeRemovalEnv(
        model=DummyModel(m_edges=3),
        reward=RewardConfig(
            step=0.0,
            terminal=0.0,
            edge_cost=0.0,
            goal_onset_cost=2.0,
            violation_penalty=100.0,
        ),
        horizon=HorizonConfig(reach_horizon=1, max_steps=2),
        constraint_type="monotone",
    )
    env.reset(initial_state=[1])  # the goal state of DummyModel
    action = bits_to_int([1, 0, 1])

    _state, reward1, _done, info1 = env.step(action)
    _state, reward2, _done, info2 = env.step(action)

    assert info1["violation"] is False
    assert info2["violation"] is False
    assert reward1 == -4.0  # two onsets at the goal
    assert reward2 == 0.0  # the same edges are only maintained


def test_goal_onset_cost_is_not_charged_outside_the_goal() -> None:
    """The cost follows the pre-transition state x_t, not the next state."""
    env = EdgeRemovalEnv(
        model=FlipToGoalModel(m_edges=1),
        reward=RewardConfig(
            step=0.0,
            terminal=0.0,
            edge_cost=0.0,
            goal_onset_cost=2.0,
            violation_penalty=0.0,
        ),
        horizon=HorizonConfig(reach_horizon=1, max_steps=2),
    )
    env.reset(initial_state=[0])

    _state, reward1, _done, _info = env.step(1)
    _state, reward2, _done, _info = env.step(1)

    assert env.trajectory == [[0], [1], [1]]
    assert reward1 == 0.0  # x_t = 0 is not a goal even though x_{t+1} = 1 is
    assert reward2 == 0.0  # the flag already prevents a new onset at x_t = 1


def test_goal_onset_cost_reapplies_after_recovery_cooldown() -> None:
    env = EdgeRemovalEnv(
        model=DummyModel(m_edges=1),
        reward=RewardConfig(
            step=0.0,
            terminal=0.0,
            edge_cost=0.0,
            goal_onset_cost=3.0,
            violation_penalty=100.0,
        ),
        horizon=HorizonConfig(reach_horizon=1, max_steps=6),
        constraint_type="recovery",
        recovery_tau=2,
    )
    env.reset(initial_state=[1])  # the goal state of DummyModel
    on, off = bits_to_int([1]), bits_to_int([0])

    rewards = [env.step(a)[1] for a in (on, on, off, off, on, off)]

    #          onset  violation      cooldown   onset again
    assert rewards == [-3.0, -100.0, 0.0, 0.0, -3.0, 0.0]


def test_goal_edge_cost_charges_maintained_removals_at_the_goal() -> None:
    """The per-step goal cost is the analogue of edge_cost inside the goal set."""
    env = EdgeRemovalEnv(
        model=DummyModel(m_edges=3),
        reward=RewardConfig(
            step=0.0,
            terminal=0.0,
            edge_cost=0.0,
            goal_edge_cost=2.0,
            violation_penalty=100.0,
        ),
        horizon=HorizonConfig(reach_horizon=1, max_steps=2),
        constraint_type="monotone",
    )
    env.reset(initial_state=[1])
    action = bits_to_int([1, 0, 1])

    _state, reward1, _done, _info = env.step(action)
    _state, reward2, _done, _info = env.step(action)

    assert reward1 == -4.0
    assert reward2 == -4.0  # maintained removals are still charged


def test_goal_edge_cost_is_not_charged_outside_the_goal() -> None:
    env = EdgeRemovalEnv(
        model=DummyModel(m_edges=1),
        reward=RewardConfig(
            step=0.0,
            terminal=0.0,
            edge_cost=0.0,
            goal_edge_cost=2.0,
            violation_penalty=0.0,
        ),
        horizon=HorizonConfig(reach_horizon=1, max_steps=2),
    )
    env.reset(initial_state=[0])

    _state, reward, _done, _info = env.step(1)

    assert reward == 0.0
