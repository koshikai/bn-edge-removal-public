from __future__ import annotations

from ai_research_template.bn_edge_removal.cortical import CorticalModel
from ai_research_template.bn_edge_removal.env import (
    EdgeRemovalEnv,
    HorizonConfig,
    RewardConfig,
)


def test_env_initialization() -> None:
    model = CorticalModel()
    reward = RewardConfig()
    horizon = HorizonConfig(reach_horizon=3, max_steps=5)
    env = EdgeRemovalEnv(model=model, reward=reward, horizon=horizon)

    assert env.n_nodes == 5
    assert env.m_edges == 6
    assert env.num_actions == 64
    assert env.num_states == 32 * (2**6)  # flag_state_size for monotone is 2^6


def test_env_reset() -> None:
    model = CorticalModel()
    reward = RewardConfig()
    horizon = HorizonConfig(reach_horizon=3, max_steps=5)
    env = EdgeRemovalEnv(model=model, reward=reward, horizon=horizon)

    state_id = env.reset()
    assert 0 <= state_id < env.num_states
    assert len(env.x) == 5
    assert env.h == [0] * 6
    assert env.t == 0
    assert env.violation_any is False

    specific_state = [1, 0, 1, 0, 1]
    state_id_2 = env.reset(initial_state=specific_state)
    assert env.x == specific_state
    assert state_id_2 == env.extended_state_id(specific_state, [0] * 6)


def test_env_monotone_constraint() -> None:
    model = CorticalModel()
    reward = RewardConfig(step=1.0, violation_penalty=10.0, edge_cost=1.0)
    horizon = HorizonConfig(reach_horizon=3, max_steps=5)
    env = EdgeRemovalEnv(
        model=model,
        reward=reward,
        horizon=horizon,
        action_masking=True,
        constraint_type="monotone",
    )

    env.reset()
    # 初期状態ではすべて許可
    allowed = env.allowed_actions()
    assert len(allowed) == 64

    # アクション 1 (バイナリ: 000001 -> エッジ5を削除) を実行
    action_1 = 1
    _, _, _, info1 = env.step(action_1)
    assert env.h == [0, 0, 0, 0, 0, 1]
    assert info1["violation"] is False

    # 単調性制約下で、h_5=1 のとき、エッジ5は削除し続けなければならない。
    # したがって、u_5=0 (action が偶数) は不許可、
    # u_5=1 (action が奇数) のみが許可される。
    allowed_after = env.allowed_actions()
    assert len(allowed_after) == 32
    assert all(a % 2 == 1 for a in allowed_after)

    # 許可されていない偶数アクション (例: 0) をあえて実行すると違反になる
    _, r2, _, info2 = env.step(0)
    assert info2["violation"] is True
    assert env.violation_any is True
    # 違反ペナルティ (-10.0) が引かれていることを確認
    assert r2 <= -10.0


def test_env_recovery_constraint() -> None:
    model = CorticalModel()
    reward = RewardConfig(step=1.0, violation_penalty=10.0, edge_cost=1.0)
    horizon = HorizonConfig(reach_horizon=3, max_steps=5)
    env = EdgeRemovalEnv(
        model=model,
        reward=reward,
        horizon=horizon,
        action_masking=True,
        constraint_type="recovery",
        recovery_tau=2,
    )

    env.reset()
    # 初期状態ではすべて許可
    assert len(env.allowed_actions()) == 64

    # アクション 1 (000001 -> エッジ5を削除) を実行
    _, _, _, info1 = env.step(1)
    assert env.h == [0, 0, 0, 0, 0, 2]  # tau=2 に設定される
    assert info1["violation"] is False

    # 休薬制約下で、h_5 > 0 の間は、エッジ5に対して再介入 (u_5=1) できない。
    # すなわち、action が奇数 (u_5=1) は不許可となり、偶数 (u_5=0) のみが許可される。
    allowed_after = env.allowed_actions()
    assert len(allowed_after) == 32
    assert all(a % 2 == 0 for a in allowed_after)

    # 許可されていないアクション 1 を実行すると違反になる
    _, r2, _, info2 = env.step(1)
    assert info2["violation"] is True
    assert r2 <= -10.0
