"""Environment for edge removal control with STL-style input constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ai_research_template.bn_edge_removal.encoding import bits_to_int, int_to_bits
from ai_research_template.bn_edge_removal.stl_monotone import (
    allowed_actions as allowed_actions_monotone,
)
from ai_research_template.bn_edge_removal.stl_monotone import (
    check_violation as check_violation_monotone,
)
from ai_research_template.bn_edge_removal.stl_monotone import (
    update_flags as update_flags_monotone,
)
from ai_research_template.bn_edge_removal.stl_recovery import (
    allowed_actions as allowed_actions_recovery,
)
from ai_research_template.bn_edge_removal.stl_recovery import (
    check_violation as check_violation_recovery,
)
from ai_research_template.bn_edge_removal.stl_recovery import (
    update_flags as update_flags_recovery,
)


class EdgeRemovalModel(Protocol):
    n_nodes: int
    m_edges: int

    def next_state(self, x: list[int], u: list[int]) -> list[int]: ...

    @staticmethod
    def goal_states() -> list[list[int]]: ...

    def is_goal(self, x: list[int]) -> bool: ...

    def state_to_id(self, x: list[int]) -> int: ...

    def id_to_state(self, value: int) -> list[int]: ...


@dataclass
class RewardConfig:
    step: float = 1.0
    terminal: float = 5.0
    edge_cost: float = 1.0
    new_edge_cost: float = 0.0
    violation_penalty: float = 5.0


@dataclass
class HorizonConfig:
    reach_horizon: int = 5
    max_steps: int = 10


@dataclass
class EdgeRemovalEnv:
    model: EdgeRemovalModel
    reward: RewardConfig
    horizon: HorizonConfig
    action_masking: bool = False
    constraint_type: str = "monotone"
    recovery_tau: int = 2
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    x: list[int] = field(init=False)
    h: list[int] = field(init=False)
    t: int = field(init=False)
    trajectory: list[list[int]] = field(init=False)
    violation_any: bool = field(init=False)
    success: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.constraint_type not in {"monotone", "recovery"}:
            raise ValueError("constraint_type must be 'monotone' or 'recovery'")
        if self.constraint_type == "recovery" and self.recovery_tau < 1:
            raise ValueError("recovery_tau must be >= 1 for recovery constraint")
        self.reset()

    @property
    def n_nodes(self) -> int:
        return self.model.n_nodes

    @property
    def m_edges(self) -> int:
        return self.model.m_edges

    @property
    def num_states(self) -> int:
        return (2**self.n_nodes) * self.flag_state_size

    @property
    def num_actions(self) -> int:
        return 2**self.m_edges

    @property
    def flag_state_size(self) -> int:
        if self.constraint_type == "monotone":
            return 2**self.m_edges
        return (self.recovery_tau + 1) ** self.m_edges

    def reset(self, initial_state: list[int] | None = None) -> int:
        if initial_state is None:
            state_id = int(self.rng.integers(0, 2**self.n_nodes))
            self.x = int_to_bits(state_id, self.n_nodes)
        else:
            if len(initial_state) != self.n_nodes:
                raise ValueError(f"initial_state must have length {self.n_nodes}")
            self.x = list(initial_state)
        self.h = [0] * self.m_edges
        self.t = 0
        self.trajectory = [self.x.copy()]
        self.violation_any = False
        self.success = False
        return self.extended_state_id(self.x, self.h)

    def allowed_actions(self) -> list[int]:
        if not self.action_masking:
            return list(range(self.num_actions))
        if self.constraint_type == "monotone":
            return allowed_actions_monotone(self.h)
        return allowed_actions_recovery(self.h)

    def goal_preserving_actions(self, actions: list[int] | None = None) -> list[int]:
        candidates = self.allowed_actions() if actions is None else list(actions)
        if not candidates:
            return candidates
        if not self.model.is_goal(self.x):
            return candidates

        preserving: list[int] = []
        for action in candidates:
            u = int_to_bits(action, self.m_edges)
            next_x = self.model.next_state(self.x, u)
            if self.model.is_goal(next_x):
                preserving.append(action)
        return preserving if preserving else candidates

    def extended_state_id(self, x: list[int], h: list[int]) -> int:
        return bits_to_int(x) * self.flag_state_size + self.flags_to_id(h)

    def flags_to_id(self, h: list[int]) -> int:
        if len(h) != self.m_edges:
            raise ValueError("h must have length m_edges")

        if self.constraint_type == "monotone":
            return bits_to_int(h)

        base = self.recovery_tau + 1
        value = 0
        for h_i in h:
            if h_i < 0 or h_i > self.recovery_tau:
                raise ValueError("recovery flags must be in [0, recovery_tau]")
            value = value * base + h_i
        return value

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        if action < 0 or action >= self.num_actions:
            raise ValueError("action out of range")

        u = int_to_bits(action, self.m_edges)
        if self.constraint_type == "monotone":
            violation = check_violation_monotone(self.h, u)
            next_h = update_flags_monotone(self.h, u)
        else:
            violation = check_violation_recovery(self.h, u)
            next_h = update_flags_recovery(self.h, u, self.recovery_tau)
        self.violation_any = self.violation_any or violation

        r_stab = self.reward.step if self.model.is_goal(self.x) else 0.0
        r_control = -self.reward.edge_cost * float(sum(u))
        newly_intervened = sum(
            1 for h_i, u_i in zip(self.h, u, strict=True) if h_i == 0 and u_i == 1
        )
        r_new_edge = -self.reward.new_edge_cost * float(newly_intervened)
        r_violation = -self.reward.violation_penalty if violation else 0.0
        reward = r_stab + r_control + r_new_edge + r_violation

        next_x = self.model.next_state(self.x, u)

        self.t += 1
        done = self.t >= self.horizon.max_steps

        self.x = next_x
        self.h = next_h
        self.trajectory.append(next_x.copy())

        if done:
            self.success = trajectory_satisfies_goal(
                self.trajectory,
                self.model.goal_states(),
                self.horizon.reach_horizon,
                self.horizon.max_steps,
            )
            if self.success:
                reward += self.reward.terminal

        info = {
            "t": self.t,
            "violation": violation,
            "success": self.success if done else False,
        }
        return self.extended_state_id(self.x, self.h), reward, done, info


def trajectory_satisfies_goal(
    trajectory: list[list[int]],
    goal_states: list[list[int]],
    reach_horizon: int,
    max_steps: int,
) -> bool:
    if len(trajectory) != max_steps + 1:
        raise ValueError("trajectory length must be max_steps + 1")
    goal_set = {tuple(state) for state in goal_states}
    for k in range(0, reach_horizon + 1):
        ok = True
        for j in range(0, max_steps - reach_horizon + 1):
            if tuple(trajectory[k + j]) not in goal_set:
                ok = False
                break
        if ok:
            return True
    return False
