"""Environment for edge removal control with STL-style input constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from bn_edge_removal.control.stl_monotone import (
    allowed_actions as allowed_actions_monotone,
)
from bn_edge_removal.control.stl_monotone import (
    check_violation as check_violation_monotone,
)
from bn_edge_removal.control.stl_monotone import (
    update_flags as update_flags_monotone,
)
from bn_edge_removal.control.stl_recovery import (
    allowed_actions as allowed_actions_recovery,
)
from bn_edge_removal.control.stl_recovery import (
    check_violation as check_violation_recovery,
)
from bn_edge_removal.control.stl_recovery import (
    update_flags as update_flags_recovery,
)
from bn_edge_removal.models.encoding import bits_to_int, int_to_bits


class EdgeRemovalModel(Protocol):
    @property
    def n_nodes(self) -> int: ...

    @property
    def m_edges(self) -> int: ...

    def next_state(self, x: list[int], u: list[int]) -> list[int]: ...

    @staticmethod
    def goal_states() -> list[list[int]]: ...

    def is_goal(self, x: list[int]) -> bool: ...

    def state_to_id(self, x: list[int]) -> int: ...

    def id_to_state(self, value: int) -> list[int]: ...


@dataclass
class RewardConfig:
    """Reward weights of ``EdgeRemovalEnv.step``.

    The reward of one step taken in extended state ``(x_t, h_t)`` with input ``u_t``
    is ``r_stab(x_t) + r_control(u_t) + r_new_edge(h_t, u_t) + r_violation(h_t, u_t)``
    (plus ``terminal`` on the last step of a successful episode).

    Attributes:
        step: Stabilization reward, paid when the *pre-transition* state ``x_t`` is
            a goal state (``r_stab(x_t)``, as defined in the ITC-CSCC 2026 paper).
            An episode therefore pays it for ``x_0 .. x_{max_steps-1}``; the final
            state ``x_{max_steps}`` is rewarded only through ``terminal``.
        terminal: Bonus added to the last step when the whole trajectory satisfies
            ``trajectory_satisfies_goal``.
        edge_cost: Cost per intervened edge per step (``-edge_cost * |u_t|_1``).
        new_edge_cost: Cost per *intervention onset*, i.e. per edge with
            ``h_i == 0 and u_i == 1``.  What ``h_i == 0`` means depends on the
            constraint type, so the name is only literal for ``monotone``:

            - ``monotone``: ``h_i == 0`` means "never removed so far", so this is
              charged once per edge, when it is removed for the first time.
            - ``recovery``: ``h_i == 0`` means "not in the cooldown window", so it
              is charged again every time an edge is re-intervened after its
              cooldown.  Every non-violating intervention is an onset, hence under
              action masking it is equivalent to adding to ``edge_cost``.
        goal_onset_cost: Extra cost per intervention onset made while the
            pre-transition state is in the goal set
            (``-goal_onset_cost * sum_i 1[x_t in X_goal and h_i == 0 and u_i == 1]``).
            Unlike ``edge_cost``, it does not charge an intervention that merely
            keeps an edge removed (``h_i > 0`` under ``monotone``), so it prices
            *new* surgery at the target rather than the control effort itself.
        goal_edge_cost: Extra per-step cost per intervened edge while the
            pre-transition state is in the goal set
            (``-goal_edge_cost * |u_t|_1``).  This is the goal-conditional
            analogue of ``edge_cost``; under ``monotone`` it also charges edges
            that are only being kept removed, so staying at the goal becomes
            expensive.
        violation_penalty: Penalty when ``u_t`` violates the input constraint.
    """

    step: float = 1.0
    terminal: float = 5.0
    edge_cost: float = 1.0
    new_edge_cost: float = 0.0
    goal_onset_cost: float = 0.0
    goal_edge_cost: float = 0.0
    violation_penalty: float = 5.0


CONSTRAINT_TYPES = ("monotone", "recovery")

GOAL_SPECS = ("window", "first_entry")
"""Success conditions of an episode (``EdgeRemovalEnv.goal_spec``).

- ``window``: ``x_t`` is in the goal set for every ``t`` in ``[H, T]``
  (``trajectory_satisfies_goal``).  Leaving the goal set before ``H`` is allowed.
- ``first_entry``: the first goal visit ``t_reach`` satisfies ``t_reach <= H`` and
  ``x_t`` stays in the goal set for every ``t`` in ``[t_reach, T]``.  Every
  ``first_entry`` success is also a ``window`` success.
"""

GOAL_PENDING = 0
"""Goal monitor status: not decided yet (``first_entry``: no goal visit so far;
``window``: ``t < H``)."""
GOAL_HOLDING = 1
"""Goal monitor status: the success condition holds up to the current time."""
GOAL_FAILED = 2
"""Goal monitor status: the success condition can no longer hold."""
GOAL_STATUS_SIZE = 3


def resolve_constraint_type(
    constraint_cfg: dict[str, Any], default: str = "monotone"
) -> str:
    """Resolve the constraint type from the ``constraint`` section of a config.

    ``constraint.type`` is authoritative.  The legacy boolean ``constraint.monotone``
    (used by the cortical / cell_cycle10 configs) selects ``"monotone"`` when true.
    Settings that contradict each other are rejected instead of silently ignored.

    Args:
        constraint_cfg: The ``constraint`` mapping of an experiment config.
        default: Constraint type used when neither key is present.

    Returns:
        ``"monotone"`` or ``"recovery"``.

    Raises:
        ValueError: If the keys are invalid or contradict each other.
    """
    explicit = constraint_cfg.get("type")
    monotone_flag = constraint_cfg.get("monotone")
    if monotone_flag is not None and not isinstance(monotone_flag, bool):
        raise ValueError("constraint.monotone must be boolean")

    if explicit is None:
        if monotone_flag is None:
            resolved = default
        elif monotone_flag:
            resolved = "monotone"
        else:
            raise ValueError(
                "constraint.monotone is false but constraint.type is not set"
            )
    else:
        resolved = str(explicit)
        if monotone_flag is not None and monotone_flag != (resolved == "monotone"):
            raise ValueError(
                f"constraint.monotone={monotone_flag} contradicts "
                f"constraint.type={resolved!r}"
            )

    if resolved not in CONSTRAINT_TYPES:
        raise ValueError("constraint type must be 'monotone' or 'recovery'")
    return resolved


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
    goal_spec: str = "window"
    """Success condition, one of ``GOAL_SPECS``."""
    time_in_state: bool = False
    """Add ``t`` and the goal monitor status to the state id.

    ``first_entry`` always does, since its success depends on the history and the
    deadline ``H``.  For ``window`` it is optional and serves as the control arm
    with the same state information.
    """

    x: list[int] = field(init=False)
    h: list[int] = field(init=False)
    t: int = field(init=False)
    trajectory: list[list[int]] = field(init=False)
    violation_any: bool = field(init=False)
    success: bool = field(init=False)
    goal_status: int = field(init=False)
    _allowed_actions_cache: dict[tuple[str, tuple[int, ...]], tuple[int, ...]] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        if self.constraint_type not in CONSTRAINT_TYPES:
            raise ValueError("constraint_type must be 'monotone' or 'recovery'")
        if self.constraint_type == "recovery" and self.recovery_tau < 1:
            raise ValueError("recovery_tau must be >= 1 for recovery constraint")
        if self.goal_spec not in GOAL_SPECS:
            raise ValueError(f"goal_spec must be one of {GOAL_SPECS}")
        if self.goal_spec == "first_entry":
            self.time_in_state = True
        self.reset()

    @property
    def n_nodes(self) -> int:
        return self.model.n_nodes

    @property
    def m_edges(self) -> int:
        return self.model.m_edges

    @property
    def num_states(self) -> int:
        base = (2**self.n_nodes) * self.flag_state_size
        if not self.time_in_state:
            return base
        return base * (self.horizon.max_steps + 1) * GOAL_STATUS_SIZE

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
        self.goal_status = GOAL_PENDING
        self._update_goal_status()
        return self.state_id()

    def state_id(self) -> int:
        """Id of the current state used by the Q-table."""
        base = self.extended_state_id(self.x, self.h)
        if not self.time_in_state:
            return base
        return (
            base * (self.horizon.max_steps + 1) + self.t
        ) * GOAL_STATUS_SIZE + self.goal_status

    def _update_goal_status(self) -> None:
        """Advance the goal monitor with the current state ``x_t``."""
        if self.goal_status == GOAL_FAILED:
            return
        in_goal = self.model.is_goal(self.x)
        deadline = self.horizon.reach_horizon
        if self.goal_spec == "window":
            if self.t >= deadline:
                self.goal_status = GOAL_HOLDING if in_goal else GOAL_FAILED
            return
        if self.goal_status == GOAL_HOLDING:
            if not in_goal:
                self.goal_status = GOAL_FAILED
        elif in_goal:
            self.goal_status = GOAL_HOLDING
        elif self.t >= deadline:
            self.goal_status = GOAL_FAILED

    def allowed_actions(self) -> list[int]:
        if not self.action_masking:
            return list(range(self.num_actions))
        # Input admissibility depends only on flags, so repeated visits need no
        # repeated enumeration of the same input combinations (or BN queries).
        key = (self.constraint_type, tuple(self.h))
        cached = self._allowed_actions_cache.get(key)
        if cached is None:
            allowed = (
                allowed_actions_monotone(self.h)
                if self.constraint_type == "monotone"
                else allowed_actions_recovery(self.h)
            )
            cached = tuple(allowed)
            self._allowed_actions_cache[key] = cached
        return list(cached)

    def goal_preserving_actions(self, actions: list[int] | None = None) -> list[int]:
        """Restrict candidate actions to those that keep a goal state in the goal set.

        Outside the goal set every candidate is returned.  In a goal state only the
        candidates whose next state is again a goal state are returned; if there is
        none (the goal state cannot be kept by any candidate), all candidates are
        returned unchanged, so the filter never empties the action set.

        This is a model-based diagnostic retained for offline analysis only.
        Training and policy evaluation must use ``allowed_actions`` instead: the
        latter depends only on intervention history, without predicting successors.
        """
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

        # r_stab is attributed to the pre-transition state x_t (see RewardConfig).
        r_stab = self.reward.step if self.model.is_goal(self.x) else 0.0
        r_control = -self.reward.edge_cost * float(sum(u))
        intervention_onsets = sum(
            1 for h_i, u_i in zip(self.h, u, strict=True) if h_i == 0 and u_i == 1
        )
        r_new_edge = -self.reward.new_edge_cost * float(intervention_onsets)
        # Goal-conditional terms are attributed to the pre-transition state x_t.
        if self.model.is_goal(self.x):
            r_goal_onset = -self.reward.goal_onset_cost * float(intervention_onsets)
            r_goal_edge = -self.reward.goal_edge_cost * float(sum(u))
        else:
            r_goal_onset = 0.0
            r_goal_edge = 0.0
        r_violation = -self.reward.violation_penalty if violation else 0.0
        reward = (
            r_stab + r_control + r_new_edge + r_goal_onset + r_goal_edge + r_violation
        )

        next_x = self.model.next_state(self.x, u)

        self.t += 1
        done = self.t >= self.horizon.max_steps

        self.x = next_x
        self.h = next_h
        self.trajectory.append(next_x.copy())
        self._update_goal_status()

        if done:
            if self.goal_spec == "window":
                self.success = trajectory_satisfies_goal(
                    self.trajectory,
                    self.model.goal_states(),
                    self.horizon.reach_horizon,
                    self.horizon.max_steps,
                )
            else:
                self.success = self.goal_status == GOAL_HOLDING
            if self.success:
                reward += self.reward.terminal

        info = {
            "t": self.t,
            "violation": violation,
            "success": self.success if done else False,
        }
        return self.state_id(), reward, done, info


def unmaintainable_goal_states(model: EdgeRemovalModel) -> list[list[int]]:
    """Goal states that leave the goal set under every control input.

    This optional offline diagnostic is not called by environment initialization,
    training, or policy evaluation. Input constraints are ignored, so a state
    returned here cannot be maintained
    under any constraint type.  An empty list means every goal state has at least one
    action keeping the next state in the goal set (true in particular when all goal
    states are fixed points for ``u = 0``).
    """
    stuck: list[list[int]] = []
    for goal in model.goal_states():
        if not any(
            model.is_goal(model.next_state(goal, int_to_bits(action, model.m_edges)))
            for action in range(2**model.m_edges)
        ):
            stuck.append(goal)
    return stuck


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
        if all(tuple(trajectory[t]) in goal_set for t in range(k, max_steps + 1)):
            return True
    return False
