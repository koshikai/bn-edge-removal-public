"""Exact reachability verification for edge-removal controlled Boolean networks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache

from ai_research_template.bn_edge_removal.cell_cycle10 import CellCycle10Model
from ai_research_template.bn_edge_removal.cortical import CorticalModel
from ai_research_template.bn_edge_removal.encoding import int_to_bits
from ai_research_template.bn_edge_removal.env import EdgeRemovalModel
from ai_research_template.bn_edge_removal.stl_monotone import (
    check_violation as check_violation_monotone,
)
from ai_research_template.bn_edge_removal.stl_monotone import (
    update_flags as update_flags_monotone,
)
from ai_research_template.bn_edge_removal.stl_recovery import (
    check_violation as check_violation_recovery,
)
from ai_research_template.bn_edge_removal.stl_recovery import (
    update_flags as update_flags_recovery,
)
from ai_research_template.bn_edge_removal.wnt5a import Wnt5aModel

type BinaryState = tuple[int, ...]
type FlagState = tuple[int, ...]
type StrictKey = tuple[int, BinaryState, FlagState]
type IgnoreKey = tuple[int, BinaryState]

_VALID_CONSTRAINT_TYPES = {"monotone", "recovery"}
_VALID_MODES = {"strict", "ignore"}


@dataclass(frozen=True)
class ReachabilityConfig:
    """Configuration for exact reachability verification.

    Args:
        max_steps: Maximum number of transitions allowed from each initial state.
        constraint_type: Input constraint type (`monotone` or `recovery`).
        recovery_tau: Recovery horizon for `recovery` constraints.
        enforce_constraints: If True, disallow violating actions (`strict` mode).
    """

    max_steps: int
    constraint_type: str
    recovery_tau: int = 2
    enforce_constraints: bool = True


@dataclass(frozen=True)
class WitnessTrajectory:
    """One successful trajectory witness.

    Args:
        states: State sequence, including initial and reached goal states.
        actions: Action IDs applied between states.
        goal_step: First step index where the state belongs to the goal set.
    """

    states: list[list[int]]
    actions: list[int]
    goal_step: int


@dataclass(frozen=True)
class InitialStateReachability:
    """Reachability result for a single initial state.

    Args:
        initial_state_id: Integer ID of the initial state.
        initial_state: Binary initial state vector.
        reachable: Whether a goal is reachable within `max_steps`.
        witness: Successful witness trajectory if reachable, else None.
    """

    initial_state_id: int
    initial_state: list[int]
    reachable: bool
    witness: WitnessTrajectory | None


@dataclass(frozen=True)
class ReachabilityResult:
    """Aggregate result for one model under one mode.

    Args:
        model_name: Model identifier (`cortical`, `wnt5a`, or `cell_cycle10`).
        mode: Verification mode (`strict` or `ignore`).
        success_count: Number of initial states that can reach the goal set.
        total_initial_states: Total number of enumerated initial states.
        global_reachable: True if all initial states are reachable.
        per_state_results: Per-initial-state details.
    """

    model_name: str
    mode: str
    success_count: int
    total_initial_states: int
    global_reachable: bool
    per_state_results: list[InitialStateReachability]


@dataclass(frozen=True)
class _SearchContext:
    model: EdgeRemovalModel
    config: ReachabilityConfig
    goal_set: frozenset[BinaryState]
    all_actions: tuple[tuple[int, ...], ...]


def verify_all_initial_states(
    model: EdgeRemovalModel, config: ReachabilityConfig
) -> ReachabilityResult:
    """Verify reachability from every initial state for one model.

    Success means: the trajectory reaches any goal state at least once within
    `max_steps`.

    Args:
        model: Boolean-network model with edge-removal control input.
        config: Reachability verification configuration.

    Returns:
        Aggregated reachability result including per-state witnesses.

    Raises:
        ValueError: If configuration values are invalid.
        RuntimeError: If witness reconstruction fails unexpectedly.
    """

    _validate_config(config)
    context = _SearchContext(
        model=model,
        config=config,
        goal_set=frozenset(tuple(state) for state in model.goal_states()),
        all_actions=tuple(
            tuple(int_to_bits(action_id, model.m_edges))
            for action_id in range(2**model.m_edges)
        ),
    )

    if config.enforce_constraints:
        per_state = _verify_strict(context)
        mode = "strict"
    else:
        per_state = _verify_ignore(context)
        mode = "ignore"

    success_count = sum(1 for result in per_state if result.reachable)
    total_initial_states = 2**model.n_nodes

    return ReachabilityResult(
        model_name=_infer_model_name(model),
        mode=mode,
        success_count=success_count,
        total_initial_states=total_initial_states,
        global_reachable=success_count == total_initial_states,
        per_state_results=per_state,
    )


def verify_all_models(
    config_by_model: dict[str, ReachabilityConfig],
    modes: list[str] | tuple[str, ...],
) -> list[ReachabilityResult]:
    """Verify multiple models and modes in one call.

    Args:
        config_by_model: Mapping from model name to base configuration.
            Supported keys are `cortical`, `wnt5a`, and `cell_cycle10`.
        modes: Subset of `strict` and `ignore`.

    Returns:
        List of per-model, per-mode reachability results.

    Raises:
        ValueError: If an unknown model or mode is specified.
    """

    model_builders: dict[str, Callable[[], EdgeRemovalModel]] = {
        "cell_cycle10": CellCycle10Model,
        "cortical": CorticalModel,
        "wnt5a": Wnt5aModel,
    }
    normalized_modes = _normalize_modes(modes)
    results: list[ReachabilityResult] = []

    for model_name, base_config in config_by_model.items():
        builder = model_builders.get(model_name)
        if builder is None:
            raise ValueError(
                "unknown model "
                f"'{model_name}' (expected one of {sorted(model_builders)})"
            )
        model = builder()
        for mode in normalized_modes:
            run_config = replace(base_config, enforce_constraints=(mode == "strict"))
            results.append(verify_all_initial_states(model, run_config))

    return results


def _verify_strict(context: _SearchContext) -> list[InitialStateReachability]:
    policy_cache: dict[StrictKey, int] = {}
    zero_flags: FlagState = tuple(0 for _ in range(context.model.m_edges))

    @cache
    def can_reach(t: int, state: BinaryState, flags: FlagState) -> bool:
        if state in context.goal_set:
            return True
        if t >= context.config.max_steps:
            return False

        key = (t, state, flags)
        for action_id, action_bits in enumerate(context.all_actions):
            transition = _strict_transition(context, state, flags, action_bits)
            if transition is None:
                continue
            next_state, next_flags = transition
            if can_reach(t + 1, next_state, next_flags):
                policy_cache[key] = action_id
                return True
        return False

    per_state_results: list[InitialStateReachability] = []
    for initial_state_id in range(2**context.model.n_nodes):
        initial_state = tuple(context.model.id_to_state(initial_state_id))
        reachable = can_reach(0, initial_state, zero_flags)
        witness = (
            _build_witness_strict(
                context=context,
                policy_cache=policy_cache,
                initial_state=initial_state,
                zero_flags=zero_flags,
            )
            if reachable
            else None
        )
        per_state_results.append(
            InitialStateReachability(
                initial_state_id=initial_state_id,
                initial_state=list(initial_state),
                reachable=reachable,
                witness=witness,
            )
        )

    return per_state_results


def _verify_ignore(context: _SearchContext) -> list[InitialStateReachability]:
    policy_cache: dict[IgnoreKey, int] = {}

    @cache
    def can_reach(t: int, state: BinaryState) -> bool:
        if state in context.goal_set:
            return True
        if t >= context.config.max_steps:
            return False

        key = (t, state)
        for action_id, action_bits in enumerate(context.all_actions):
            next_state = tuple(context.model.next_state(list(state), list(action_bits)))
            if can_reach(t + 1, next_state):
                policy_cache[key] = action_id
                return True
        return False

    per_state_results: list[InitialStateReachability] = []
    for initial_state_id in range(2**context.model.n_nodes):
        initial_state = tuple(context.model.id_to_state(initial_state_id))
        reachable = can_reach(0, initial_state)
        witness = (
            _build_witness_ignore(
                context=context,
                policy_cache=policy_cache,
                initial_state=initial_state,
            )
            if reachable
            else None
        )
        per_state_results.append(
            InitialStateReachability(
                initial_state_id=initial_state_id,
                initial_state=list(initial_state),
                reachable=reachable,
                witness=witness,
            )
        )

    return per_state_results


def _build_witness_strict(
    context: _SearchContext,
    policy_cache: dict[StrictKey, int],
    initial_state: BinaryState,
    zero_flags: FlagState,
) -> WitnessTrajectory:
    if initial_state in context.goal_set:
        return WitnessTrajectory(
            states=[list(initial_state)],
            actions=[],
            goal_step=0,
        )

    state = initial_state
    flags = zero_flags
    states: list[list[int]] = [list(initial_state)]
    actions: list[int] = []

    for t in range(context.config.max_steps):
        key = (t, state, flags)
        action_id = policy_cache.get(key)
        if action_id is None:
            raise RuntimeError(
                "internal inconsistency: missing strict policy for reachable state"
            )
        action_bits = context.all_actions[action_id]
        transition = _strict_transition(context, state, flags, action_bits)
        if transition is None:
            raise RuntimeError(
                "internal inconsistency: strict witness selected violating action"
            )
        next_state, next_flags = transition
        actions.append(action_id)
        states.append(list(next_state))
        state = next_state
        flags = next_flags
        if state in context.goal_set:
            return WitnessTrajectory(states=states, actions=actions, goal_step=t + 1)

    raise RuntimeError(
        "internal inconsistency: strict witness reconstruction did not reach a goal"
    )


def _build_witness_ignore(
    context: _SearchContext,
    policy_cache: dict[IgnoreKey, int],
    initial_state: BinaryState,
) -> WitnessTrajectory:
    if initial_state in context.goal_set:
        return WitnessTrajectory(
            states=[list(initial_state)],
            actions=[],
            goal_step=0,
        )

    state = initial_state
    states: list[list[int]] = [list(initial_state)]
    actions: list[int] = []

    for t in range(context.config.max_steps):
        key = (t, state)
        action_id = policy_cache.get(key)
        if action_id is None:
            raise RuntimeError(
                "internal inconsistency: missing ignore policy for reachable state"
            )
        action_bits = context.all_actions[action_id]
        next_state = tuple(context.model.next_state(list(state), list(action_bits)))
        actions.append(action_id)
        states.append(list(next_state))
        state = next_state
        if state in context.goal_set:
            return WitnessTrajectory(states=states, actions=actions, goal_step=t + 1)

    raise RuntimeError(
        "internal inconsistency: ignore witness reconstruction did not reach a goal"
    )


def _strict_transition(
    context: _SearchContext,
    state: BinaryState,
    flags: FlagState,
    action_bits: tuple[int, ...],
) -> tuple[BinaryState, FlagState] | None:
    if context.config.constraint_type == "monotone":
        if check_violation_monotone(list(flags), list(action_bits)):
            return None
        next_flags = tuple(update_flags_monotone(list(flags), list(action_bits)))
    else:
        if check_violation_recovery(list(flags), list(action_bits)):
            return None
        next_flags = tuple(
            update_flags_recovery(
                list(flags), list(action_bits), context.config.recovery_tau
            )
        )

    next_state = tuple(context.model.next_state(list(state), list(action_bits)))
    return next_state, next_flags


def _infer_model_name(model: EdgeRemovalModel) -> str:
    if isinstance(model, CellCycle10Model):
        return "cell_cycle10"
    if isinstance(model, CorticalModel):
        return "cortical"
    if isinstance(model, Wnt5aModel):
        return "wnt5a"
    return model.__class__.__name__.lower()


def _validate_config(config: ReachabilityConfig) -> None:
    if config.max_steps < 0:
        raise ValueError("max_steps must be >= 0")
    if config.constraint_type not in _VALID_CONSTRAINT_TYPES:
        raise ValueError(
            f"constraint_type must be one of {sorted(_VALID_CONSTRAINT_TYPES)}"
        )
    if config.constraint_type == "recovery" and config.recovery_tau < 1:
        raise ValueError("recovery_tau must be >= 1 for recovery constraint")


def _normalize_modes(modes: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for mode in modes:
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}")
        if mode not in normalized:
            normalized.append(mode)
    if not normalized:
        raise ValueError("modes must be non-empty")
    return tuple(normalized)


__all__ = [
    "InitialStateReachability",
    "ReachabilityConfig",
    "ReachabilityResult",
    "WitnessTrajectory",
    "verify_all_initial_states",
    "verify_all_models",
]
