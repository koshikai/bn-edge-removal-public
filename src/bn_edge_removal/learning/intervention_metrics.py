"""Intervention-count metrics computed from evaluation action records.

These are the metrics reported in
``docs/experiments/2026-02-18_itc_edge_cost_sweep.md`` and in the
``outputs/sweep_*.csv`` summaries.  They were originally produced by ad-hoc code
that was never committed; the definitions below reproduce the recorded values from
the ``artifacts/actions.csv`` and ``artifacts/trajectories.csv`` of each run.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from typing import Any


def compute_intervention_metrics(
    actions: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
    goal_state_ids: Collection[int],
) -> dict[str, float]:
    """Compute intervention-count metrics over all evaluated initial states.

    One episode is the set of records sharing ``init_state``.  ``bits(a)`` is the
    number of intervened (removed) edges in action ``a``, and the hitting time
    ``T`` of an episode is the first ``t >= 0`` with ``x(t)`` in the goal set.

    - ``avg_total_interventions``: mean over episodes of ``sum_t bits(a_t)``.
    - ``avg_interventions_to_goal``: mean over episodes of
      ``sum_{t < T} bits(a_t)``, the interventions applied strictly before the
      goal is first hit.  An episode that starts in a goal state contributes 0; an
      episode that never hits the goal contributes all of its interventions.
    - ``avg_unique_removed_edges``: mean over episodes of the number of distinct
      edges removed at least once (popcount of the bitwise OR of its actions).
    - ``avg_bits_per_action_step``: mean of ``bits(a_t)`` over all action records
      (equals ``avg_total_interventions / max_steps`` for equal-length episodes).

    Args:
        actions: Records with ``init_state``, ``t`` and ``action`` keys, as in
            ``EvaluationResult.actions`` / ``actions.csv``.
        trajectories: Records with ``init_state``, ``t`` and ``state_id`` keys, as
            in ``EvaluationResult.trajectories`` / ``trajectories.csv``.
        goal_state_ids: Integer ids of the goal states.

    Returns:
        The four metrics plus ``num_initial_states``.

    Raises:
        ValueError: If there are no actions, or an episode's trajectory does not
            cover its actions.
    """
    if not actions:
        raise ValueError("no actions to compute intervention metrics from")

    goal_ids = {int(state_id) for state_id in goal_state_ids}

    actions_by_init: dict[int, dict[int, int]] = defaultdict(dict)
    for row in actions:
        actions_by_init[int(row["init_state"])][int(row["t"])] = int(row["action"])

    states_by_init: dict[int, dict[int, int]] = defaultdict(dict)
    for row in trajectories:
        states_by_init[int(row["init_state"])][int(row["t"])] = int(row["state_id"])

    total_sum = 0
    to_goal_sum = 0
    unique_sum = 0
    for init, episode_actions in actions_by_init.items():
        states = states_by_init.get(init, {})
        if any(t not in states for t in episode_actions):
            raise ValueError(f"trajectory of init_state {init} does not cover actions")

        hit_times = [t for t, state_id in states.items() if state_id in goal_ids]
        hitting_time = min(hit_times) if hit_times else None

        removed_edges = 0
        for t, action in episode_actions.items():
            bits = action.bit_count()
            total_sum += bits
            if hitting_time is None or t < hitting_time:
                to_goal_sum += bits
            removed_edges |= action
        unique_sum += removed_edges.bit_count()

    episodes = len(actions_by_init)
    return {
        "avg_total_interventions": total_sum / episodes,
        "avg_interventions_to_goal": to_goal_sum / episodes,
        "avg_unique_removed_edges": unique_sum / episodes,
        "avg_bits_per_action_step": total_sum / len(actions),
        "num_initial_states": float(episodes),
    }
