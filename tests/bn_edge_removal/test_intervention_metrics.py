"""Tests pinning the intervention-count metric definitions."""

from __future__ import annotations

import pytest

from bn_edge_removal.learning.intervention_metrics import (
    compute_intervention_metrics,
)


def _records(
    episodes: dict[int, tuple[list[int], list[int]]],
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    actions: list[dict[str, int]] = []
    trajectories: list[dict[str, int]] = []
    for init, (states, acts) in episodes.items():
        trajectories += [
            {"init_state": init, "t": t, "state_id": s} for t, s in enumerate(states)
        ]
        actions += [
            {"init_state": init, "t": t, "action": a} for t, a in enumerate(acts)
        ]
    return actions, trajectories


def test_intervention_metric_definitions() -> None:
    goal = 9
    actions, trajectories = _records(
        {
            # reaches the goal at t=2: actions at t=0,1 count "to goal".
            0: ([0, 1, goal, goal], [0b011, 0b100, 0b111]),
            # starts at the goal: nothing counts "to goal".
            goal: ([goal, goal, goal, goal], [0b001, 0b001, 0b001]),
            # never reaches the goal: every action counts "to goal".
            5: ([5, 5, 5, 5], [0b000, 0b010, 0b010]),
        }
    )
    metrics = compute_intervention_metrics(actions, trajectories, {goal})

    assert metrics["num_initial_states"] == 3.0
    # total bits per episode: 6, 3, 2
    assert metrics["avg_total_interventions"] == pytest.approx(11 / 3)
    # bits before the first goal hit (t >= 0): 3, 0, 2
    assert metrics["avg_interventions_to_goal"] == pytest.approx(5 / 3)
    # distinct edges ever removed: 3, 1, 1
    assert metrics["avg_unique_removed_edges"] == pytest.approx(5 / 3)
    # bits per (episode, step) row: 11 bits over 9 rows
    assert metrics["avg_bits_per_action_step"] == pytest.approx(11 / 9)


def test_intervention_metrics_ignore_record_order() -> None:
    actions, trajectories = _records({0: ([0, 1, 7], [0b01, 0b11])})
    forward = compute_intervention_metrics(actions, trajectories, {7})
    backward = compute_intervention_metrics(actions[::-1], trajectories[::-1], {7})

    assert forward == backward
    assert forward["avg_interventions_to_goal"] == 3.0


def test_intervention_metrics_reject_inconsistent_records() -> None:
    actions, trajectories = _records({0: ([0, 1, 7], [0b01, 0b11])})
    with pytest.raises(ValueError, match="no actions"):
        compute_intervention_metrics([], trajectories, {7})
    with pytest.raises(ValueError, match="trajectory"):
        compute_intervention_metrics(actions, trajectories[:1], {7})
