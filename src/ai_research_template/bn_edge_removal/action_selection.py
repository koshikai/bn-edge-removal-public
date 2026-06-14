"""Action-selection helpers for sparse edge intervention policies."""

from __future__ import annotations

import numpy as np


def action_intervention_count(action: int) -> int:
    """Return the number of intervened edges in one action."""
    return int(action).bit_count()


def select_greedy_action_with_sparse_tiebreak(
    q_values: np.ndarray,
    actions: list[int],
) -> int:
    """Pick argmax Q action, preferring fewer interventions on ties.

    Tie-break order:
    1) higher Q-value
    2) fewer intervened edges (lower bit-count)
    3) smaller action id (deterministic)
    """
    if not actions:
        raise ValueError("actions must be non-empty")

    candidate_q = q_values[actions]
    max_q = float(np.max(candidate_q))
    best = [a for a, q in zip(actions, candidate_q, strict=True) if q == max_q]
    return min(best, key=lambda action: (action_intervention_count(action), action))
