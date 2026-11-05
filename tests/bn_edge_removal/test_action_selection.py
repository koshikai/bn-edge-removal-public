import numpy as np

from bn_edge_removal.control.action_selection import (
    action_intervention_count,
    select_greedy_action_with_sparse_tiebreak,
)


def test_action_intervention_count_returns_bit_count():
    assert action_intervention_count(0b0) == 0
    assert action_intervention_count(0b1011) == 3


def test_select_greedy_action_prefers_fewer_interventions_on_tie():
    q_values = np.array([0.0, 1.0, 1.0, 1.0], dtype=float)
    actions = [1, 2, 3]
    # all have same Q=1.0; bit-counts are 1,1,2 -> choose 1 then smaller id.
    assert select_greedy_action_with_sparse_tiebreak(q_values, actions) == 1


def test_select_greedy_action_prefers_higher_q_before_sparsity():
    q_values = np.array([0.0, 0.9, 1.1, 1.0], dtype=float)
    actions = [1, 2, 3]
    assert select_greedy_action_with_sparse_tiebreak(q_values, actions) == 2
