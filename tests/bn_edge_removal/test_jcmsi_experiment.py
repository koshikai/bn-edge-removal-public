import pytest

from bn_edge_removal.experiment.jcmsi import _initial_states
from bn_edge_removal.models.cortical import CorticalModel
from bn_edge_removal.models.wnt5a import Wnt5aModel


def test_wnt5a_defaults_to_the_controllable_pool():
    states, pool = _initial_states(Wnt5aModel(), {})

    assert pool == "controllable"
    assert len(states) == 96


def test_wnt5a_all_pool_covers_every_state():
    states, pool = _initial_states(Wnt5aModel(), {"initial_states": "all"})

    assert pool == "all"
    assert len(states) == 128


def test_cortical_always_uses_every_state():
    states, pool = _initial_states(CorticalModel(), {})

    assert pool == "all"
    assert len(states) == 32


def test_unknown_pool_is_rejected():
    with pytest.raises(ValueError):
        _initial_states(Wnt5aModel(), {"initial_states": "some_states"})


def test_evaluation_pool_falls_back_to_the_training_pool():
    settings = {"initial_states": "controllable"}

    states, pool = _initial_states(Wnt5aModel(), settings, key="evaluation_initial_states")

    assert pool == "controllable"
    assert len(states) == 96


def test_evaluation_pool_can_be_wider_than_the_training_pool():
    settings = {"initial_states": "controllable", "evaluation_initial_states": "all"}

    train_states, train_pool = _initial_states(Wnt5aModel(), settings)
    eval_states, eval_pool = _initial_states(
        Wnt5aModel(), settings, key="evaluation_initial_states"
    )

    assert (train_pool, len(train_states)) == ("controllable", 96)
    assert (eval_pool, len(eval_states)) == ("all", 128)


def test_goal_spec_diagnostics_scores_both_success_conditions():
    from bn_edge_removal.experiment.jcmsi import goal_spec_diagnostics

    # H = 2, T = 4.
    flags = [
        [False, True, True, True, True],  # reaches at 1 and stays: both succeed
        [True, False, True, True, True],  # leaves and re-enters: window only
        [False, False, False, True, True],  # reaches after H: neither
        [False, False, False, False, False],  # never reaches
    ]

    result = goal_spec_diagnostics(flags, reach_horizon=2)

    assert result == {
        "window_success_rate": 0.5,
        "first_entry_success_rate": 0.25,
        "avg_first_entry_time": 1.0,
        "avg_post_reach_exits": 0.25,
    }
