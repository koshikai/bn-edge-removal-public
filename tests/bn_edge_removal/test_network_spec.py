import warnings

import pytest

from bn_edge_removal.models.network_spec import (
    load_network_spec,
    parse_network_spec,
)


def test_cortical_network_spec():
    expected_n_nodes = 5
    expected_m_edges = 6
    expected_total_edges = 14
    fixed_activation_edge = (5, 1)
    spec = load_network_spec("cortical")

    assert spec.name == "cortical"
    assert spec.n_nodes == expected_n_nodes
    assert spec.m_edges == expected_m_edges
    assert spec.goal_states == ((1, 0, 1, 0, 1), (0, 1, 0, 1, 0))
    assert [edge.index for edge in spec.removable_edges] == [1, 2, 3, 4, 5, 6]
    assert len(spec.all_edges) == expected_total_edges
    assert sum(int(edge.removable) for edge in spec.all_edges) == expected_m_edges
    assert sorted(
        edge.removable_index for edge in spec.all_edges if edge.removable
    ) == [1, 2, 3, 4, 5, 6]
    assert {edge.sign for edge in spec.all_edges} == {"activation", "inhibition"}
    assert any(
        edge.source == fixed_activation_edge[0]
        and edge.target == fixed_activation_edge[1]
        and edge.sign == "activation"
        and edge.removable is False
        for edge in spec.all_edges
    )
    assert len(spec.update_equations) == expected_n_nodes
    assert spec.update_equations[0].startswith("x1'")


def test_wnt5a_network_spec():
    expected_n_nodes = 7
    expected_m_edges = 8
    expected_total_edges = 12
    fixed_self_loop = (7, 7)
    spec = load_network_spec("wnt5a")

    assert spec.name == "wnt5a"
    assert spec.n_nodes == expected_n_nodes
    assert spec.m_edges == expected_m_edges
    assert spec.goal_states == ((1, 0, 0, 0, 0, 0, 1), (0, 1, 0, 1, 1, 1, 1))
    assert [edge.sign for edge in spec.removable_edges] == [
        "activation",
        "activation",
        "activation",
        "activation",
        "inhibition",
        "activation",
        "activation",
        "inhibition",
    ]
    assert len(spec.all_edges) == expected_total_edges
    assert sum(int(edge.removable) for edge in spec.all_edges) == expected_m_edges
    assert sorted(
        edge.removable_index for edge in spec.all_edges if edge.removable
    ) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert {edge.sign for edge in spec.all_edges} == {"activation", "inhibition"}
    assert any(
        edge.source == fixed_self_loop[0]
        and edge.target == fixed_self_loop[1]
        and edge.sign == "activation"
        and edge.removable is False
        for edge in spec.all_edges
    )
    assert len(spec.update_equations) == expected_n_nodes
    assert "u1" in spec.update_equations[1]


def test_cell_cycle10_network_spec():
    spec = load_network_spec("cell_cycle10")

    assert spec.name == "cell_cycle10"
    assert spec.n_nodes == 10
    assert spec.m_edges == 3
    assert spec.goal_states == ((1, 0, 0, 0, 0, 1, 0, 0, 1, 0),)
    assert [edge.index for edge in spec.removable_edges] == [1, 2, 3]
    assert len(spec.all_edges) == 27
    assert sum(int(edge.removable) for edge in spec.all_edges) == spec.m_edges
    assert sorted(
        edge.removable_index for edge in spec.all_edges if edge.removable
    ) == [1, 2, 3]
    assert {edge.sign for edge in spec.all_edges} == {"activation", "inhibition"}
    assert len(spec.update_equations) == 10
    assert spec.update_equations[0].startswith("x01'")


def _minimal_spec_data() -> dict:
    return {
        "name": "tiny",
        "n_nodes": 2,
        "m_edges": 1,
        "goal_states": [[1, 0]],
        "removable_edges": [
            {"index": 1, "source": 1, "target": 2, "sign": "inhibition"}
        ],
        "update_equations": ["x1' = x1", "x2' = not(x1 and not u1)"],
    }


def test_parse_network_spec_accepts_minimal_spec():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = parse_network_spec(_minimal_spec_data(), "tiny")

    assert spec.goal_states == ((1, 0),)
    assert spec.node_labels == ()
    assert spec.reference is None
    assert spec.controllable_initial_states_note is None


@pytest.mark.parametrize("mutation", ["missing", "misspelled", "empty"])
def test_parse_network_spec_requires_goal_states(mutation):
    data = _minimal_spec_data()
    goal_states = data.pop("goal_states")
    if mutation == "misspelled":
        data["goal_state"] = goal_states
    elif mutation == "empty":
        data["goal_states"] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="goal_states"):
            parse_network_spec(data, "tiny")


def test_parse_network_spec_warns_on_unknown_keys():
    data = _minimal_spec_data()
    data["removeable_edges"] = []
    data["extra_note"] = "x"

    with pytest.warns(UserWarning, match="extra_note.*removeable_edges"):
        parse_network_spec(data, "tiny")


def test_parse_network_spec_reads_optional_metadata():
    data = _minimal_spec_data()
    data["node_labels"] = ["A", "B"]
    data["reference"] = "Someone (2026)"
    data["controllable_initial_states_note"] = "x1 == 0"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = parse_network_spec(data, "tiny")

    assert spec.node_labels == ("A", "B")
    assert spec.reference == "Someone (2026)"
    assert spec.controllable_initial_states_note == "x1 == 0"

    data["node_labels"] = ["A"]
    with pytest.raises(ValueError, match="node_labels"):
        parse_network_spec(data, "tiny")


@pytest.mark.parametrize("name", ["cortical", "wnt5a", "cell_cycle10", "s_pombe"])
def test_packaged_specs_load_without_warnings(name):
    load_network_spec.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = load_network_spec(name)

    assert spec.goal_states
    assert spec.m_edges == len(spec.removable_edges)


def test_packaged_spec_optional_metadata():
    assert load_network_spec("wnt5a").controllable_initial_states_note == (
        "x2 == 0 or x7 == 0"
    )
    assert load_network_spec("cell_cycle10").node_labels[0] == "Rb"
    assert load_network_spec("cortical").node_labels == ()
