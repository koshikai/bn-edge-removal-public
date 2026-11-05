import pytest

from bn_edge_removal.models.cell_cycle10 import CellCycle10Model


def test_cell_cycle10_next_state_matches_equations():
    model = CellCycle10Model()
    x = [1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
    u = [0, 0, 0]

    next_state = model.next_state(x, u)

    assert next_state == x


def test_cell_cycle10_edge_removal_changes_cycd():
    model = CellCycle10Model()
    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    next_active = model.next_state(x, [0, 0, 0])
    next_removed = model.next_state(x, [1, 0, 0])

    assert next_active[9] == 1
    assert next_removed[9] == 0


def test_cell_cycle10_rejects_wrong_action_length():
    model = CellCycle10Model()
    x = [0] * model.n_nodes

    with pytest.raises(ValueError, match="u must have length 3"):
        model.next_state(x, [])


def test_cell_cycle10_all_removed_has_non_goal_fixed_point():
    model = CellCycle10Model()
    goal = tuple(model.goal_states()[0])
    all_removed = [1, 1, 1]

    fixed_points: list[tuple[int, ...]] = []
    for sid in range(2**model.n_nodes):
        x = tuple(model.id_to_state(sid))
        if tuple(model.next_state(list(x), all_removed)) == x:
            fixed_points.append(x)

    assert goal in fixed_points
    assert any(fp != goal for fp in fixed_points)
