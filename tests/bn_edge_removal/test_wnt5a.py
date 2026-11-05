from bn_edge_removal.models.wnt5a import Wnt5aModel


def test_wnt5a_goal_state_is_fixed_point():
    model = Wnt5aModel()
    x = [1, 0, 0, 0, 0, 0, 1]
    u = [0, 0, 0, 0, 0, 0, 0, 0]

    next_state = model.next_state(x, u)

    assert next_state == x


def test_wnt5a_edge_removal_changes_x2():
    model = Wnt5aModel()
    x = [0, 1, 0, 1, 0, 1, 0]

    u_active = [0, 0, 0, 0, 0, 0, 0, 0]
    next_active = model.next_state(x, u_active)
    assert next_active[1] == 1

    u_removed = [1, 1, 1, 0, 0, 0, 0, 0]
    next_removed = model.next_state(x, u_removed)
    assert next_removed[1] == 0
