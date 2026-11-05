from bn_edge_removal.models.cortical import CorticalModel


def test_edge_removal_effect_on_x1():
    model = CorticalModel()
    x = [0, 1, 0, 0, 1]

    u_active = [0, 0, 0, 0, 0, 0]
    next_active = model.next_state(x, u_active)
    assert next_active[0] == 0

    u_removed = [1, 0, 0, 0, 0, 0]
    next_removed = model.next_state(x, u_removed)
    assert next_removed[0] == 1
