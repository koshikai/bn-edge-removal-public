from bn_edge_removal.control.stl_monotone import (
    check_violation,
    update_flags,
)


def test_monotone_update_and_violation():
    h = [0, 1, 0]
    u = [1, 0, 1]
    assert check_violation(h, u) is True
    assert update_flags(h, u) == [1, 1, 1]
