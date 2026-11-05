from bn_edge_removal.control.stl_recovery import (
    allowed_actions,
    check_violation,
    update_flags,
)


def test_recovery_update_and_no_violation():
    h = [0, 2, 1]
    u = [1, 0, 0]

    assert check_violation(h, u) is False
    assert update_flags(h, u, tau=2) == [2, 1, 0]


def test_recovery_violation_and_allowed_actions():
    h = [1, 0]
    u = [1, 1]

    assert check_violation(h, u) is True
    assert allowed_actions([0, 2]) == [0, 2]
