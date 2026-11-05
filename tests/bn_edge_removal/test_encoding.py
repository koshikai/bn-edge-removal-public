import pytest

from bn_edge_removal.models.encoding import all_states, int_to_bits


def test_int_to_bits_width_zero():
    assert int_to_bits(0, 0) == []


def test_int_to_bits_width_zero_rejects_nonzero_value():
    with pytest.raises(ValueError, match="value does not fit in width"):
        int_to_bits(1, 0)


def test_all_states_width_zero():
    assert all_states(0) == [[]]
