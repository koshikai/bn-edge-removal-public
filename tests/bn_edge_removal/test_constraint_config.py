"""Tests for resolving constraint.type / constraint.monotone from configs."""

from __future__ import annotations

import pytest

from bn_edge_removal.control.env import resolve_constraint_type


def test_explicit_type_wins() -> None:
    assert resolve_constraint_type({"type": "recovery"}, default="monotone") == (
        "recovery"
    )
    assert resolve_constraint_type({"type": "monotone", "monotone": True}) == (
        "monotone"
    )


def test_monotone_flag_selects_monotone() -> None:
    assert resolve_constraint_type({"monotone": True}, default="recovery") == (
        "monotone"
    )


def test_default_used_when_nothing_is_specified() -> None:
    assert resolve_constraint_type({}, default="recovery") == "recovery"
    assert resolve_constraint_type({"action_masking": True}) == "monotone"


@pytest.mark.parametrize(
    "cfg",
    [
        {"monotone": False},
        {"monotone": True, "type": "recovery"},
        {"monotone": False, "type": "monotone"},
        {"monotone": "yes"},
        {"type": "bogus"},
    ],
)
def test_contradictory_or_invalid_settings_are_rejected(cfg: dict) -> None:
    with pytest.raises(ValueError, match="constraint"):
        resolve_constraint_type(cfg)
