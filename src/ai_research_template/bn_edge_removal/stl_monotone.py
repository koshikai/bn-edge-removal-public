"""Monotone input specification utilities."""

from __future__ import annotations

from ai_research_template.bn_edge_removal.encoding import int_to_bits


def update_flags(h: list[int], u: list[int]) -> list[int]:
    if len(h) != len(u):
        raise ValueError("h and u must have the same length")
    return [int(h_i or u_i) for h_i, u_i in zip(h, u, strict=True)]


def check_violation(h: list[int], u: list[int]) -> bool:
    if len(h) != len(u):
        raise ValueError("h and u must have the same length")
    return any((h_i == 1 and u_i == 0) for h_i, u_i in zip(h, u, strict=True))


def allowed_actions(h: list[int]) -> list[int]:
    m = len(h)
    allowed: list[int] = []
    for action in range(2**m):
        u = int_to_bits(action, m)
        if all((h_i == 0 or u_i == 1) for h_i, u_i in zip(h, u, strict=True)):
            allowed.append(action)
    return allowed
