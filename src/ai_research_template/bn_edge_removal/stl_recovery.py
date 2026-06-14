"""Recovery-period STL input specification utilities."""

from __future__ import annotations

from ai_research_template.bn_edge_removal.encoding import int_to_bits


def update_flags(h: list[int], u: list[int], tau: int) -> list[int]:
    if tau < 1:
        raise ValueError("tau must be >= 1")
    if len(h) != len(u):
        raise ValueError("h and u must have the same length")

    updated: list[int] = []
    for h_i, u_i in zip(h, u, strict=True):
        if h_i < 0 or h_i > tau:
            raise ValueError("h elements must be in [0, tau]")
        if u_i == 1:
            updated.append(tau)
        else:
            updated.append(max(0, h_i - 1))
    return updated


def check_violation(h: list[int], u: list[int]) -> bool:
    if len(h) != len(u):
        raise ValueError("h and u must have the same length")
    return any((h_i > 0 and u_i == 1) for h_i, u_i in zip(h, u, strict=True))


def allowed_actions(h: list[int]) -> list[int]:
    m = len(h)
    allowed: list[int] = []
    for action in range(2**m):
        u = int_to_bits(action, m)
        if all((h_i == 0 or u_i == 0) for h_i, u_i in zip(h, u, strict=True)):
            allowed.append(action)
    return allowed
