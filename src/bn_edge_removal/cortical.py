"""Cortical Development Boolean network with edge removal control."""

from __future__ import annotations

from dataclasses import dataclass

from bn_edge_removal.encoding import bits_to_int, int_to_bits
from bn_edge_removal.network_spec import load_network_spec

_CORTICAL_SPEC = load_network_spec("cortical")


@dataclass(frozen=True)
class CorticalModel:
    n_nodes: int = _CORTICAL_SPEC.n_nodes
    m_edges: int = _CORTICAL_SPEC.m_edges

    def next_state(self, x: list[int], u: list[int]) -> list[int]:
        if len(x) != self.n_nodes:
            raise ValueError(f"x must have length {self.n_nodes}")
        if len(u) != self.m_edges:
            raise ValueError(f"u must have length {self.m_edges}")

        x1, x2, x3, x4, x5 = [bool(v) for v in x]
        u1, u2, u3, u4, u5, u6 = [bool(v) for v in u]

        n1 = x5 and not (x2 and (not u1)) and not (x4 and (not u2))
        n2 = not (x5 and (not u3)) and (not x1) and (not x3)
        n3 = x1 and not (x2 and (not u4)) and not (x4 and (not u5))
        n4 = not (x5 and (not u6)) and (not x1) and (not x3)
        n5 = (not x2) and (not x4)

        return [int(n1), int(n2), int(n3), int(n4), int(n5)]

    @staticmethod
    def goal_states() -> list[list[int]]:
        return [list(state) for state in _CORTICAL_SPEC.goal_states]

    def is_goal(self, x: list[int]) -> bool:
        return x in self.goal_states()

    def state_to_id(self, x: list[int]) -> int:
        return bits_to_int(x)

    def id_to_state(self, value: int) -> list[int]:
        return int_to_bits(value, self.n_nodes)
