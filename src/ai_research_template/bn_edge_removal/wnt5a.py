"""Wnt5a Boolean network with edge removal control."""

from __future__ import annotations

from dataclasses import dataclass

from ai_research_template.bn_edge_removal.encoding import bits_to_int, int_to_bits
from ai_research_template.bn_edge_removal.network_spec import load_network_spec

_WNT5A_SPEC = load_network_spec("wnt5a")


@dataclass(frozen=True)
class Wnt5aModel:
    n_nodes: int = _WNT5A_SPEC.n_nodes
    m_edges: int = _WNT5A_SPEC.m_edges

    def next_state(self, x: list[int], u: list[int]) -> list[int]:
        if len(x) != self.n_nodes:
            raise ValueError(f"x must have length {self.n_nodes}")
        if len(u) != self.m_edges:
            raise ValueError(f"u must have length {self.m_edges}")

        _x1, x2, x3, x4, _x5, x6, x7 = [bool(v) for v in x]
        u1, u2, u3, u4, u5, u6, u7, u8 = [bool(v) for v in u]

        n1 = not x6
        n2 = (
            ((x2 and (not u1)) and (x4 and (not u2)))
            or ((x4 and (not u2)) and (x6 and (not u3)))
            or ((x6 and (not u3)) and (x2 and (not u1)))
        )
        n3 = not x7
        n4 = x4
        n5 = (x2 and (not u4)) or (not (x7 and (not u5)))
        n6 = (x3 and (not u6)) or (x4 and (not u7))
        n7 = (not (x2 and (not u8))) or x7

        return [int(n1), int(n2), int(n3), int(n4), int(n5), int(n6), int(n7)]

    @staticmethod
    def goal_states() -> list[list[int]]:
        return [list(state) for state in _WNT5A_SPEC.goal_states]

    def is_goal(self, x: list[int]) -> bool:
        return x in self.goal_states()

    def state_to_id(self, x: list[int]) -> int:
        return bits_to_int(x)

    def id_to_state(self, value: int) -> list[int]:
        return int_to_bits(value, self.n_nodes)

    def controllable_initial_states(self) -> list[list[int]]:
        states: list[list[int]] = []
        for i in range(2**self.n_nodes):
            x = self.id_to_state(i)
            if x[1] == 0 or x[6] == 0:
                states.append(x)
        return states
