"""10-node cell-cycle Boolean network model."""

from __future__ import annotations

from dataclasses import dataclass

from bn_edge_removal.encoding import bits_to_int, int_to_bits
from bn_edge_removal.network_spec import load_network_spec

_CELL_CYCLE10_SPEC = load_network_spec("cell_cycle10")


@dataclass(frozen=True)
class CellCycle10Model:
    n_nodes: int = _CELL_CYCLE10_SPEC.n_nodes
    m_edges: int = _CELL_CYCLE10_SPEC.m_edges

    def next_state(self, x: list[int], u: list[int]) -> list[int]:
        if len(x) != self.n_nodes:
            raise ValueError(f"x must have length {self.n_nodes}")
        if len(u) != self.m_edges:
            raise ValueError(f"u must have length {self.m_edges}")

        x1, x2, x3, x4, x5, x6, x7, x8, _x9, x10 = [bool(v) for v in x]
        u1, u2, u3 = [bool(v) for v in u]

        n1 = (not x10) and (not x3) and (not x4) and (not x8)
        n2 = (not x1) and (not x4) and (not x8)
        n3 = x2 and (not x1)
        n4 = x2 and (not x1) and (not x5) and (not x6) and (not x7)
        n5 = x8
        n6 = (
            (not x4)
            and (not x8)
            and (not (x3 and (not u2)))
            and (not (x5 and (not u3)))
        )
        n7 = not x6
        n8 = (not x5) and (not x6)
        n9 = (not x10) and (not x3) and (not x4) and (not x8)
        n10 = x10 and (not u1)

        return [
            int(n1),
            int(n2),
            int(n3),
            int(n4),
            int(n5),
            int(n6),
            int(n7),
            int(n8),
            int(n9),
            int(n10),
        ]

    @staticmethod
    def goal_states() -> list[list[int]]:
        return [list(state) for state in _CELL_CYCLE10_SPEC.goal_states]

    def is_goal(self, x: list[int]) -> bool:
        return x in self.goal_states()

    def state_to_id(self, x: list[int]) -> int:
        return bits_to_int(x)

    def id_to_state(self, value: int) -> list[int]:
        return int_to_bits(value, self.n_nodes)
