"""Fission yeast (S. pombe) cell cycle Boolean network with edge removal control.

Based on: Davidich & Bornholdt (2008) "Boolean Network Model Predicts Cell Cycle
Sequence of Fission Yeast" PLoS ONE 3(2): e1672.
Boolean update rules from biodivine-boolean-models repository.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_research_template.bn_edge_removal.encoding import bits_to_int, int_to_bits
from ai_research_template.bn_edge_removal.network_spec import load_network_spec

_S_POMBE_SPEC = load_network_spec("s_pombe")


@dataclass(frozen=True)
class SPombeModel:
    n_nodes: int = _S_POMBE_SPEC.n_nodes
    m_edges: int = _S_POMBE_SPEC.m_edges

    def next_state(self, x: list[int], u: list[int]) -> list[int]:
        if len(x) != self.n_nodes:
            raise ValueError(f"x must have length {self.n_nodes}")
        if len(u) != self.m_edges:
            raise ValueError(f"u must have length {self.m_edges}")

        start, sk, cdc2, cdc2s, ste9, rum1, slp1, pp, cdc25, wee1 = [bool(v) for v in x]
        u1, u2, u3, u4, u5, _u6, _u7, _u8 = [bool(v) for v in u]

        n_start = start

        n_sk = start

        inhibited_cdc2 = (ste9 and not u1) or (rum1 and not u2) or (slp1 and not u3)
        n_cdc2 = not inhibited_cdc2

        inhibited_cdc2s = (
            (ste9 and not u1)
            or (rum1 and not u2)
            or (slp1 and not u3)
            or (wee1 and not u5)
            or not cdc25
            or (pp and not u4)
        )
        n_cdc2s = not inhibited_cdc2s

        n_ste9 = self._update_ste9_rum1(sk, cdc2, ste9, cdc2s, pp)

        n_rum1 = self._update_ste9_rum1(sk, cdc2, rum1, cdc2s, pp)

        n_slp1 = cdc2s

        n_pp = slp1

        n_cdc25 = ((not cdc2 and cdc25) or (cdc2 and cdc25)) and not pp

        n_wee1 = self._update_wee1(cdc2, wee1, pp)

        return [
            int(n_start),
            int(n_sk),
            int(n_cdc2),
            int(n_cdc2s),
            int(n_ste9),
            int(n_rum1),
            int(n_slp1),
            int(n_pp),
            int(n_cdc25),
            int(n_wee1),
        ]

    @staticmethod
    def _update_ste9_rum1(
        sk: bool, cdc2: bool, current: bool, cdc2s: bool, pp: bool
    ) -> bool:
        return (
            ((not sk and not cdc2 and not current and not cdc2s) and pp)
            or ((not sk and not cdc2 and current) and not cdc2s)
            or ((not sk and not cdc2 and current and cdc2s) and pp)
            or ((not sk and cdc2 and current and not cdc2s) and pp)
            or ((sk and not cdc2 and current and not cdc2s) and pp)
        )

    @staticmethod
    def _update_wee1(cdc2: bool, wee1: bool, pp: bool) -> bool:
        return (
            (not cdc2 and not wee1 and pp)
            or (not cdc2 and wee1)
            or (cdc2 and wee1 and pp)
        )

    @staticmethod
    def goal_states() -> list[list[int]]:
        return [list(state) for state in _S_POMBE_SPEC.goal_states]

    def is_goal(self, x: list[int]) -> bool:
        return x in self.goal_states()

    def state_to_id(self, x: list[int]) -> int:
        return bits_to_int(x)

    def id_to_state(self, value: int) -> list[int]:
        return int_to_bits(value, self.n_nodes)
