"""Wnt5a Boolean network with edge removal control."""

from __future__ import annotations

from dataclasses import dataclass

from bn_edge_removal.models.base import BooleanNetworkModel
from bn_edge_removal.models.network_spec import load_network_spec


@dataclass(frozen=True)
class Wnt5aModel(BooleanNetworkModel):
    spec = load_network_spec("wnt5a")

    def _update(self, x: list[bool], u: list[bool]) -> list[bool]:
        _x1, x2, x3, x4, _x5, x6, x7 = x
        u1, u2, u3, u4, u5, u6, u7, u8 = u

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

        return [n1, n2, n3, n4, n5, n6, n7]

    def controllable_initial_states(self) -> list[list[int]]:
        states: list[list[int]] = []
        for i in range(2**self.n_nodes):
            x = self.id_to_state(i)
            if x[1] == 0 or x[6] == 0:
                states.append(x)
        return states
