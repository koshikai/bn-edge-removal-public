"""10-node cell-cycle Boolean network model."""

from __future__ import annotations

from dataclasses import dataclass

from bn_edge_removal.models.base import BooleanNetworkModel
from bn_edge_removal.models.network_spec import load_network_spec


@dataclass(frozen=True)
class CellCycle10Model(BooleanNetworkModel):
    spec = load_network_spec("cell_cycle10")

    def _update(self, x: list[bool], u: list[bool]) -> list[bool]:
        x1, x2, x3, x4, x5, x6, x7, x8, _x9, x10 = x
        u1, u2, u3 = u

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

        return [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]
