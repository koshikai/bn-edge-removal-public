"""Cortical Development Boolean network with edge removal control."""

from __future__ import annotations

from dataclasses import dataclass

from bn_edge_removal.models.base import BooleanNetworkModel
from bn_edge_removal.models.network_spec import load_network_spec


@dataclass(frozen=True)
class CorticalModel(BooleanNetworkModel):
    spec = load_network_spec("cortical")

    def _update(self, x: list[bool], u: list[bool]) -> list[bool]:
        x1, x2, x3, x4, x5 = x
        u1, u2, u3, u4, u5, u6 = u

        n1 = x5 and not (x2 and (not u1)) and not (x4 and (not u2))
        n2 = not (x5 and (not u3)) and (not x1) and (not x3)
        n3 = x1 and not (x2 and (not u4)) and not (x4 and (not u5))
        n4 = not (x5 and (not u6)) and (not x1) and (not x3)
        n5 = (not x2) and (not x4)

        return [n1, n2, n3, n4, n5]
