"""Fission yeast (S. pombe) cell cycle Boolean network with edge removal control.

Source of the update rules: Davidich & Bornholdt (2008) "Boolean Network Model
Predicts Cell Cycle Sequence of Fission Yeast", PLoS ONE 3(2): e1672 (Table 1,
Figure 1, thresholds read as +-0.5).

With ``u = 0`` the update rules equal the paper's threshold network for all 10
nodes and all 1024 states (27 dependencies, all listed in
``network_specs/s_pombe.yaml``; checked in
``tests/bn_edge_removal/test_s_pombe_source_paper.py``).  ``Start`` has a
self-degradation loop and no other input, so it is off after one step.

biodivine-boolean-models id-095 ``FISSION-YEAST-2008`` (``model.bnet``, from GINsim)
has the same rules for the other 9 nodes but treats ``Start`` as an input that keeps
its value; this module followed biodivine until 2026-10-05.

Control (project-defined, not part of the source): four inhibitions are removable,
one control bit per edge.

    u1: Ste9 -| Cdc2_Cdc13        u3: Slp1 -| Cdc2_Cdc13
    u2: Rum1 -| Cdc2_Cdc13        u4: Wee1_Mik1 -| Cdc2_Cdc13_star

``u_k`` changes only the update of the target of its own edge.  In particular the
inhibitions of ``Cdc2_Cdc13_star`` by Ste9 / Rum1 / Slp1 are separate, fixed edges.

Goal: the G1 stationary state ``[0, 0, 0, 0, 1, 1, 0, 0, 0, 1]`` (Ste9, Rum1 and
Wee1_Mik1 on).  It is a fixed point for ``u = 0``.  With the environment's success
rule (``trajectory_satisfies_goal``) the objective has two parts:

- reach: ``x(k)`` is the goal for some ``k <= reach_horizon``;
- maintain: ``x(t)`` stays in the goal set for every ``t`` in ``k..max_steps``.

Without control the goal is reached from 762 of the 1024 states (within 9 steps);
with control it is reachable from every state (within 9 steps).

This model is not used by any run script.
"""

from __future__ import annotations

from dataclasses import dataclass

from bn_edge_removal.models.base import BooleanNetworkModel
from bn_edge_removal.models.network_spec import load_network_spec


@dataclass(frozen=True)
class SPombeModel(BooleanNetworkModel):
    spec = load_network_spec("s_pombe")

    def _update(self, x: list[bool], u: list[bool]) -> list[bool]:
        start, sk, cdc2, cdc2s, ste9, rum1, slp1, pp, cdc25, wee1 = x
        u1, u2, u3, u4 = u

        # Self-degradation (a_ii = -1, no other input): off after one step.
        n_start = False

        n_sk = start

        n_cdc2 = (
            not (ste9 and not u1) and not (rum1 and not u2) and not (slp1 and not u3)
        )

        n_cdc2s = not ste9 and not rum1 and not slp1 and not (wee1 and not u4) and cdc25

        n_ste9 = self._update_ste9_rum1(sk, cdc2, ste9, cdc2s, pp)

        n_rum1 = self._update_ste9_rum1(sk, cdc2, rum1, cdc2s, pp)

        n_slp1 = cdc2s

        n_pp = slp1

        n_cdc25 = self._update_cdc25(cdc2, cdc25, pp)

        n_wee1 = self._update_wee1(cdc2, wee1, pp)

        return [
            n_start,
            n_sk,
            n_cdc2,
            n_cdc2s,
            n_ste9,
            n_rum1,
            n_slp1,
            n_pp,
            n_cdc25,
            n_wee1,
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
    def _update_cdc25(cdc2: bool, cdc25: bool, pp: bool) -> bool:
        return (
            (not cdc2 and cdc25 and not pp)
            or (cdc2 and not cdc25 and not pp)
            or (cdc2 and cdc25)
        )

    @staticmethod
    def _update_wee1(cdc2: bool, wee1: bool, pp: bool) -> bool:
        return (
            (not cdc2 and not wee1 and pp)
            or (not cdc2 and wee1)
            or (cdc2 and wee1 and pp)
        )
