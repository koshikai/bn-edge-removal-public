"""Common behaviour of the Boolean network models with edge-removal inputs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

from bn_edge_removal.models.encoding import bits_to_int, int_to_bits
from bn_edge_removal.models.network_spec import NetworkSpec


@dataclass(frozen=True)
class BooleanNetworkModel:
    """A Boolean network ``x(t+1) = f(x(t), u(t))`` defined by a network spec.

    Subclasses set ``spec`` and implement ``_update`` with the update rules.
    ``u_k = 1`` removes the ``k``-th removable edge for one step.
    """

    spec: ClassVar[NetworkSpec]

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self.spec.n_nodes

    @property
    def m_edges(self) -> int:
        """Number of removable edges (control inputs)."""
        return self.spec.m_edges

    def _update(self, x: list[bool], u: list[bool]) -> list[bool]:
        """Update rules on Boolean values (lengths already checked)."""
        raise NotImplementedError

    def next_state(self, x: list[int], u: list[int]) -> list[int]:
        """Return the successor of ``x`` under input ``u``.

        Raises:
            ValueError: If ``x`` or ``u`` has the wrong length.
        """
        if len(x) != self.n_nodes:
            raise ValueError(f"x must have length {self.n_nodes}")
        if len(u) != self.m_edges:
            raise ValueError(f"u must have length {self.m_edges}")
        return [
            int(v) for v in self._update([bool(v) for v in x], [bool(v) for v in u])
        ]

    @classmethod
    def goal_states(cls) -> list[list[int]]:
        """Goal states from the network spec."""
        return [list(state) for state in cls.spec.goal_states]

    @cached_property
    def _goal_set(self) -> frozenset[tuple[int, ...]]:
        return frozenset(tuple(state) for state in self.spec.goal_states)

    def is_goal(self, x: list[int]) -> bool:
        """Whether ``x`` is a goal state."""
        return tuple(x) in self._goal_set

    def state_to_id(self, x: list[int]) -> int:
        """Encode a state as an integer (first node is the most significant bit)."""
        return bits_to_int(x)

    def id_to_state(self, value: int) -> list[int]:
        """Decode an integer produced by ``state_to_id``."""
        return int_to_bits(value, self.n_nodes)
