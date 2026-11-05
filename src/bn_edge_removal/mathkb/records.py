"""Row and definition types of the `mathkb` seed tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One source row for the `sources` table."""

    source_key: str
    source_type: str
    title: str
    location: str
    anchor: str
    citation_text: str
    trust_level: str
    last_verified_on: str


@dataclass(frozen=True, slots=True)
class NetworkRecord:
    """One network row for the `networks` table."""

    network_key: str
    display_name: str
    domain: str
    n_nodes: int
    m_edges: int
    goal_state_count: int
    origin_type: str
    initial_state_condition_note: str
    notes: str


@dataclass(frozen=True, slots=True)
class ConstraintTypeRecord:
    """One constraint row for the `constraint_types` table."""

    constraint_key: str
    display_name: str
    parameter_schema: str
    flag_state_domain: str
    update_rule: str
    violation_rule: str
    allowed_action_rule: str
    default_parameters: str


@dataclass(frozen=True, slots=True)
class NetworkNodeRecord:
    """One node row for the `network_nodes` table."""

    node_key: str
    network_key: str
    node_index: int
    label: str
    description: str
    source_key: str


@dataclass(frozen=True, slots=True)
class NetworkEdgeRecord:
    """One edge row for the `network_edges` table."""

    edge_key: str
    network_key: str
    source_node: str
    target_node: str
    sign: str
    removable: bool
    removable_index: int | None
    edge_note: str
    source_key: str


@dataclass(frozen=True, slots=True)
class NetworkGoalStateRecord:
    """One goal-state row for the `network_goal_states` table."""

    goal_key: str
    network_key: str
    goal_label: str
    state_vector: str
    semantic_meaning: str
    source_key: str


@dataclass(frozen=True, slots=True)
class KnowledgeEntryRecord:
    """One atomic knowledge row for the `kb_entries` table."""

    canonical_slug: str
    title_ja: str
    title_en: str
    entry_type: str
    domain_area: str
    scope: str
    canonical_statement: str
    formal_expression: str
    plain_explanation: str
    applicability: str
    exception_note: str
    status: str
    confidence: float
    aliases: list[str]
    tags: list[str]
    updated_from_source_at: str
    source_keys: list[str]
    network_keys: list[str]
    constraint_keys: list[str]
    prerequisite_slugs: list[str]
    related_slugs: list[str]
    contradict_slugs: list[str]


@dataclass(frozen=True, slots=True)
class MathKbSeedBundle:
    """Container for all seed tables."""

    sources: list[SourceRecord]
    networks: list[NetworkRecord]
    constraint_types: list[ConstraintTypeRecord]
    network_nodes: list[NetworkNodeRecord]
    network_edges: list[NetworkEdgeRecord]
    network_goal_states: list[NetworkGoalStateRecord]
    kb_entries: list[KnowledgeEntryRecord]

    def as_table_mapping(self) -> dict[str, list[dict[str, Any]]]:
        """Convert the bundle to a table-to-row mapping."""

        return {
            "sources": [asdict(record) for record in self.sources],
            "networks": [asdict(record) for record in self.networks],
            "constraint_types": [asdict(record) for record in self.constraint_types],
            "network_nodes": [asdict(record) for record in self.network_nodes],
            "network_edges": [asdict(record) for record in self.network_edges],
            "network_goal_states": [
                asdict(record) for record in self.network_goal_states
            ],
            "kb_entries": [asdict(record) for record in self.kb_entries],
        }


@dataclass(frozen=True, slots=True)
class _RepoSourceDefinition:
    """Static definition for one repo-backed source row."""

    source_key: str
    source_type: str
    title: str
    relative_path: str
    anchor: str
    citation_text: str
    trust_level: str


@dataclass(frozen=True, slots=True)
class _PaperSourceDefinition:
    """Static definition for one external paper source row."""

    source_key: str
    title: str
    location: str
    citation_text: str
    trust_level: str


@dataclass(frozen=True, slots=True)
class _NetworkStaticMetadata:
    """Static metadata that is not encoded directly in YAML."""

    display_name: str
    domain: str
    origin_type: str
    notes: str


@dataclass(frozen=True, slots=True)
class _KnowledgeEntryDefinition:
    """Specification for one `kb_entries` row before timestamp expansion."""

    canonical_slug: str
    title_ja: str
    title_en: str
    entry_type: str
    domain_area: str
    scope: str
    canonical_statement: str
    formal_expression: str
    plain_explanation: str
    applicability: str
    exception_note: str
    status: str
    confidence: float
    aliases: list[str]
    tags: list[str]
    source_keys: list[str]
    network_keys: list[str]
    constraint_keys: list[str]
    prerequisite_slugs: list[str]
    related_slugs: list[str]
    contradict_slugs: list[str]
