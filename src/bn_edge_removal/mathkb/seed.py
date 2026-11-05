"""Build import-ready seed data for the `mathkb` NocoDB base.

The static content (sources, network metadata, constraint types, knowledge
entries) lives in ``mathkb/data/*.yaml``; this module combines it with the
network specs and writes the import tables.
"""

from __future__ import annotations

import csv
import json
from dataclasses import fields
from datetime import UTC, date, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from bn_edge_removal.mathkb.records import (
    ConstraintTypeRecord,
    KnowledgeEntryRecord,
    MathKbSeedBundle,
    NetworkEdgeRecord,
    NetworkGoalStateRecord,
    NetworkNodeRecord,
    NetworkRecord,
    SourceRecord,
    _KnowledgeEntryDefinition,
    _NetworkStaticMetadata,
    _PaperSourceDefinition,
    _RepoSourceDefinition,
)
from bn_edge_removal.models.network_spec import load_network_spec

REPO_ROOT = Path(__file__).resolve().parents[3]
NETWORK_SPEC_DIR = REPO_ROOT / "src" / "bn_edge_removal" / "models" / "network_specs"

CSV_LIST_DELIMITER = " | "


def _load_data(name: str) -> Any:
    """Load one YAML file from ``mathkb/data``."""
    text = files("bn_edge_removal.mathkb").joinpath("data", name).read_text("utf-8")
    return yaml.safe_load(text)


_SOURCES = _load_data("sources.yaml")
REPO_SOURCES: tuple[_RepoSourceDefinition, ...] = tuple(
    _RepoSourceDefinition(**row) for row in _SOURCES["repo"]
)
PAPER_SOURCES: tuple[_PaperSourceDefinition, ...] = tuple(
    _PaperSourceDefinition(**row) for row in _SOURCES["papers"]
)
_NETWORKS = _load_data("networks.yaml")
NETWORK_METADATA: dict[str, _NetworkStaticMetadata] = {
    key: _NetworkStaticMetadata(**row) for key, row in _NETWORKS["metadata"].items()
}
NETWORK_NODE_DESCRIPTIONS: dict[str, dict[int, str]] = _NETWORKS["node_descriptions"]
NETWORK_GOAL_MEANINGS: dict[str, dict[int, str]] = _NETWORKS["goal_meanings"]
_KB_ENTRY_DEFINITIONS: tuple[_KnowledgeEntryDefinition, ...] = tuple(
    _KnowledgeEntryDefinition(**row) for row in _load_data("kb_entries.yaml")
)


def build_mathkb_seed_bundle(today: date | None = None) -> MathKbSeedBundle:
    """Build the complete `mathkb` seed bundle.

    Args:
        today: Override for the `last_verified_on` date. Defaults to today.

    Returns:
        Fully populated seed bundle across all import tables.
    """

    effective_today = today or date.today()
    source_records = build_source_records(effective_today)
    source_map = {record.source_key: record for record in source_records}
    networks, nodes, edges, goal_states = build_network_records()
    constraints = build_constraint_type_records()
    kb_entries = build_kb_entry_records(source_map)
    return MathKbSeedBundle(
        sources=source_records,
        networks=networks,
        constraint_types=constraints,
        network_nodes=nodes,
        network_edges=edges,
        network_goal_states=goal_states,
        kb_entries=kb_entries,
    )


def build_source_records(today: date) -> list[SourceRecord]:
    """Build source rows for repo files and paper citations.

    Args:
        today: Date to record in `last_verified_on`.

    Returns:
        Source rows sorted by source key.
    """

    records: list[SourceRecord] = []
    for definition in REPO_SOURCES:
        path = REPO_ROOT / definition.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"source path not found: {path}")
        records.append(
            SourceRecord(
                source_key=definition.source_key,
                source_type=definition.source_type,
                title=definition.title,
                location=definition.relative_path,
                anchor=definition.anchor,
                citation_text=definition.citation_text,
                trust_level=definition.trust_level,
                last_verified_on=today.isoformat(),
            )
        )

    for definition in PAPER_SOURCES:
        records.append(
            SourceRecord(
                source_key=definition.source_key,
                source_type="paper",
                title=definition.title,
                location=definition.location,
                anchor="",
                citation_text=definition.citation_text,
                trust_level=definition.trust_level,
                last_verified_on=today.isoformat(),
            )
        )

    return sorted(records, key=lambda record: record.source_key)


def build_network_records() -> tuple[
    list[NetworkRecord],
    list[NetworkNodeRecord],
    list[NetworkEdgeRecord],
    list[NetworkGoalStateRecord],
]:
    """Build network-related rows from YAML specs.

    Returns:
        Tuple of rows for `networks`, `network_nodes`, `network_edges`,
        and `network_goal_states`.
    """

    networks: list[NetworkRecord] = []
    nodes: list[NetworkNodeRecord] = []
    edges: list[NetworkEdgeRecord] = []
    goal_states: list[NetworkGoalStateRecord] = []

    for network_key, metadata in sorted(NETWORK_METADATA.items()):
        raw_spec = _load_raw_network_yaml(network_key)
        spec = load_network_spec(network_key)
        source_key = f"code:{network_key}_yaml"
        networks.append(
            NetworkRecord(
                network_key=network_key,
                display_name=metadata.display_name,
                domain=metadata.domain,
                n_nodes=spec.n_nodes,
                m_edges=spec.m_edges,
                goal_state_count=len(spec.goal_states),
                origin_type=metadata.origin_type,
                initial_state_condition_note=str(
                    raw_spec.get("controllable_initial_states_note", "")
                ),
                notes=metadata.notes,
            )
        )

        labels = _network_node_labels(network_key, raw_spec, spec.n_nodes)
        descriptions = NETWORK_NODE_DESCRIPTIONS.get(network_key, {})
        for index, label in enumerate(labels, start=1):
            nodes.append(
                NetworkNodeRecord(
                    node_key=f"{network_key}:x{index:02d}",
                    network_key=network_key,
                    node_index=index,
                    label=label,
                    description=descriptions.get(index, ""),
                    source_key=source_key,
                )
            )

        for edge_index, edge in enumerate(spec.all_edges, start=1):
            source_label = labels[edge.source - 1]
            target_label = labels[edge.target - 1]
            removable_note = (
                f"Removable edge u{edge.removable_index}."
                if edge.removable and edge.removable_index is not None
                else "Fixed edge."
            )
            edges.append(
                NetworkEdgeRecord(
                    edge_key=f"{network_key}:edge{edge_index:03d}",
                    network_key=network_key,
                    source_node=f"{edge.source}:{source_label}",
                    target_node=f"{edge.target}:{target_label}",
                    sign=edge.sign,
                    removable=edge.removable,
                    removable_index=edge.removable_index,
                    edge_note=removable_note,
                    source_key=source_key,
                )
            )

        goal_meanings = NETWORK_GOAL_MEANINGS.get(network_key, {})
        for goal_index, goal_state in enumerate(spec.goal_states, start=1):
            goal_states.append(
                NetworkGoalStateRecord(
                    goal_key=f"{network_key}:goal{goal_index:02d}",
                    network_key=network_key,
                    goal_label=f"goal_{goal_index}",
                    state_vector=json.dumps(list(goal_state)),
                    semantic_meaning=goal_meanings.get(goal_index, ""),
                    source_key=source_key,
                )
            )

    return networks, nodes, edges, goal_states


def build_constraint_type_records() -> list[ConstraintTypeRecord]:
    """Build the fixed constraint-type rows.

    Returns:
        Rows for the `constraint_types` table.
    """

    return [ConstraintTypeRecord(**row) for row in _load_data("constraint_types.yaml")]


def build_kb_entry_records(
    source_map: dict[str, SourceRecord],
) -> list[KnowledgeEntryRecord]:
    """Build `kb_entries` rows with source-derived timestamps.

    Args:
        source_map: Map of source key to the corresponding source record.

    Returns:
        Knowledge entry rows sorted by canonical slug.
    """

    records: list[KnowledgeEntryRecord] = []
    for definition in _KB_ENTRY_DEFINITIONS:
        updated_from_source_at = _latest_source_timestamp(
            definition.source_keys, source_map
        )
        records.append(
            KnowledgeEntryRecord(
                canonical_slug=definition.canonical_slug,
                title_ja=definition.title_ja,
                title_en=definition.title_en,
                entry_type=definition.entry_type,
                domain_area=definition.domain_area,
                scope=definition.scope,
                canonical_statement=definition.canonical_statement,
                formal_expression=definition.formal_expression,
                plain_explanation=definition.plain_explanation,
                applicability=definition.applicability,
                exception_note=definition.exception_note,
                status=definition.status,
                confidence=definition.confidence,
                aliases=definition.aliases,
                tags=definition.tags,
                updated_from_source_at=updated_from_source_at,
                source_keys=definition.source_keys,
                network_keys=definition.network_keys,
                constraint_keys=definition.constraint_keys,
                prerequisite_slugs=definition.prerequisite_slugs,
                related_slugs=definition.related_slugs,
                contradict_slugs=definition.contradict_slugs,
            )
        )
    return sorted(records, key=lambda record: record.canonical_slug)


def export_mathkb_seed(
    output_dir: Path, today: date | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Export `mathkb` seed data to JSON and CSV files.

    Args:
        output_dir: Destination directory for exported seed files.
        today: Override for the `last_verified_on` date. Defaults to today.

    Returns:
        Serialized table mapping that was written to disk.
    """

    bundle = build_mathkb_seed_bundle(today=today)
    table_mapping = bundle.as_table_mapping()
    output_dir.mkdir(parents=True, exist_ok=True)

    for table_name, rows in table_mapping.items():
        _write_json(output_dir / f"{table_name}.json", rows)
        _write_csv(output_dir / f"{table_name}.csv", rows)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "tables": {table_name: len(rows) for table_name, rows in table_mapping.items()},
        "csv_list_delimiter": CSV_LIST_DELIMITER,
        "notes": [
            "List-valued columns are serialized to CSV with ' | ' as the delimiter.",
            "Link fields are represented as key lists and should be "
            "connected in NocoDB after import.",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return table_mapping


def _load_raw_network_yaml(network_key: str) -> dict[str, Any]:
    """Load one raw network YAML mapping.

    Args:
        network_key: Name of the network YAML file without extension.

    Returns:
        Parsed YAML mapping.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML content is not a mapping.
    """

    path = NETWORK_SPEC_DIR / f"{network_key}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"network YAML not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"network YAML must be a mapping: {path}")
    return raw


def _network_node_labels(
    network_key: str, raw_spec: dict[str, Any], n_nodes: int
) -> list[str]:
    """Resolve node labels, falling back to x1...xN.

    Args:
        network_key: Network identifier.
        raw_spec: Raw YAML mapping for the network.
        n_nodes: Number of nodes in the network.

    Returns:
        Node labels in index order.
    """

    labels = raw_spec.get("node_labels")
    if isinstance(labels, list) and len(labels) == n_nodes:
        return [str(label) for label in labels]
    return [f"x{index}" for index in range(1, n_nodes + 1)]


def _latest_source_timestamp(
    source_keys: list[str], source_map: dict[str, SourceRecord]
) -> str:
    """Compute the latest filesystem timestamp among repo-backed sources.

    Args:
        source_keys: Source keys linked from one KB entry.
        source_map: Map of source key to source metadata.

    Returns:
        ISO-8601 timestamp string in UTC.
    """

    timestamps: list[datetime] = []
    for source_key in source_keys:
        source = source_map[source_key]
        if source.source_type == "paper":
            continue
        path = REPO_ROOT / source.location
        if not path.exists():
            raise FileNotFoundError(f"linked source path not found: {path}")
        timestamps.append(datetime.fromtimestamp(path.stat().st_mtime, tz=UTC))

    if not timestamps:
        return datetime.now(UTC).isoformat()
    return max(timestamps).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable formatting.

    Args:
        path: Output file path.
        payload: Serializable object to write.
    """

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a CSV export for one table.

    Args:
        path: Output file path.
        rows: Serialized rows for one table.
    """

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row[key]) for key in fieldnames})


def _csv_value(value: Any) -> str | int | float | bool:
    """Convert one JSON value to a CSV-friendly scalar.

    Args:
        value: Source value from a serialized row.

    Returns:
        Scalar value suitable for csv.DictWriter.
    """

    if isinstance(value, list):
        return CSV_LIST_DELIMITER.join(str(item) for item in value)
    if value is None:
        return ""
    if isinstance(value, bool | int | float):
        return value
    return str(value)


def csv_headers_for_dataclass(cls: type[Any]) -> list[str]:
    """Expose field names for tests and docs.

    Args:
        cls: Dataclass type.

    Returns:
        Ordered field names from the dataclass definition.
    """

    return [field.name for field in fields(cls)]
