from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from ai_research_template.bn_edge_removal.reachability import (
    ReachabilityConfig,
    ReachabilityResult,
    verify_all_models,
)
from ai_research_template.utils import (
    current_timestamp,
    load_config,
    prepare_output_dir,
    setup_logger,
    write_report,
)


@dataclass(frozen=True)
class ConfigOverride:
    """Optional override values extracted from a YAML config file.

    Args:
        model_name: Target model name declared in the YAML (if any).
        max_steps: Override for maximum transition steps.
        constraint_type: Override for constraint type.
        recovery_tau: Override for recovery horizon.
    """

    model_name: str | None
    max_steps: int | None
    constraint_type: str | None
    recovery_tau: int | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify exact edge-removal reachability from all initial states "
            "with witness trajectories."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["cortical", "wnt5a", "cell_cycle10", "all"],
        help="Model to verify.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config for overriding max_steps / constraint settings.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional override for maximum transition steps.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["strict", "ignore", "both"],
        help="Verification mode.",
    )
    return parser.parse_args()


def _default_configs() -> dict[str, ReachabilityConfig]:
    return {
        "cell_cycle10": ReachabilityConfig(
            max_steps=20,
            constraint_type="monotone",
            recovery_tau=2,
            enforce_constraints=True,
        ),
        "cortical": ReachabilityConfig(
            max_steps=10,
            constraint_type="monotone",
            recovery_tau=2,
            enforce_constraints=True,
        ),
        "wnt5a": ReachabilityConfig(
            max_steps=10,
            constraint_type="recovery",
            recovery_tau=2,
            enforce_constraints=True,
        ),
    }


def _selected_models(model_option: str) -> list[str]:
    if model_option == "all":
        return ["cortical", "wnt5a", "cell_cycle10"]
    return [model_option]


def _selected_modes(mode_option: str) -> list[str]:
    if mode_option == "both":
        return ["strict", "ignore"]
    return [mode_option]


def _load_override(config_path: str | None) -> ConfigOverride:
    if config_path is None:
        return ConfigOverride(
            model_name=None,
            max_steps=None,
            constraint_type=None,
            recovery_tau=None,
        )

    raw = load_config(config_path)
    model_cfg = raw.get("model", {})
    horizon_cfg = raw.get("horizon", {})
    constraint_cfg = raw.get("constraint", {})

    model_name_raw = model_cfg.get("name")
    model_name = str(model_name_raw) if isinstance(model_name_raw, str) else None

    max_steps_raw = horizon_cfg.get("max_steps")
    max_steps = int(max_steps_raw) if max_steps_raw is not None else None

    constraint_type: str | None = None
    if "type" in constraint_cfg:
        constraint_type = str(constraint_cfg["type"])
    elif bool(constraint_cfg.get("monotone")):
        constraint_type = "monotone"

    recovery_tau_raw = constraint_cfg.get("recovery_tau")
    recovery_tau = int(recovery_tau_raw) if recovery_tau_raw is not None else None

    return ConfigOverride(
        model_name=model_name,
        max_steps=max_steps,
        constraint_type=constraint_type,
        recovery_tau=recovery_tau,
    )


def _build_configs(
    selected_models: list[str],
    override: ConfigOverride,
    max_steps_override: int | None,
) -> dict[str, ReachabilityConfig]:
    defaults = _default_configs()
    config_by_model: dict[str, ReachabilityConfig] = {}

    for model_name in selected_models:
        config = defaults[model_name]
        apply_override = override.model_name in {None, model_name}

        if apply_override:
            if override.max_steps is not None:
                config = replace(config, max_steps=override.max_steps)
            if override.constraint_type is not None:
                config = replace(config, constraint_type=override.constraint_type)
            if override.recovery_tau is not None:
                config = replace(config, recovery_tau=override.recovery_tau)

        if max_steps_override is not None:
            config = replace(config, max_steps=max_steps_override)

        config_by_model[model_name] = config

    return config_by_model


def _build_metrics(results: list[ReachabilityResult]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for result in results:
        failed_initial_state_ids = [
            item.initial_state_id
            for item in result.per_state_results
            if not item.reachable
        ]
        entries.append(
            {
                "model_name": result.model_name,
                "mode": result.mode,
                "success_count": result.success_count,
                "total_initial_states": result.total_initial_states,
                "global_reachable": result.global_reachable,
                "success_rate": (
                    result.success_count / result.total_initial_states
                    if result.total_initial_states
                    else 0.0
                ),
                "failed_initial_state_ids": failed_initial_state_ids,
            }
        )

    return {
        "results": entries,
        "all_global_reachable": all(item["global_reachable"] for item in entries),
    }


def _write_per_state_csv(output_dir: Path, results: list[ReachabilityResult]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        for item in result.per_state_results:
            witness = item.witness
            rows.append(
                {
                    "model_name": result.model_name,
                    "mode": result.mode,
                    "initial_state_id": item.initial_state_id,
                    "initial_state": "".join(str(bit) for bit in item.initial_state),
                    "reachable": item.reachable,
                    "goal_step": witness.goal_step if witness is not None else None,
                    "action_count": (
                        len(witness.actions) if witness is not None else None
                    ),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "artifacts" / "per_state_results.csv", index=False)


def _write_witness_jsonl(output_dir: Path, results: list[ReachabilityResult]) -> None:
    witness_path = output_dir / "artifacts" / "witnesses.jsonl"
    with witness_path.open("w", encoding="utf-8") as f:
        for result in results:
            for item in result.per_state_results:
                if item.witness is None:
                    continue
                payload = {
                    "model_name": result.model_name,
                    "mode": result.mode,
                    "initial_state_id": item.initial_state_id,
                    "initial_state": item.initial_state,
                    "witness": asdict(item.witness),
                }
                f.write(json.dumps(payload) + "\n")


def _write_metrics_json(output_dir: Path, metrics: dict[str, Any]) -> None:
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _write_summary_report(output_dir: Path, results: list[ReachabilityResult]) -> None:
    lines = [
        "# Reachability Verification Report",
        "",
        f"- Timestamp: {output_dir.name}",
        f"- Output: {output_dir}",
        "",
        "## Results",
    ]
    for result in results:
        lines.append(
            f"- {result.model_name} ({result.mode}): "
            f"{result.success_count}/{result.total_initial_states} "
            f"(global_reachable={result.global_reachable})"
        )
    write_report(lines, output_dir)


def main() -> None:
    args = _parse_args()
    selected_models = _selected_models(args.model)
    selected_modes = _selected_modes(args.mode)
    override = _load_override(args.config)
    config_by_model = _build_configs(selected_models, override, args.max_steps)

    timestamp = current_timestamp()
    output_dir = prepare_output_dir(
        experiment_name="reachability",
        output_root="outputs",
        timestamp=timestamp,
    )
    logger = setup_logger(output_dir, name="reachability")
    logger.info("Starting reachability verification")
    logger.info(f"Models: {selected_models}")
    logger.info(f"Modes: {selected_modes}")
    logger.info(f"Configs: {config_by_model}")

    results = verify_all_models(config_by_model=config_by_model, modes=selected_modes)

    metrics = _build_metrics(results)
    _write_metrics_json(output_dir, metrics)
    _write_per_state_csv(output_dir, results)
    _write_witness_jsonl(output_dir, results)
    _write_summary_report(output_dir, results)

    logger.info("Verification complete")
    logger.info(f"Output: {output_dir}")
    for result in results:
        logger.info(
            f"{result.model_name} ({result.mode}): "
            f"{result.success_count}/{result.total_initial_states}"
        )


if __name__ == "__main__":
    main()
