"""Utility functions for experiment management.

This module provides helper functions for:
- Configuration loading (YAML files)
- Logging setup (file and console output)
- Result saving (JSON format)
- Experiment summary tracking (CSV format)
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def current_timestamp() -> str:
    """Return a filesystem-safe timestamp for experiment folders."""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def update_latest_symlink(latest_link: Path, target_dir: Path) -> None:
    """Update a symlink to point at the latest experiment output.

    Errors are ignored to keep the experiment flow robust on systems
    that don't support symlinks.
    """
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(
            target_dir.relative_to(latest_link.parent), target_is_directory=True
        )
    except Exception:
        # Fallback if symlink fails or filesystem doesn't support it.
        pass


def prepare_output_dir(
    experiment_name: str,
    output_root: Path | str = Path("outputs"),
    timestamp: str | None = None,
    update_latest: bool = True,
) -> Path:
    """Create a standardized output directory for an experiment.

    Layout:
      outputs/<experiment_name>/<timestamp>/
        logs/
        artifacts/
        metrics.json
        params.json
        report.md
    """
    output_root_path = Path(output_root)
    exp_root = output_root_path / experiment_name
    ts = timestamp or current_timestamp()
    output_dir = exp_root / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "artifacts").mkdir(exist_ok=True)

    if update_latest:
        update_latest_symlink(exp_root / "latest", output_dir)
        update_latest_symlink(output_root_path / "latest", output_dir)

    return output_dir


def setup_logger(output_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger that writes to both file and console.

    Creates a logger with two handlers:
    - File handler: writes detailed logs to {output_dir}/logs/experiment.log
    - Console handler: writes simplified messages to stdout

    Args:
        output_dir: Directory where the log file will be created.
        name: Name of the logger (default: "experiment").

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logger(Path("outputs/exp1"))
        >>> logger.info("Starting experiment")
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # File handler with detailed format
    fh = logging.FileHandler(log_dir / "experiment.log")
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(fh)

    # Console handler with simple format
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


def update_experiment_summary(results: dict[str, Any], output_dir: Path) -> None:
    """Update the global experiment summary CSV file.

    Appends a new row to experiments.csv with flattened results.
    Keys are prefixed to indicate their source:
    - param_*: Configuration parameters
    - metric_*: Evaluation metrics
    - model_*: Model parameters

    Args:
        results: Dictionary containing 'config', 'metrics', and 'model_params' keys.
        output_dir: Directory of the current experiment (used to derive CSV path).

    Note:
        The CSV file is located at outputs/experiments.csv.
        If the CSV already exists with different columns, new columns will be
        added but existing rows may have empty values for new columns.
    """
    summary_path = Path("outputs/experiments.csv")

    # Flatten the results for CSV
    flat_results: dict[str, Any] = {}

    # Add timestamp and path
    flat_results["timestamp"] = output_dir.name
    flat_results["path"] = str(output_dir)

    # Add config params (prefix with param_)
    for key, value in results.get("config", {}).items():
        if isinstance(value, (int, float, str, bool)):
            flat_results[f"param_{key}"] = value

    # Add metrics (prefix with metric_)
    for key, value in results.get("metrics", {}).items():
        if isinstance(value, (int, float, str, bool)):
            flat_results[f"metric_{key}"] = value

    # Add model params (prefix with model_)
    for key, value in results.get("model_params", {}).items():
        if isinstance(value, (int, float, str, bool)):
            flat_results[f"model_{key}"] = value

    # Read existing headers if file exists
    existing_headers: list[str] = []
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            reader = csv.reader(f)
            existing_headers = next(reader, [])

    # Merge headers: existing + new columns
    all_keys = set(existing_headers) | set(flat_results.keys())
    fieldnames = sorted(all_keys)

    # If file exists, read all rows and rewrite with updated headers
    rows: list[dict[str, Any]] = []
    if summary_path.exists() and existing_headers:
        with open(summary_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    # Append new result
    rows.append(flat_results)

    # Write all rows with complete fieldnames
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.

    Example:
        >>> config = load_config("configs/sample.yaml")
        >>> config["n_samples"]
        200
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_results(results: dict[str, Any], output_dir: Path) -> None:
    """Save experiment results as a JSON file.

    Creates the output directory if it doesn't exist and saves
    the results to metrics.json with pretty formatting.

    Args:
        results: Dictionary containing experiment results.
        output_dir: Directory where metrics.json will be saved.

    Example:
        >>> results = {"metrics": {"mse": 0.05}, "config": {"lr": 0.01}}
        >>> save_results(results, Path("outputs/exp1"))
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=4)


def save_params(params: dict[str, Any], output_dir: Path) -> None:
    """Save experiment parameters as a JSON file."""
    with open(output_dir / "params.json", "w") as f:
        json.dump(params, f, indent=4)


def write_report(lines: list[str], output_dir: Path) -> None:
    """Write a plain-text markdown report."""
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_daily_report_request(
    output_dir: Path,
    experiment_name: str,
    timestamp: str,
    config_path: str | None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Write a request file for an AI-authored daily report.

    The request includes paths and metadata needed to draft the daily log entry.
    """
    request = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "output_dir": str(output_dir),
        "report_path": str(output_dir / "report.md"),
        "params_path": str(output_dir / "params.json"),
        "metrics_path": str(output_dir / "metrics.json"),
        "config_path": config_path,
        "metrics": metrics or {},
    }
    request_path = output_dir / "report_request.json"
    with request_path.open("w", encoding="utf-8") as f:
        json.dump(request, f, indent=2)

    prompt_path = output_dir / "report_request.md"
    prompt_lines = [
        "# Daily Report Request",
        "",
        f"- Timestamp: {timestamp}",
        f"- Experiment: {experiment_name}",
        f"- Output: {output_dir}",
        f"- Report: {output_dir / 'report.md'}",
        f"- Params: {output_dir / 'params.json'}",
        f"- Metrics: {output_dir / 'metrics.json'}",
        f"- Config: {config_path or 'n/a'}",
        "",
        "Next step:",
        f"- Run: uv run poe daily-report --request {request_path}",
    ]
    prompt_path.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")

    return request_path


def append_daily_log(entry_lines: list[str], log_date: str | None = None) -> Path:
    """Append an entry to a date-based research log in docs/experiments.

    Args:
        entry_lines: Markdown lines describing the experiment outcome.
        log_date: Date string in YYYY-MM-DD format (default: today).

    Returns:
        The path to the log file that was written.
    """
    date_str = log_date or datetime.now().strftime("%Y-%m-%d")
    logs_dir = Path("docs/experiments")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{date_str}.md"

    if not log_path.exists():
        log_path.write_text(f"# {date_str}\n\n", encoding="utf-8")

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(entry_lines).rstrip() + "\n\n")

    return log_path
