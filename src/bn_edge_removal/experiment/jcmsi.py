"""Reproducible hard-constraint experiments without transition lookahead."""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bn_edge_removal.control.env import EdgeRemovalEnv, HorizonConfig, RewardConfig
from bn_edge_removal.experiment.utils import current_timestamp
from bn_edge_removal.learning.evaluation import (
    EvaluationResult,
    evaluate_policy,
    evaluate_policy_sparse,
)
from bn_edge_removal.learning.q_learning import (
    EpsilonSchedule,
    QLearningConfig,
    train_q_learning,
    train_q_learning_sparse,
)
from bn_edge_removal.models.cortical import CorticalModel
from bn_edge_removal.models.wnt5a import Wnt5aModel


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def goal_spec_diagnostics(
    goal_flags_by_episode: list[list[bool]], reach_horizon: int
) -> dict[str, float]:
    """Score evaluation trajectories under both success conditions.

    The policy is scored as rolled out, whatever ``goal_spec`` it was trained
    with.  Input violations are not taken into account (they are zero under
    action masking; see ``violation_rate``).

    Args:
        goal_flags_by_episode: ``flags[t]`` tells whether ``x(t)`` is a goal state,
            one list per evaluated initial state.
        reach_horizon: Deadline ``H``.

    Returns:
        ``window_success_rate`` (in the goal on ``[H, T]``),
        ``first_entry_success_rate`` (first visit ``t_reach <= H`` and no exit
        afterwards), ``avg_first_entry_time`` (mean ``t_reach`` over first-entry
        successes, -1.0 when there is none) and ``avg_post_reach_exits`` (exits
        from the goal set after the first visit, averaged over all initial
        states).
    """
    window = 0
    first_entry_times: list[int] = []
    exits = 0
    for flags in goal_flags_by_episode:
        window += all(flags[reach_horizon:])
        t_reach = next((t for t, flag in enumerate(flags) if flag), None)
        if t_reach is None:
            continue
        exits += sum(
            1 for t in range(t_reach, len(flags) - 1) if flags[t] and not flags[t + 1]
        )
        if t_reach <= reach_horizon and all(flags[t_reach:]):
            first_entry_times.append(t_reach)
    total = len(goal_flags_by_episode)
    return {
        "window_success_rate": window / total,
        "first_entry_success_rate": len(first_entry_times) / total,
        "avg_first_entry_time": (
            float(np.mean(first_entry_times)) if first_entry_times else -1.0
        ),
        "avg_post_reach_exits": exits / total,
    }


def _save_evaluation(
    output: Path,
    result: EvaluationResult,
    history: list[dict[str, float]],
    env: EdgeRemovalEnv,
) -> dict[str, float]:
    """Save observed trajectories and compute intervention and goal metrics."""
    artifacts = output / "artifacts"
    artifacts.mkdir(parents=True)
    actions = pd.DataFrame(result.actions)
    pd.DataFrame(history).to_csv(artifacts / "training.csv", index=False)
    pd.DataFrame(result.trajectories).to_csv(
        artifacts / "trajectories.csv", index=False
    )
    actions.to_csv(artifacts / "actions.csv", index=False)
    interventions = actions["action"].map(lambda action: int(action).bit_count())
    metrics = dict(result.metrics)
    metrics["avg_total_interventions"] = float(
        interventions.sum() / metrics["num_initial_states"]
    )
    metrics["training_violation_rate"] = float(
        np.mean([row["violation"] for row in history])
    )
    goal_ids = {env.model.state_to_id(state) for state in env.model.goal_states()}
    trajectories = pd.DataFrame(result.trajectories).sort_values(["init_state", "t"])
    goal_flags = [
        group["state_id"].isin(goal_ids).tolist()
        for _, group in trajectories.groupby("init_state", sort=False)
    ]
    metrics.update(goal_spec_diagnostics(goal_flags, env.horizon.reach_horizon))
    return metrics


def _initial_states(
    model: CorticalModel | Wnt5aModel,
    settings: dict[str, Any],
    key: str = "initial_states",
) -> tuple[list[list[int]], str]:
    """Select the initial-state pool used for training or evaluation.

    Args:
        model: The network under control.
        settings: Per-model settings from the config.
        key: ``"initial_states"`` for the training pool, ``"evaluation_initial_states"``
            for the evaluation pool.

    Returns:
        The initial states and the name of the selected pool.

    Notes:
        The pool is ``"all"`` (every state of the network) or ``"controllable"`` (the
        states listed in the network spec).  Wnt5a defaults to ``"controllable"``;
        the cortical model always uses all its states.  The evaluation pool falls back
        to the training pool when it is not given.
    """
    if isinstance(model, CorticalModel):
        return [model.id_to_state(i) for i in range(2**model.n_nodes)], "all"
    default = (
        "controllable"
        if key == "initial_states"
        else settings.get("initial_states", "controllable")
    )
    pool = str(settings.get(key, default))
    if pool == "all":
        return [model.id_to_state(i) for i in range(2**model.n_nodes)], "all"
    if pool == "controllable":
        return model.controllable_initial_states(), "controllable"
    raise ValueError(f"{key} must be 'all' or 'controllable'")


def _run_seed(
    config: dict[str, Any], model_name: str, seed: int, output: Path
) -> dict[str, float]:
    """Train and exhaustively evaluate one independent seeded policy."""
    settings = config["models"][model_name]
    model = CorticalModel() if model_name == "cortical" else Wnt5aModel()
    initial_states, initial_state_pool = _initial_states(model, settings)
    evaluation_states, evaluation_state_pool = _initial_states(
        model, settings, key="evaluation_initial_states"
    )
    rng = np.random.default_rng(seed)
    env = EdgeRemovalEnv(
        model=model,
        reward=RewardConfig(**config["reward"]),
        horizon=HorizonConfig(
            reach_horizon=int(settings["reach_horizon"]),
            max_steps=int(config["horizon"]["max_steps"]),
        ),
        action_masking=True,
        constraint_type=str(settings["constraint_type"]),
        recovery_tau=int(settings.get("recovery_tau", 2)),
        rng=rng,
        goal_spec=str(config["horizon"].get("goal_spec", "window")),
        time_in_state=bool(config["horizon"].get("time_in_state", False)),
    )
    learning = QLearningConfig(
        alpha=float(config["learning"]["alpha"]),
        gamma=float(config["learning"]["gamma"]),
        episodes=int(settings["episodes"]),
        epsilon=EpsilonSchedule(**config["learning"]["epsilon"]),
        n_step=int(config["learning"].get("n_step", 1)),
    )
    output.mkdir(parents=True)
    if isinstance(model, CorticalModel):
        dense, history = train_q_learning(
            env, learning, rng, initial_states, initial_state_strategy="cycle"
        )
        result = evaluate_policy(env, dense, evaluation_states)
        np.save(output / "q_table.npy", dense)
        baseline = evaluate_policy(env, np.zeros_like(dense), evaluation_states)
    else:
        sparse, history = train_q_learning_sparse(
            env, learning, rng, initial_states, initial_state_strategy="cycle"
        )
        result = evaluate_policy_sparse(env, sparse, evaluation_states)
        ids = sorted(sparse)
        np.savez_compressed(
            output / "q_table_sparse.npz",
            state_ids=np.array(ids),
            q_values=np.vstack([sparse[state_id] for state_id in ids]),
        )
        baseline = evaluate_policy_sparse(env, {}, evaluation_states)
    metrics = _save_evaluation(output, result, history, env)
    run_config = {
        "seed": seed,
        "model": {
            "name": model_name,
            "n_nodes": model.n_nodes,
            "m_edges": model.m_edges,
            "goal_states": model.goal_states(),
        },
        "horizon": {
            "reach_horizon": env.horizon.reach_horizon,
            "max_steps": env.horizon.max_steps,
            "goal_spec": env.goal_spec,
            "time_in_state": env.time_in_state,
        },
        "constraint": {
            "type": env.constraint_type,
            "recovery_tau": env.recovery_tau,
            "action_masking": True,
        },
        "reward": config["reward"],
        "learning": {
            **config["learning"],
            "episodes": learning.episodes,
            "n_step": learning.n_step,
        },
        "initial_state_strategy": "cycle",
        "initial_state_pool": initial_state_pool,
        "evaluation_initial_state_pool": evaluation_state_pool,
        "goal_transition_lookahead": False,
    }
    _save_json(
        output / "metrics.json",
        {"config": run_config, "metrics": metrics, "baseline": baseline.metrics},
    )
    print(f"{model_name} seed={seed}: {json.dumps(metrics)}", flush=True)
    return metrics


def run_experiments(config_path: Path) -> None:
    """Run every configured seed and save parameters and aggregate results.

    Args:
        config_path: YAML file containing learning parameters and both models.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_dir"]) / current_timestamp()
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    package_root = Path(__file__).resolve().parents[1]
    control_files = (
        "control/env.py",
        "learning/q_learning.py",
        "learning/evaluation.py",
        "control/action_selection.py",
        "control/stl_monotone.py",
        "control/stl_recovery.py",
    )
    _save_json(
        output / "provenance.json",
        {
            "base_commit": revision,
            "config": config,
            "control_source_sha256": {
                name: sha256((package_root / name).read_bytes()).hexdigest()
                for name in control_files
            },
        },
    )
    patch = subprocess.run(
        ["git", "diff", "HEAD"], capture_output=True, text=True, check=True
    ).stdout
    (output / "source.patch").write_text(patch, encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for model_name in config["models"]:
        if model_name not in {"cortical", "wnt5a"}:
            raise ValueError(f"Unsupported model: {model_name}")
        for seed in config["seeds"]:
            metrics = _run_seed(
                config, model_name, int(seed), output / model_name / f"seed{seed}"
            )
            rows.append({"model": model_name, "seed": seed, **metrics})
            pd.DataFrame(rows).to_csv(output / "per_seed.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.drop(columns="seed").groupby("model").agg(["mean", "std"]).to_csv(
        output / "summary.csv"
    )
    print(f"Results: {output}", flush=True)


def export_manuscript_results(run_dir: Path, destination: Path) -> None:
    """Export measured results and their provenance for the manuscript.

    Args:
        run_dir: Completed experiment suite directory.
        destination: Manuscript results directory.
    """
    frame = pd.read_csv(run_dir / "per_seed.csv")
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "per_seed.csv", index=False)
    for name in ("config.yaml", "provenance.json"):
        (destination / name).write_text(
            (run_dir / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    lines = [
        "% Generated from " + run_dir.as_posix() + "/per_seed.csv; do not edit values.",
        r"Table~\ref{tab:results} reports the learned masked policies without",
        r"successor-state filtering. Each metric is first averaged over the",
        r"prescribed initial states, then summarized across the five seeds",
        r"using the sample standard deviation ($N-1$ denominator).",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Model-free masked-policy evaluation (mean $\pm$ SD).",
        r"  Success requires arrival by $H$, maintenance through $T=10$, and no",
        r"  input violations. First step counts the first goal visit at $t\ge1$;",
        r"  interventions are removed edges summed over the ten control steps.}",
        r"  \label{tab:results}",
        r"  \begin{tabular}{lcccc}",
        r"    \hline",
        r"    Model & Initial states & Success (\%) & First step & Interventions \\",
        r"    \hline",
    ]
    for name, group in frame.groupby("model", sort=False):
        first = group["avg_first_goal_step"]
        interventions = group["avg_total_interventions"]
        success = 100 * group["success_rate"]
        lines.append(
            f"    {str(name).capitalize()} & {int(group['num_initial_states'].iloc[0])}"
            f" & ${success.mean():.1f}\\pm{success.std(ddof=1):.1f}$"
            f" & ${first.mean():.2f}\\pm{first.std(ddof=1):.2f}$"
            f" & ${interventions.mean():.1f}\\pm{interventions.std(ddof=1):.1f}$ \\\\"
        )
    lines.extend([r"    \hline", r"  \end{tabular}", r"\end{table}", ""])
    if (frame[["training_violation_rate", "violation_rate"]] == 0).all().all():
        lines.extend(
            [
                "No input violations occurred during training or evaluation"
                " in any run.",
                "These observations agree with the input-admissibility guarantee;",
                "target-set satisfaction remains an empirical result of learning.",
            ]
        )
    (destination / "hard_results.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
