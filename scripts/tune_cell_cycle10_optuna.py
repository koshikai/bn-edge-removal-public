"""Hyperparameter tuning for CellCycle10 Q-learning with Optuna."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import optuna
import yaml

from ai_research_template.bn_edge_removal.cell_cycle10 import CellCycle10Model
from ai_research_template.bn_edge_removal.env import (
    EdgeRemovalEnv,
    HorizonConfig,
    RewardConfig,
)
from ai_research_template.bn_edge_removal.evaluation import evaluate_policy
from ai_research_template.bn_edge_removal.q_learning import (
    EpsilonSchedule,
    QLearningConfig,
    train_q_learning,
)
from ai_research_template.utils import (
    current_timestamp,
    load_config,
    prepare_output_dir,
    save_params,
    save_results,
    setup_logger,
    update_experiment_summary,
    write_daily_report_request,
    write_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune CellCycle10 Q-learning hyperparameters with Optuna."
    )
    parser.add_argument(
        "--config", type=str, default="configs/cell_cycle10_optuna.yaml"
    )
    return parser.parse_args()


def _resolve_initial_state_settings(
    model: CellCycle10Model, config: dict[str, Any]
) -> tuple[list[list[int]] | None, str]:
    init_strategy = str(config["learning"].get("initial_state_strategy", "env"))
    if init_strategy == "random_all":
        all_states = [model.id_to_state(i) for i in range(2**model.n_nodes)]
        return all_states, "random"
    if init_strategy == "cycle_all":
        all_states = [model.id_to_state(i) for i in range(2**model.n_nodes)]
        return all_states, "cycle"
    if init_strategy == "env":
        return None, "random"
    raise ValueError("initial_state_strategy must be env, random_all, or cycle_all")


def _suggest_space(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}

    alpha_cfg = search_space["alpha"]
    params["alpha"] = trial.suggest_float(
        "alpha", float(alpha_cfg["low"]), float(alpha_cfg["high"])
    )

    gamma_cfg = search_space["gamma"]
    params["gamma"] = trial.suggest_float(
        "gamma", float(gamma_cfg["low"]), float(gamma_cfg["high"])
    )

    episodes_cfg = search_space["episodes"]
    params["episodes"] = trial.suggest_int(
        "episodes",
        int(episodes_cfg["low"]),
        int(episodes_cfg["high"]),
        step=int(episodes_cfg.get("step", 1)),
    )

    eps_cfg = search_space["epsilon_end"]
    params["epsilon_end"] = trial.suggest_float(
        "epsilon_end",
        float(eps_cfg["low"]),
        float(eps_cfg["high"]),
        log=bool(eps_cfg.get("log", False)),
    )

    terminal_cfg = search_space["terminal"]
    params["terminal"] = trial.suggest_float(
        "terminal",
        float(terminal_cfg["low"]),
        float(terminal_cfg["high"]),
        step=float(terminal_cfg.get("step", 1.0)),
    )

    edge_cfg = search_space["edge_cost"]
    params["edge_cost"] = trial.suggest_float(
        "edge_cost",
        float(edge_cfg["low"]),
        float(edge_cfg["high"]),
        step=float(edge_cfg.get("step", 0.01)),
    )

    new_edge_cfg = search_space["new_edge_cost"]
    params["new_edge_cost"] = trial.suggest_float(
        "new_edge_cost",
        float(new_edge_cfg["low"]),
        float(new_edge_cfg["high"]),
        step=float(new_edge_cfg.get("step", 0.01)),
    )

    init_cfg = search_space["initial_state_strategy"]
    params["initial_state_strategy"] = trial.suggest_categorical(
        "initial_state_strategy", list(init_cfg["choices"])
    )

    return params


def _apply_trial_params(
    base_config: dict[str, Any], params: dict[str, Any], seed: int
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg["seed"] = seed
    cfg["learning"]["alpha"] = float(params["alpha"])
    cfg["learning"]["gamma"] = float(params["gamma"])
    cfg["learning"]["episodes"] = int(params["episodes"])
    cfg["learning"]["epsilon"]["end"] = float(params["epsilon_end"])
    cfg["learning"]["initial_state_strategy"] = str(params["initial_state_strategy"])
    cfg["reward"]["terminal"] = float(params["terminal"])
    cfg["reward"]["edge_cost"] = float(params["edge_cost"])
    cfg["reward"]["new_edge_cost"] = float(params["new_edge_cost"])
    return cfg


def _run_one(config: dict[str, Any]) -> dict[str, float]:
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)

    model = CellCycle10Model()
    reward = RewardConfig(
        step=float(config["reward"]["step"]),
        terminal=float(config["reward"]["terminal"]),
        edge_cost=float(config["reward"]["edge_cost"]),
        new_edge_cost=float(config["reward"].get("new_edge_cost", 0.0)),
        violation_penalty=float(config["reward"]["violation_penalty"]),
    )
    horizon = HorizonConfig(
        reach_horizon=int(config["horizon"]["reach_horizon"]),
        max_steps=int(config["horizon"]["max_steps"]),
    )
    env = EdgeRemovalEnv(
        model=model,
        reward=reward,
        horizon=horizon,
        action_masking=bool(config["constraint"].get("action_masking", False)),
        rng=rng,
    )

    q_config = QLearningConfig(
        alpha=float(config["learning"]["alpha"]),
        gamma=float(config["learning"]["gamma"]),
        episodes=int(config["learning"]["episodes"]),
        epsilon=EpsilonSchedule(
            start=float(config["learning"]["epsilon"]["start"]),
            end=float(config["learning"]["epsilon"]["end"]),
        ),
    )

    initial_states, initial_state_strategy = _resolve_initial_state_settings(
        model, config
    )

    q_table, _history = train_q_learning(
        env,
        q_config,
        rng=rng,
        initial_states=initial_states,
        initial_state_strategy=initial_state_strategy,
    )
    eval_result = evaluate_policy(env, q_table)
    return {
        "success_rate": float(eval_result.metrics["success_rate"]),
        "violation_rate": float(eval_result.metrics["violation_rate"]),
        "avg_first_goal_step": float(eval_result.metrics["avg_first_goal_step"]),
    }


def _storage_url(storage_path: str | Path) -> str:
    db_path = Path(storage_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.resolve()}"


def _create_study(optuna_cfg: dict[str, Any]) -> tuple[optuna.Study, str]:
    storage_url = _storage_url(optuna_cfg["storage"])
    sampler = optuna.samplers.TPESampler(seed=int(optuna_cfg.get("sampler_seed", 42)))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(optuna_cfg.get("n_startup_trials", 5))
    )
    study = optuna.create_study(
        study_name=str(optuna_cfg["study_name"]),
        storage=storage_url,
        load_if_exists=bool(optuna_cfg.get("load_if_exists", True)),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    return study, storage_url


def _build_objective(
    config: dict[str, Any],
    search_space: dict[str, Any],
    eval_seeds: list[int],
    objective_cfg: dict[str, float],
) -> Callable[[optuna.Trial], float]:
    success_weight = objective_cfg["success_weight"]
    violation_weight = objective_cfg["violation_weight"]
    speed_weight = objective_cfg["speed_weight"]
    max_steps = objective_cfg["max_steps"]

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_space(trial, search_space)
        metrics_across_seeds: list[dict[str, float]] = []
        for idx, seed in enumerate(eval_seeds):
            trial_config = _apply_trial_params(config, params, seed)
            metrics = _run_one(trial_config)
            metrics_across_seeds.append(metrics)

            running_success = mean(m["success_rate"] for m in metrics_across_seeds)
            trial.report(running_success, idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        success_rate = mean(m["success_rate"] for m in metrics_across_seeds)
        violation_rate = mean(m["violation_rate"] for m in metrics_across_seeds)
        avg_first_goal = mean(m["avg_first_goal_step"] for m in metrics_across_seeds)

        score = (
            (success_weight * success_rate)
            - (violation_weight * violation_rate)
            - (speed_weight * (avg_first_goal / max_steps))
        )

        trial.set_user_attr("success_rate", success_rate)
        trial.set_user_attr("violation_rate", violation_rate)
        trial.set_user_attr("avg_first_goal_step", avg_first_goal)
        trial.set_user_attr("score", score)
        return score

    return objective


def _collect_best_results(
    study: optuna.Study,
    config: dict[str, Any],
    eval_seeds: list[int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete_trials:
        raise RuntimeError("No completed trials found in study.")

    best_trial = study.best_trial
    if best_trial.value is None:
        raise RuntimeError("Best trial has no objective value.")
    best_params = dict(best_trial.params)
    best_config = _apply_trial_params(config, best_params, int(eval_seeds[0]))

    metrics = {
        "best_score": float(best_trial.value),
        "best_success_rate": float(best_trial.user_attrs["success_rate"]),
        "best_violation_rate": float(best_trial.user_attrs["violation_rate"]),
        "best_avg_first_goal_step": float(best_trial.user_attrs["avg_first_goal_step"]),
        "n_trials_total": float(len(study.trials)),
        "n_trials_complete": float(len(complete_trials)),
    }
    return best_params, best_config, metrics


def _write_artifacts(
    study: optuna.Study, best_config: dict[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    trials_df = study.trials_dataframe(
        attrs=("number", "value", "params", "state", "user_attrs")
    )
    trials_csv_path = output_dir / "artifacts" / "trials.csv"
    trials_df.to_csv(trials_csv_path, index=False)

    best_config_path = output_dir / "artifacts" / "best_config.yaml"
    best_config_path.write_text(
        yaml.safe_dump(best_config, sort_keys=False),
        encoding="utf-8",
    )
    return trials_csv_path, best_config_path


def _write_tuning_report(output_dir: Path, report_ctx: dict[str, Any]) -> None:
    write_report(
        [
            "# CellCycle10 Optuna Tuning Report",
            "",
            f"- Timestamp: {report_ctx['timestamp']}",
            f"- Study Name: {report_ctx['study_name']}",
            f"- Storage: {report_ctx['storage_url']}",
            f"- Best Score: {report_ctx['metrics']['best_score']:.6f}",
            f"- Best Success Rate: {report_ctx['metrics']['best_success_rate']:.6f}",
            (
                "- Best Violation Rate: "
                f"{report_ctx['metrics']['best_violation_rate']:.6f}"
            ),
            (
                "- Best Avg First Goal Step: "
                f"{report_ctx['metrics']['best_avg_first_goal_step']:.6f}"
            ),
            f"- Completed Trials: {int(report_ctx['metrics']['n_trials_complete'])}",
            f"- Output: {output_dir}",
            "",
            "## Best Params",
            f"- {report_ctx['best_params']}",
            "",
            f"- Best Config: {report_ctx['best_config_path']}",
            f"- Trials CSV: {report_ctx['trials_csv_path']}",
        ],
        output_dir,
    )


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    optuna_cfg = config["optuna"]
    search_space = config["search_space"]

    experiment_name = str(optuna_cfg.get("experiment_name", "cell_cycle10_optuna"))
    timestamp = current_timestamp()
    output_dir = prepare_output_dir(
        experiment_name=experiment_name,
        output_root=config.get("output_dir", "outputs"),
        timestamp=timestamp,
    )
    logger = setup_logger(output_dir, name="cell_cycle10_optuna")
    logger.info("Starting Optuna tuning run")
    logger.info(f"Config path: {args.config}")

    study, storage_url = _create_study(optuna_cfg)

    eval_seeds = [int(v) for v in optuna_cfg.get("eval_seeds", [int(config["seed"])])]
    objective = _build_objective(
        config=config,
        search_space=search_space,
        eval_seeds=eval_seeds,
        objective_cfg={
            "max_steps": float(config["horizon"]["max_steps"]),
            "success_weight": float(optuna_cfg["objective"]["success_weight"]),
            "violation_weight": float(optuna_cfg["objective"]["violation_weight"]),
            "speed_weight": float(optuna_cfg["objective"]["speed_weight"]),
        },
    )

    study.optimize(
        objective,
        n_trials=int(optuna_cfg["n_trials"]),
        timeout=(
            None
            if optuna_cfg.get("timeout_sec") is None
            else int(optuna_cfg["timeout_sec"])
        ),
        n_jobs=int(optuna_cfg.get("n_jobs", 1)),
        show_progress_bar=bool(optuna_cfg.get("show_progress_bar", True)),
    )

    best_params, best_config, metrics = _collect_best_results(study, config, eval_seeds)
    trials_csv_path, best_config_path = _write_artifacts(study, best_config, output_dir)

    results = {
        "config": config,
        "metrics": metrics,
        "model_params": {
            "n_nodes": int(config["model"]["n_nodes"]),
            "m_edges": int(config["model"]["m_edges"]),
        },
    }
    save_results(results, output_dir)
    save_params(config, output_dir)
    update_experiment_summary(results, output_dir)

    _write_tuning_report(
        output_dir,
        report_ctx={
            "timestamp": timestamp,
            "study_name": str(optuna_cfg["study_name"]),
            "storage_url": storage_url,
            "metrics": metrics,
            "best_params": best_params,
            "best_config_path": best_config_path,
            "trials_csv_path": trials_csv_path,
        },
    )

    request_path = write_daily_report_request(
        output_dir=output_dir,
        experiment_name=experiment_name,
        timestamp=timestamp,
        config_path=args.config,
        metrics=metrics,
    )

    logger.info("Tuning complete")
    logger.info(f"Best score: {metrics['best_score']:.6f}")
    logger.info(f"Best success rate: {metrics['best_success_rate']:.6f}")
    logger.info(f"Best config: {best_config_path}")
    logger.info(f"Trials CSV: {trials_csv_path}")
    logger.info(f"Run: uv run poe daily-report --request {request_path}")


if __name__ == "__main__":
    main()
