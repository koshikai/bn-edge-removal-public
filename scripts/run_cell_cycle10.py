import argparse

import numpy as np
import pandas as pd

from ai_research_template.bn_edge_removal.cell_cycle10 import CellCycle10Model
from ai_research_template.bn_edge_removal.env import (
    EdgeRemovalEnv,
    HorizonConfig,
    RewardConfig,
)
from ai_research_template.bn_edge_removal.evaluation import evaluate_policy
from ai_research_template.bn_edge_removal.plotting import (
    plot_training_curves,
    plot_trajectories,
)
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
        description="Train Q-learning on CellCycle10 model."
    )
    parser.add_argument(
        "--config", type=str, default="configs/cell_cycle10_monotone.yaml"
    )
    return parser.parse_args()


def _resolve_initial_state_settings(
    model: CellCycle10Model, config: dict
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


def main() -> None:
    args = _parse_args()

    config = load_config(args.config)
    output_dir_config = config.get("output_dir", "outputs")
    experiment_name = config.get("model", {}).get("name", "cell_cycle10")

    timestamp = current_timestamp()
    output_dir = prepare_output_dir(
        experiment_name=experiment_name,
        output_root=output_dir_config,
        timestamp=timestamp,
    )

    logger = setup_logger(output_dir)
    logger.info("Starting CellCycle10 training run")
    logger.info(f"Config: {config}")

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

    q_table, history = train_q_learning(
        env,
        q_config,
        rng=rng,
        initial_states=initial_states,
        initial_state_strategy=initial_state_strategy,
    )

    artifacts_dir = output_dir / "artifacts"
    history_df = pd.DataFrame(history)
    history_df.to_csv(artifacts_dir / "training.csv", index=False)

    eval_result = evaluate_policy(env, q_table)

    trajectories_df = pd.DataFrame(eval_result.trajectories)
    trajectories_df.to_csv(artifacts_dir / "trajectories.csv", index=False)

    actions_df = pd.DataFrame(eval_result.actions)
    actions_df.to_csv(artifacts_dir / "actions.csv", index=False)

    np.save(artifacts_dir / "q_table.npy", q_table)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_training_curves(history, figures_dir / "training_curves.png")
    plot_trajectories(eval_result.trajectories, figures_dir / "trajectories.png")

    results = {
        "config": config,
        "metrics": eval_result.metrics,
        "model_params": {
            "n_nodes": model.n_nodes,
            "m_edges": model.m_edges,
        },
    }
    save_results(results, output_dir)
    save_params(config, output_dir)
    update_experiment_summary(results, output_dir)

    write_report(
        [
            "# CellCycle10 Training Report",
            "",
            f"- Timestamp: {timestamp}",
            f"- Success Rate: {eval_result.metrics['success_rate']:.3f}",
            f"- Violation Rate: {eval_result.metrics['violation_rate']:.3f}",
            f"- Output: {output_dir}",
        ],
        output_dir,
    )

    request_path = write_daily_report_request(
        output_dir=output_dir,
        experiment_name=experiment_name,
        timestamp=timestamp,
        config_path=args.config,
        metrics=results.get("metrics", {}),
    )

    logger.info("Run complete")
    logger.info(f"Success rate: {eval_result.metrics['success_rate']:.3f}")
    logger.info(f"Violation rate: {eval_result.metrics['violation_rate']:.3f}")
    logger.info(f"Run: uv run poe daily-report --request {request_path}")


if __name__ == "__main__":
    main()
