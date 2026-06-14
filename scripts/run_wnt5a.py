import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bn_edge_removal.env import (
    EdgeRemovalEnv,
    HorizonConfig,
    RewardConfig,
)
from bn_edge_removal.evaluation_sparse import (
    evaluate_policy_sparse,
)
from bn_edge_removal.plotting import (
    plot_training_curves,
    plot_trajectories,
)
from bn_edge_removal.q_learning import (
    EpsilonSchedule,
    QLearningConfig,
)
from bn_edge_removal.q_learning_sparse import (
    SparseQTable,
    train_q_learning_sparse,
)
from bn_edge_removal.utils import (
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
from bn_edge_removal.wnt5a import Wnt5aModel


def _state_pool(model: Wnt5aModel, pool_name: str) -> list[list[int]]:
    if pool_name == "all":
        return [model.id_to_state(i) for i in range(2**model.n_nodes)]
    if pool_name == "controllable":
        return model.controllable_initial_states()
    raise ValueError("state pool must be 'all' or 'controllable'")


def _resolve_initial_state_config(
    model: Wnt5aModel, strategy: str
) -> tuple[list[list[int]] | None, str]:
    if strategy == "env":
        return None, "random"
    if strategy == "random_all":
        return _state_pool(model, "all"), "random"
    if strategy == "cycle_all":
        return _state_pool(model, "all"), "cycle"
    if strategy == "random_controllable":
        return _state_pool(model, "controllable"), "random"
    if strategy == "cycle_controllable":
        return _state_pool(model, "controllable"), "cycle"
    raise ValueError(
        "initial_state_strategy must be one of "
        "env/random_all/cycle_all/random_controllable/cycle_controllable"
    )


def _save_sparse_q_table(
    q_table: SparseQTable, num_actions: int, output_path: Path
) -> None:
    if q_table:
        state_ids = np.array(sorted(q_table.keys()), dtype=np.int64)
        q_values = np.vstack([q_table[int(s)] for s in state_ids]).astype(float)
    else:
        state_ids = np.array([], dtype=np.int64)
        q_values = np.empty((0, num_actions), dtype=float)
    np.savez_compressed(output_path, state_ids=state_ids, q_values=q_values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Q-learning on Wnt5a model.")
    parser.add_argument("--config", type=str, default="configs/wnt5a_recovery.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir_config = config.get("output_dir", "outputs")
    experiment_name = config.get("model", {}).get("name", "wnt5a")

    timestamp = current_timestamp()
    output_dir = prepare_output_dir(
        experiment_name=experiment_name,
        output_root=output_dir_config,
        timestamp=timestamp,
    )

    logger = setup_logger(output_dir)
    logger.info("Starting Wnt5a training run")
    logger.info(f"Config: {config}")

    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)

    model = Wnt5aModel()
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

    constraint_cfg = config.get("constraint", {})
    env = EdgeRemovalEnv(
        model=model,
        reward=reward,
        horizon=horizon,
        action_masking=bool(constraint_cfg.get("action_masking", False)),
        constraint_type=str(constraint_cfg.get("type", "recovery")),
        recovery_tau=int(constraint_cfg.get("recovery_tau", 2)),
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

    init_strategy = str(config["learning"].get("initial_state_strategy", "env"))
    train_initial_states, train_state_strategy = _resolve_initial_state_config(
        model, init_strategy
    )

    q_table, history = train_q_learning_sparse(
        env,
        q_config,
        rng=rng,
        initial_states=train_initial_states,
        initial_state_strategy=train_state_strategy,
    )

    eval_pool = str(config.get("evaluation", {}).get("initial_states", "controllable"))
    eval_initial_states = _state_pool(model, eval_pool)
    eval_result = evaluate_policy_sparse(
        env, q_table, initial_states=eval_initial_states
    )

    artifacts_dir = output_dir / "artifacts"
    history_df = pd.DataFrame(history)
    history_df.to_csv(artifacts_dir / "training.csv", index=False)

    trajectories_df = pd.DataFrame(eval_result.trajectories)
    trajectories_df.to_csv(artifacts_dir / "trajectories.csv", index=False)

    actions_df = pd.DataFrame(eval_result.actions)
    actions_df.to_csv(artifacts_dir / "actions.csv", index=False)

    _save_sparse_q_table(q_table, env.num_actions, artifacts_dir / "q_table_sparse.npz")

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
            "constraint_type": env.constraint_type,
            "recovery_tau": env.recovery_tau,
        },
    }
    save_results(results, output_dir)
    save_params(config, output_dir)
    update_experiment_summary(results, output_dir)

    write_report(
        [
            "# Wnt5a Training Report",
            "",
            f"- Timestamp: {timestamp}",
            f"- Success Rate: {eval_result.metrics['success_rate']:.3f}",
            f"- Violation Rate: {eval_result.metrics['violation_rate']:.3f}",
            f"- Initial States: {int(eval_result.metrics['num_initial_states'])}",
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
