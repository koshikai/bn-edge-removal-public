import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell
def __():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from ai_research_template.bn_edge_removal.encoding import int_to_bits

    MODEL_INFO = {
        "cortical": {
            "title": "Cortical (Monotone Constraint)",
            "n_nodes": 5,
            "m_edges": 6,
            "edge_labels": [
                "x2->x1",
                "x4->x1",
                "x5->x2",
                "x2->x3",
                "x4->x3",
                "x5->x4",
            ],
        },
        "wnt5a": {
            "title": "Wnt5a (Recovery Constraint)",
            "n_nodes": 7,
            "m_edges": 8,
            "edge_labels": [
                "x2->x2",
                "x4->x2",
                "x6->x2",
                "x2->x5",
                "x7->x5",
                "x3->x6",
                "x4->x6",
                "x2->x7",
            ],
        },
    }

    def bits_to_int(bits: list[int]) -> int:
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return value

    def moving_average(values: np.ndarray, window: int) -> np.ndarray:
        if len(values) == 0:
            return values
        if window <= 1:
            return values
        cumsum = np.cumsum(np.insert(values, 0, 0.0))
        out = (cumsum[window:] - cumsum[:-window]) / float(window)
        pad = np.full(window - 1, out[0])
        return np.concatenate([pad, out])

    def parse_run(model_name: str, run_dir: Path) -> dict | None:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            return None
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        metrics = payload.get("metrics", {})
        model_cfg = cfg.get("model", {})
        learning_cfg = cfg.get("learning", {})
        horizon_cfg = cfg.get("horizon", {})
        constraint_cfg = cfg.get("constraint", {})
        report_request_path = run_dir / "report_request.json"
        config_path = ""
        if report_request_path.exists():
            req = json.loads(report_request_path.read_text(encoding="utf-8"))
            config_path = str(req.get("config_path", ""))
        return {
            "model": model_name,
            "run_id": run_dir.name,
            "timestamp": run_dir.name,
            "path": str(run_dir),
            "config_path": config_path,
            "n_nodes": int(model_cfg.get("n_nodes", MODEL_INFO[model_name]["n_nodes"])),
            "m_edges": int(model_cfg.get("m_edges", MODEL_INFO[model_name]["m_edges"])),
            "goal_states": model_cfg.get("goal_states", []),
            "episodes": int(learning_cfg.get("episodes", 0)),
            "alpha": float(learning_cfg.get("alpha", 0.0)),
            "gamma": float(learning_cfg.get("gamma", 0.0)),
            "epsilon_start": float(learning_cfg.get("epsilon", {}).get("start", 0.0)),
            "epsilon_end": float(learning_cfg.get("epsilon", {}).get("end", 0.0)),
            "initial_state_strategy": str(
                learning_cfg.get("initial_state_strategy", "")
            ),
            "reach_horizon": int(horizon_cfg.get("reach_horizon", 0)),
            "max_steps": int(horizon_cfg.get("max_steps", 0)),
            "constraint_type": str(
                constraint_cfg.get(
                    "type", "monotone" if constraint_cfg.get("monotone") else ""
                )
            ),
            "action_masking": bool(constraint_cfg.get("action_masking", False)),
            "recovery_tau": int(constraint_cfg.get("recovery_tau", 0)),
            "success_rate": float(metrics.get("success_rate", np.nan)),
            "violation_rate": float(metrics.get("violation_rate", np.nan)),
            "avg_first_goal_step": float(metrics.get("avg_first_goal_step", np.nan)),
            "num_initial_states": float(metrics.get("num_initial_states", np.nan)),
        }

    def collect_runs(model_name: str) -> pd.DataFrame:
        root = Path("outputs") / model_name
        if not root.exists():
            return pd.DataFrame()
        rows: list[dict] = []
        for run_dir in sorted(root.iterdir()):
            if run_dir.name in {"latest", "comparisons"}:
                continue
            if not run_dir.is_dir():
                continue
            parsed = parse_run(model_name, run_dir)
            if parsed is not None:
                rows.append(parsed)
        if not rows:
            return pd.DataFrame()
        runs_df = pd.DataFrame(rows)
        runs_df = runs_df.sort_values("timestamp", ascending=False).reset_index(
            drop=True
        )
        return runs_df

    return (
        MODEL_INFO,
        Path,
        bits_to_int,
        collect_runs,
        int_to_bits,
        mo,
        moving_average,
        np,
        pd,
        plt,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # BN Edge Removal Experiments Dashboard

        このノートブックでは、以下の実験ログを比較・可視化できます。

        - **Cortical**: 単調性制約（一度除去したエッジは戻さない）
        - **Wnt5a**: 休薬制約（`tau=2` ステップ再介入禁止）

        画面上部でモデルとランを選ぶと、学習曲線・軌道・行動タイムラインを確認できます。
        """
    )


@app.cell
def __(mo):
    model_options = {
        "Cortical (Monotone Constraint)": "cortical",
        "Wnt5a (Recovery Constraint)": "wnt5a",
    }
    model_selector = mo.ui.dropdown(
        options=model_options,
        value="Cortical (Monotone Constraint)",
        label="Experiment Set",
    )
    return (model_selector,)


@app.cell
def __(collect_runs, model_selector, pd):
    runs_df = collect_runs(model_selector.value)
    if runs_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = runs_df[
            [
                "timestamp",
                "success_rate",
                "violation_rate",
                "avg_first_goal_step",
                "num_initial_states",
                "episodes",
                "reach_horizon",
                "max_steps",
                "constraint_type",
                "action_masking",
                "config_path",
            ]
        ].copy()
    return runs_df, summary_df


@app.cell
def __(mo, model_selector, runs_df, summary_df):
    if runs_df.empty:
        mo.md(
            f"`outputs/{model_selector.value}` に有効なランが見つかりませんでした。"
        )
    else:
        latest = runs_df.iloc[0]
        mo.md(
            f"""
            ## 実験一覧（{model_selector.value}）

            - 総ラン数: **{len(runs_df)}**
            - 最新ラン: **{latest["timestamp"]}**
            - 最新 Success / Violation: **{latest["success_rate"]:.3f} / {latest["violation_rate"]:.3f}**
            """
        )
        summary_df


@app.cell
def __(mo, runs_df):
    run_selector = None
    if not runs_df.empty:
        run_options = {}
        for run_row in runs_df.itertuples(index=False):
            label = (
                f'{run_row.timestamp} | success={run_row.success_rate:.3f} | '
                f'violation={run_row.violation_rate:.3f}'
            )
            run_options[label] = run_row.path
        run_first_key = next(iter(run_options.keys()))
        run_selector = mo.ui.dropdown(
            options=run_options, value=run_first_key, label="Run"
        )
    return (run_selector,)


@app.cell
def __(Path, model_selector, run_selector, runs_df):
    selected_run_dir = None
    selected_row = None
    if run_selector is not None and run_selector.value is not None and not runs_df.empty:
        selected_run_dir = Path(run_selector.value)
        selected_row = runs_df[runs_df["path"] == str(selected_run_dir)].iloc[0]
    model_name = model_selector.value
    return model_name, selected_row, selected_run_dir


@app.cell
def __(mo, selected_row):
    if selected_row is None:
        mo.md("ラン未選択です。")
    else:
        mo.md(
            f"""
            ### 選択中のラン

            - timestamp: `{selected_row["timestamp"]}`
            - config: `{selected_row["config_path"]}`
            - episodes: `{int(selected_row["episodes"])}`
            - horizon: `H={int(selected_row["reach_horizon"])}, T={int(selected_row["max_steps"])}`
            - success / violation: `{selected_row["success_rate"]:.3f} / {selected_row["violation_rate"]:.3f}`
            - avg first goal step: `{selected_row["avg_first_goal_step"]:.3f}`
            - action masking: `{bool(selected_row["action_masking"])}`
            """
        )


@app.cell
def __(mo):
    mo.md("### 保存済み Figure ファイル（run スクリプト出力）")


@app.cell
def __(mo, selected_run_dir):
    if selected_run_dir is None:
        mo.md("ラン未選択のため、保存済み figure を表示できません。")
    else:
        saved_training_png_path = selected_run_dir / "figures" / "training_curves.png"
        saved_trajectories_png_path = selected_run_dir / "figures" / "trajectories.png"
        mo.md(
            f"""
            - `{saved_training_png_path}`
            - `{saved_trajectories_png_path}`
            """
        )


@app.cell
def __(plt, selected_run_dir):
    saved_training_png_fig = None
    saved_trajectories_png_fig = None

    if selected_run_dir is not None:
        saved_training_png_path_for_plot = (
            selected_run_dir / "figures" / "training_curves.png"
        )
        saved_trajectories_png_path_for_plot = (
            selected_run_dir / "figures" / "trajectories.png"
        )

        if saved_training_png_path_for_plot.exists():
            saved_training_png_img = plt.imread(saved_training_png_path_for_plot)
            saved_training_png_fig, saved_training_png_ax = plt.subplots(
                figsize=(8, 4)
            )
            saved_training_png_ax.imshow(saved_training_png_img)
            saved_training_png_ax.set_title("Saved training_curves.png")
            saved_training_png_ax.axis("off")
            saved_training_png_fig.tight_layout()

        if saved_trajectories_png_path_for_plot.exists():
            saved_trajectories_png_img = plt.imread(
                saved_trajectories_png_path_for_plot
            )
            saved_trajectories_png_fig, saved_trajectories_png_ax = plt.subplots(
                figsize=(8, 5)
            )
            saved_trajectories_png_ax.imshow(saved_trajectories_png_img)
            saved_trajectories_png_ax.set_title("Saved trajectories.png")
            saved_trajectories_png_ax.axis("off")
            saved_trajectories_png_fig.tight_layout()

    saved_training_png_fig
    saved_trajectories_png_fig


@app.cell
def __(pd, selected_run_dir):
    training_df = pd.DataFrame()
    trajectories_df = pd.DataFrame()
    actions_df = pd.DataFrame()
    if selected_run_dir is not None:
        artifacts = selected_run_dir / "artifacts"
        train_path = artifacts / "training.csv"
        traj_path = artifacts / "trajectories.csv"
        action_path = artifacts / "actions.csv"
        if train_path.exists():
            training_df = pd.read_csv(train_path)
        if traj_path.exists():
            trajectories_df = pd.read_csv(traj_path)
        if action_path.exists():
            actions_df = pd.read_csv(action_path)
    return actions_df, training_df, trajectories_df


@app.cell
def __(mo):
    smooth_slider = mo.ui.slider(
        20,
        2000,
        step=20,
        value=200,
        label="Smoothing Window (episodes)",
    )
    return (smooth_slider,)


@app.cell
def __(moving_average, np, plt, smooth_slider, training_df):
    training_fig = None
    if not training_df.empty:
        window = int(smooth_slider.value)
        reward = training_df["reward"].to_numpy(dtype=float)
        success = training_df["success"].to_numpy(dtype=float)
        violation = training_df["violation"].to_numpy(dtype=float)
        eps = training_df["epsilon"].to_numpy(dtype=float)

        reward_ma = moving_average(reward, window)
        success_ma = moving_average(success, window)
        violation_ma = moving_average(violation, window)

        episode_index = np.arange(len(reward))
        training_fig, train_axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        train_axes[0].plot(episode_index, reward_ma, color="#1f77b4", linewidth=1.2)
        train_axes[0].set_ylabel("reward")
        train_axes[0].grid(alpha=0.3)
        train_axes[1].plot(episode_index, success_ma, color="#2ca02c", linewidth=1.2)
        train_axes[1].set_ylabel("success")
        train_axes[1].grid(alpha=0.3)
        train_axes[2].plot(
            episode_index, violation_ma, color="#d62728", linewidth=1.2
        )
        train_axes[2].set_ylabel("violation")
        train_axes[2].grid(alpha=0.3)
        train_axes[3].plot(episode_index, eps, color="#7f7f7f", linewidth=1.2)
        train_axes[3].set_ylabel("epsilon")
        train_axes[3].set_xlabel("episode")
        train_axes[3].grid(alpha=0.3)
        training_fig.tight_layout()
    training_fig


@app.cell
def __(MODEL_INFO, actions_df, int_to_bits, mo, selected_row):
    init_selector = None
    if not actions_df.empty and selected_row is not None:
        state_width = int(selected_row["n_nodes"])
        init_options = {}
        for candidate_state in sorted(actions_df["init_state"].unique().tolist()):
            state_bits = int_to_bits(int(candidate_state), state_width)
            init_options[f"{int(candidate_state)} ({state_bits})"] = int(
                candidate_state
            )
        init_first_key = next(iter(init_options.keys()))
        init_selector = mo.ui.dropdown(
            options=init_options, value=init_first_key, label="Initial State"
        )
    return (init_selector,)


@app.cell
def __(init_selector, mo, model_selector, run_selector, selected_row, smooth_slider):
    run_status_md = (
        mo.md(
            f"""
            **Selected run**
            - `{selected_row["timestamp"]}`
            - success: `{selected_row["success_rate"]:.3f}`
            - violation: `{selected_row["violation_rate"]:.3f}`
            """
        )
        if selected_row is not None
        else mo.md("**Selected run**: none")
    )
    controls = [
        mo.md("## Controls"),
        model_selector,
        run_selector if run_selector is not None else mo.md("No runs available"),
        init_selector
        if init_selector is not None
        else mo.md("No initial states available"),
        smooth_slider,
        run_status_md,
    ]
    sidebar_footer = mo.md(
        "ヒント: `Initial State` を変えるとエッジ軸タイムラインが切り替わります。"
    )
    mo.sidebar(mo.vstack(controls, gap=1.0), footer=sidebar_footer, width="320px")


@app.cell
def __(
    actions_df,
    init_selector,
    selected_row,
    trajectories_df,
):
    selected_init_state_id = None
    selected_action_df = actions_df.head(0).copy()
    selected_trajectory_df = trajectories_df.head(0).copy()
    if (
        selected_row is not None
        and init_selector is not None
        and init_selector.value is not None
        and not actions_df.empty
        and not trajectories_df.empty
    ):
        selected_init_state_id = int(init_selector.value)
        selected_action_df = actions_df[
            actions_df["init_state"] == selected_init_state_id
        ].copy()
        selected_action_df = selected_action_df.sort_values("t")
        selected_trajectory_df = trajectories_df[
            trajectories_df["init_state"] == selected_init_state_id
        ].copy()
        selected_trajectory_df = selected_trajectory_df.sort_values("t")
    return selected_action_df, selected_init_state_id, selected_trajectory_df


@app.cell
def __(int_to_bits, mo, selected_init_state_id, selected_row):
    if selected_row is None or selected_init_state_id is None:
        mo.md("初期状態を選ぶと、対応するタイムラインを表示します。")
    else:
        selected_state_bits = int_to_bits(
            int(selected_init_state_id), int(selected_row["n_nodes"])
        )
        mo.md(
            f"""
            ### 初期状態別タイムライン
            - init_state: `{selected_init_state_id}`
            - bits: `{selected_state_bits}`
            """
        )


@app.cell
def __(bits_to_int, mo, plt, selected_init_state_id, selected_row, selected_trajectory_df):
    trajectory_fig = None
    if (
        selected_row is not None
        and selected_init_state_id is not None
        and not selected_trajectory_df.empty
    ):
        goal_states = selected_row["goal_states"]
        goal_state_ids = [bits_to_int(list(bits)) for bits in goal_states]
        trajectory_fig, trajectory_ax = plt.subplots(figsize=(10, 3.5))
        trajectory_ax.plot(
            selected_trajectory_df["t"],
            selected_trajectory_df["state_id"],
            marker="o",
            linewidth=1.5,
            ms=4,
            color="#1f77b4",
        )
        for target_id in goal_state_ids:
            trajectory_ax.axhline(
                target_id, color="#9467bd", linestyle="--", alpha=0.5
            )
        trajectory_ax.set_xlabel("time")
        trajectory_ax.set_ylabel("state_id")
        trajectory_ax.set_title(
            f"State trajectory (init_state={selected_init_state_id})"
        )
        trajectory_ax.grid(alpha=0.3)
        trajectory_fig.tight_layout()
    trajectory_output = (
        trajectory_fig
        if trajectory_fig is not None
        else mo.md("選択中の初期状態に対する軌道データがありません。")
    )
    trajectory_output


@app.cell
def __(
    MODEL_INFO,
    int_to_bits,
    mo,
    model_name,
    np,
    plt,
    selected_action_df,
    selected_init_state_id,
    selected_row,
):
    edge_timeline_fig = None
    if (
        selected_row is not None
        and selected_init_state_id is not None
        and not selected_action_df.empty
    ):
        m_edges_for_timeline = int(selected_row["m_edges"])
        timeline_edge_labels = MODEL_INFO[model_name]["edge_labels"]
        max_time_step = int(selected_action_df["t"].max())
        edge_timeline_matrix = np.zeros((m_edges_for_timeline, max_time_step + 1), dtype=int)
        for edge_action_row in selected_action_df.itertuples(index=False):
            edge_action_bits = int_to_bits(int(edge_action_row.action), m_edges_for_timeline)
            edge_timeline_matrix[:, int(edge_action_row.t)] = np.array(
                edge_action_bits, dtype=int
            )

        edge_timeline_fig, edge_timeline_ax = plt.subplots(figsize=(10, 4.5))
        edge_timeline_im = edge_timeline_ax.imshow(
            edge_timeline_matrix, aspect="auto", cmap="Greys", vmin=0, vmax=1
        )
        edge_timeline_ax.set_xlabel("time")
        edge_timeline_ax.set_ylabel("edge")
        edge_timeline_ax.set_yticks(range(m_edges_for_timeline))
        edge_timeline_ax.set_yticklabels(timeline_edge_labels)
        edge_timeline_ax.set_title(
            f"Edge-axis timeline (init_state={selected_init_state_id}, 1=removed)"
        )
        edge_timeline_fig.colorbar(
            edge_timeline_im, ax=edge_timeline_ax, fraction=0.03, pad=0.02
        )
        edge_timeline_fig.tight_layout()
    edge_timeline_output = (
        edge_timeline_fig
        if edge_timeline_fig is not None
        else mo.md("選択中の初期状態に対する行動データがありません。")
    )
    edge_timeline_output


@app.cell
def __(np, plt, runs_df):
    compare_fig = None
    if not runs_df.empty:
        plot_df = runs_df.sort_values("timestamp").reset_index(drop=True)
        run_index = np.arange(len(plot_df))
        compare_fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(
            run_index, plot_df["success_rate"], marker="o", label="success_rate"
        )
        ax.plot(
            run_index, plot_df["violation_rate"], marker="o", label="violation_rate"
        )
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(run_index)
        ax.set_xticklabels(plot_df["timestamp"], rotation=45, ha="right")
        ax.set_ylabel("rate")
        ax.set_title("Run-to-run comparison")
        ax.grid(alpha=0.3)
        ax.legend()
        compare_fig.tight_layout()
    compare_fig


if __name__ == "__main__":
    app.run()
