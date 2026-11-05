import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
async def __():
    import sys

    if sys.platform in ("emscripten", "pyodide"):
        import importlib
        import io
        import zipfile

        js = importlib.import_module("js")
        pyfetch = importlib.import_module("pyodide.http").pyfetch

        # WASM ではセルは Worker の中で走るため、URL は
        # <app root>/assets/worker-xxxx.js の形になる。上位へ順に辿って zip を探す。
        current = str(js.location.href).split("?")[0].split("#")[0]
        parts = current.split("/")
        candidate_urls: list[str] = []
        for cut in (2, 1, 3):
            if len(parts) > cut:
                url = "/".join(parts[:-cut]) + "/bn_edge_removal.zip"
                if url not in candidate_urls:
                    candidate_urls.append(url)

        loaded = False
        last_error: Exception | None = None
        for zip_url in candidate_urls:
            try:
                response = await pyfetch(zip_url)
                if response.status != 200:
                    raise ValueError(f"{zip_url} returned HTTP {response.status}")
                zip_bytes = await response.bytes()
                if not zip_bytes.startswith(b"PK\x03\x04"):
                    raise ValueError(
                        f"Content from {zip_url} is not a zip archive "
                        "(magic header check failed)"
                    )
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_ref:
                    zip_ref.extractall(".")
                sys.path.insert(0, ".")
                print(f"Loaded bn_edge_removal package from {zip_url}.")
                loaded = True
                break
            except Exception as e:
                last_error = e

        if not loaded:
            print(f"Failed to load bn_edge_removal package: {last_error!r}")

    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from bn_edge_removal.models.encoding import bits_to_int, int_to_bits
    from bn_edge_removal.models.network_spec import load_network_spec

    OUTPUTS_ROOT = Path("outputs")

    def parse_run(metrics_path: Path) -> dict | None:
        """Read one Q-learning run. Reachability runs have no config and are skipped."""
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        model_cfg = cfg.get("model", {})
        if "name" not in model_cfg:
            return None
        run_dir = metrics_path.parent
        metrics = payload.get("metrics", {})
        learning_cfg = cfg.get("learning", {})
        horizon_cfg = cfg.get("horizon", {})
        constraint_cfg = cfg.get("constraint", {})
        return {
            "model": str(model_cfg["name"]),
            "group": str(run_dir.parent.relative_to(OUTPUTS_ROOT)),
            "run": run_dir.name,
            "path": str(run_dir),
            "modified": metrics_path.stat().st_mtime,
            "seed": cfg.get("seed"),
            "n_nodes": int(model_cfg.get("n_nodes", 0)),
            "m_edges": int(model_cfg.get("m_edges", 0)),
            "goal_states": model_cfg.get("goal_states", []),
            "episodes": int(learning_cfg.get("episodes", 0)),
            "reach_horizon": int(horizon_cfg.get("reach_horizon", 0)),
            "max_steps": int(horizon_cfg.get("max_steps", 0)),
            "constraint": str(constraint_cfg.get("type", "")),
            "action_masking": bool(constraint_cfg.get("action_masking", False)),
            "success_rate": float(metrics.get("success_rate", np.nan)),
            "violation_rate": float(metrics.get("violation_rate", np.nan)),
            "avg_first_goal_step": float(metrics.get("avg_first_goal_step", np.nan)),
            "num_initial_states": float(metrics.get("num_initial_states", np.nan)),
        }

    def discover_runs(root: Path) -> pd.DataFrame:
        """Find every run under root, including outputs/papers/, skipping latest symlinks."""
        rows = []
        for metrics_path in sorted(root.rglob("metrics.json")):
            if "latest" in metrics_path.relative_to(root).parts:
                continue
            row = parse_run(metrics_path)
            if row is not None:
                rows.append(row)
        return pd.DataFrame(rows)

    def moving_average(values: np.ndarray, window: int) -> np.ndarray:
        """Mean over each full window; the result has len(values) - window + 1 points."""
        window = max(1, min(window, len(values)))
        cumsum = np.cumsum(np.insert(values, 0, 0.0))
        return (cumsum[window:] - cumsum[:-window]) / window

    def read_artifact(run_dir: Path, name: str) -> pd.DataFrame:
        path = run_dir / "artifacts" / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    return (
        OUTPUTS_ROOT,
        Path,
        bits_to_int,
        discover_runs,
        int_to_bits,
        load_network_spec,
        mo,
        moving_average,
        np,
        plt,
        read_artifact,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # BN Edge Removal Experiments Dashboard

    `outputs/` 以下の Q-learning run（`outputs/papers/` を含む）を比較・可視化します。

    サイドバーで **モデル → run グループ → run** の順に選びます。run グループは
    同じ親ディレクトリにある run のまとまりで、seed 違いの run が並びます。
    """)
    return


@app.cell
def _(OUTPUTS_ROOT, discover_runs, mo):
    all_runs = discover_runs(OUTPUTS_ROOT)
    mo.stop(
        all_runs.empty,
        mo.callout(
            mo.md(
                "`outputs/` に run がありません。`uv run poe exp` などで実験を実行してください。"
            ),
            kind="warn",
        ),
    )
    model_selector = mo.ui.dropdown(
        options=sorted(all_runs["model"].unique()),
        value=sorted(all_runs["model"].unique())[0],
        label="Model",
    )
    return all_runs, model_selector


@app.cell
def _(all_runs, mo, model_selector):
    model_runs = all_runs[all_runs["model"] == model_selector.value]
    # Newest group first, so the default is the latest experiment.
    group_summary = (
        model_runs.groupby("group")
        .agg(count=("run", "size"), modified=("modified", "max"))
        .sort_values("modified", ascending=False)
    )
    group_options = {
        f"{group} ({row.count} runs)": group for group, row in group_summary.iterrows()
    }
    group_selector = mo.ui.dropdown(
        options=group_options,
        value=next(iter(group_options)),
        label="Run group",
        full_width=True,
    )
    return group_selector, model_runs


@app.cell
def _(group_selector, mo, model_runs):
    group_runs = (
        model_runs[model_runs["group"] == group_selector.value]
        .sort_values("run")
        .reset_index(drop=True)
    )
    run_selector = mo.ui.dropdown(
        options={
            f"{row.run} | success={row.success_rate:.3f}": row.path
            for row in group_runs.itertuples(index=False)
        },
        value=f"{group_runs.iloc[0]['run']} | success={group_runs.iloc[0]['success_rate']:.3f}",
        label="Run",
        full_width=True,
    )
    return group_runs, run_selector


@app.cell
def _(Path, group_runs, load_network_spec, read_artifact, run_selector):
    selected_run = group_runs[group_runs["path"] == run_selector.value].iloc[0]
    selected_run_dir = Path(selected_run["path"])
    network_spec = load_network_spec(selected_run["model"])
    edge_labels = [
        f"x{edge.source}→x{edge.target}"
        for edge in sorted(network_spec.removable_edges, key=lambda edge: edge.index)
    ]
    training_df = read_artifact(selected_run_dir, "training.csv")
    trajectories_df = read_artifact(selected_run_dir, "trajectories.csv")
    actions_df = read_artifact(selected_run_dir, "actions.csv")
    return (
        actions_df,
        edge_labels,
        selected_run,
        selected_run_dir,
        training_df,
        trajectories_df,
    )


@app.cell
def _(actions_df, int_to_bits, mo, selected_run):
    init_options = (
        {
            f"{state} ({''.join(map(str, int_to_bits(int(state), int(selected_run['n_nodes']))))})": int(
                state
            )
            for state in sorted(actions_df["init_state"].unique())
        }
        if not actions_df.empty
        else {"(no actions.csv)": None}
    )
    init_selector = mo.ui.dropdown(
        options=init_options, value=next(iter(init_options)), label="Initial state"
    )
    smooth_slider = mo.ui.slider(
        1, 2000, step=1, value=200, label="Smoothing window (episodes)"
    )
    return init_selector, smooth_slider


@app.cell
def _(
    group_selector,
    init_selector,
    mo,
    model_selector,
    run_selector,
    smooth_slider,
):
    mo.sidebar(
        mo.vstack(
            [
                mo.md("## Controls"),
                model_selector,
                group_selector,
                run_selector,
                init_selector,
                smooth_slider,
            ],
            gap=1.0,
        ),
        footer=mo.md(
            "`Initial state` を変えると軌道と行動タイムラインが切り替わります。"
        ),
        width="340px",
    )
    return


@app.cell
def _(group_runs, mo, selected_run):
    metric_columns = ["success_rate", "violation_rate", "avg_first_goal_step"]
    group_stats = group_runs[metric_columns].agg(["mean", "std"]).T
    stat_cards = mo.hstack(
        [
            mo.stat(
                value=f"{group_stats.loc[column, 'mean']:.3f}",
                label=column,
                caption=(
                    f"± {group_stats.loc[column, 'std']:.3f} (SD, n={len(group_runs)})"
                    if len(group_runs) > 1
                    else "1 run"
                ),
                bordered=True,
            )
            for column in metric_columns
        ],
        widths="equal",
    )
    run_table = mo.ui.table(
        group_runs[
            [
                "run",
                "seed",
                *metric_columns,
                "episodes",
                "reach_horizon",
                "max_steps",
                "constraint",
                "action_masking",
            ]
        ],
        selection=None,
        show_column_summaries=False,
    )
    mo.vstack(
        [
            mo.md(f"## Run group: `{selected_run['group']}`"),
            stat_cards,
            run_table,
        ]
    )
    return


@app.cell
def _(group_runs, np, plt):
    compare_fig, compare_ax = plt.subplots(figsize=(10, 3.2))
    run_index = np.arange(len(group_runs))
    bar_width = 0.4
    compare_ax.bar(
        run_index - bar_width / 2,
        group_runs["success_rate"],
        width=bar_width,
        label="success_rate",
        color="#2ca02c",
    )
    compare_ax.bar(
        run_index + bar_width / 2,
        group_runs["violation_rate"],
        width=bar_width,
        label="violation_rate",
        color="#d62728",
    )
    compare_ax.set_xticks(run_index)
    compare_ax.set_xticklabels(group_runs["run"], rotation=30, ha="right")
    compare_ax.set_ylim(0, 1.05)
    compare_ax.set_ylabel("rate")
    compare_ax.set_title("Runs in this group")
    compare_ax.grid(axis="y", alpha=0.3)
    compare_ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    compare_fig.tight_layout()
    compare_fig
    return


@app.cell
def _(mo, selected_run, selected_run_dir):
    mo.md(f"""
    ## Selected run: `{selected_run["run"]}`

    - path: `{selected_run_dir}`
    - seed: `{selected_run["seed"]}` / episodes: `{selected_run["episodes"]}`
    - horizon: `H={selected_run["reach_horizon"]}, T={selected_run["max_steps"]}`
    - constraint: `{selected_run["constraint"]}`（action masking: `{selected_run["action_masking"]}`）
    - success / violation: **{selected_run["success_rate"]:.3f} / {selected_run["violation_rate"]:.3f}**
    - avg first goal step: **{selected_run["avg_first_goal_step"]:.3f}**
    """)
    return


@app.cell
def _(mo, moving_average, np, plt, smooth_slider, training_df):
    mo.stop(
        training_df.empty,
        mo.md("この run には `artifacts/training.csv` がありません。"),
    )
    training_series = [
        ("reward", "#1f77b4", True),
        ("success", "#2ca02c", True),
        ("violation", "#d62728", True),
        ("epsilon", "#7f7f7f", False),
    ]
    training_fig, training_axes = plt.subplots(
        len(training_series), 1, figsize=(10, 8), sharex=True
    )
    for training_ax, (column, color, smooth) in zip(
        training_axes, training_series, strict=True
    ):
        values = training_df[column].to_numpy(dtype=float)
        if smooth:
            values = moving_average(values, int(smooth_slider.value))
        # A smoothed point is plotted at the last episode of its window.
        episode_index = np.arange(len(training_df) - len(values), len(training_df))
        training_ax.plot(episode_index, values, color=color, linewidth=1.2)
        training_ax.set_ylabel(column)
        training_ax.grid(alpha=0.3)
    training_axes[-1].set_xlabel("episode")
    training_axes[0].set_title(
        f"Training curves (moving average over {int(smooth_slider.value)} episodes)"
    )
    training_fig.tight_layout()
    training_fig
    return


@app.cell
def _(actions_df, init_selector, mo, trajectories_df):
    mo.stop(
        init_selector.value is None or trajectories_df.empty,
        mo.md(
            "この run には評価軌道（`artifacts/trajectories.csv` / `actions.csv`）がありません。"
        ),
    )
    init_state = int(init_selector.value)
    init_actions = actions_df[actions_df["init_state"] == init_state].sort_values("t")
    init_trajectory = trajectories_df[
        trajectories_df["init_state"] == init_state
    ].sort_values("t")
    return init_actions, init_state, init_trajectory


@app.cell
def _(bits_to_int, init_state, init_trajectory, plt, selected_run):
    goal_state_ids = [bits_to_int(bits) for bits in selected_run["goal_states"]]
    trajectory_fig, trajectory_ax = plt.subplots(figsize=(10, 3.2))
    trajectory_ax.plot(
        init_trajectory["t"],
        init_trajectory["state_id"],
        marker="o",
        linewidth=1.5,
        ms=4,
        color="#1f77b4",
    )
    for goal_id in goal_state_ids:
        trajectory_ax.axhline(goal_id, color="#9467bd", linestyle="--", alpha=0.6)
    trajectory_ax.set_xlabel("t")
    trajectory_ax.set_ylabel("state_id")
    trajectory_ax.set_title(
        f"State trajectory (init_state={init_state}; dashed = goal states)"
    )
    trajectory_ax.grid(alpha=0.3)
    trajectory_fig.tight_layout()
    trajectory_fig
    return


@app.cell
def _(edge_labels, init_actions, init_state, int_to_bits, np, plt):
    edge_timeline = np.array(
        [
            int_to_bits(int(action), len(edge_labels))
            for action in init_actions["action"]
        ],
        dtype=int,
    ).T
    timeline_fig, timeline_ax = plt.subplots(figsize=(10, 1.2 + 0.4 * len(edge_labels)))
    timeline_ax.imshow(edge_timeline, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    timeline_ax.set_xticks(range(len(init_actions)))
    timeline_ax.set_xticklabels(init_actions["t"].tolist())
    timeline_ax.set_xlabel("t")
    timeline_ax.set_yticks(range(len(edge_labels)))
    timeline_ax.set_yticklabels(edge_labels)
    timeline_ax.set_title(
        f"Edge removal timeline (init_state={init_state}; black = removed)"
    )
    timeline_fig.tight_layout()
    timeline_fig
    return


@app.cell
def _(mo, selected_run_dir):
    saved_figures = sorted((selected_run_dir / "figures").glob("*.png"))
    mo.stop(not saved_figures)
    mo.accordion(
        {
            "保存済み figure（run スクリプトの出力）": mo.vstack(
                [
                    mo.vstack([mo.md(f"`{path.name}`"), mo.image(src=path)])
                    for path in saved_figures
                ]
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
