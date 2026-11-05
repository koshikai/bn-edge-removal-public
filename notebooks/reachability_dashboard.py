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

    from bn_edge_removal.models.encoding import int_to_bits
    from bn_edge_removal.models.network_spec import load_network_spec

    REACHABILITY_ROOT = Path("outputs/reachability")

    def collect_reachability_runs(root: Path) -> pd.DataFrame:
        """One row per (run, model, mode) from each run's metrics.json."""
        rows = []
        for metrics_path in sorted(root.glob("*/metrics.json")):
            run_dir = metrics_path.parent
            if run_dir.name == "latest":
                continue
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            for entry in payload.get("results", []):
                rows.append(
                    {
                        "run_id": run_dir.name,
                        "path": str(run_dir),
                        "model_name": str(entry.get("model_name", "")),
                        "mode": str(entry.get("mode", "")),
                        "success_count": int(entry.get("success_count", 0)),
                        "total_initial_states": int(
                            entry.get("total_initial_states", 0)
                        ),
                        "success_rate": float(entry.get("success_rate", 0.0)),
                        "global_reachable": bool(entry.get("global_reachable", False)),
                        "failed_count": len(entry.get("failed_initial_state_ids", [])),
                    }
                )
        return pd.DataFrame(rows)

    def load_per_state(run_dir: Path) -> pd.DataFrame:
        path = run_dir / "artifacts" / "per_state_results.csv"
        if not path.exists():
            return pd.DataFrame()
        # Keep initial_state as a bit string such as "00101".
        return pd.read_csv(path, dtype={"initial_state": str})

    def load_witnesses(run_dir: Path) -> pd.DataFrame:
        path = run_dir / "artifacts" / "witnesses.jsonl"
        if not path.exists():
            return pd.DataFrame()
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            witness = item.get("witness", {})
            rows.append(
                {
                    "model_name": str(item.get("model_name", "")),
                    "mode": str(item.get("mode", "")),
                    "initial_state_id": int(item.get("initial_state_id", -1)),
                    "states": witness.get("states", []),
                    "actions": witness.get("actions", []),
                    "goal_step": int(witness.get("goal_step", -1)),
                }
            )
        return pd.DataFrame(rows)

    return (
        Path,
        REACHABILITY_ROOT,
        collect_reachability_runs,
        int_to_bits,
        load_network_spec,
        load_per_state,
        load_witnesses,
        mo,
        np,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Reachability Verification Dashboard

    `scripts/verify_reachability.py`（`uv run poe verify-reach`）の結果を可視化します。

    - model / mode ごとの到達率
    - 初期状態ごとの到達可否と、ゴール到達ステップの分布
    - 到達できなかった初期状態の一覧
    - witness 軌道（状態列と行動列）
    """)
    return


@app.cell
def _(REACHABILITY_ROOT, collect_reachability_runs, mo):
    runs_df = collect_reachability_runs(REACHABILITY_ROOT)
    mo.stop(
        runs_df.empty,
        mo.callout(
            mo.md(
                f"`{REACHABILITY_ROOT}` に run がありません。"
                "`uv run poe verify-reach` で作成してください。"
            ),
            kind="warn",
        ),
    )
    run_ids = sorted(runs_df["run_id"].unique(), reverse=True)
    run_selector = mo.ui.dropdown(options=run_ids, value=run_ids[0], label="Run")
    return run_selector, runs_df


@app.cell
def _(Path, load_per_state, load_witnesses, mo, run_selector, runs_df):
    run_summary_df = (
        runs_df[runs_df["run_id"] == run_selector.value]
        .sort_values(["model_name", "mode"])
        .reset_index(drop=True)
    )
    selected_run_dir = Path(run_summary_df.iloc[0]["path"])
    per_state_df = load_per_state(selected_run_dir)
    witness_df = load_witnesses(selected_run_dir)

    model_mode_options = {
        f"{row.model_name} ({row.mode})": (row.model_name, row.mode)
        for row in run_summary_df.itertuples(index=False)
    }
    model_mode_selector = mo.ui.dropdown(
        options=model_mode_options,
        value=next(iter(model_mode_options)),
        label="Model / Mode",
    )
    return (
        model_mode_selector,
        per_state_df,
        run_summary_df,
        selected_run_dir,
        witness_df,
    )


@app.cell
def _(model_mode_selector, per_state_df, witness_df):
    selected_model, selected_mode = model_mode_selector.value
    filtered_state_df = (
        per_state_df[
            (per_state_df["model_name"] == selected_model)
            & (per_state_df["mode"] == selected_mode)
        ].sort_values("initial_state_id")
        if not per_state_df.empty
        else per_state_df
    )
    model_witnesses = (
        witness_df[
            (witness_df["model_name"] == selected_model)
            & (witness_df["mode"] == selected_mode)
        ]
        .sort_values("initial_state_id")
        .reset_index(drop=True)
        if not witness_df.empty
        else witness_df
    )
    return filtered_state_df, model_witnesses, selected_mode, selected_model


@app.cell
def _(mo, model_witnesses):
    witness_options = (
        {
            f"init={row.initial_state_id} | goal_step={row.goal_step}": row.initial_state_id
            for row in model_witnesses.itertuples(index=False)
        }
        if not model_witnesses.empty
        else {"(no witnesses)": None}
    )
    witness_selector = mo.ui.dropdown(
        options=witness_options,
        value=next(iter(witness_options)),
        label="Witness initial state",
    )
    return (witness_selector,)


@app.cell
def _(mo, model_mode_selector, run_selector, witness_selector):
    mo.sidebar(
        mo.vstack(
            [mo.md("## Controls"), run_selector, model_mode_selector, witness_selector],
            gap=1.0,
        ),
        width="320px",
    )
    return


@app.cell
def _(mo, run_summary_df, selected_run_dir):
    mo.vstack(
        [
            mo.md(f"## Run summary\n\n`{selected_run_dir}`"),
            mo.ui.table(
                run_summary_df[
                    [
                        "model_name",
                        "mode",
                        "success_count",
                        "total_initial_states",
                        "success_rate",
                        "failed_count",
                        "global_reachable",
                    ]
                ],
                selection=None,
                show_column_summaries=False,
            ),
        ]
    )
    return


@app.cell
def _(plt, run_summary_df):
    rate_labels = [
        f"{row.model_name}\n{row.mode}"
        for row in run_summary_df.itertuples(index=False)
    ]
    rate_fig, rate_ax = plt.subplots(figsize=(8.0, 3.8))
    rate_bars = rate_ax.bar(
        rate_labels,
        run_summary_df["success_rate"],
        color="#2a9d8f",
        edgecolor="#1f5f58",
    )
    rate_ax.bar_label(rate_bars, fmt="%.3f", fontsize=9, padding=2)
    rate_ax.set_ylim(0.0, 1.1)
    rate_ax.set_ylabel("success_rate")
    rate_ax.set_title("Reachable fraction of initial states, by model / mode")
    rate_ax.grid(axis="y", alpha=0.25)
    rate_fig.tight_layout()
    rate_fig
    return


@app.cell
def _(filtered_state_df, mo, selected_mode, selected_model):
    mo.stop(
        filtered_state_df.empty,
        mo.md("この run には `artifacts/per_state_results.csv` がありません。"),
    )
    reachable_count = int(filtered_state_df["reachable"].sum())
    unreachable_df = filtered_state_df[~filtered_state_df["reachable"]]
    unreachable_view = (
        mo.callout("すべての初期状態からゴールに到達できます。", kind="success")
        if unreachable_df.empty
        else mo.vstack(
            [
                mo.callout(
                    f"到達できない初期状態が {len(unreachable_df)} 件あります。",
                    kind="warn",
                ),
                mo.ui.table(
                    unreachable_df[["initial_state_id", "initial_state"]],
                    selection=None,
                ),
            ]
        )
    )
    mo.vstack(
        [
            mo.md(f"## Per-state results: {selected_model} ({selected_mode})"),
            mo.hstack(
                [
                    mo.stat(
                        len(filtered_state_df), label="initial states", bordered=True
                    ),
                    mo.stat(reachable_count, label="reachable", bordered=True),
                    mo.stat(
                        f"{filtered_state_df['goal_step'].min()}–{filtered_state_df['goal_step'].max()}",
                        label="goal_step range",
                        bordered=True,
                    ),
                ],
                widths="equal",
            ),
            unreachable_view,
            mo.accordion(
                {
                    "初期状態ごとの結果（全件）": mo.ui.table(
                        filtered_state_df[
                            [
                                "initial_state_id",
                                "initial_state",
                                "reachable",
                                "goal_step",
                                "action_count",
                            ]
                        ],
                        selection=None,
                    )
                }
            ),
        ]
    )
    return


@app.cell
def _(filtered_state_df, mo, np, plt):
    mo.stop(filtered_state_df.empty or not filtered_state_df["reachable"].any())
    goal_steps = filtered_state_df.loc[
        filtered_state_df["reachable"], "goal_step"
    ].to_numpy(dtype=int)
    step_bins = np.arange(goal_steps.min(), goal_steps.max() + 2) - 0.5
    goal_step_fig, goal_step_ax = plt.subplots(figsize=(8.0, 3.2))
    goal_step_ax.hist(goal_steps, bins=step_bins, color="#264653", edgecolor="white")
    goal_step_ax.set_xticks(np.arange(goal_steps.min(), goal_steps.max() + 1))
    goal_step_ax.set_xlabel("goal_step")
    goal_step_ax.set_ylabel("initial states")
    goal_step_ax.set_title("Steps needed to reach the goal (reachable initial states)")
    goal_step_ax.grid(axis="y", alpha=0.25)
    goal_step_fig.tight_layout()
    goal_step_fig
    return


@app.cell
def _(mo, model_witnesses, witness_selector):
    mo.stop(
        witness_selector.value is None,
        mo.md("この model / mode には witness がありません。"),
    )
    chosen_witness = model_witnesses[
        model_witnesses["initial_state_id"] == witness_selector.value
    ].iloc[0]
    return (chosen_witness,)


@app.cell
def _(chosen_witness, int_to_bits, load_network_spec, np, plt):
    witness_spec = load_network_spec(str(chosen_witness["model_name"]))
    witness_edge_labels = [
        f"u{edge.index}: x{edge.source}→x{edge.target}"
        for edge in sorted(witness_spec.removable_edges, key=lambda edge: edge.index)
    ]
    witness_states = np.array(chosen_witness["states"], dtype=int)
    witness_actions = np.array(
        [
            int_to_bits(int(action), len(witness_edge_labels))
            for action in chosen_witness["actions"]
        ],
        dtype=int,
    ).reshape(-1, len(witness_edge_labels))

    witness_fig, (state_ax, action_ax) = plt.subplots(
        2,
        1,
        figsize=(9, 2.0 + 0.3 * (witness_states.shape[1] + len(witness_edge_labels))),
        sharex=True,
        height_ratios=[witness_states.shape[1], max(len(witness_edge_labels), 1)],
    )
    state_ax.imshow(witness_states.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    state_ax.set_yticks(range(witness_states.shape[1]))
    state_ax.set_yticklabels(
        witness_spec.node_labels
        or [f"x{index}" for index in range(1, witness_states.shape[1] + 1)]
    )
    state_ax.axvline(
        int(chosen_witness["goal_step"]), color="#e76f51", linestyle="--", linewidth=1.5
    )
    state_ax.set_title(
        f"Witness from init={chosen_witness['initial_state_id']} "
        f"(blue = node on; dashed = goal step {chosen_witness['goal_step']})"
    )
    # The action at step t moves the state from t to t + 1.
    action_ax.imshow(
        witness_actions.T,
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
        extent=(-0.5, len(witness_actions) - 0.5, len(witness_edge_labels) - 0.5, -0.5),
    )
    action_ax.set_yticks(range(len(witness_edge_labels)))
    action_ax.set_yticklabels(witness_edge_labels)
    action_ax.set_xticks(range(witness_states.shape[0]))
    action_ax.set_xlabel("step")
    action_ax.set_title("Edge removals applied at each step (black = removed)")
    witness_fig.tight_layout()
    witness_fig
    return


if __name__ == "__main__":
    app.run()
