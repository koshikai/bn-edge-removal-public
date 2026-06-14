import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def __():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from bn_edge_removal.network_spec import load_network_spec

    def get_model_dimensions(model_name: str) -> tuple[int, int] | None:
        try:
            spec = load_network_spec(model_name)
        except FileNotFoundError:
            return None
        return spec.n_nodes, spec.m_edges

    def collect_reachability_runs(root: Path) -> pd.DataFrame:
        if not root.exists():
            return pd.DataFrame()

        rows: list[dict[str, str | int | float | bool]] = []
        for run_dir in sorted(root.iterdir()):
            if run_dir.name == "latest" or not run_dir.is_dir():
                continue

            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            entries = payload.get("results", [])
            for entry in entries:
                failed_ids = entry.get("failed_initial_state_ids", [])
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
                        "global_reachable": bool(
                            entry.get("global_reachable", False)
                        ),
                        "failed_count": len(failed_ids),
                    }
                )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return df.sort_values(
            ["run_id", "model_name", "mode"],
            ascending=[False, True, True],
        )

    def load_run_artifacts(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        per_state_path = run_dir / "artifacts" / "per_state_results.csv"
        witnesses_path = run_dir / "artifacts" / "witnesses.jsonl"

        if per_state_path.exists():
            per_state_df = pd.read_csv(per_state_path)
            per_state_df["reachable"] = per_state_df["reachable"].astype(bool)
            per_state_df["model_name"] = per_state_df["model_name"].astype(str)
            per_state_df["mode"] = per_state_df["mode"].astype(str)
            per_state_df["initial_state"] = per_state_df["initial_state"].astype(str)
        else:
            per_state_df = pd.DataFrame()

        witness_rows: list[dict[str, object]] = []
        if witnesses_path.exists():
            for line in witnesses_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                witness = item.get("witness", {})
                witness_rows.append(
                    {
                        "model_name": str(item.get("model_name", "")),
                        "mode": str(item.get("mode", "")),
                        "initial_state_id": int(item.get("initial_state_id", -1)),
                        "initial_state": witness.get("states", [[None]])[0],
                        "states": witness.get("states", []),
                        "actions": witness.get("actions", []),
                        "goal_step": int(witness.get("goal_step", -1)),
                    }
                )

        witness_df = pd.DataFrame(witness_rows)
        return per_state_df, witness_df

    return (
        Path,
        collect_reachability_runs,
        get_model_dimensions,
        load_run_artifacts,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def __(mo):
    mo.md(
        "# Reachability Verification Dashboard\n\n"
        "`scripts/verify_reachability.py` の結果を可視化します。\n\n"
        "- model/mode ごとの到達率\n"
        "- 初期状態ごとの到達可否と到達ステップ分布\n"
        "- 失敗初期状態(存在する場合)の一覧\n"
        "- witness 軌道(状態列・行動列)\n"
    )


@app.cell
def __(Path, collect_reachability_runs, mo):
    runs_df = collect_reachability_runs(Path("outputs/reachability"))

    run_selector = None
    if runs_df.empty:
        _ = mo.md("`outputs/reachability` に run がありません。")
    else:
        _run_ids = runs_df["run_id"].drop_duplicates().tolist()
        _options = {run_id: run_id for run_id in _run_ids}
        _default_key = next(iter(_options.keys()))
        run_selector = mo.ui.dropdown(
            options=_options,
            value=_default_key,
            label="Run",
        )
    return run_selector, runs_df


@app.cell
def __(Path, load_run_artifacts, pd, run_selector, runs_df):
    selected_run_dir = None
    run_summary_df = pd.DataFrame()
    per_state_df = pd.DataFrame()
    witness_df = pd.DataFrame()

    if (
        run_selector is not None
        and run_selector.value is not None
        and not runs_df.empty
    ):
        _selected_run_id = run_selector.value
        run_summary_df = runs_df[runs_df["run_id"] == _selected_run_id].copy()
        run_summary_df = run_summary_df.sort_values(["model_name", "mode"])
        selected_run_dir = Path(str(run_summary_df.iloc[0]["path"]))
        per_state_df, witness_df = load_run_artifacts(selected_run_dir)

    return per_state_df, run_summary_df, selected_run_dir, witness_df


@app.cell
def __(mo, pd, run_summary_df, selected_run_dir):
    if run_summary_df.empty or selected_run_dir is None:
        summary_view = mo.md("run を選択するとサマリを表示します。")
        summary_table_df = pd.DataFrame()
    else:
        summary_view = mo.md(
            f"""
            ## Run Summary

            - run_id: `{selected_run_dir.name}`
            - path: `{selected_run_dir}`
            - entries: `{len(run_summary_df)}`
            """
        )
        summary_table_df = run_summary_df[
            [
                "model_name",
                "mode",
                "success_count",
                "total_initial_states",
                "success_rate",
                "failed_count",
                "global_reachable",
            ]
        ].copy()

    summary_view
    summary_table_df
    return (summary_table_df,)


@app.cell
def __(mo, np, plt, run_summary_df):
    success_rate_fig = None
    if not run_summary_df.empty:
        _labels = [
            f"{row.model_name}\n{row.mode}"
            for row in run_summary_df.itertuples(index=False)
        ]
        _values = run_summary_df["success_rate"].to_numpy(dtype=float)

        success_rate_fig, _ax = plt.subplots(figsize=(8.0, 4.2))
        _bars = _ax.bar(
            _labels,
            _values,
            color="#2a9d8f",
            edgecolor="#1f5f58",
        )
        _ax.set_ylim(0.0, 1.05)
        _ax.set_ylabel("success_rate")
        _ax.set_title("Success Rate by Model/Mode")
        _ax.grid(axis="y", alpha=0.25)

        for _bar, _value in zip(_bars, _values, strict=True):
            _ax.text(
                _bar.get_x() + _bar.get_width() / 2.0,
                min(1.02, _value + 0.03),
                f"{_value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        success_rate_fig.tight_layout()

    success_view = (
        success_rate_fig
        if success_rate_fig is not None
        else mo.md("可視化できるサマリがありません。")
    )
    success_view
    return (success_rate_fig,)


@app.cell
def __(mo, per_state_df):
    model_mode_selector = None
    if per_state_df.empty:
        _ = mo.md("per-state データがありません。")
    else:
        _pairs = (
            per_state_df[["model_name", "mode"]]
            .drop_duplicates()
            .sort_values(["model_name", "mode"])
        )
        _options: dict[str, str] = {}
        for _row in _pairs.itertuples(index=False):
            _key = f"{_row.model_name}::{_row.mode}"
            _options[f"{_row.model_name} ({_row.mode})"] = _key

        _default_key = next(iter(_options.keys()))
        model_mode_selector = mo.ui.dropdown(
            options=_options,
            value=_default_key,
            label="Model / Mode",
        )
    return (model_mode_selector,)


@app.cell
def __(model_mode_selector, pd, per_state_df):
    filtered_state_df = pd.DataFrame()
    selected_model = None
    selected_mode = None

    if model_mode_selector is not None and model_mode_selector.value is not None:
        selected_model, selected_mode = model_mode_selector.value.split("::")
        _mask = (per_state_df["model_name"] == selected_model) & (
            per_state_df["mode"] == selected_mode
        )
        filtered_state_df = per_state_df[_mask].copy().sort_values("initial_state_id")

    return filtered_state_df, selected_model, selected_mode


@app.cell
def __(filtered_state_df, mo, pd):
    if filtered_state_df.empty:
        per_state_overview = mo.md("model/mode を選択すると初期状態分析を表示します。")
        per_state_table_df = pd.DataFrame()
    else:
        _total = len(filtered_state_df)
        _success = int(filtered_state_df["reachable"].sum())
        _fail = _total - _success
        _min_goal_step = int(filtered_state_df["goal_step"].min())
        _max_goal_step = int(filtered_state_df["goal_step"].max())

        per_state_overview = mo.md(
            f"""
            ## Per-State Overview

            - total initial states: **{_total}**
            - reachable: **{_success}**
            - unreachable: **{_fail}**
            - goal_step range: **{_min_goal_step} .. {_max_goal_step}**
            """
        )

        per_state_table_df = filtered_state_df[
            [
                "initial_state_id",
                "initial_state",
                "reachable",
                "goal_step",
                "action_count",
            ]
        ].copy()

    per_state_overview
    per_state_table_df
    return (per_state_table_df,)


@app.cell
def __(filtered_state_df, mo, np, plt):
    goal_step_fig = None
    if not filtered_state_df.empty:
        _goal_steps = filtered_state_df["goal_step"].to_numpy(dtype=int)
        _min_step = int(np.min(_goal_steps))
        _max_step = int(np.max(_goal_steps))
        _bins = np.arange(_min_step, _max_step + 2) - 0.5

        goal_step_fig, _ax = plt.subplots(figsize=(8.0, 3.8))
        _ax.hist(_goal_steps, bins=_bins, color="#264653", edgecolor="white")
        _ax.set_xticks(np.arange(_min_step, _max_step + 1))
        _ax.set_xlabel("goal_step")
        _ax.set_ylabel("count")
        _ax.set_title("Goal Step Distribution")
        _ax.grid(axis="y", alpha=0.25)
        goal_step_fig.tight_layout()

    goal_step_view = (
        goal_step_fig
        if goal_step_fig is not None
        else mo.md("goal_step 分布を描画できるデータがありません。")
    )
    goal_step_view
    return (goal_step_fig,)


@app.cell
def __(filtered_state_df, mo, pd):
    failed_table_df = pd.DataFrame()
    if filtered_state_df.empty:
        failed_view = mo.md("")
    else:
        failed_table_df = filtered_state_df[~filtered_state_df["reachable"]].copy()
        if failed_table_df.empty:
            failed_view = mo.callout(
                "この model/mode では失敗初期状態はありません。",
                kind="success",
            )
        else:
            failed_view = mo.callout(
                f"失敗初期状態が {len(failed_table_df)} 件あります。",
                kind="warn",
            )
            failed_table_df = failed_table_df[
                ["initial_state_id", "initial_state", "goal_step", "action_count"]
            ].copy()

    failed_view
    failed_table_df
    return (failed_table_df,)


@app.cell
def __(filtered_state_df, mo, pd, witness_df):
    selected_witness_df = pd.DataFrame()
    witness_selector = None

    if not filtered_state_df.empty and not witness_df.empty:
        _model_name = str(filtered_state_df.iloc[0]["model_name"])
        _mode_name = str(filtered_state_df.iloc[0]["mode"])
        selected_witness_df = witness_df[
            (witness_df["model_name"] == _model_name)
            & (witness_df["mode"] == _mode_name)
        ].copy()

    if selected_witness_df.empty:
        _ = mo.md("witness データがありません。")
    else:
        _options = {
            f"init={int(_row.initial_state_id)} | goal_step={int(_row.goal_step)}": int(
                _row.initial_state_id
            )
            for _row in selected_witness_df.itertuples(index=False)
        }
        _default_key = next(iter(_options.keys()))
        witness_selector = mo.ui.dropdown(
            options=_options,
            value=_default_key,
            label="Witness Initial State",
        )
    return selected_witness_df, witness_selector


@app.cell
def __(mo, model_mode_selector, run_selector, witness_selector):
    controls: list[object] = [mo.md("## Controls")]
    if run_selector is None:
        controls.append(mo.md("- Run: (not available)"))
    else:
        controls.append(run_selector)

    if model_mode_selector is None:
        controls.append(mo.md("- Model / Mode: (not available)"))
    else:
        controls.append(model_mode_selector)

    if witness_selector is None:
        controls.append(mo.md("- Witness: (not available)"))
    else:
        controls.append(witness_selector)

    sidebar = mo.sidebar(mo.vstack(controls, gap=1))
    sidebar
    return (sidebar,)


@app.cell
def __(selected_witness_df, witness_selector):
    chosen_witness = None
    if (
        witness_selector is not None
        and witness_selector.value is not None
        and not selected_witness_df.empty
    ):
        chosen_witness = selected_witness_df[
            selected_witness_df["initial_state_id"] == witness_selector.value
        ].iloc[0]

    return (chosen_witness,)


@app.cell
def __(chosen_witness, get_model_dimensions, mo, np, pd, plt):
    state_timeline_fig = None
    action_timeline_fig = None
    witness_trajectory_df = pd.DataFrame()

    if chosen_witness is None:
        witness_detail_view = mo.md("witness を選択すると軌道詳細を表示します。")
    else:
        _model_name = str(chosen_witness["model_name"])
        _states = list(chosen_witness["states"])
        _actions = list(chosen_witness["actions"])
        _goal_step = int(chosen_witness["goal_step"])
        _initial_state_id = int(chosen_witness["initial_state_id"])
        _dimensions = get_model_dimensions(_model_name)
        _m_edges = None
        if _dimensions is not None:
            _, _m_edges = _dimensions
        elif _actions:
            _m_edges = len(np.binary_repr(int(_actions[0])))

        _state_matrix = np.array(_states, dtype=int)
        state_timeline_fig, _state_ax = plt.subplots(figsize=(8.8, 3.8))
        _state_im = _state_ax.imshow(
            _state_matrix.T,
            aspect="auto",
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
        _state_ax.set_xlabel("step")
        _state_ax.set_ylabel("node")
        _state_ax.set_yticks(range(_state_matrix.shape[1]))
        _state_ax.set_yticklabels(
            [f"x{i}" for i in range(1, _state_matrix.shape[1] + 1)]
        )
        _state_ax.set_xticks(range(_state_matrix.shape[0]))
        _state_ax.set_title(
            "Witness State Timeline "
            f"(init={_initial_state_id}, goal_step={_goal_step})"
        )
        state_timeline_fig.colorbar(_state_im, ax=_state_ax, fraction=0.03, pad=0.02)
        state_timeline_fig.tight_layout()

        if len(_actions) > 0:
            if _m_edges is None:
                _m_edges = len(np.binary_repr(int(max(_actions))))
            _action_matrix = np.zeros((_m_edges, len(_actions)), dtype=int)
            for _action_t, _action_id in enumerate(_actions):
                _bits = np.binary_repr(int(_action_id), width=_m_edges)
                _action_matrix[:, _action_t] = np.array(
                    [int(_ch) for _ch in _bits],
                    dtype=int,
                )

            action_timeline_fig, _action_ax = plt.subplots(figsize=(8.8, 3.8))
            _action_im = _action_ax.imshow(
                _action_matrix,
                aspect="auto",
                cmap="Greys",
                vmin=0,
                vmax=1,
            )
            _action_ax.set_xlabel("t")
            _action_ax.set_ylabel("edge control")
            _action_ax.set_yticks(range(_m_edges))
            _action_ax.set_yticklabels([f"u{i}" for i in range(1, _m_edges + 1)])
            _action_ax.set_xticks(range(len(_actions)))
            _action_ax.set_title("Witness Action Timeline (1=remove)")
            action_timeline_fig.colorbar(
                _action_im,
                ax=_action_ax,
                fraction=0.03,
                pad=0.02,
            )
            action_timeline_fig.tight_layout()

        _trajectory_rows = [
            {
                "step": _step,
                "state": "".join(str(_bit) for _bit in _state),
                "is_goal_step": _step == _goal_step,
                "action_to_next": _actions[_step] if _step < len(_actions) else None,
            }
            for _step, _state in enumerate(_states)
        ]
        witness_trajectory_df = pd.DataFrame(_trajectory_rows)
        witness_detail_view = mo.md("### Witness Trajectory")

    witness_detail_view
    state_view = (
        state_timeline_fig
        if state_timeline_fig is not None
        else mo.md("state timeline を描画できません。")
    )
    state_view
    if action_timeline_fig is not None:
        action_timeline_fig
    witness_trajectory_df
    return action_timeline_fig, state_timeline_fig, witness_trajectory_df


if __name__ == "__main__":
    app.run()
