# ruff: noqa: PLR1711, PLR1714
import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def __():
    import re

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    from bn_edge_removal.network_spec import (
        NetworkEdge,
        load_network_spec,
    )

    def circular_node_positions(
        n_nodes: int, radius: float = 1.4
    ) -> dict[int, tuple[float, float]]:
        angles = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False) + (np.pi / 2.0)
        return {
            node_id: (radius * float(np.cos(theta)), radius * float(np.sin(theta)))
            for node_id, theta in enumerate(angles, start=1)
        }

    def filter_edges(
        edges: tuple[NetworkEdge, ...], sign_filter: str, removable_filter: str
    ) -> list[NetworkEdge]:
        filtered: list[NetworkEdge] = []
        for edge in edges:
            if sign_filter != "all" and edge.sign != sign_filter:
                continue
            if removable_filter == "removable" and not edge.removable:
                continue
            if removable_filter == "fixed" and edge.removable:
                continue
            filtered.append(edge)
        return filtered

    def edge_style(edge: NetworkEdge) -> tuple[str, str, float, float]:
        color = "#1f77b4" if edge.sign == "activation" else "#d62728"
        linestyle = "--" if edge.removable else "-"
        linewidth = 2.0
        alpha = 0.95
        return color, linestyle, linewidth, alpha

    def edge_table(edges: list[NetworkEdge]) -> pd.DataFrame:
        rows: list[dict[str, str | int | bool]] = []
        for idx, edge in enumerate(edges, start=1):
            control_id = (
                f"u{edge.removable_index}"
                if edge.removable and edge.removable_index is not None
                else "-"
            )
            rows.append(
                {
                    "edge_id": idx,
                    "source": edge.source,
                    "target": edge.target,
                    "sign": edge.sign,
                    "removable": edge.removable,
                    "control_id": control_id,
                }
            )
        return pd.DataFrame(rows)

    def control_mapping_table(edges: tuple[NetworkEdge, ...]) -> pd.DataFrame:
        rows: list[dict[str, str | int]] = []
        removable_edges: list[tuple[int, NetworkEdge]] = []
        for edge in edges:
            if edge.removable and edge.removable_index is not None:
                removable_edges.append((edge.removable_index, edge))

        for control_idx, edge in sorted(removable_edges, key=lambda item: item[0]):
            rows.append(
                {
                    "u_k": f"u{control_idx}",
                    "u_(j,i)(t)": f"u_{{{edge.source},{edge.target}}}(t)",
                    "edge": f"x{edge.source} -> x{edge.target}",
                    "sign": edge.sign,
                }
            )
        return pd.DataFrame(rows)

    token_re = re.compile(r"x\d+'?|u\d+|and|or|not|\(|\)")

    def _tokenize_boolean_expr(expr: str) -> list[str]:
        normalized = " ".join(expr.strip().split())
        tokens: list[str] = []
        cursor = 0
        for match in token_re.finditer(normalized):
            if match.start() > cursor:
                remainder = normalized[cursor : match.start()].strip()
                if remainder:
                    tokens.append(remainder)
            tokens.append(match.group(0))
            cursor = match.end()
        tail = normalized[cursor:].strip()
        if tail:
            tokens.append(tail)
        return tokens

    def _control_lookup(edges: tuple[NetworkEdge, ...]) -> dict[int, tuple[int, int]]:
        lookup: dict[int, tuple[int, int]] = {}
        for edge in edges:
            if edge.removable and edge.removable_index is not None:
                lookup[edge.removable_index] = (edge.source, edge.target)
        return lookup

    def boolean_expr_to_latex(
        expr: str, control_lookup: dict[int, tuple[int, int]]
    ) -> str:
        latex_tokens: list[str] = []
        for token in _tokenize_boolean_expr(expr):
            if token == "and":
                latex_tokens.append(r"\land")
                continue
            if token == "or":
                latex_tokens.append(r"\lor")
                continue
            if token == "not":
                latex_tokens.append(r"\neg")
                continue
            if token in {"(", ")"}:
                latex_tokens.append(token)
                continue

            x_match = re.fullmatch(r"x(\d+)'?", token)
            if x_match is not None:
                node_id = x_match.group(1)
                if token.endswith("'"):
                    latex_tokens.append(f"x_{{{node_id}}}(t+1)")
                else:
                    latex_tokens.append(f"x_{{{node_id}}}(t)")
                continue

            u_match = re.fullmatch(r"u(\d+)", token)
            if u_match is not None:
                control_idx = int(u_match.group(1))
                if control_idx in control_lookup:
                    source, target = control_lookup[control_idx]
                    latex_tokens.append(f"u_{{{source},{target}}}(t)")
                else:
                    latex_tokens.append(f"u_{{{control_idx}}}(t)")
                continue

            latex_tokens.append(token)

        result = " ".join(latex_tokens)
        return result.replace("( ", "(").replace(" )", ")")

    def equation_to_latex(
        equation: str, control_lookup: dict[int, tuple[int, int]]
    ) -> str:
        normalized = " ".join(equation.strip().split())
        if "=" not in normalized:
            return normalized
        lhs_raw, rhs_raw = normalized.split("=", maxsplit=1)
        lhs = lhs_raw.strip()
        lhs_match = re.fullmatch(r"x(\d+)'", lhs)
        lhs_tex = lhs if lhs_match is None else f"x_{{{lhs_match.group(1)}}}(t+1)"
        rhs_tex = boolean_expr_to_latex(rhs_raw.strip(), control_lookup)
        return f"{lhs_tex} &= {rhs_tex}"

    def equations_markdown(
        equations: tuple[str, ...], edges: tuple[NetworkEdge, ...]
    ) -> str:
        control_lookup = _control_lookup(edges)
        latex_lines = [
            equation_to_latex(equation, control_lookup) for equation in equations
        ]
        return (
            "$$\n"
            "\\begin{aligned}\n"
            + " \\\\\n".join(latex_lines)
            + "\n\\end{aligned}\n"
            "$$"
        )

    def draw_network_graph(
        model_name: str,
        n_nodes: int,
        edges: list[NetworkEdge],
        show_control_ids: bool,
    ):
        positions = circular_node_positions(n_nodes)
        fig, ax = plt.subplots(figsize=(8.6, 7.2))

        for node_id, (x_pos, y_pos) in positions.items():
            node_circle = plt.Circle(
                (x_pos, y_pos),
                0.13,
                facecolor="#f6f6f6",
                edgecolor="#222222",
                linewidth=1.2,
                zorder=3,
            )
            ax.add_patch(node_circle)
            ax.text(
                x_pos,
                y_pos,
                f"x{node_id}",
                ha="center",
                va="center",
                fontsize=11,
                zorder=4,
            )

        edge_keys = {(edge.source, edge.target) for edge in edges if edge.source != edge.target}

        for edge in edges:
            color, linestyle, linewidth, alpha = edge_style(edge)
            source_pos = positions[edge.source]
            target_pos = positions[edge.target]

            if edge.source == edge.target:
                loop_patch = FancyArrowPatch(
                    posA=(source_pos[0] + 0.11, source_pos[1] + 0.03),
                    posB=(source_pos[0] + 0.11, source_pos[1] - 0.03),
                    connectionstyle="arc3,rad=1.2",
                    arrowstyle="-|>",
                    mutation_scale=12.0,
                    shrinkA=4.0,
                    shrinkB=4.0,
                    zorder=2,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                )
                ax.add_patch(loop_patch)
                label_x = source_pos[0] + 0.36
                label_y = source_pos[1] + 0.23
            else:
                is_bidirectional = (edge.target, edge.source) in edge_keys
                # For bidirectional pairs, using the same rad sign for both
                # directions yields opposite sides geometrically.
                curve = 0.22 if is_bidirectional else 0.0

                edge_patch = FancyArrowPatch(
                    posA=source_pos,
                    posB=target_pos,
                    connectionstyle=f"arc3,rad={curve}",
                    arrowstyle="-|>",
                    mutation_scale=12.0,
                    shrinkA=14.0,
                    shrinkB=14.0,
                    zorder=2,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                )
                ax.add_patch(edge_patch)

                path_vertices = np.asarray(edge_patch.get_path().vertices, dtype=float)
                mid_idx = int(path_vertices.shape[0] // 2)
                midpoint_x = float(path_vertices[mid_idx, 0])
                midpoint_y = float(path_vertices[mid_idx, 1])
                direction_x = target_pos[0] - source_pos[0]
                direction_y = target_pos[1] - source_pos[1]
                norm = max((direction_x**2 + direction_y**2) ** 0.5, 1e-9)
                normal_x = -direction_y / norm
                normal_y = direction_x / norm
                offset = 0.05 + abs(curve) * 0.22
                signed = 1.0 if curve >= 0 else -1.0
                label_x = midpoint_x + normal_x * offset * signed
                label_y = midpoint_y + normal_y * offset * signed

            if show_control_ids and edge.removable and edge.removable_index is not None:
                ax.text(
                    label_x,
                    label_y,
                    f"u{edge.removable_index}",
                    fontsize=8.8,
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": color,
                        "alpha": 0.85,
                    },
                    color=color,
                    zorder=5,
                )

        legend_handles = [
            Line2D([0], [0], color="#1f77b4", lw=2.2, ls="-", label="Activation"),
            Line2D([0], [0], color="#d62728", lw=2.2, ls="-", label="Inhibition"),
            Line2D(
                [0],
                [0],
                color="#555555",
                lw=2.0,
                ls="--",
                alpha=0.95,
                label="Removable edge",
            ),
            Line2D(
                [0],
                [0],
                color="#555555",
                lw=2.0,
                ls="-",
                alpha=0.95,
                label="Fixed edge",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.00),
            frameon=False,
        )
        ax.set_title(f"{model_name.upper()} network structure", fontsize=13)
        ax.set_aspect("equal")
        ax.set_xlim(-2.0, 2.1)
        ax.set_ylim(-1.9, 1.9)
        ax.axis("off")
        fig.tight_layout()
        return fig

    return (
        control_mapping_table,
        draw_network_graph,
        edge_table,
        equations_markdown,
        filter_edges,
        load_network_spec,
        mo,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # Network Graph Dashboard

        YAML で管理されたネットワーク定義をもとに、構造グラフを可視化します。

        - **色**: 活性化（青）/ 抑制（赤）
        - **線種**: 除去可能（破線）/ 除去不可（実線）
        - **ラベル**: 除去可能エッジの制御ID `u_k`（表示ON/OFF可能）
        - **補助表示**: `u_k ↔ edge` 対応表と論理式（LaTeX）
        """
    )
    return


@app.cell
def __(mo):
    model_selector = mo.ui.dropdown(
        options={
            "Cortical": "cortical",
            "Wnt5a": "wnt5a",
            "Cell Cycle 10": "cell_cycle10",
        },
        value="Cortical",
        label="Model",
    )
    sign_filter_selector = mo.ui.dropdown(
        options={"All": "all", "Activation": "activation", "Inhibition": "inhibition"},
        value="All",
        label="Sign filter",
    )
    removable_filter_selector = mo.ui.dropdown(
        options={"All": "all", "Removable": "removable", "Fixed": "fixed"},
        value="All",
        label="Removability filter",
    )
    show_control_id_checkbox = mo.ui.checkbox(
        value=True,
        label="Show removable control IDs (u_k)",
    )
    return (
        model_selector,
        removable_filter_selector,
        show_control_id_checkbox,
        sign_filter_selector,
    )


@app.cell
def __(
    model_selector,
    mo,
    removable_filter_selector,
    show_control_id_checkbox,
    sign_filter_selector,
):
    controls = [
        mo.md("## Controls"),
        model_selector,
        sign_filter_selector,
        removable_filter_selector,
        show_control_id_checkbox,
    ]
    mo.sidebar(
        mo.vstack(controls, gap=1.0),
        footer=mo.md("表示対象を絞るとグラフとテーブルが連動して更新されます。"),
        width="320px",
    )
    return


@app.cell
def __(load_network_spec, model_selector):
    model_name = model_selector.value
    network_spec = load_network_spec(model_name)
    return model_name, network_spec


@app.cell
def __(filter_edges, network_spec, removable_filter_selector, sign_filter_selector):
    filtered_edges = filter_edges(
        network_spec.all_edges,
        sign_filter_selector.value,
        removable_filter_selector.value,
    )
    return (filtered_edges,)


@app.cell
def __(filtered_edges, mo, network_spec):
    total_edges = len(network_spec.all_edges)
    removable_total = sum(int(edge.removable) for edge in network_spec.all_edges)
    activation_total = sum(
        int(edge.sign == "activation") for edge in network_spec.all_edges
    )
    inhibition_total = sum(
        int(edge.sign == "inhibition") for edge in network_spec.all_edges
    )
    shown_edges = len(filtered_edges)

    mo.md(
        f"""
        ## Summary

        - `n_nodes`: **{network_spec.n_nodes}**
        - `total_edges`: **{total_edges}**
        - `removable_edges`: **{removable_total}**
        - `activation_count`: **{activation_total}**
        - `inhibition_count`: **{inhibition_total}**
        - `displayed_edges`: **{shown_edges}**
        """
    )
    return


@app.cell
def __(control_mapping_table, mo, network_spec):
    mapping_title = mo.md("## Control input mapping")
    mapping_description = mo.md("`u_k` がどのエッジに対応するかを固定表で確認できます。")
    mapping_df = control_mapping_table(network_spec.all_edges)
    if mapping_df.empty:
        mapping_body = mo.md("このモデルには除去可能エッジがありません。")
    else:
        mapping_body = mapping_df
    mo.vstack([mapping_title, mapping_description, mapping_body], gap=0.7)
    return


@app.cell
def __(equations_markdown, mo, network_spec):
    equations_title = mo.md("## Logical update equations")
    equations_description = mo.md(
        "論文読みのために、制御入力を `u_{j,i}(t)`（`x_j -> x_i` に対応）で表示します。"
    )
    equations_body = mo.md(
        equations_markdown(network_spec.update_equations, network_spec.all_edges)
    )
    mo.vstack([equations_title, equations_description, equations_body], gap=0.7)
    return


@app.cell
def __(
    draw_network_graph,
    filtered_edges,
    model_name,
    network_spec,
    show_control_id_checkbox,
):
    network_fig = draw_network_graph(
        model_name=model_name,
        n_nodes=network_spec.n_nodes,
        edges=filtered_edges,
        show_control_ids=bool(show_control_id_checkbox.value),
    )
    network_fig
    return


@app.cell
def __(edge_table, filtered_edges, mo):
    mo.md("## Edge table")
    edge_df = edge_table(filtered_edges)
    if edge_df.empty:
        mo.md("該当するエッジがありません。フィルタ条件を変更してください。")
    else:
        edge_df
    return


if __name__ == "__main__":
    app.run()
