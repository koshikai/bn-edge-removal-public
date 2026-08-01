# ruff: noqa: PLR1711, PLR1714
import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def __():
    import sys

    if sys.platform in ("emscripten", "pyodide"):
        import importlib
        import io
        import urllib.request
        import zipfile

        zip_url = "./bn_edge_removal.zip"
        try:
            js = importlib.import_module("js")

            href = str(js.window.location.href)
            base_dir = href.split("?", maxsplit=1)[0].split("#", maxsplit=1)[0].rsplit("/", 1)[0]
            zip_url = f"{base_dir}/bn_edge_removal.zip"
        except Exception:
            zip_url = "./bn_edge_removal.zip"

        try:
            req = urllib.request.Request(
                zip_url,
                headers={"User-Agent": "Mozilla/5.0 (MarimoWASM)"},
            )
            with urllib.request.urlopen(req) as response:
                zip_bytes = response.read()

            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_ref:
                zip_ref.extractall(".")
            print(f"Loaded bn_edge_removal package successfully from {zip_url}.")
        except Exception as e:
            print(f"Failed to load bn_edge_removal package from {zip_url}: {e}")

    import re

    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
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

    def networkx_node_positions(
        n_nodes: int,
        edges: list[NetworkEdge],
        layout_type: str = "spring",
        radius: float = 1.4,
    ) -> dict[int, tuple[float, float]]:
        G = nx.Graph()
        for i in range(1, n_nodes + 1):
            G.add_node(i)
        for edge in edges:
            if edge.source != edge.target:
                G.add_edge(edge.source, edge.target)

        raw_pos = None
        if layout_type == "kamada_kawai":
            try:
                raw_pos = nx.kamada_kawai_layout(G)
            except (ImportError, ModuleNotFoundError, Exception):
                raw_pos = None

        if raw_pos is None:
            raw_pos = nx.spring_layout(G, seed=42, k=radius / np.sqrt(max(n_nodes, 1)))

        coords = np.array([raw_pos[i] for i in range(1, n_nodes + 1)], dtype=float)
        coords -= np.mean(coords, axis=0)
        max_r = np.max(np.linalg.norm(coords, axis=1))
        if max_r > 1e-4:
            coords = coords * (radius / max_r)

        return {i + 1: (float(coords[i, 0]), float(coords[i, 1])) for i in range(n_nodes)}

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

    def edge_style(edge: NetworkEdge) -> tuple[str, str, float, float, int]:
        color = "#1f77b4" if edge.sign == "activation" else "#d62728"
        if edge.removable:
            linestyle = "--"
            linewidth = 2.5
            alpha = 0.95
            zorder = 3
        else:
            linestyle = "-"
            linewidth = 1.2
            alpha = 0.35
            zorder = 2
        return color, linestyle, linewidth, alpha, zorder

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
            "\\begin{aligned}\n" + " \\\\\n".join(latex_lines) + "\n\\end{aligned}\n"
            "$$"
        )

    def draw_network_graph(
        model_name: str,
        n_nodes: int,
        edges: list[NetworkEdge],
        show_control_ids: bool,
        layout_type: str = "circular",
        label_pos_ratio: float = 0.30,
        label_offset_dist: float = 0.14,
    ):
        if layout_type in ("spring", "kamada_kawai"):
            positions = networkx_node_positions(n_nodes, edges, layout_type=layout_type)
        else:
            positions = circular_node_positions(n_nodes)

        fig, ax = plt.subplots(figsize=(8.6, 7.2))

        for node_id, (x_pos, y_pos) in positions.items():
            node_circle = plt.Circle(
                (x_pos, y_pos),
                0.14,
                facecolor="#ffffff",
                edgecolor="#1f2937",
                linewidth=1.6,
                zorder=4,
            )
            ax.add_patch(node_circle)
            ax.text(
                x_pos,
                y_pos,
                f"x{node_id}",
                ha="center",
                va="center",
                fontsize=11.5,
                fontweight="bold",
                color="#111827",
                zorder=5,
            )

        edge_keys = {
            (edge.source, edge.target) for edge in edges if edge.source != edge.target
        }

        # Track placed label coordinates for collision avoidance
        placed_label_coords: list[tuple[float, float]] = []

        for edge in edges:
            color, linestyle, linewidth, alpha, zorder = edge_style(edge)
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
                    zorder=zorder,
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
                curve = 0.25 if is_bidirectional else 0.0

                edge_patch = FancyArrowPatch(
                    posA=source_pos,
                    posB=target_pos,
                    connectionstyle=f"arc3,rad={curve}",
                    arrowstyle="-|>",
                    mutation_scale=13.0,
                    shrinkA=15.0,
                    shrinkB=15.0,
                    zorder=zorder,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                )
                ax.add_patch(edge_patch)

                path_vertices = np.asarray(edge_patch.get_path().vertices, dtype=float)
                # Compute sample index along edge path based on label_pos_ratio
                clamped_ratio = max(0.1, min(0.9, label_pos_ratio))
                sample_idx = int((path_vertices.shape[0] - 1) * clamped_ratio)
                anchor_x = float(path_vertices[sample_idx, 0])
                anchor_y = float(path_vertices[sample_idx, 1])
                direction_x = target_pos[0] - source_pos[0]
                direction_y = target_pos[1] - source_pos[1]
                norm = max((direction_x**2 + direction_y**2) ** 0.5, 1e-9)
                normal_x = -direction_y / norm
                normal_y = direction_x / norm
                offset = label_offset_dist + abs(curve) * 0.20
                signed = 1.0 if curve >= 0 else -1.0
                label_x = anchor_x + normal_x * offset * signed
                label_y = anchor_y + normal_y * offset * signed

            if show_control_ids and edge.removable and edge.removable_index is not None:
                # Collision avoidance check with previously placed labels
                for px, py in placed_label_coords:
                    dist = ((label_x - px) ** 2 + (label_y - py) ** 2) ** 0.5
                    if dist < 0.18:
                        # Shift outward along normal vector
                        label_x += normal_x * 0.16 * signed
                        label_y += normal_y * 0.16 * signed

                placed_label_coords.append((label_x, label_y))

                ax.text(
                    label_x,
                    label_y,
                    f"u{edge.removable_index}",
                    fontsize=9.0,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.22",
                        "facecolor": "#ffffff",
                        "edgecolor": color,
                        "linewidth": 1.4,
                        "alpha": 0.95,
                    },
                    color=color,
                    zorder=6,
                )

        legend_handles = [
            Line2D([0], [0], color="#1f77b4", lw=2.4, ls="-", label="Activation"),
            Line2D([0], [0], color="#d62728", lw=2.4, ls="-", label="Inhibition"),
            Line2D(
                [0],
                [0],
                color="#374151",
                lw=2.5,
                ls="--",
                alpha=0.95,
                label="Removable edge (u_k)",
            ),
            Line2D(
                [0],
                [0],
                color="#9ca3af",
                lw=1.2,
                ls="-",
                alpha=0.45,
                label="Fixed edge",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.00),
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#e5e7eb",
        )
        ax.set_title(f"{model_name.upper()} network structure", fontsize=13.5, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(-2.1, 2.2)
        ax.set_ylim(-2.0, 2.0)
        ax.axis("off")
        fig.tight_layout()
        return fig

    return (
        circular_node_positions,
        control_mapping_table,
        draw_network_graph,
        edge_table,
        equations_markdown,
        filter_edges,
        load_network_spec,
        mo,
        networkx_node_positions,
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
    layout_selector = mo.ui.dropdown(
        options={
            "Circular": "circular",
            "Spring (Force-directed)": "spring",
            "Kamada-Kawai": "kamada_kawai",
        },
        value="Circular",
        label="Layout",
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
    label_pos_slider = mo.ui.slider(
        start=0.15,
        stop=0.85,
        step=0.05,
        value=0.30,
        label="Label Position Ratio (t)",
    )
    label_offset_slider = mo.ui.slider(
        start=0.05,
        stop=0.30,
        step=0.01,
        value=0.14,
        label="Label Distance Offset",
    )
    return (
        label_offset_slider,
        label_pos_slider,
        layout_selector,
        model_selector,
        removable_filter_selector,
        show_control_id_checkbox,
        sign_filter_selector,
    )


@app.cell
def __(
    label_offset_slider,
    label_pos_slider,
    layout_selector,
    model_selector,
    mo,
    removable_filter_selector,
    show_control_id_checkbox,
    sign_filter_selector,
):
    controls = [
        mo.md("## Controls"),
        model_selector,
        layout_selector,
        sign_filter_selector,
        removable_filter_selector,
        show_control_id_checkbox,
        mo.md("---"),
        mo.md("### Label Fine Tuning"),
        label_pos_slider,
        label_offset_slider,
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
    mapping_description = mo.md(
        "`u_k` がどのエッジに対応するかを固定表で確認できます。"
    )
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
    label_offset_slider,
    label_pos_slider,
    layout_selector,
    model_name,
    network_spec,
    show_control_id_checkbox,
):
    network_fig = draw_network_graph(
        model_name=model_name,
        n_nodes=network_spec.n_nodes,
        edges=filtered_edges,
        show_control_ids=bool(show_control_id_checkbox.value),
        layout_type=str(layout_selector.value),
        label_pos_ratio=float(label_pos_slider.value),
        label_offset_dist=float(label_offset_slider.value),
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
