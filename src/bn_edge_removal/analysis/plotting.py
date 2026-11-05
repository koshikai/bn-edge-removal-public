"""Plotting utilities for training curves and trajectories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _moving_average(series: list[float], window: int) -> list[float]:
    if window <= 1:
        return series
    values = []
    for i in range(len(series)):
        start = max(0, i - window + 1)
        values.append(sum(series[start : i + 1]) / (i - start + 1))
    return values


def plot_training_curves(
    history: list[dict[str, float]],
    output_path: Path,
    window: int = 100,
) -> None:
    df = pd.DataFrame(history)
    rewards = df["reward"].tolist()
    success = df["success"].tolist()

    rewards_ma = _moving_average(rewards, window)
    success_ma = _moving_average(success, window)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(rewards_ma, color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel("Avg Reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(success_ma, color="#2ca02c", linewidth=1.5)
    axes[1].set_ylabel("Success Rate")
    axes[1].set_xlabel("Episode")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_trajectories(trajectories: list[dict[str, int]], output_path: Path) -> None:
    df = pd.DataFrame(trajectories)
    fig, ax = plt.subplots(figsize=(9, 6))
    for _init_state, group in df.groupby("init_state"):
        ax.plot(group["t"], group["state_id"], linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("State ID")
    ax.set_title("All Initial-State Trajectories")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
