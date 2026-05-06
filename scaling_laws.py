#!/usr/bin/env python3
"""Fit scaling laws to data_neurips.csv."""

import argparse
import ast
import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from tueplots import bundles

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_list_column(value: str) -> float:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, list):
        return float(parsed[0])
    return float(parsed)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["flops", "loss"])
    df["params"] = df["params"].apply(parse_list_column)
    df["tokens"] = df["tokens"].apply(parse_list_column)
    df["flops"] = pd.to_numeric(df["flops"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df = df.dropna(subset=["flops", "loss", "params", "tokens"])
    df = df.sort_values("flops").reset_index(drop=True)
    logger.info(
        f"Loaded {len(df)} points | FLOPs {df['flops'].min():.2e}–{df['flops'].max():.2e} | "
        f"loss {df['loss'].min():.4f}–{df['loss'].max():.4f}"
    )
    return df


def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit L = a * C^b via log-linear regression. Returns (a, b)."""
    log_x, log_y = np.log10(x), np.log10(y)
    b, log_a = np.polyfit(log_x, log_y, 1)
    return 10 ** log_a, b


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-flops",
        type=float,
        default=1.0e16,
        help="Lower bound on FLOPs included in the fit. Default: 1e16",
    )
    parser.add_argument(
        "--holdout-flops",
        type=float,
        default=7.0e17,
        help="Points with FLOPs > this are held out as validation. Default: 7e17",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data_neurips.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "cursor_reports",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)

    df_in_range = df[df["flops"] >= args.min_flops]
    df_fit = df_in_range[df_in_range["flops"] <= args.holdout_flops]
    df_holdout = df_in_range[df_in_range["flops"] > args.holdout_flops]
    df_excluded = df[df["flops"] < args.min_flops]
    logger.info(
        f"Splits: fit={len(df_fit)} ({args.min_flops:.0e} <= C <= {args.holdout_flops:.0e}) | "
        f"holdout={len(df_holdout)} (C > {args.holdout_flops:.0e}) | "
        f"excluded={len(df_excluded)} (C < {args.min_flops:.0e})"
    )

    a, b = fit_power_law(df_fit["flops"].values, df_fit["loss"].values)
    y_pred_fit = a * np.power(df_fit["flops"].values, b)
    r2_fit = compute_r2(df_fit["loss"].values, y_pred_fit)
    logger.info(f"Fit: L = {a:.4f} * C^{b:.4f} | R^2 (fit) = {r2_fit:.4f}")

    if len(df_holdout) > 0:
        y_pred_holdout = a * np.power(df_holdout["flops"].values, b)
        residuals = df_holdout["loss"].values - y_pred_holdout
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        r2_holdout = compute_r2(df_holdout["loss"].values, y_pred_holdout)
        logger.info(
            f"Holdout: MAE={mae:.4f} RMSE={rmse:.4f} R^2={r2_holdout:.4f} "
            f"(n={len(df_holdout)})"
        )

    plt.rcParams.update(bundles.neurips2024(usetex=False, ncols=1, nrows=1))
    plt.rcParams["legend.fontsize"] = plt.rcParams["font.size"] + 3
    plt.rcParams["image.cmap"] = "viridis"
    viridis = plt.get_cmap("viridis")
    line_color = viridis(0.25)
    model_sizes = [2_000_000, 5_000_000, 13_000_000, 30_000_000, 80_000_000]
    size_to_color = {
        n: viridis(i / (len(model_sizes) - 1)) for i, n in enumerate(model_sizes)
    }

    def fmt_params(n: float) -> str:
        return f"{int(n / 1_000_000)}M"

    fig, ax = plt.subplots(1, 1)

    ax.grid(True, which="major", alpha=0.4, linestyle="-", linewidth=0.5, zorder=0)
    ax.grid(True, which="minor", alpha=0.2, linestyle="--", linewidth=0.3, zorder=0)

    # Pareto front points: per-size colors, small filled circles
    for n in model_sizes:
        sub = df_fit[df_fit["params"] == n]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["flops"],
            sub["loss"],
            s=15,
            color=size_to_color[n],
            alpha=0.85,
            zorder=3,
        )

    # Held-out points: large stars in 80M (yellow) with black edge — clearly distinct
    if len(df_holdout) > 0:
        holdout_tokens = df_holdout["tokens"].iloc[0] / 1e9
        holdout_params = df_holdout["params"].iloc[0]
        ax.scatter(
            df_holdout["flops"],
            df_holdout["loss"],
            s=120,
            color=size_to_color[holdout_params],
            edgecolors="black",
            linewidths=0.8,
            marker="*",
            zorder=4,
        )

    flops_min_line = df_fit["flops"].min() * 0.7
    flops_max_line = df["flops"].max() * 1.3
    flops_grid = np.logspace(np.log10(flops_min_line), np.log10(flops_max_line), 200)
    ax.plot(
        flops_grid,
        a * np.power(flops_grid, b),
        "--",
        linewidth=2,
        zorder=1,
        color="black",
        label=rf"$L = {a:.4f} \cdot C^{{{b:.4f}}}$",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"FLOPs ($C$)")
    ax.set_ylabel(r"Loss ($L$)")

    # Custom legend: a representative Pareto marker, the held-out marker, and the fit line
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o", linestyle="none",
            markerfacecolor="black", markeredgecolor="none", markersize=6,
            label="Compute-optimal frontier",
        ),
    ]
    if len(df_holdout) > 0:
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="*", linestyle="none",
                markerfacecolor=size_to_color[holdout_params],
                markeredgecolor="black",
                markeredgewidth=0.8, markersize=12,
                label=f"Held-out ({fmt_params(holdout_params)}, {holdout_tokens:.0f}B tokens)",
            )
        )
    legend_handles.append(
        Line2D(
            [0], [0],
            color="black", linestyle="--", linewidth=2,
            label=rf"$L = {a:.4f} \cdot C^{{{b:.4f}}}$",
        )
    )
    ax.legend(handles=legend_handles)

    # Discrete colorbar mapping color -> model size (shared with learning-curves plot)
    colors = [size_to_color[n] for n in model_sizes]
    cmap_discrete = ListedColormap(colors)
    boundaries = np.arange(len(model_sizes) + 1) - 0.5
    norm = BoundaryNorm(boundaries, cmap_discrete.N)
    sm = plt.cm.ScalarMappable(cmap=cmap_discrete, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=ax,
        ticks=np.arange(len(model_sizes)),
        spacing="proportional",
    )
    cbar.ax.set_yticklabels([fmt_params(n) for n in model_sizes])
    cbar.set_label("Model size")

    out_pdf = args.output_dir / "scaling_laws_fit_neurips.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    main()
