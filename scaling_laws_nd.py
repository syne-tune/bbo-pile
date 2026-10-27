#!/usr/bin/env python3
"""Chinchilla-style N/D decomposition of the scaling data in data_neurips.csv.

Fits L(N, D) = E + A / N^alpha + B / D^beta (Hoffmann et al. 2022, approach 3):
Huber loss on log-loss residuals with a log-sum-exp parameterization, minimized
from a grid of initializations. D is the number of tokens actually consumed at
the checkpoint (C / 6N), not the run's token budget.
"""

import argparse
import ast
import itertools
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from scipy.special import huber, logsumexp
from tueplots import bundles

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_list_column(value: str) -> float:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, list):
        return float(parsed[0])
    return float(parsed)


def load_data(csv_path: Path, completion_warn: float = 0.9) -> pd.DataFrame:
    """Load scaling points and derive D = tokens consumed at each checkpoint.

    IMPORTANT: the fit uses D = tokens_seen = C / (6 N), the tokens *actually
    consumed* at the checkpoint — NOT the run's nominal token budget. For a
    fully-trained final checkpoint the two coincide; for a mid-training
    checkpoint tokens_seen is the tokens-so-far, so feeding a non-final CSV
    silently fits D = partial progress rather than the intended budget. We
    check completion (tokens_seen / tokens_budget) and warn loudly if any point
    is well below 1.0, so this can never happen unnoticed.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["flops", "loss"])
    df["params"] = df["params"].apply(parse_list_column)
    df["tokens_budget"] = df["tokens"].apply(parse_list_column)
    df["flops"] = pd.to_numeric(df["flops"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df = df.dropna(subset=["flops", "loss", "params"])
    # Tokens consumed at this checkpoint, from C = 6 N D
    df["tokens_seen"] = df["flops"] / (6 * df["params"])
    df["completion"] = df["tokens_seen"] / df["tokens_budget"]
    df = df.sort_values("flops").reset_index(drop=True)

    n_partial = int((df["completion"] < completion_warn).sum())
    if n_partial:
        logger.warning(
            f"⚠ {n_partial}/{len(df)} points are below {completion_warn:.0%} of their "
            f"token budget (min completion {df['completion'].min():.2f}). The fit uses "
            f"D = tokens_seen = C/(6N), so for these points D is the tokens-so-far, "
            f"NOT the nominal budget. Feed a final-checkpoint (best-per-cell) CSV for a "
            f"clean L(N, D) fit."
        )
    logger.info(
        f"Loaded {len(df)} points | N {df['params'].min():.0e}-{df['params'].max():.0e} | "
        f"D_seen {df['tokens_seen'].min():.2e}-{df['tokens_seen'].max():.2e} | "
        f"completion {df['completion'].min():.2f}-{df['completion'].max():.2f}"
    )
    return df


def predict_log_loss(theta: np.ndarray, log_n: np.ndarray, log_d: np.ndarray) -> np.ndarray:
    """log L = LSE(a - alpha*log N, b - beta*log D, e), with A=e^a, B=e^b, E=e^e."""
    a, b, e, alpha, beta = theta
    stacked = np.stack([a - alpha * log_n, b - beta * log_d, np.full_like(log_n, e)])
    return logsumexp(stacked, axis=0)


def fit_chinchilla(
    n: np.ndarray,
    d: np.ndarray,
    loss: np.ndarray,
    delta: float = 1e-3,
    init: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Grid of initializations + L-BFGS on Huber(log L_pred - log L_obs).

    Pass `init` to skip the grid and warm-start from a known solution
    (used by the bootstrap).
    """
    log_n, log_d, log_loss = np.log(n), np.log(d), np.log(loss)

    def objective(theta: np.ndarray) -> float:
        residuals = predict_log_loss(theta, log_n, log_d) - log_loss
        return float(np.sum(huber(delta, residuals)))

    if init is not None:
        result = minimize(objective, init, method="L-BFGS-B")
        return result.x, result.fun

    best_theta, best_val = None, np.inf
    # Initialisation grid exactly as Hoffmann et al. (2022), Appendix D.2:
    # log A, log B in {0,5,...,25}; log E in {-1,-0.5,0,0.5,1}; alpha, beta in {0,0.5,1,1.5,2}.
    init_grid = itertools.product(
        np.arange(0, 30, 5),        # a = log A
        np.arange(0, 30, 5),        # b = log B
        np.arange(-1.0, 1.5, 0.5),  # e = log E
        np.arange(0.0, 2.5, 0.5),   # alpha
        np.arange(0.0, 2.5, 0.5),   # beta
    )
    for grid_init in init_grid:
        result = minimize(objective, np.array(grid_init), method="L-BFGS-B")
        if result.fun < best_val:
            best_val, best_theta = result.fun, result.x
    return best_theta, best_val


def compute_optimal_allocation(A, B, E, alpha, beta):
    """N_opt(C), D_opt(C) minimizing L subject to C = 6 N D."""
    exp_n = beta / (alpha + beta)
    exp_d = alpha / (alpha + beta)
    g = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    n_opt = lambda c: g * (c / 6) ** exp_n
    d_opt = lambda c: (1 / g) * (c / 6) ** exp_d
    return n_opt, d_opt, exp_n, exp_d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).parent / "data_neurips.csv"
    )
    parser.add_argument(
        "--min-flops",
        type=float,
        default=1.0e16,
        help="Lower bound on FLOPs included in the fit (same as scaling_laws.py).",
    )
    parser.add_argument(
        "--holdout-flops",
        type=float,
        default=7.0e17,
        help="Points with FLOPs > this are held out as validation.",
    )
    parser.add_argument(
        "--dataset-tokens",
        type=float,
        default=2.5e9,
        help="Total tokens available in BBO-Pile, to locate where data binds.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=200,
        help="Bootstrap resamples for exponent confidence intervals.",
    )
    parser.add_argument(
        "--final-losses",
        action="store_true",
        help="Reduce the data to one point per (N, token-budget) cell: the best "
        "loss among near-final checkpoints (>=90%% of budget consumed). Use with "
        "full-sweep data to avoid the mid-training (un-decayed LR) bias.",
    )
    parser.add_argument(
        "--assert-final",
        type=float,
        default=None,
        metavar="MIN_COMPLETION",
        help="Error out if any point is below this completion fraction "
        "(tokens_seen/budget). Use e.g. 0.9 to guarantee the L(N,D) fit sees only "
        "final-checkpoint losses. Off by default so isoFLOP-style full-curve CSVs "
        "still load.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "cursor_reports"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)
    if args.assert_final is not None:
        partial = df[df["completion"] < args.assert_final]
        if len(partial):
            raise SystemExit(
                f"--assert-final={args.assert_final}: {len(partial)} point(s) below "
                f"that completion (min {df['completion'].min():.2f}). The N/D fit would "
                f"use D=tokens-so-far for these. Pass a final-checkpoint CSV or lower "
                f"the threshold."
            )
    if args.final_losses:
        near_final = df[df["tokens_seen"] >= 0.9 * df["tokens_budget"]]
        df = (
            near_final.loc[
                near_final.groupby(["params", "tokens_budget"])["loss"].idxmin()
            ]
            .sort_values("flops")
            .reset_index(drop=True)
        )
        logger.info(
            f"Final-loss mode: {len(df)} (N, D) cells "
            f"({df['params'].nunique()} sizes x {df['tokens_budget'].nunique()} budgets)"
        )
    df_in_range = df[df["flops"] >= args.min_flops]
    df_fit = df_in_range[df_in_range["flops"] <= args.holdout_flops]
    df_holdout = df_in_range[df_in_range["flops"] > args.holdout_flops]
    logger.info(f"Splits: fit={len(df_fit)} | holdout={len(df_holdout)}")

    n = df_fit["params"].values
    d = df_fit["tokens_seen"].values
    loss = df_fit["loss"].values

    theta, val = fit_chinchilla(n, d, loss)
    a, b, e, alpha, beta = theta
    A, B, E = np.exp(a), np.exp(b), np.exp(e)
    logger.info(
        f"Fit: L = {E:.4f} + {A:.3g}/N^{alpha:.3f} + {B:.3g}/D^{beta:.3f} "
        f"(huber objective={val:.3g})"
    )

    pred_fit = np.exp(predict_log_loss(theta, np.log(n), np.log(d)))
    ss_res = np.sum((loss - pred_fit) ** 2)
    ss_tot = np.sum((loss - loss.mean()) ** 2)
    logger.info(f"R^2 (fit) = {1 - ss_res / ss_tot:.4f}")

    if len(df_holdout) > 0:
        pred_hold = np.exp(
            predict_log_loss(
                theta,
                np.log(df_holdout["params"].values),
                np.log(df_holdout["tokens_seen"].values),
            )
        )
        mae = float(np.mean(np.abs(df_holdout["loss"].values - pred_hold)))
        logger.info(
            f"Holdout (n={len(df_holdout)}): MAE={mae:.4f} | "
            f"pred={np.round(pred_hold, 4)} actual={np.round(df_holdout['loss'].values, 4)}"
        )

    # Bootstrap CIs on the exponents
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(args.n_bootstrap):
        idx = rng.integers(0, len(n), len(n))
        try:
            th, _ = fit_chinchilla(n[idx], d[idx], loss[idx], init=theta)
            boots.append([np.exp(th[2]), th[3], th[4]])
        except Exception:
            continue
    boots = np.array(boots)
    for name, i in [("E", 0), ("alpha", 1), ("beta", 2)]:
        lo, hi = np.percentile(boots[:, i], [2.5, 97.5])
        logger.info(f"{name}: 95% CI [{lo:.3f}, {hi:.3f}]")

    # Compute-optimal allocation and where the dataset binds
    n_opt, d_opt, exp_n, exp_d = compute_optimal_allocation(A, B, E, alpha, beta)
    logger.info(f"Compute-optimal: N_opt ~ C^{exp_n:.3f}, D_opt ~ C^{exp_d:.3f}")
    c_grid = np.logspace(15, 20, 400)
    d_curve = d_opt(c_grid)
    c_max_studied = df["flops"].max()
    logger.info(
        f"D_opt at max studied C ({c_max_studied:.2e}): {d_opt(c_max_studied):.2e} tokens "
        f"(dataset: {args.dataset_tokens:.1e})"
    )
    if d_curve[-1] > args.dataset_tokens > d_curve[0]:
        c_bind = c_grid[np.searchsorted(d_curve, args.dataset_tokens)]
        logger.info(
            f"D_opt reaches the {args.dataset_tokens:.1e}-token dataset at C ≈ {c_bind:.2e} FLOPs; "
            f"N_opt there ≈ {n_opt(c_bind):.2e} params"
        )
    else:
        c_bind = None
        logger.info(
            "D_opt does not cross the dataset size on the probed compute grid "
            f"(D_opt spans {d_curve[0]:.2e}-{d_curve[-1]:.2e})"
        )

    # Marginal contributions at the frontier endpoints
    for label, row in [("min-fit", df_fit.iloc[0]), ("max-fit", df_fit.iloc[-1])]:
        term_n = A / row["params"] ** alpha
        term_d = B / row["tokens_seen"] ** beta
        logger.info(
            f"{label}: N-term={term_n:.4f}, D-term={term_d:.4f}, E={E:.4f} "
            f"({'N (model size)' if term_n > term_d else 'D (data)'} is the larger reducible term)"
        )

    # ---- Plot: observed vs predicted + compute-optimal allocation ----
    plt.rcParams.update(bundles.neurips2024(usetex=False, ncols=2, nrows=1))
    viridis = plt.get_cmap("viridis")
    model_sizes = sorted(df["params"].unique())
    size_to_color = {
        s: viridis(i / max(len(model_sizes) - 1, 1)) for i, s in enumerate(model_sizes)
    }

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for axis in (ax1, ax2):
        axis.grid(True, which="major", alpha=0.4, linestyle="-", linewidth=0.5, zorder=0)
        axis.grid(True, which="minor", alpha=0.2, linestyle="--", linewidth=0.3, zorder=0)

    # Left: loss vs C, points + per-size fitted curves L(N, C/6N)
    for s in model_sizes:
        sub = df_in_range[df_in_range["params"] == s]
        if len(sub) == 0:
            continue
        ax1.scatter(
            sub["flops"], sub["loss"], s=15, color=size_to_color[s], alpha=0.85, zorder=3
        )
        c_line = np.logspace(
            np.log10(sub["flops"].min() * 0.8), np.log10(sub["flops"].max() * 1.2), 100
        )
        d_line = c_line / (6 * s)
        pred = np.exp(
            predict_log_loss(theta, np.full_like(c_line, np.log(s)), np.log(d_line))
        )
        ax1.plot(c_line, pred, "-", color=size_to_color[s], linewidth=1.2, zorder=2)
    # Keep the y-axis on the data. If the fitted floor E lands well below the
    # observed losses, the 3-term fit is not identifiable over this compute
    # range: annotate rather than stretch the axis down to a spurious floor.
    loss_lo, loss_hi = df_in_range["loss"].min(), df_in_range["loss"].max()
    pad = 0.05 * (loss_hi - loss_lo)
    degenerate = E < loss_lo - 3 * pad
    if degenerate:
        ax1.set_ylim(loss_lo - pad, loss_hi + pad)
        ax1.text(
            0.03, 0.05,
            f"fit not identifiable over this range\n"
            rf"(floor $E={E:.2f}$ off-axis, $\alpha={alpha:.2f}$)",
            transform=ax1.transAxes, fontsize=7, color="firebrick", va="bottom",
        )
    else:
        ax1.axhline(E, color="black", linestyle=":", linewidth=1.2, zorder=1)
        ax1.text(
            df_in_range["flops"].min(), E * 1.002, rf"$E = {E:.3f}$",
            fontsize=8, va="bottom",
        )
        ax1.set_ylim(min(E, loss_lo) - pad, loss_hi + pad)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"FLOPs ($C$)")
    ax1.set_ylabel(r"Loss ($L$)")
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="-",
            color=size_to_color[s], markersize=4,
            label=f"{int(s / 1e6)}M",
        )
        for s in model_sizes
    ]
    ax1.legend(handles=handles, title=None, ncol=2)
    if degenerate:
        ax1.set_title(r"$L = E + A/N^{\alpha} + B/D^{\beta}$ (fit unstable)", fontsize=8)
    else:
        ax1.set_title(
            rf"$L = {E:.3f} + {A:.3g}/N^{{{alpha:.2f}}} + {B:.3g}/D^{{{beta:.2f}}}$",
            fontsize=8,
        )

    # Right: compute-optimal N and D vs C, with dataset-size line
    ax2.plot(c_grid, d_opt(c_grid), "-", color=viridis(0.25), label=r"$D_{opt}(C)$")
    ax2.plot(c_grid, n_opt(c_grid), "-", color=viridis(0.75), label=r"$N_{opt}(C)$")
    ax2.axhline(
        args.dataset_tokens, color="black", linestyle="--", linewidth=1.2,
        label=f"BBO-Pile ({args.dataset_tokens / 1e9:.1f}B tokens)",
    )
    ax2.axvspan(df["flops"].min(), df["flops"].max(), color="gray", alpha=0.15, zorder=0)
    if c_bind is not None:
        ax2.axvline(c_bind, color="firebrick", linestyle=":", linewidth=1.2)
        ax2.text(
            c_bind * 1.3, args.dataset_tokens * 3e-3,
            f"data binds\n$C \\approx$ {c_bind:.1e}",
            fontsize=7, color="firebrick",
        )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"FLOPs ($C$)")
    ax2.set_ylabel(r"Optimal $N$, $D$")
    ax2.legend(loc="upper left")
    if degenerate:
        ax2.set_title("Compute-optimal allocation (unreliable)", fontsize=8)
    else:
        ax2.set_title(
            rf"$N_{{opt}} \propto C^{{{exp_n:.2f}}}$, $D_{{opt}} \propto C^{{{exp_d:.2f}}}$",
            fontsize=8,
        )

    suffix = "_final" if args.final_losses else ""
    out_pdf = args.output_dir / f"scaling_laws_nd_decomposition{suffix}.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    logger.info(f"Saved plot to {out_pdf}")


if __name__ == "__main__":
    main()
