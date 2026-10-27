#!/usr/bin/env python3
"""Rebuttal scaling figure: N/D decomposition, both panels from a SINGLE fit.

Panel (a): validation loss vs compute, one fitted curve per model size
           (solid = data range, dashed = extrapolated).
Panel (b): Chinchilla Fig.-4-style iso-loss contour of L(N,D) over the (N,D)
           plane, with the compute-efficient frontier and the BBO-Pile token limit.

Both panels use the same fitted theta so the figure is internally consistent.
Prints the fitted parameters and the subset-robustness ranges for the caption.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from tueplots import bundles

from scaling_laws_nd import (
    compute_optimal_allocation,
    fit_chinchilla,
    load_data,
    predict_log_loss,
)


def sci(x, sig=2):
    exp = int(np.floor(np.log10(abs(x))))
    if -2 <= exp <= 3:
        return f"{x:.{sig}g}"
    return rf"{x / 10**exp:.{sig}g}\times10^{{{exp}}}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path(__file__).parent / "data_filter_bestcell_2B.csv")
    p.add_argument("--dataset-tokens", type=float, default=2.5e9)
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent / "cursor_reports")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    fit = df  # keep all final-checkpoint points (see load_data completion check)
    theta, val = fit_chinchilla(fit["params"].values, fit["tokens_seen"].values, fit["loss"].values)
    a, b, e, alpha, beta = theta
    A, B, E = np.exp(a), np.exp(b), np.exp(e)
    n_opt, d_opt, exp_n, exp_d = compute_optimal_allocation(A, B, E, alpha, beta)
    pred = np.exp(predict_log_loss(theta, np.log(fit["params"].values), np.log(fit["tokens_seen"].values)))
    r2 = 1 - np.sum((fit["loss"].values - pred) ** 2) / np.sum((fit["loss"].values - fit["loss"].mean()) ** 2)
    print(f"FIT  L = {E:.3f} + {A:.3g} N^-{alpha:.3f} + {B:.3g} D^-{beta:.3f}  | R2={r2:.4f} huber={val:.2e}")
    print(f"     N_opt ~ C^{exp_n:.3f}, D_opt ~ C^{exp_d:.3f}")

    sizes = sorted(df["params"].unique())
    ramp = plt.get_cmap("viridis")
    col = {s: ramp(t) for s, t in zip(sizes, np.linspace(0, 1, len(sizes)))}

    # ---------------- Panel (a): loss vs compute ----------------
    plt.rcParams.update(bundles.neurips2024(usetex=False, ncols=1, nrows=1))
    figa, ax = plt.subplots(1, 1)
    ax.grid(True, which="major", alpha=0.35, linewidth=0.5, zorder=0)
    ax.grid(True, which="minor", alpha=0.15, linestyle=":", linewidth=0.4, zorder=0)
    cmin, cmax = fit["flops"].min(), fit["flops"].max()
    for s in sizes:
        sub = fit[fit["params"] == s].sort_values("flops")
        if len(sub) == 0:
            continue
        c = col[s]
        cl = np.logspace(np.log10(cmin), np.log10(cmax), 240)
        ax.plot(cl, np.exp(predict_log_loss(theta, np.full_like(cl, np.log(s)), np.log(cl / (6 * s)))),
                "--", color=c, linewidth=0.9, alpha=0.35, zorder=2)
        ci = np.logspace(np.log10(sub["flops"].min()), np.log10(sub["flops"].max()), 120)
        ax.plot(ci, np.exp(predict_log_loss(theta, np.full_like(ci, np.log(s)), np.log(ci / (6 * s)))),
                "-", color=c, linewidth=1.8, zorder=3, solid_capstyle="round")
        ax.scatter(sub["flops"], sub["loss"], s=34, color=c, edgecolors="white", linewidths=0.6, zorder=4)
    lo, hi = fit["loss"].min(), fit["loss"].max()
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.set_xlabel(r"FLOPs ($C$)")
    ax.set_ylabel(r"Validation loss ($L$)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    cmap_d = ListedColormap([col[s] for s in sizes])
    norm = BoundaryNorm(np.arange(len(sizes) + 1) - 0.5, cmap_d.N)
    sm = plt.cm.ScalarMappable(cmap=cmap_d, norm=norm)
    sm.set_array([])
    cbar = figa.colorbar(sm, ax=ax, ticks=np.arange(len(sizes)), pad=0.02, fraction=0.046)
    cbar.ax.set_yticklabels([f"{int(s / 1e6)}M" for s in sizes])
    cbar.set_label("Model size")
    cbar.outline.set_visible(False)
    figa.savefig(args.outdir / "rebuttal_nd_curves.pdf")
    figa.savefig(args.outdir / "rebuttal_nd_curves.png", dpi=200)
    plt.close(figa)

    # ---------------- Panel (b): iso-loss contour, FLOPs (x) vs model size (y) ----------------
    # Chinchilla efficient-frontier view: L(N, C) with D = C / (6 N).
    n_lo, n_hi = fit["params"].min() / 2, fit["params"].max() * 2
    c_lo, c_hi = fit["flops"].min() / 3, fit["flops"].max() * 3
    cg = np.logspace(np.log10(c_lo), np.log10(c_hi), 400)
    ngrid = np.logspace(np.log10(n_lo), np.log10(n_hi), 400)
    CC, NN = np.meshgrid(cg, ngrid)
    LL = np.exp(predict_log_loss(theta, np.log(NN), np.log(CC / (6 * NN))))
    figb, ax = plt.subplots(1, 1)
    lmin = np.floor(fit["loss"].min() * 20) / 20
    lmax = np.ceil(fit["loss"].max() * 20) / 20
    cf = ax.contourf(CC, NN, LL, levels=np.linspace(lmin, lmax, 11), cmap="viridis", alpha=0.8, extend="both")
    # Efficient frontier across the full compute box (solid black throughout).
    c_all = np.logspace(np.log10(c_lo), np.log10(c_hi), 300)
    ax.plot(c_all, n_opt(c_all), color="black", linewidth=2.4, zorder=6, solid_capstyle="round",
            label=rf"efficient frontier ($N_{{opt}}\!\propto\! C^{{{exp_n:.2f}}}$)")
    ax.scatter(fit["flops"], fit["params"], s=30, facecolors="white", edgecolors="black", linewidths=0.8, zorder=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(c_lo, c_hi)
    ax.set_ylim(n_lo, n_hi)
    ax.set_xlabel(r"FLOPs ($C$)")
    ax.set_ylabel(r"Model size ($N$)")
    # Tick at the actual trained model sizes (log scale) so dots map to sizes.
    ax.set_yticks(sizes)
    ax.set_yticklabels([f"{int(s / 1e6)}M" for s in sizes])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=6)
    cbar = figb.colorbar(cf, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Loss ($L$)")
    cbar.outline.set_visible(False)
    figb.savefig(args.outdir / "rebuttal_nd_contour.pdf")
    figb.savefig(args.outdir / "rebuttal_nd_contour.png", dpi=200)
    plt.close(figb)
    print(f"saved -> {args.outdir}/rebuttal_nd_curves.pdf and rebuttal_nd_contour.pdf")


if __name__ == "__main__":
    main()
