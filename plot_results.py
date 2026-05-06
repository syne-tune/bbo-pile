import argparse
import json
import logging
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles, figsizes
from matplotlib import cm

from scipy.interpolate import make_interp_spline

logger = logging.getLogger(__name__)


# Style definitions
rs_color = "black"
cqr_color = "tab:purple"
multifidelity_style = "dashed"
multifidelity_style2 = "dashdot"

show_seeds = False
marker_ours = "*"

cmap = cm.get_cmap("viridis")
method_styles = {
    'RS': dict(color=rs_color, linestyle="solid", marker="o"),
    'CQR': dict(color=cqr_color, linestyle="solid", marker="v"),
    'OptformerVLLM': dict(color="blue", linestyle=multifidelity_style, marker="v"),
    'OptformerLitGPT': dict(color="red", linestyle=multifidelity_style, marker="v"),
}

# Fallback style for methods not in method_styles
fallback_colors = ["tab:cyan", "tab:pink", "tab:olive", "tab:gray"]

def get_style(method, fallback_idx):
    if method in method_styles:
        return method_styles[method]
    return dict(
        color=fallback_colors[fallback_idx % len(fallback_colors)],
        linestyle="solid",
        marker="o",
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot runtime results from saved per-run JSON files."
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=pathlib.Path("/Users/lucathale-bombien/optformer_runtimes"),
        help="Directory containing per-run JSON result files.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("./fig-multi-searcher-runtime-comparison.png"),
        help="Where to save the output plot.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Restrict plot to these methods. Defaults to all found in results dir.",
    )
    return parser.parse_args()


def load_results(results_dir: pathlib.Path, methods_filter: list | None):
    """Load all JSON results, grouped by method. Returns {method: [runtimes_per_seed]}."""
    all_runtimes = defaultdict(list)  # method -> list of runtime arrays (one per seed)

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON result files found in {results_dir}")

    for path in json_files:
        with open(path) as f:
            data = json.load(f)

        method = data["method"]
        if methods_filter and method not in methods_filter:
            continue

        all_runtimes[method].append(np.array(data["runtimes"]))

    return all_runtimes


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    logger.info("Loading results from %s", args.results_dir)
    all_runtimes = load_results(args.results_dir, args.methods)

    if not all_runtimes:
        logger.error("No results loaded. Check --results-dir and --methods.")
        return

    # Determine n_trials from data (consistent across seeds/methods)
    n_trials = len(next(iter(all_runtimes.values()))[0])
    trials_x = np.arange(n_trials)

    # Summary table
    logger.info("%-25s  %6s  %10s  %10s", "Method", "Seeds", "Mean RT (s)", "Std RT (s)")
    logger.info("-" * 60)
    for method, seed_runtimes in sorted(all_runtimes.items()):
        stacked = np.stack(seed_runtimes)  # (n_seeds, n_trials)
        logger.info(
            "%-25s  %6d  %10.4f  %10.4f",
            method, len(seed_runtimes), stacked.mean(), stacked.std(),
        )

    # tueplots layout
    n_cols = 1
    rc = bundles.neurips2024(usetex=True)
    rc.update(figsizes.neurips2024(ncols=n_cols))
    rc["font.size"] = 10
    rc["axes.labelsize"] = 10
    rc["axes.titlesize"] = 10
    rc["xtick.labelsize"] = 10
    rc["ytick.labelsize"] = 10
    rc["legend.fontsize"] = 10

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, n_cols, squeeze=False)
        ax = axes[0, 0]

        for idx, (method, seed_runtimes) in enumerate(sorted(all_runtimes.items())):
            style = get_style(method, idx)

            stacked = np.stack(seed_runtimes)  # (n_seeds, n_trials)
            mean_rt = stacked.mean(axis=0)
            std_rt = stacked.std(axis=0)
            n_seeds = len(seed_runtimes)

            x_smooth = np.linspace(trials_x[0], trials_x[-1], 300)
            spline_mean = make_interp_spline(trials_x, mean_rt, k=3)(x_smooth)
            spline_lo = make_interp_spline(trials_x, mean_rt - std_rt, k=3)(x_smooth)
            spline_hi = make_interp_spline(trials_x, mean_rt + std_rt, k=3)(x_smooth)

            ax.plot(
                x_smooth,
                spline_mean,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                label=f"{method.replace("Optformer", "")}",
            )

            ax.fill_between(
                x_smooth,
                spline_lo,
                spline_hi,
                color=style["color"],
                alpha=0.15,
            )

        ax.set_yscale("log")
        ax.set_xlabel("function evaluations")
        ax.set_ylabel("runtime (s)")
        ax.set_title(
            "Runtime per Trial - 30 seeds, FCNet-protein",
            fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize="small")

        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        plt.close(fig)

    logger.info("Plot saved to %s", args.output)


if __name__ == "__main__":
    main()