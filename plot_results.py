import argparse
import json
import logging
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot runtime results from saved per-run JSON files."
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=pathlib.Path("/runtime_results"),
        help="Directory containing per-run JSON result files.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("/fig-multi-searcher-runtime-comparison.png"),
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

    # Plot (log-scale)
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (method, seed_runtimes) in enumerate(sorted(all_runtimes.items())):
        color = colors[idx % len(colors)]

        stacked = np.stack(seed_runtimes)  # (n_seeds, n_trials)
        mean_rt = stacked.mean(axis=0)
        std_rt = stacked.std(axis=0)
        n_seeds = len(seed_runtimes)

        ax.plot(
            trials_x,
            mean_rt,
            marker="o",
            color=color,
            linewidth=2,
            markersize=3,
            label=f"{method} (n={n_seeds}, mean ± 1 std)",
        )

        ax.fill_between(
            trials_x,
            mean_rt - std_rt,
            mean_rt + std_rt,
            color=color,
            alpha=0.15,
        )

    # Log scale
    ax.set_yscale("log")

    # Labels and title
    ax.set_xlabel("Trial ID", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title(
        "Runtime per Trial — 30 seeds, FCNet-protein",
        fontsize=14,
        fontweight="bold",
    )

    # Legend and grid
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    logger.info("Plot saved to %s", args.output)

if __name__ == "__main__":
    main()
