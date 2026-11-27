import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import dill
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from benchmarks.counting_ones.benchmarking_counting_ones.baselines import (
    Methods,
)
from benchmarks.counting_ones.benchmarking_counting_ones.results_analysis.load_experiments_parallel import (
    load_benchmark_results,
)
from benchmarks.counting_ones.benchmarking_counting_ones.results_analysis.method_styles import (
    plot_range,
)
from syne_tune.util import catchtime

def figure_folder(path):
    root = Path('./')
    figure_path = root / path
    figure_path.mkdir(exist_ok=True, parents=True)
    print(figure_path)
    return figure_path

lw = 2.5
alpha = 0.7
matplotlib.rcParams.update({"font.size": 15})
benchmark_families = [
    "co",
]
benchmark_names = {
    "co": "\\Counting-Ones{}",
}

def plot_result_benchmark(
    num_trials: int,
    method_dict: Dict[str, np.array],
    title: str,
    rename_dict: dict,
    ax=None,
    methods_to_show: list = None,
    plot_regret: bool = True,
):
    agg_results = {}

    if plot_regret:
        mode = "min"
        min_value = min([v.min() for v in method_dict.values()])
        max_value = max([v.max() for v in method_dict.values()])
        best_result, worse_result = (
            (min_value, max_value) if mode == "min" else (max_value, min_value)
        )

    if len(method_dict) > 0:
        if ax is None:
            fig, ax = plt.subplots()
        x_range = np.arange(num_trials)
        for algorithm in method_dict.keys():
            if methods_to_show is not None and algorithm not in methods_to_show:
                continue
            renamed_algorithm = rename_dict.get(algorithm, algorithm)

            y_ranges = method_dict[algorithm]
            if plot_regret:
                y_ranges = (y_ranges - best_result) / (worse_result - best_result)
            mean = y_ranges.mean(axis=0)
            std = y_ranges.std(axis=0, ddof=1) / np.sqrt(y_ranges.shape[0])
            ax.fill_between(
                x_range,
                mean - std,
                mean + std,
                alpha=0.1,
            )
            ax.plot(
                x_range,
                mean,
                label=renamed_algorithm,
                alpha=alpha,
            )

            agg_results[algorithm] = mean

        ax.set_xlabel("Trials")
        ax.legend()
        ax.set_title(title)
    return ax

def plot_task_performance_over_trials(
    benchmark_results: Dict[str, Tuple[np.array, Dict[str, np.array]]],
    rename_dict: dict,
    result_folder: Path,
    title: str = None,
    ax=None,
    methods_to_show: list = None,
    plot_regret: bool = False,
):
    print(f"plot rank through trials on {result_folder}")
    for benchmark, (t_range, method_dict) in benchmark_results.items():
        num_trials = t_range.shape[0] if hasattr(t_range, "shape") else len(t_range)
        ax = plot_result_benchmark(
            num_trials=num_trials,
            method_dict=method_dict,
            title=benchmark,
            ax=ax,
            methods_to_show=methods_to_show,
            rename_dict=rename_dict,
            plot_regret=plot_regret,
        )
        ax.set_ylabel("objective")
        if title is not None:
            ax.set_title(title)
        if not plot_regret:
            if benchmark in plot_range:
                plotargs = plot_range[benchmark]
                ax.set_ylim([plotargs.ymin, plotargs.ymax])
                ax.set_xlim([plotargs.xmin, plotargs.xmax])

        if ax is not None:
            plt.tight_layout()
            filepath = result_folder / f"{benchmark}.pdf"
            plt.savefig(filepath)
        ax = None

def load_and_cache(
    path: Path,
    methods: Optional[List[str]] = None,
    load_cache_if_exists: bool = True,
    num_time_steps=100,
    max_seed=10,
    experiment_filter=None,
):
    result_file = (Path(path) / "results-cache.dill").expanduser()
    if load_cache_if_exists and result_file.exists():
        with catchtime(f"loading results from {result_file}"):
            with open(result_file, "rb") as f:
                benchmark_results = dill.load(f)
    else:
        print(f"regenerating results to {result_file}")
        with catchtime("load benchmark results"):
            benchmark_results = load_benchmark_results(
                path=path,
                methods=methods,
                num_time_steps=num_time_steps,
                max_seed=max_seed,
                experiment_filter=experiment_filter,
            )

        with open(result_file, "wb") as f:
            dill.dump(benchmark_results, f)

    return benchmark_results

def plot_ranks(
    ranks,
    benchmark_results,
    title: str,
    rename_dict: dict,
    result_folder: Path,
    methods_to_show: List[str],
):
    plt.figure()
    ys = ranks.reshape(benchmark_results.shape).mean(axis=(1, 2))
    xs = np.arange(ys.shape[-1])
    for i, method in enumerate(methods_to_show):
        plt.plot(
            xs,
            ys[i],
            label=rename_dict.get(method, method),
            alpha=alpha,
            lw=lw,
        )
    plt.xlabel("Trials")
    plt.ylabel("Method rank")
    plt.xlim(0, xs[-1])
    plt.grid()
    plt.title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(result_folder / f"{title}.pdf")

def stack_benchmark_results(
    benchmark_results_dict: Dict[str, Tuple[np.array, Dict[str, np.array]]],
    methods_to_show: Optional[List[str]],
    benchmark_families: List[str],
) -> Dict[str, np.array]:
    for benchmark, (t_range, method_dict) in benchmark_results_dict.items():
        for method in list(methods_to_show):
            if method not in method_dict:
                print(
                    f"removing method {method} from methods to show as it is not present in all benchmarks"
                )
                methods_to_show.remove(method)
    if len(methods_to_show) == 0:
        return {}
    else:
        res = {}
        for benchmark_family in benchmark_families:
            benchmarks_family = [
                benchmark
                for benchmark in benchmark_results_dict.keys()
                if benchmark_family in benchmark
            ]

            benchmark_results = []
            for benchmark in benchmarks_family:
                benchmark_result = [
                    benchmark_results_dict[benchmark][1][method]
                    for method in methods_to_show
                ]
                benchmark_result = np.stack(benchmark_result)
                benchmark_results.append(benchmark_result)

            benchmark_results = np.stack(benchmark_results)

            if benchmark_family in ["lcbench", "yahpo", "co"]:
                benchmark_results *= -1

            res[benchmark_family] = benchmark_results.swapaxes(0, 1)

        return res

def generate_rank_results(
    benchmark_families: List[str],
    stacked_benchmark_results: Dict[str, np.array],
    methods_to_show: Optional[List[str]],
    rename_dict: dict,
    result_folder: Path,
):
    rows = []
    for benchmark_family in benchmark_families:
        print(benchmark_family)
        benchmark_results = stacked_benchmark_results[benchmark_family]

        ranks = pd.DataFrame(
            benchmark_results.reshape(len(benchmark_results), -1)
        ).rank()
        ranks = ranks.values.reshape(benchmark_results.shape)
        avg_ranks_per_tasks = ranks.mean(axis=(2, 3))
        for i in range(benchmark_results.shape[1]):
            row = {"benchmark": f"{benchmark_family}-{i}"}
            row.update(dict(zip(methods_to_show, avg_ranks_per_tasks[:, i])))
            rows.append(row)

        plot_ranks(
            ranks,
            benchmark_results,
            benchmark_family,
            rename_dict,
            result_folder,
            methods_to_show,
        )

    all_results = np.concatenate(list(stacked_benchmark_results.values()), axis=1)
    all_ranks = pd.DataFrame(all_results.reshape(len(all_results), -1)).rank()
    plot_ranks(
        all_ranks.values,
        all_results,
        "Average-rank",
        rename_dict,
        result_folder,
        methods_to_show,
    )

def plot_average_normalized_regret(
    stacked_benchmark_results,
    rename_dict: dict,
    result_folder: Path,
    title: str = None,
    show_ci: bool = False,
    ax=None,
    methods_to_show: list = None,
):
    normalized_regrets = []
    for benchmark_family in benchmark_families:
        benchmark_results = stacked_benchmark_results[benchmark_family]
        benchmark_results_best = benchmark_results.min(axis=(0, 2, 3), keepdims=True)
        benchmark_results_worse = benchmark_results.max(axis=(0, 2, 3), keepdims=True)
        normalized_regret = (benchmark_results - benchmark_results_best) / (
            benchmark_results_worse - benchmark_results_best
        )
        normalized_regrets.append(normalized_regret)

    normalized_regrets = np.concatenate(normalized_regrets, axis=1)
    avg_regret = normalized_regrets.mean(axis=(1, 2))
    std_regret = normalized_regrets.std(axis=2).mean(axis=1) if show_ci else None

    if ax is None:
        fig, ax = plt.subplots()
    for i, algorithm in enumerate(methods_to_show):
        renamed_algorithm = rename_dict.get(algorithm, algorithm)
        mean = avg_regret[i]
        xs = np.arange(len(mean))
        ax.plot(
            xs,
            mean,
            label=renamed_algorithm,
            lw=lw,
            alpha=alpha,
        )
        if show_ci:
            std = std_regret[i]
            ax.fill_between(
                xs,
                mean - std,
                mean + std,
                alpha=0.1,
            )
        ax.set_yscale("log")

    plt.xlabel("Trials")
    ax.set_ylabel("Average normalized regret")
    plt.xlim(0, xs[-1])
    plt.ylim(6e-3, None)
    plt.grid()
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(result_folder / f"{title}.pdf")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="path where to find the results",
    )
    parser.add_argument(
        "--max_seed",
        type=int,
        required=False,
        default=30,
    )
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        required=False,
    )

    methods_selected = [
        Methods.RS,
        Methods.OPT_RS,
        Methods.OPT_LS,
        Methods.LS,
    ]

    single_fidelity = [x for x in methods_selected if not ("ASHA" in x or "BOHB" in x)]

    methods_to_show = single_fidelity
    groups = {
        "single-fidelity": single_fidelity,
    }

    args, _ = parser.parse_known_args()

    print(args.__dict__)
    assert Path(args.path).exists()
    max_seed = args.max_seed
    num_time_steps = 200

    with catchtime("load benchmark results"):
        benchmark_results = load_and_cache(
            path=args.path,
            load_cache_if_exists=args.reuse_cache,
            max_seed=max_seed,
            num_time_steps=num_time_steps,
            methods=methods_to_show,
        )

    assert (
        len(benchmark_results) > 0
    ), f"Could not find results in path provided {args.path}."
    for group_name, methods in groups.items():
        if len(methods) > 0:
            folder_name = Path(args.path).parent.name
            result_folder = figure_folder(Path("figures") / folder_name / group_name)
            result_folder.mkdir(parents=True, exist_ok=True)
            stacked_benchmark_results = stack_benchmark_results(
                benchmark_results_dict=benchmark_results,
                methods_to_show=methods,
                benchmark_families=benchmark_families,
            )
            if len(stacked_benchmark_results) > 0:
                rename_dict = {}
                with catchtime("generating rank table"):
                    generate_rank_results(
                        stacked_benchmark_results=stacked_benchmark_results,
                        benchmark_families=benchmark_families,
                        methods_to_show=methods,
                        rename_dict=rename_dict,
                        result_folder=result_folder,
                    )

                with catchtime("generating plots per task"):
                    plot_task_performance_over_trials(
                        benchmark_results=benchmark_results,
                        methods_to_show=methods,
                        rename_dict=rename_dict,
                        result_folder=result_folder,
                    )

                plot_average_normalized_regret(
                    stacked_benchmark_results=stacked_benchmark_results,
                    methods_to_show=methods,
                    rename_dict=rename_dict,
                    result_folder=result_folder,
                    title="Normalized-regret",
                )