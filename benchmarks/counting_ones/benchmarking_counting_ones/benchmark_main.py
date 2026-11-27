import itertools
import logging
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from baselines import (
    MethodArguments,
    methods,
)
from benchmarks import (
    benchmark_definitions,
)
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, choice


def run(
    method_names,
    benchmark_names,
    seeds,
    checkpoint_dir,
    max_num_evaluations=None,
    n_workers: int = 1,
):
    logging.getLogger("syne_tune.optimizer.schedulers").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend.simulator_backend.simulator_backend").setLevel(
        logging.WARNING
    )

    combinations = list(itertools.product(method_names, seeds, benchmark_names))

    print(f"Going to evaluate: {combinations}")
    exp_names = []
    for method, seed, benchmark_name in tqdm(combinations):
        np.random.seed(seed)
        benchmark = benchmark_definitions[benchmark_name]

        print(f"Starting experiment ({method}/{benchmark_name}/{seed})")

        config_space = {f"x_{i}": choice([0, 1]) for i in range(benchmark.dim)}
        config_space['dim'] = benchmark.dim

        trial_backend = LocalBackend(entry_point=benchmark.objective)

        # 5 candidates initially to be evaluated
        num_random_candidates = 1
        random_state = np.random.RandomState(seed)
        points_to_evaluate = [
            {
                k: 0 if hasattr(v, "sample") else v
                for k, v in config_space.items()
            }
            for _ in range(num_random_candidates)
        ]
        scheduler = methods[method](
            MethodArguments(
                config_space=config_space,
                metric=benchmark.metric,
                mode=benchmark.mode,
                random_seed=seed,
                checkpoint_dir=checkpoint_dir,
                points_to_evaluate=points_to_evaluate,
            )
        )

        stop_criterion = StoppingCriterion(
            max_num_trials_completed=benchmark.max_num_evaluations,
        )
        tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=n_workers,
 #           sleep_time=1,
            results_update_interval=600,
            print_update_interval=30,
            # we set a convenient name for tuner to retrieve results easily
            tuner_name=f"results/{method}-{seed}-{benchmark_name}".replace("_", "-"),
            save_tuner=False,
            suffix_tuner_name=False,
            metadata={
                "seed": seed,
                "algorithm": method,
                "benchmark": benchmark_name,
            },
        )
        tuner.run()
        exp_names.append(tuner.name)
    return exp_names


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
        help="seed to run",
    )
    parser.add_argument(
        "--run_all_seeds",
        type=int,
        required=False,
        default=0,
        help="If 1 runs all seeds in [0, args.seed - 1], if 0 runs only args.seed.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
        default="",
        help="directory for optformer model checkpoints",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=False,
        help="a method to run from baselines.py, run all by default.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=False,
        help="a benchmark to run from blackbox_benchmarks.py, run all by default.",
    )
    parser.add_argument(
        "--n_workers",
        help="number of workers to use when tuning.",
        type=int,
        default=1,
    )

    args, _ = parser.parse_known_args()
    if args.run_all_seeds:
        seeds = list(range(args.seed))
    else:
        seeds = [args.seed]
    method_names = [args.method] if args.method is not None else list(methods.keys())
    benchmark_names = (
        [args.benchmark]
        if args.benchmark is not None
        else list(benchmark_definitions.keys())
    )
    run(
        method_names=method_names,
        checkpoint_dir=args.checkpoint_dir,
        benchmark_names=benchmark_names,
        seeds=seeds,
        n_workers=args.n_workers,
    )
