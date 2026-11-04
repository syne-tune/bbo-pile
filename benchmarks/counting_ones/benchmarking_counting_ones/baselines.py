from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers.asha import AsynchronousSuccessiveHalving
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)
from syne_tune.optimizer.schedulers.searchers.fmbo.fmbo_searcher import FMBOSearcher
from open_optformer.local_search import LocalSearch

@dataclass
class MethodArguments:
    config_space: dict
    metric: str
    mode: str
    random_seed: int
    points_to_evaluate: List[dict]
    verbose: Optional[bool] = False
    checkpoint_dir: Optional[str] = None


class Methods:
    # single fidelity
    RS = "RS"
    LS = "LS"
    OPT_RS = "OPT-RS"
    OPT_LS = "OPT-LS"

methods = {
    Methods.RS: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="random_search",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.LS: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher=LocalSearch(config_space=method_arguments.config_space,
                             points_to_evaluate=method_arguments.points_to_evaluate,
                             random_seed=method_arguments.random_seed),
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
    ),
    Methods.OPT_RS: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher=FMBOSearcher(
            config_space=method_arguments.config_space,
            checkpoint_dir=Path(method_arguments.checkpoint_dir),
            tokenizer_dir=Path(method_arguments.checkpoint_dir),
            use_vllm=False,
            task_info={'name': f'counting_ones_{method_arguments.config_space["dim"]}D',
                       'algorithm': "random_search",
                       'metric_names': "feval"},
            random_seed=method_arguments.random_seed,
            points_to_evaluate=method_arguments.points_to_evaluate,
        ),
    ),

    Methods.OPT_LS: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher=FMBOSearcher(
            config_space=method_arguments.config_space,
            checkpoint_dir=Path(method_arguments.checkpoint_dir),
            tokenizer_dir=Path(method_arguments.checkpoint_dir),
            use_vllm=False,
            task_info={'name': f'counting_ones_{method_arguments.config_space["dim"]}D',
                       'algorithm': "local_search",
                       'metric_names': "feval"},
            random_seed=method_arguments.random_seed,
            points_to_evaluate=method_arguments.points_to_evaluate,
        ),
    ),
}


if __name__ == "__main__":
    # Run a loop that initializes all schedulers on all benchmark to see if they all work
    from benchmarks import (
        benchmark_definitions,
    )

    print(f"Checking initialization of {list(methods.keys())[::-1]}")

    benchmarks = [
        "fcnet-protein",
        "nas201-cifar10",
        "lcbench-Fashion-MNIST",
        "tabrepo-RandomForest-2dplanes",
        "hpob_5636_3492",
    ]
    for benchmark_name in benchmarks:
        benchmark = benchmark_definitions[benchmark_name]
        backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
        )
        points_to_evaluate = [
            {
                k: v.sample() if hasattr(v, "sample") else v
                for k, v in backend.blackbox.configuration_space.items()
            }
            for _ in range(4)
        ]
        print(f"Checking initialization of {list(methods.keys())[::-1]}")
        for method_name, method_fun in list(methods.items())[::-1]:
            print(f"checking initialization of: {method_name}, {benchmark_name}")
            # if method_name != Methods.QHB_XGB:
            #     continue

            scheduler = method_fun(
                MethodArguments(
                    config_space=backend.blackbox.configuration_space,
                    metric=benchmark.metric,
                    mode=benchmark.mode,
                    random_seed=0,
                    points_to_evaluate=points_to_evaluate,
                )
            )
            if isinstance(scheduler, TrialScheduler):
                print(scheduler.suggest())
                print(scheduler.suggest())
            else:
                print(scheduler.suggest(0))
                print(scheduler.suggest(1))
