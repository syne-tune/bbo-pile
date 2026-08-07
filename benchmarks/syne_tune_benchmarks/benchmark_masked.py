import itertools
import logging
from argparse import ArgumentParser
from typing import Any

import numpy as np
from tqdm import tqdm

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
    UserBlackboxBackend,
)
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner

from baselines import MethodArguments, methods


def _per_config_best(blackbox, metric_idx: int, mode: str) -> np.ndarray:
    """
    Collapse seeds and fidelities to a single scalar per config.
    `blackbox` must be a BlackboxTabular.
    Returns array of shape (n_configs,).
    """
    flat = blackbox.objectives_evaluations[:, :, :, metric_idx].reshape(
        blackbox.objectives_evaluations.shape[0], -1
    )
    return np.nanmax(flat, axis=1) if mode == "max" else np.nanmin(flat, axis=1)


def _best_value_from_groups(groups: dict, mode: str):
    if mode == "max":
        return max(groups, key=lambda v: np.nanmean(groups[v]))
    else:
        return min(groups, key=lambda v: np.nanmean(groups[v]))


def _marginal_best_single(
    blackbox, hp_name: str, metric_idx: int, mode: str, config_mask=None
) -> Any:
    """
    Group configs by hp_name value and return the value with the best mean.
    `blackbox` must be a BlackboxTabular.
    """
    per_config = _per_config_best(blackbox, metric_idx, mode)
    hps = blackbox.hyperparameters

    if config_mask is not None:
        hps = hps[config_mask]
        per_config = per_config[config_mask]

    groups: dict = {}
    for val, perf in zip(hps[hp_name].values, per_config):
        groups.setdefault(val, []).append(perf)

    return _best_value_from_groups(groups, mode)


def fixed_value_global_optimum(
    blackbox,
    metric: str,
    mode: str,
    masked_params: tuple,
    already_fixed: dict,
) -> dict:
    """
    Fix each masked HP to its value in the best
    config.
    """
    objectives = blackbox.objectives_evaluations
    metric_idx = list(blackbox.objectives_names).index(metric)

    flat = objectives[:, :, :, metric_idx].reshape(objectives.shape[0], -1)
    best_idx = (
        int(np.nanargmax(np.nanmax(flat, axis=1)))
        if mode == "max"
        else int(np.nanargmin(np.nanmin(flat, axis=1)))
    )
    best_row = blackbox.hyperparameters.iloc[best_idx].to_dict()
    return {hp: best_row[hp] for hp in masked_params}


def fixed_value_marginal_best(
    blackbox,
    metric: str,
    mode: str,
    masked_params: tuple,
    already_fixed: dict,
) -> dict:
    """
    For each masked HP independently, pick the value with the best mean
    performance marginalising over all other HPs.
    """
    metric_idx = list(blackbox.objectives_names).index(metric)
    return {
        hp: _marginal_best_single(blackbox, hp, metric_idx, mode, config_mask=None)
        for hp in masked_params
    }


def fixed_value_conditional_marginal(
    blackbox,
    metric: str,
    mode: str,
    masked_params: tuple,
    already_fixed: dict,
) -> dict:
    """
    Fix masked HPs one by one, conditioning each on all previously fixed HPs.
    Falls back to the unconditional marginal if the conditioning set is empty.
    """
    metric_idx = list(blackbox.objectives_names).index(metric)
    hps_df = blackbox.hyperparameters
    result: dict = {}
    running_fixed = dict(already_fixed)

    for hp in masked_params:
        if running_fixed:
            config_mask = np.ones(len(hps_df), dtype=bool)
            for k, v in running_fixed.items():
                if k in hps_df.columns:
                    config_mask &= hps_df[k].values == v
        else:
            config_mask = None

        if config_mask is not None and config_mask.sum() == 0:
            config_mask = None  # fall back to unconditional

        best_val = _marginal_best_single(
            blackbox, hp, metric_idx, mode, config_mask=config_mask
        )
        result[hp] = best_val
        running_fixed[hp] = best_val

    return result


HP_VALUE_STRATEGIES = {
    "global_optimum": fixed_value_global_optimum,
    "marginal_best": fixed_value_marginal_best,
    "conditional_marginal": fixed_value_conditional_marginal,
}


def make_reduced_config_space(config_space: dict, masked_params: tuple) -> dict:
    return {k: v for k, v in config_space.items() if k not in masked_params}


def make_reduced_points(points_to_evaluate: list, masked_params: tuple) -> list:
    return [
        {k: v for k, v in point.items() if k not in masked_params}
        for point in points_to_evaluate
    ]


class _ReducedConfigSpaceProxy:
    """
    Wraps a Blackbox, exposing a reduced configuration_space to the
    scheduler/tuner while forwarding everything else to the real blackbox.
    """

    def __init__(self, blackbox, reduced_config_space: dict):
        self._blackbox = blackbox
        self._reduced_config_space = reduced_config_space

    @property
    def configuration_space(self):
        return self._reduced_config_space

    def __getattr__(self, name):
        return getattr(self._blackbox, name)


class MaskedBlackboxBackend(UserBlackboxBackend):
    """
    Accepts a pre-loaded Blackbox and pins a subset of
    hyperparameters to fixed values before every blackbox query.

    - Scheduler sees `reduced_config_space` via the `blackbox` property.
    - `config_objectives` injects fixed_params before calling super().
    - `_filter_config` uses the real full config space so injected fixed
      params are not stripped before the table/surrogate lookup.
    """

    def __init__(
        self,
        blackbox,
        fixed_params: dict,
        reduced_config_space: dict,
        elapsed_time_attr: str,
        **kwargs,
    ):
        super().__init__(
            blackbox=blackbox,
            elapsed_time_attr=elapsed_time_attr,
            **kwargs,
        )
        self._fixed_params = fixed_params
        self._reduced_config_space = reduced_config_space

    @property
    def blackbox(self):
        real_bb = super().blackbox
        if self._fixed_params:
            return _ReducedConfigSpaceProxy(real_bb, self._reduced_config_space)
        return real_bb

    def _filter_config(self, config: dict[str, Any]) -> dict[str, Any]:
        # Use the full config space so fixed_params survive to the lookup.
        real_config_space = super().blackbox.configuration_space
        return {k: v for k, v in config.items() if k in real_config_space}

    def config_objectives(self, config: dict[str, Any], seed: int) -> list[dict]:
        return super().config_objectives({**config, **self._fixed_params}, seed)


def run(
    method_names,
    benchmark_names,
    seeds,
    checkpoint_dir,
    max_num_evaluations=None,
    n_workers: int = 4,
    mask_n: int = 0,
    hp_value_strategy: str = "global_optimum",
):
    assert hp_value_strategy in HP_VALUE_STRATEGIES, (
        f"Unknown strategy '{hp_value_strategy}'. "
        f"Choose from: {list(HP_VALUE_STRATEGIES.keys())}"
    )
    strategy_fn = HP_VALUE_STRATEGIES[hp_value_strategy]

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

        if benchmark_name.startswith("hpob"):
            from hpob_benchmarks import hpob_benchmark_definitions

            benchmark = hpob_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("tabrepo"):
            from tabrepo_benchmarks import tabrepo_benchmark_definitions

            benchmark = tabrepo_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("pd1"):
            from pd1_benchmarks import pd1_benchmark_definitions

            benchmark = pd1_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("deepar"):
            from deepar_benchmarks import deepar_benchmark_definitions

            benchmark = deepar_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("fcnet"):
            from fcnet_benchmarks import fcnet_benchmark_definitions

            benchmark = fcnet_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("nas201"):
            from nas201_benchmarks import nas201_benchmark_definitions

            benchmark = nas201_benchmark_definitions[benchmark_name]
        elif benchmark_name.startswith("lcbench"):
            from lcbench_benchmarks import lcbench_benchmark_definitions

            benchmark = lcbench_benchmark_definitions[benchmark_name]
        else:
            raise NotImplementedError(f"Unknown benchmark name: {benchmark_name}")

        # Some blackboxes use the surrogate wrapper.
        # In these cases we cant access the objective evaluations
        # and need to instantiate another tabular blackbox
        _eval_probe = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
            surrogate=benchmark.surrogate,
            surrogate_kwargs=benchmark.surrogate_kwargs,
        )
        loaded_blackbox = _eval_probe.blackbox

        _tabular_probe = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
            surrogate=None,
        )
        tabular_blackbox = _tabular_probe.blackbox

        # Config space and HP names come from the evaluation blackbox so that
        # the scheduler sees the correct (possibly continuous surrogate) space.
        full_config_space = loaded_blackbox.configuration_space
        hp_names = list(full_config_space.keys())

        mask_combinations = (
            [()] if mask_n == 0 else list(itertools.combinations(hp_names, mask_n))
        )
        print(
            f"  {len(mask_combinations)} mask combination(s) "
            f"(mask_n={mask_n}, strategy={hp_value_strategy}) "
            f"for ({method}/{benchmark_name}/{seed})"
        )

        shared_backend_kwargs = dict(elapsed_time_attr=benchmark.elapsed_time_attr)

        for masked_params in tqdm(mask_combinations, leave=False):
            # Strategy functions always receive the tabular blackbox.
            fixed_params = (
                strategy_fn(
                    blackbox=tabular_blackbox,
                    metric=benchmark.metric,
                    mode=benchmark.mode,
                    masked_params=masked_params,
                    already_fixed={},
                )
                if masked_params
                else {}
            )

            reduced_config_space = make_reduced_config_space(
                full_config_space, masked_params
            )
            scheduler_config_space = (
                reduced_config_space if masked_params else full_config_space
            )

            num_random_candidates = 5
            random_state = np.random.RandomState(seed)
            points_to_evaluate = make_reduced_points(
                [
                    {
                        k: v.sample(random_state=random_state)
                        if hasattr(v, "sample")
                        else v
                        for k, v in full_config_space.items()
                    }
                    for _ in range(num_random_candidates)
                ],
                masked_params,
            )

            # Backends receive the evaluation blackbox (surrogate if applicable).
            if masked_params:
                backend = MaskedBlackboxBackend(
                    blackbox=loaded_blackbox,
                    fixed_params=fixed_params,
                    reduced_config_space=reduced_config_space,
                    **shared_backend_kwargs,
                )
                mask_label = f"mask-{'_'.join(masked_params)}-{hp_value_strategy}"
            else:
                backend = UserBlackboxBackend(
                    blackbox=loaded_blackbox,
                    **shared_backend_kwargs,
                )
                mask_label = "nomask"

            print(
                f"  ({method}/{benchmark_name}/{seed}) "
                f"strategy={hp_value_strategy} "
                f"masked={masked_params or 'none'} "
                f"fixed={fixed_params or {}}"
            )

            scheduler = methods[method](
                MethodArguments(
                    benchmark_name=benchmark.blackbox_name
                    + "_"
                    + benchmark.dataset_name,
                    config_space=scheduler_config_space,
                    metric=benchmark.metric,
                    mode=benchmark.mode,
                    random_seed=seed,
                    num_brackets=1,
                    checkpoint_dir=checkpoint_dir,
                    use_surrogates=benchmark.use_surrogate,
                    points_to_evaluate=points_to_evaluate,
                )
            )

            stop_criterion = StoppingCriterion(
                max_num_trials_completed=(
                    max_num_evaluations
                    if max_num_evaluations
                    else benchmark.max_num_evaluations
                ),
            )

            exp_label = (
                f"results/{method}-{seed}-{benchmark_name}-{mask_label}".replace(
                    "_", "-"
                )
            )

            tuner = Tuner(
                trial_backend=backend,
                scheduler=scheduler,
                stop_criterion=stop_criterion,
                n_workers=n_workers,
                sleep_time=0,
                callbacks=[SimulatorCallback()],
                results_update_interval=600,
                print_update_interval=30,
                tuner_name=exp_label,
                save_tuner=False,
                suffix_tuner_name=False,
                metadata={
                    "seed": seed,
                    "algorithm": method,
                    "benchmark": benchmark_name,
                    "masked_params": list(masked_params),
                    "fixed_params": {k: str(v) for k, v in fixed_params.items()},
                    "mask_n": mask_n,
                    "hp_value_strategy": hp_value_strategy,
                },
            )
            tuner.run()
            exp_names.append(tuner.name)

    return exp_names


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_all_seeds", type=int, default=0)
    parser.add_argument("--method", type=str, required=False)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument(
        "--mask_n",
        type=int,
        default=1,
        help=(
            "Number of HPs to mask per run. 0 = no masking. "
            "k = run every C(|HPs|, k) combination of k masked HPs."
        ),
    )
    parser.add_argument(
        "--hp_value_strategy",
        type=str,
        default="marginal_best",
        choices=list(HP_VALUE_STRATEGIES.keys()),
        help=(
            "How to choose the fixed value for each masked HP:\n"
            "  global_optimum       — value from the single best joint config (oracle).\n"
            "  marginal_best        — value with the best mean performance, "
            "marginalising over other HPs.\n"
            "  conditional_marginal — like marginal_best but conditions on previously "
            "fixed HPs to account for interactions."
        ),
    )
    args, _ = parser.parse_known_args()

    seeds = list(range(args.seed)) if args.run_all_seeds else [args.seed]

    if args.method is None or args.method.startswith("OptFormer"):
        from original_optformer_methods import original_optformer_methods

        methods = original_optformer_methods | methods

    method_names = [args.method] if args.method is not None else list(methods.keys())

    run(
        method_names=method_names,
        checkpoint_dir=args.checkpoint_dir,
        benchmark_names=[args.benchmark],
        seeds=seeds,
        n_workers=args.n_workers,
        mask_n=args.mask_n,
        hp_value_strategy=args.hp_value_strategy,
    )
