import json
import os
from pathlib import Path

import numpy as np
import pathlib
import argparse

from scipy.stats import entropy
from collections import Counter
from typing import Dict, Any, List

from syne_tune.blackbox_repository import add_surrogate
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular
from syne_tune.backend.trial_status import Trial
from syne_tune.blackbox_repository.repository import load_blackbox
from syne_tune.config_space import Categorical
from syne_tune.optimizer.baselines import CQR
from syne_tune.optimizer.schedulers.single_objective_scheduler import SingleObjectiveScheduler
from syne_tune.optimizer.schedulers.searchers.fmbo.fmbo_searcher import FMBOSearcher

from benchmarks.syne_tune_benchmarks.fcnet_benchmarks import fcnet_benchmark_definitions
from benchmarks.syne_tune_benchmarks.nas201_benchmarks import nas201_benchmark_definitions
from benchmarks.syne_tune_benchmarks.lcbench_benchmarks import lcbench_benchmark_definitions
from benchmarks.syne_tune_benchmarks.tabrepo_benchmarks import tabrepo_benchmark_definitions
from benchmarks.syne_tune_benchmarks.hpob_benchmarks import hpob_benchmark_definitions
from benchmarks.syne_tune_benchmarks.pd1_benchmarks import pd1_benchmark_definitions


# --- Helper Functions ---
def objective_function(config: Dict[str, Any], blackbox: BlackboxTabular) -> float:
    """
    Evaluates a configuration using the Syne Tune blackbox backend.
    """
    # The evaluate method returns a dictionary of metrics, we need to extract the one we care about
    result = blackbox(config, fidelity=max_fidelity)
    return result[metric_name]

def collect_samples(scheduler,
                    initial_design: List[Dict[str, Any]],
                    observations: List[float],
                    num_samples: int,
                    ) -> List[Dict[str, Any]]:
    """
    Collects configurations suggested by a scheduler over a fixed number of iterations.
    """
    for i, (config, yi) in enumerate(zip(initial_design, observations)):

        # Inform the scheduler about the trial completion
        scheduler.on_trial_complete(
            Trial(trial_id=i, config=config, creation_time=0.0), # creation_time is dummy
            {metric_name: yi}
        )
    return [scheduler.suggest().config for _ in range(num_samples)]


def config_to_tuple(config: Dict[str, Any]) -> tuple:
    """Converts a configuration dictionary to a hashable tuple for counting."""
    # Ensure consistent order by sorting keys
    return tuple(config[key] for key in sorted(config.keys()))

def compute_kl_divergence(configs1: List[Dict[str, Any]], configs2: List[Dict[str, Any]]) -> float:
    """
    Computes the KL divergence between the sampling distributions of two sets of configurations.
    """
    all_configs = set(config_to_tuple(c) for c in configs1).union(
        set(config_to_tuple(c) for c in configs2)
    )

    # Create frequency counts for each distribution
    counts1 = compute_hyperparameter_value_counts(configs1, config_space)
    counts2 = compute_hyperparameter_value_counts(configs2, config_space)

    # Convert counts to probability distributions, ensuring all_configs are present
    # Add a small epsilon to avoid log(0) for configurations not sampled by one method
 #   epsilon = 1e-10
 #   p1 = np.array([counts1.get(c, 0) + epsilon for c in all_configs])
 #   p2 = np.array([counts2.get(c, 0) + epsilon for c in all_configs])

    kl_div = 0
    for hp_name in counts1.keys():
        p1 = np.array(tuple(counts1[hp_name].values()))
        p1 = p1 / p1.sum()
        p2 = np.array(tuple(counts2[hp_name].values()))
        p2 = p2 / p2.sum()

        kl_div += entropy(p1, p2)
    return kl_div / len(counts1.keys())

def compute_hyperparameter_value_counts(
    configs: List[Dict[str, Any]], config_space: Dict[str, Any]
) -> Dict[str, Dict[Any, int]]:
    """
    Computes the frequency of each value for each hyperparameter across a list of configurations.
    """
    hp_value_counts = {}
    for hp_name, hp in config_space.items():
        if isinstance(hp, Categorical):
            values = hp.categories
        else:
            values = hp.values
        hp_value_counts[hp_name] = Counter(values)

    for config in configs:
        for hp_name, hp_value in config.items():
            if hp_name in hp_value_counts:
                hp_value_counts[hp_name][hp_value] += 1
    return {hp_name: dict(counts) for hp_name, counts in hp_value_counts.items()}

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute KL Divergence between CQR and OptFormer sampling distributions.")
    parser.add_argument("--benchmark_name", type=str, default="fcnet-protein",
                        help="Name of the benchmark to use (e.g., 'fcnet-protein').")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of random seeds for reproducibility.")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to collect from each scheduler for KL divergence computation.")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="qwen3_150M_token_2B_lr_1e-2_bsz_16_ws_2288_seed_0",
                        help="Path to a valid trained OptFormer model checkpoint directory.")
    parser.add_argument("--output_dir", type=str,
                        default="./results",
                        help="Path to store the results json.")
    parser.add_argument("--method", type=str,
                        default="CQR",
                        help="Optimization method.")

    args = parser.parse_args()

    benchmark_name = args.benchmark_name
    num_samples = args.num_samples
    optformer_checkpoint_dir = pathlib.Path(args.checkpoint_dir)

    benchmark_definitions = {**fcnet_benchmark_definitions,
                             **nas201_benchmark_definitions,
                             **tabrepo_benchmark_definitions,
                             **lcbench_benchmark_definitions,
                             **pd1_benchmark_definitions,
                             **tabrepo_benchmark_definitions}
    print(benchmark_definitions.keys())
    # Load benchmark definition and setup backend
    try:
        benchmark = benchmark_definitions[benchmark_name]
    except KeyError:
        print(f"Error: Benchmark '{benchmark_name}' not found in benchmark_definitions.")
        print("Available benchmarks: ", list(benchmark_definitions.keys()))
        exit()

    blackbox = load_blackbox(benchmark.blackbox_name)[benchmark.dataset_name]
    blackbox = add_surrogate(blackbox=blackbox, predict_curves=False)
    config_space = blackbox.configuration_space
    metric_name = benchmark.metric
    mode = benchmark.mode
    max_fidelity = max(blackbox.fidelity_values)

    # Initial Design
    for seed in range(args.num_seeds):
        initial_design = []
        observations = []
        rng = np.random.RandomState(seed)
        for num_iterations in [5, 10, 20, 50, 90]:
            print(
                f"Starting KL Divergence computation for benchmark '{benchmark_name}' with {num_iterations} iterations...")
            for i in range(len(initial_design), num_iterations - 1):
                config = {k: v.sample(random_state=rng) for k, v in config_space.items()}
                initial_design.append(config)
                observation = objective_function(config, blackbox)
                observations.append(observation)

            # 1. Initialize Scheduler
            print("Initializing Scheduler...")
            if args.method == 'CQR':
                scheduler = CQR(
                    config_space=config_space,
                    metric=metric_name,
                    do_minimize= mode == "min",
                    random_seed=seed,
                )
            print(f"Collecting {num_samples} samples for {args.method}...")
            original_samples = collect_samples(scheduler, initial_design, observations, num_samples)
            print(f"Collected {len(original_samples)} samples for {args.method}.")

            # 2. Initialize OptFormer Scheduler
            print("Initializing OptFormer Scheduler...")
            optformer_scheduler = SingleObjectiveScheduler(
                config_space=config_space,
                metric=metric_name,
                do_minimize=mode == "min",
                random_seed=seed,
                searcher=FMBOSearcher(
                    config_space=config_space,
                    checkpoint_dir=optformer_checkpoint_dir,
                    tokenizer_dir=optformer_checkpoint_dir,
                    use_vllm=False,
                    task_info={
                        'name': benchmark_name,
                        'algorithm': args.method, # This is important for OptFormer's internal prompt generation
                        'metric_names': metric_name
                    },
                    random_seed=seed,
                    n_sample_configurations=1, # We want the final suggested config for comparison
                ),
            )

            print(f"Collecting {num_samples} samples for OptFormer...")
            optformer_samples = collect_samples(optformer_scheduler, initial_design, observations, num_samples=num_samples)
            print(f"Collected {len(optformer_samples)} samples for OptFormer.")

            # Ensure both lists have samples before proceeding
            if not original_samples or not optformer_samples:
                print("Error: One or both schedulers failed to collect samples. Cannot compute KL divergence.")
            else:
                # 4. Compute KL Divergence
                print("Computing KL Divergence...")
                kl_divergence = compute_kl_divergence(original_samples, optformer_samples)
                print(f"KL Divergence (CQR vs OptFormer): {kl_divergence:.4f}")

            print("Script finished.")

            results = dict()
            results['method'] = args.method
            results['benchmark_name'] = benchmark_name
            results['kl_divergence'] = kl_divergence
            results['seed'] = seed
            results['num_iterations'] = num_iterations
            os.makedirs(args.output_dir, exist_ok=True)
            json.dump(results, open(Path(args.output_dir) / f'results_{args.method}_{args.benchmark_name}_iters_{num_iterations}_seed_{seed}.json', 'w'))