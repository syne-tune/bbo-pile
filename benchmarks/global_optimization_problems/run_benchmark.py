import json
import os
import time

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import pandas

from syne_tune.optimizer.schedulers.ask_tell_scheduler import AskTellScheduler
from syne_tune.tuning_status import Status
from syne_tune.config_space import config_space_to_json_dict
from baselines import methods, MethodArguments
from benchmarks_definitions import benchmark_definitions

if __name__ == "__main__":
    import logging
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--max_trials", type=int, default=100)
    parser.add_argument("--output_path", type=str, default='./')
    parser.add_argument("--num_start_points", type=int, default=5)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
        default="",
        help="directory for optformer model checkpoints",
    )
    args, _ = parser.parse_known_args()

    seed = args.seed
    method = args.method
    benchmark = args.benchmark

    output_path = Path(args.output_path) / f'{benchmark}_{method}_seed_{seed}'
    os.makedirs(output_path, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    max_trials = args.max_trials
    metric = "y"
    mode = "min"

    blackbox = benchmark_definitions[benchmark]
    config_space = blackbox.configuration_space
    points_to_evaluate = [{k: v.sample() for k, v in config_space.items()} for _ in range(args.num_start_points)]
    scheduler = methods[method](MethodArguments(
            metric=metric,
            random_seed=seed,
            mode='min',
            config_space=config_space,
            points_to_evaluate=points_to_evaluate,
            checkpoint_dir=args.checkpoint_dir,
            benchmark_name=args.benchmark
        )
    )

    scheduler = AskTellScheduler(
        base_scheduler=scheduler,
    )

    start_time = time.time()

    results = defaultdict(list)

    for iter in range(max_trials):
        trial_suggestion = scheduler.ask()
        result = blackbox(trial_suggestion.config)
        scheduler.tell(trial_suggestion, result)
        runtime = time.time() - start_time
        print(f'iteration: {iter}, evaluated x={trial_suggestion.config}, objective={result[metric]}, runtime={runtime}')

        for hp_name, hp_value in trial_suggestion.config.items():
            results[f'config_{hp_name}'].append(hp_value)
        results['objective'].append(result[metric])
        results['st_tuner_time'].append(runtime)
        results['st_decision'].append(Status.completed)
        results['trial_id'].append(trial_suggestion.trial_id)

    results = pandas.DataFrame(results)
    results.to_csv(output_path / 'results.csv.zip',  compression={'method': 'zip'})
    metadata = {"algorithm": method,
                "benchmark": 'global-optimization_' + benchmark,
                "seed": seed,
                "config_space":  json.dumps(config_space_to_json_dict(config_space)),
                'metric_names': ['objective']}
    json.dump(metadata, open(output_path / 'metadata.json', 'w'))
