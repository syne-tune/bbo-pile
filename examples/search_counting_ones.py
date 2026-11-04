import os
import time
import argparse
from collections import defaultdict

import numpy as np
import pathlib

import pandas
from syne_tune.config_space import choice

from syne_tune.optimizer.schedulers.searchers.fmbo.fmbo_searcher import FMBOSearcher




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_trials",
        type=int,
        default=20,
        help="",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default='./',
        help="",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./',
        help="",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=5,
        help="",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=20,
        help="",
    )
    args, _ = parser.parse_known_args()
    config_space = {f"x_{i}": choice([0, 1]) for i in range(args.dim)}

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)


    for method in ['random_search', 'local_search']:
        results = defaultdict(list)

        for i in range(args.repetitions):
            print(i)
            st = time.time()
            searcher = FMBOSearcher(config_space=config_space,
                                    task_info={"name": "counting_ones",
                                               'algorithm': "random_search",
                                               "metric_names": 'error'},
                                    checkpoint_dir=checkpoint_dir,
                                    tokenizer_dir=checkpoint_dir,
                                    use_vllm=False)

            curr_best = None
            for trial_id in range(args.max_trials):
                config = searcher.suggest()
                metric = np.sum(list(config.values()))
                searcher.on_trial_complete(trial_id, config, metric)

                if curr_best is None or curr_best < metric:
                    curr_best = metric

                results['repetition'].append(i)
                results['trial_id'].append(trial_id)

                for hp_name in config_space:
                    results[hp_name].append(config[hp_name])

                results['runtime'].append(time.time() - st)
                results['best'].append(curr_best)
            print(time.time() - st)
        os.makedirs(args.output_path, exist_ok=True)
        output_file = pathlib.Path(args.output_path) /  f'counting_ones_results_{method}_dim_{args.dim}.csv'
        pandas.DataFrame(results).to_csv(output_file)