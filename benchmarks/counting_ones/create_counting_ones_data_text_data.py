import os
import argparse
from pathlib import Path
from copy import deepcopy
import numpy as np
from syne_tune.config_space import choice

from other_searchers.history import History


def preprocess(prompt: str):
    prompt = prompt.replace('parameter', "")
    prompt = prompt.replace('trial', "")
    prompt = prompt.replace('\"', "")
    prompt = prompt.replace(' ', "")
    return prompt


def random_trajectories(iterations, dimensionality):

    trajectory = np.random.randint(0, 2, size=(iterations, dimensionality))
    metric = np.sum(trajectory, axis=1)

    config_space = {f"x_{i}": choice([0, 1]) for i in range(dimensionality)}
    study = History(config_space=config_space,
                    name=f"counting_ones_{dimensionality}D",
                    algorithm="random_search",
                    metric_names=["feval"],
                    )
    prompt = study.get_prompt()

    for i in range(iterations):
        Ti = trajectory[i]
        mi = metric[i]
        prompt += ','.join(str(tij) for tij in Ti) + '*' + str(mi) + '|'

    return prompt


def local_search_trajectories(iterations, dimensionality):
    config_space = {f"x_{i}": choice([0, 1]) for i in range(dimensionality)}
    study = History(config_space=config_space,
                    name=f"counting_ones_{dimensionality}D",
                    algorithm="local_search",
                    metric_names=["feval"],
                    )
    prompt = study.get_prompt()

#    start_point = {k: v.sample() for k, v in config_space.items()}
    start_point = {k: 0 for k, v in config_space.items()}
    metric = np.sum(list(start_point.values()))
    prompt += ','.join(str(tij) for tij in list(start_point.values())) + '*' + str(metric) + '|'
    for i in range(1, iterations):
        # get actual hyperparameters from the search space
        config = deepcopy(start_point)

        hp_name = np.random.choice(list(config_space.keys()))
        config[hp_name] = 1 - config[hp_name]
        new_metric = np.sum(list(config.values()))
        prompt += ','.join(str(tij) for tij in list(config.values())) + '*' + str(new_metric) + '|'
        if new_metric > metric:
            metric = new_metric
            start_point = config

    return prompt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./',
        help="",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=60000,
        help="",
    )
    args, _ = parser.parse_known_args()

    num_trajectories = args.num_trajectories
    iterations = args.iterations
#    dims = [5, 10, 20]
    dims = [5]

    d = Path(args.output_path) / 'data'
    os.makedirs(d, exist_ok=True)

    for mode in ['train', 'valid']:
        with open(d / f'counting_ones_{mode}.txt', 'w') as fh:

            if mode == 'train':
                N = num_trajectories
            else:
                N = 500
            for i in range(N):
                dim = np.random.choice(dims)
#                if np.random.rand() >= args.ratio:
#                    prompt = random_trajectories(iterations, dim)
 #               else:
                prompt = local_search_trajectories(iterations, dim)
                prompt = preprocess(prompt)
                fh.write(prompt + "\n")