from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from syne_tune.config_space import Domain
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)

logger = logging.getLogger(__name__)


MAX_SAMPLES = 1000


@dataclass
class PopulationElement:
    """Internal PBT state tracked per-trial."""

    trial_id: int
    config: dict
    result: dict


class LocalSearch(SingleObjectiveBaseSearcher):
    """
    Local Search algorithm for hyperparameter optimization.

    This searcher uses a local search strategy to explore the configuration space.
    It extends the StochasticSearcher and used searcher input parameter in LS scheduler.

    Args:
        config_space: Configuration space for the evaluation function.
        points_to_evaluate: Initial points to evaluate. Defaults to None.
        random_seed: Seed for the random number generator.
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        points_to_evaluate: list[dict] | None = None,
        random_seed: int | None = None,
    ):
        self._current_best = None

        if points_to_evaluate is None:
            start_point = {
                k: v.sample() if isinstance(v, Domain) else v
                for k, v in config_space.items()
            }
            points_to_evaluate = [start_point]

        self._current_best = points_to_evaluate[0]  # take the starting point as the current best
        self._current_best_metrics = None

        super().__init__(
            config_space,
            points_to_evaluate=points_to_evaluate,
            random_seed=random_seed,
        )
        self.random_state = np.random.RandomState(self.random_seed)

    def _sample_random_neighbour(self, start_point) -> dict | None:
        # get actual hyperparameters from the search space
        config = deepcopy(start_point)
        hypers = []
        for k, v in self.config_space.items():
            if isinstance(v, Domain):
                hypers.append(k)

        hp_name = np.random.choice(hypers)
        hp = self.config_space[hp_name]
        for i in range(MAX_SAMPLES):
            new_value = hp.sample()
            if new_value != start_point[hp_name]:
                config[hp_name] = new_value
                return config
        return config

    def suggest(self, **kwargs) -> dict | None:
        config = self._next_points_to_evaluate()

        if config is not None:
            return config

        return self._sample_random_neighbour(self._current_best)

    def on_trial_complete(
            self,
            trial_id: int,
            config: Dict[str, Any],
            metric: float,
            resource_level: int = None,
    ):

        if self._current_best_metrics is None or self._current_best_metrics > metric:
            self._current_best = config
            self._current_best_metrics = metric
