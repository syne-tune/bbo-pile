import copy
import logging
import os

import gc
import jax
from pathlib import Path

from typing import Optional, List, Dict, Any

from vizier.service import pyvizier as vz
from optformer.t5x import inference_utils
from optformer.t5x import policies

from syne_tune.config_space import FiniteRange
from syne_tune.config_space import choice, Categorical, Integer, Float, is_log_space
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import SingleObjectiveBaseSearcher

logger = logging.getLogger(__name__)

BBOB_INFERENCE_MODEL_KWARGS = {
    'checkpoint_path_or_model_dir': os.environ['CHECKPOINT_DIR'] + 'bbob/checkpoint_700000',
    'model_gin_file': str(Path(__file__).parent / 'bbob.gin'),
    'batch_size': 1,
}

HPOB_INFERENCE_MODEL_KWARGS = {
    'checkpoint_path_or_model_dir': os.environ['CHECKPOINT_DIR'] + 'hpob/checkpoint_400000',
    'model_gin_file': str(Path(__file__).parent / 'hpob.gin'),
    'batch_size': 1,
}

class OriginalOptFormerSearcher(SingleObjectiveBaseSearcher):
    """

    :param config_space: Configuration space
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order). Each config in the list can be partially
        specified, or even be an empty dict. For each hyperparameter not
        specified, the default value is determined using a midpoint heuristic.
        If ``None`` (default), this is mapped to ``[dict()]``, a single default config
        determined by the midpoint heuristic. If ``[]`` (empty list), no initial
        configurations are specified.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        random_seed: int = None,
        task_info: Dict = None,
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
        designer_name: str = 'designer_hill_climb',
        model = 'bbob',
        searcher_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super().__init__(config_space=config_space,
                         points_to_evaluate=points_to_evaluate,
                         random_seed=random_seed)

        self.metric = "bbob_eval"
        self.random_seed = random_seed
        self.designer_name = designer_name
        self.config_space = config_space

        problem = vz.ProblemStatement()
        for name, hp in config_space.items():
            print(f"Adding {name} to search space")
            if isinstance(hp, Categorical):
                problem.search_space.root.add_categorical_param(name, hp.categories)
            elif isinstance(hp, Integer):
                scale_type = vz.ScaleType.LOG if is_log_space(hp) else vz.ScaleType.LINEAR
                problem.search_space.root.add_int_param(name, hp.lower, hp.upper, scale_type=scale_type)
            elif isinstance(hp, Float):
                scale_type = vz.ScaleType.LOG if is_log_space(hp) else vz.ScaleType.LINEAR
                problem.search_space.root.add_float_param(name, hp.lower, hp.upper, scale_type=scale_type)
            elif isinstance(hp, FiniteRange):
                scale_type = vz.ScaleType.LOG if is_log_space(hp) else vz.ScaleType.LINEAR
                problem.search_space.root.add_float_param(name, hp.lower, hp.upper, scale_type=scale_type)
            else:
                raise Exception(f"Unsupported hyperparameter type {type(hp)} for {name}. ")

        do_minimize = vz.ObjectiveMetricGoal.MINIMIZE

        problem.metric_information.append(vz.MetricInformation(name=self.metric, goal=do_minimize))

        if model == 'bbob':
            inference_model = inference_utils.InferenceModel.from_checkpoint(
                **BBOB_INFERENCE_MODEL_KWARGS
            )
        elif model == 'hpob':
            inference_model = inference_utils.InferenceModel.from_checkpoint(
                **HPOB_INFERENCE_MODEL_KWARGS
            )
        self.model = policies.OptFormerDesigner(
            problem, inference_model=inference_model, designer_name=self.designer_name, temperature=0.9
        )

        self.history = []

    def suggest(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Suggest a new configuration.

        Note: Query :meth:`_next_initial_config` for initial configs to return
        first.

        :param kwargs: Extra information may be passed from scheduler to
            searcher
        :return: New configuration. The searcher may return None if a new
            configuration cannot be suggested. In this case, the tuning will
            stop. This happens if searchers never suggest the same config more
            than once, and all configs in the (finite) search space are
            exhausted.

        """

        config = self._next_points_to_evaluate()
        if config is not None:
            # If there are initial configs, return them first
            suggestion = vz.TrialSuggestion(parameters=config,
            )
        else:
            suggestion = self.model.suggest(1)[0]
            jax.clear_caches()  # Clear JAX caches to avoid memory leaks
            gc.collect()  # Collect garbage to free memory
        self.history.append(suggestion)

        return suggestion.to_trial().parameters.as_dict()

    def on_trial_complete(
            self,
            trial_id: int,
            config: Dict[str, Any],
            metric: float,
            resource_level: int = None,
    ):
        """Inform searcher about result

        The scheduler passes every result. If ``update == True``, the searcher
        should update its surrogate model (if any), otherwise ``result`` is an
        intermediate result not modelled.

        The default implementation calls :meth:`_update` if ``update == True``.
        It can be overwritten by searchers which also react to intermediate
        results.

        :param trial_id: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param config: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param metric: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        """
        trial = self.history[trial_id].to_trial()
        if isinstance(metric, list):
            metric = metric[0]  # If multiple metrics, take the first one
        trial.complete(vz.Measurement({self.metric: metric}))

        copied_trial = copy.deepcopy(trial)
        if self.model._metric_flipped:
            policies.flip_trial_metric_values(copied_trial)
        self.model._historical_study.trials.append(copied_trial)


    def on_trial_error(self, trial_id: int):
        """Called by scheduler if an evaluation job for a trial failed.

        The searcher should react appropriately (e.g., remove pending evaluations
        for this trial, not suggest the configuration again).

        :param trial_id: ID of trial whose evaluated failed
        """
        return
    

if __name__ == '__main__':

    import numpy as np
    import time

    config_space = {"a": choice([0, 1, 2, 3, 4])}
    searcher = OriginalOptFormerSearcher(config_space=config_space)

    for trial_id in range(5):
        start = time.time()
        config = searcher.suggest()
        print(f"Iteration {trial_id}: ", config, "time :", time.time() - start)
        metric = np.random.rand()
        searcher.on_trial_complete(trial_id, config, metric)
