import logging
import numpy as np
import torch
import torch.nn.functional as F

from typing import Optional, List, Dict, Any
from pathlib import Path


from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from litgpt.model import GPT

from open_optformer.history import History, preprocess

from syne_tune.config_space import Integer, Categorical, Float, FiniteRange
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import SingleObjectiveBaseSearcher
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)

logger = logging.getLogger(__name__)


def select_token(logits, pos):
    #m = logits[:, pos].argmax(dim=-1)
    probs = torch.nn.functional.softmax(logits[:, pos], dim=-1).detach().numpy()[0, :]
    m = np.random.choice(np.arange(pos.shape[0]), p=probs)
    token = pos[m]
    return token


def get_probability(logits, classes):
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=0)  # shape: [vocab_size]

    # --- Vectorized approach ---

    # 1. Pad classes to same length (max length)
    max_len = max(len(c) for c in classes)
    padded_classes = [c + [-1] * (max_len - len(c)) for c in classes]  # pad with -1
    class_tensor = torch.tensor(padded_classes)  # shape: [num_classes, max_len]

    # 2. Mask for padded tokens
    mask = class_tensor != -1  # True for real tokens, False for padding

    # 3. Replace -1 with 0 for indexing (will be masked out)
    class_tensor_for_index = class_tensor.clone()
    class_tensor_for_index[class_tensor_for_index == -1] = 0

    # 4. Gather probabilities
    token_probs = probs[class_tensor_for_index]  # shape: [num_classes, max_len]

    # 5. Apply mask to ignore padding (set padding probs to 1)
    token_probs = torch.where(mask, token_probs, torch.ones_like(token_probs))

    # 6. Compute joint probability (product across token dimension)
    joint_probs = token_probs.prod(dim=1)  # shape: [num_classes]
    
    return joint_probs

def sample_category_hparam(logits, positions_per_category):
    probs_per_category = get_probability(logits, positions_per_category).detach().numpy()
    probs_per_category /= probs_per_category.sum()
    category_index = np.random.choice(np.arange(len(positions_per_category)), p=probs_per_category)
    return torch.tensor(positions_per_category[category_index])


class OptformerScheduler(SingleObjectiveScheduler):
    """
   """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        checkpoint_dir: Path,
        task_info: Dict = None,
        do_minimize: Optional[bool] = True,
        random_seed: Optional[int] = None,
        points_to_evaluate: Optional[List[dict]] = None,
    ):
        super(OptformerScheduler, self).__init__(
            config_space=config_space,
            metric=metric,
            do_minimize=do_minimize,
            searcher=OptFormerSearcher(
                config_space=config_space,
                points_to_evaluate=points_to_evaluate,
                random_seed=random_seed,
                checkpoint_dir=checkpoint_dir,
                task_info=task_info
            ),
            random_seed=random_seed,
        )




class OptFormerSearcher(SingleObjectiveBaseSearcher):
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
        checkpoint_dir: Path,
        config_space: Dict[str, Any],
        task_info: Dict = None,
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
        random_seed: int = None,
    ):
        super().__init__(config_space, points_to_evaluate, random_seed)

        config = Config.from_file(str(checkpoint_dir / 'model_config.yaml'))
        self.model = GPT(config)

#        self.tokenizer = Tokenizer(str(Path(__file__).parent / "data" / "tokenizer"))
        self.tokenizer = Tokenizer(str(checkpoint_dir))
        state_dict = torch.load(str(checkpoint_dir / 'lit_model.pth'), weights_only=True)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        self.model.load_state_dict(state_dict)
        self.history = []

        if task_info is None:
            self.task_info = {'name': "tst",
                              "algorithm": "BORE",
                              "metric_names": "error"}
        else:
            self.task_info = task_info

        self.study = History(config_space=config_space,
                             name=self.task_info['name'],
                             algorithm=self.task_info['algorithm'],
                             metric_names=[self.task_info['metric_names']],
                             )

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
            return config

        prompt = self.study.get_prompt()
        prompt = preprocess(prompt)
        token = self.tokenizer.encode(prompt)[-self.model.max_seq_length:]
        prompt_size = token.size(0)
        input_pos = torch.arange(0, token.size(0))
        input_pos_maxp1 = torch.tensor([prompt_size])

        self.model.set_kv_cache(batch_size=1)
        prefill_token = True
        config = {}

        converter = self.study.converter
        q = converter.q

        for hp_name, hp in self.config_space.items():
            logits = self.model(token.view(1, -1),
                                input_pos,
                                input_pos_maxp1=input_pos_maxp1)[:, -1]

            if isinstance(hp, (Float, Integer, FiniteRange)):
                # pick value in [0, Q] with the highest probability
                idx = torch.tensor([self.tokenizer.encode(str(i))[-1] for i in range(q + 1)], dtype=torch.int)
                token = select_token(logits, idx)
                value = int(self.tokenizer.decode(token))
                token = torch.tensor([token])
                # Use converter's txt_to_value method
                config[hp_name] = converter.txt_to_value(str(value), hp=hp, hp_name=hp_name)

            elif isinstance(hp, Categorical):
                #  pick the category with the highest probability
                tokens_per_category = [self.tokenizer.encode(str(cat)).tolist() for cat in hp.categories]
                if len(logits.shape) == 2:
                    logits = logits.squeeze(0)
                token = sample_category_hparam(logits, tokens_per_category)
                category = self.tokenizer.decode(token)
                config[hp_name] = category
            if prefill_token:
                prefill_token = False
#                input_pos = torch.tensor([prompt_size],dtype=torch.int64)
#            else:
            input_pos = torch.arange(start=int(input_pos[-1]) + 1, end=int(input_pos[-1] + token.size(0) + 1))
            input_pos_maxp1 = input_pos.max() + 1
            self.model(token.view(1, -1), input_pos, input_pos_maxp1=input_pos_maxp1)
            input_pos = torch.tensor([input_pos_maxp1])
            token = self.tokenizer.encode(',')[-1:]
        return config

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
        if isinstance(metric, list):
            self.study.add_trial(config, metric[0])
        else:
            self.study.add_trial(config, metric)

    def on_trial_error(self, trial_id: int):
        """Called by scheduler if an evaluation job for a trial failed.

        The searcher should react appropriately (e.g., remove pending evaluations
        for this trial, not suggest the configuration again).

        :param trial_id: ID of trial whose evaluated failed
        """
        return

if __name__ == '__main__':

    import pathlib
    from syne_tune.config_space import randint, choice

    config_space = {"a": choice([0, 1, 2, 3, 4])}

    checkpoint_dir = pathlib.Path(__file__).parent / "models" / "small_custom_model" / "step-00000800"
    searcher = OptFormerSearcher(config_space=config_space, checkpoint_dir=checkpoint_dir)

    trial_id = 0
    config = searcher.suggest()
    print(config)
    metric = np.random.rand()
    searcher.on_trial_complete(trial_id, config, metric)
