import json
import random
import warnings
from dataclasses import dataclass, field

import numpy as np
from syne_tune.config_space import Categorical, Float, Integer, Domain, config_space_from_json_dict, FiniteRange, is_log_space
from syne_tune.experiments import ExperimentResult


def preprocess(prompt: str):
    prompt = prompt.replace('parameter', "")
    prompt = prompt.replace('trial', "")
    prompt = prompt.replace('\"', "")
    prompt = prompt.replace(' ', "")
    return prompt


class Converter:
    """Converter: converts real values to text tokens (e.g., "500")"""
    
    def __init__(self, q=1000):
        self.q = q
        # separators for prompt formatting
        self.hp_sep = ","
        self.metric_sep = "*"
        self.trial_sep = "|"
        self.header_sep = "&"
    
    def _format_token(self, token: int) -> str:
        """Format token as string (override in subclasses)."""
        return str(token)

    def _parse_token(self, token_str: str) -> int:
        """Parse string token back to int (override in subclasses)."""
        return int(token_str)
    
    def value_to_txt(self, x, hp: Domain = None, hp_name: str = "", x_min=None, x_max=None):
        """
        Convert a value x to a text token based on hp (categorical or numeric).
        For metrics, pass hp=None and provide x_min/x_max.
        """
        # Categorical
        if isinstance(hp, Categorical):
            if hp_name == 'proc.skew_threshold' and np.isnan(x):
                x = 'None'
            if hp_name == 'proc.skew_threshold' and isinstance(x, float):
                x = str(x)
            token = hp.categories.index(x)
            return self._format_token(token)

        # Numeric (Float / Int / FiniteRange) or metric (hp is None)
        if hp is not None:
            x_min = hp.lower
            x_max = hp.upper
            log_scale = getattr(hp, 'log_scale', False)
        else:
            if x_min is None or x_max is None:
                raise ValueError("x_min and x_max must be provided when hp is None")
            log_scale = False

        if x_min == x_max:
            token = 0
        else:
            if log_scale:
                x = np.log(x + 1e-10)
                x_min = np.log(x_min + 1e-10)
                x_max = np.log(x_max + 1e-10)
            x_norm = (x - x_min) / (x_max - x_min)
            token = int(x_norm * self.q)

        return self._format_token(token)
    
    def txt_to_value(self, token_str, hp: Domain = None, hp_name: str = "", x_min=None, x_max=None):
        """
        Convert a text token back to a value based on hp (categorical or numeric).
        For metrics, pass hp=None and provide x_min/x_max.
        """
        token = self._parse_token(token_str)

        # Categorical
        if isinstance(hp, Categorical):
            return hp.categories[token]

        # Numeric or metric
        if hp is not None:
            x_min = hp.lower
            x_max = hp.upper
            log_scale = getattr(hp, 'log_scale', False)
        else:
            if x_min is None or x_max is None:
                raise ValueError("x_min and x_max must be provided when hp is None")
            log_scale = False

        if log_scale:
            x_min = np.log(x_min + 1e-10)
            x_max = np.log(x_max + 1e-10)
            return np.exp(token / self.q * (x_max - x_min) + x_min)
        return token / self.q * (x_max - x_min) + x_min
    
    def get_user_defined_symbols(self):
        """Return list of user-defined symbols for SentencePiece tokenizer"""
        base_symbols = ['name', 'algorithm', 'benchmark', 'type',
                       'CAT', 'UNI', 'INT', "|", "&", "*", ","]
        quant_tokens = [self._format_token(i) for i in range(self.q + 1)]
        return base_symbols + quant_tokens
    
    def encode_problem_definition(self, history, shuffle=False):
        """
        Encode the problem definition to match the OptFormer tokenizer format.
        https://github.com/google-research/optformer/blob/12e2639954b0cd9bf824aab2d040650e6b32089c/optformer/data/converters_test.py#L59

        Args:
            history: History object containing the data to format
            shuffle: Whether to shuffle the order of hyperparameters
            
        Returns:
            Formatted problem definition string (without header separator)
        """
        string = f"benchmark:{history.name},"
        string += f"algorithm:{history.algorithm},"

        hypers = list(history.config_space.items())
        if shuffle:
            random.shuffle(hypers)
        for hp_name, hp in hypers:
            string += f"parameter:"
            string += "{"
            string += f"name:{hp_name},"

            # Log-scale is not supported by this converter.
            if is_log_space(hp):
                warnings.warn(f"Log-scale is not supported by this converter. Skipping {hp_name}")

            if isinstance(hp, Categorical):
                string += f"type:CAT,"
                string += f"categories:{hp.categories},".replace(" ", "")
            elif isinstance(hp, Float):
                string += f"type:UNI,"
                string += f"min_value:{hp.lower},"
                string += f"max_value:{hp.upper},"
            elif isinstance(hp, Integer):
                string += f"type:INT,"
                string += f"min_value:{hp.lower},"
                string += f"max_value:{hp.upper},"
            elif isinstance(hp, FiniteRange):
                if hp.cast_int:
                    string += f"type:INT,"
                else:
                    string += f"type:UNI,"
                string += f"min_value:{hp.lower},"
                string += f"max_value:{hp.upper},"
            else:
                raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")
            string += "}"

        return string, hypers
    
    def encode_trials(self, history, hypers):
        """
        Encode the trials data.
        
        Args:
            history: History object containing the trials to format
            hypers: List of (hp_name, hp) tuples in the order they should be encoded
            
        Returns:
            Formatted trials string
        """
        string = ""
        if len(history.trials) > 0:
            y_min = min(trial.metric for trial in history.trials)
            y_max = max(trial.metric for trial in history.trials)
            if y_min == y_max:
                y_max += 1  # Avoid division by zero in quantization
            for trial in history.trials:
                for i, (hp_name, hp) in enumerate(hypers):
                    if not isinstance(hp, Domain):
                        warnings.warn(f"Skipping unsupported hyperparameter type: {type(hp)}")
                        continue
                    if i > 0:
                        string += self.hp_sep
                    hp_encoded = history.encode(trial.config[hp_name], hp, hp_name)
                    string += str(hp_encoded)
                string += self.metric_sep
                metric_token = self.value_to_txt(trial.metric, hp=None, hp_name="metric", x_min=y_min, x_max=y_max)
                string += str(metric_token)
                string += self.trial_sep
        return string
    

class OptformerConverter(Converter):
    """Converter that formats tokens as <VALUE> instead of plain numbers"""

    def __init__(self, q=1000):
        super().__init__(q=q)
        self.hp_sep = "" # no separator between hyperparameter values

    def _format_token(self, token: int) -> str:
        return f"<{token}>"

    def _parse_token(self, token_str: str) -> int:
        assert token_str.startswith("<") and token_str.endswith(">"), f"Token string must start with '<' and end with '>': {token_str}"
        token_str = token_str[1:-1]
        return int(token_str)

    def get_user_defined_symbols(self):
        base_symbols = ['name', 'algorithm', 'benchmark', 'type',
                       'CAT', 'UNI', 'INT', "LOGUNI", "LOGINT", "|", "&", "*", ",", "<", ">"]
        special_optformer_tokens = [f"<{i}>" for i in range(self.q + 1)]
        return base_symbols + special_optformer_tokens
    
    def encode_problem_definition(self, history, shuffle=False):
        """
        Encode the problem definition (benchmark, algorithm, hyperparameters).
        
        Args:
            history: History object containing the data to format
            shuffle: Whether to shuffle the order of hyperparameters
            
        Returns:
            Formatted problem definition string (without header separator)
        """
        string = f"benchmark:{history.name},"
        string += f"algorithm:{history.algorithm}"
        # ignore objective and goal as they are fixed for all problems
        # string += f"objective:single_objective,"
        # string += f"goal:mimization"

        string += "&&"

        hypers = list(history.config_space.items())
        if shuffle:
            random.shuffle(hypers)
        for idx, (hp_name, hp) in enumerate(hypers):
            if idx > 0:
                string += "*"
            string += f"name:{hp_name},"

            if isinstance(hp, Categorical):
                string += "type:CAT,"
                string += f"categories:{hp.categories}".replace(" ", "")
            elif isinstance(hp, (Float, Integer)):
                hp_type = "LOGUNI" if is_log_space(hp) and isinstance(hp, Float) else \
                          "LOGINT" if is_log_space(hp) and isinstance(hp, Integer) else \
                          "UNI" if isinstance(hp, Float) else "INT"
                string += f"type:{hp_type},"
                string += f"min_value:{hp.lower},"
                string += f"max_value:{hp.upper}"
            elif isinstance(hp, FiniteRange):
                string += "type:INT," if getattr(hp, 'cast_int', False) else "type:UNI,"
                string += f"min_value:{hp.lower},"
                string += f"max_value:{hp.upper}"
            else:
                raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")

        return string, hypers
    
    



# ============================================================================
# History Class
# ============================================================================

@dataclass
class Trial:
    config: dict
    metric: float


@dataclass
class History:
    name: str
    algorithm: str
    config_space: dict
    metric_names: list = field(default_factory=list)
    trials: list = field(default_factory=list)
    converter: Converter = field(default_factory=lambda: Converter())

    def add_trial(self, config, result):
        trial = Trial(config, result)
        self.trials.append(trial)

    def encode(self, x, hp: Domain, hp_name: str = ""):
        """Encode a value x based on the type of hyperparameter hp"""
        return self.converter.value_to_txt(x, hp=hp, hp_name=hp_name)

    def get_prompt(self, shuffle=False):
        """Generate a prompt string using the converter's prompt generation logic."""
        problem_def, hypers = self.converter.encode_problem_definition(self, shuffle=shuffle)
        trials_str = self.converter.encode_trials(self, hypers)
        return problem_def + self.converter.header_sep + trials_str
    
    @classmethod
    def from_syne_tune_experiment(cls, experiment: ExperimentResult, 
                                   max_num_trials: int = None,
                                   converter: Converter = Converter()):
        """
        Create a History object from a Syne Tune ExperimentResult.
        """
        metadata = experiment.metadata
        config_space = config_space_from_json_dict(json.loads(metadata['config_space']))
        metric_name = metadata["metric_names"][0]
        results = experiment.results

        benchmark_name = metadata['benchmark'] if 'benchmark' in metadata else metadata['entrypoint']
        algorithm_name = metadata['algorithm'] if 'algorithm' in metadata else metadata['scheduler_name']
        hist = cls(config_space=config_space,
                   name=benchmark_name,
                   algorithm=algorithm_name,
                   metric_names=metric_name,
                   converter=converter)

        for i, (trial_id, trial) in enumerate(results.groupby('trial_id')):
            row = trial.iloc[-1]
            config = {k: row[f"config_{k}"] for k in config_space.keys()}
            result = row[metric_name]
            hist.add_trial(config, result)
            if max_num_trials is not None and i >= max_num_trials - 1:
                break

        return hist
