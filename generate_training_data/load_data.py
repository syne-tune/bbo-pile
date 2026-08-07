import json
import pandas as pd

from pathlib import Path
from pyparfor import parfor

from json import JSONDecodeError

from syne_tune.experiments import ExperimentResult
from syne_tune.config_space import config_space_from_json_dict
from open_optformer.history import History


def load_result(name, metric_name, config_space, path):
    usecols = [metric_name, "st_tuner_time", "trial_id", "st_decision"]
    usecols.extend(['config_{}'.format(k) for k in config_space.keys()])
    try:
        return pd.read_csv(path / name / "results.csv.zip", usecols=usecols)
    except Exception:
        return None


def create_history_from_results(name, metadata, path: Path,
                                max_num_trials: int,
                                num_numeric_tokens: int = 1000,
                                remove_names: bool = False,
                                n_permutation: int = 0) -> list[str]:
    config_space = config_space_from_json_dict(json.loads(metadata['config_space']))
    metric_name = metadata["metric_names"][0]
    res = load_result(name, metric_name, config_space, path)

    hist = History.from_syne_tune_experiment(ExperimentResult(name=name,
                                                              metadata=metadata,
                                                              results=res,
                                                              path=path,
                                                              tuner=None),
                                             num_numeric_tokens=num_numeric_tokens,
                                             remove_names=remove_names,
                                             max_num_trials=max_num_trials)
    traj = list()
    traj.append(hist.get_prompt())
    for i in range(n_permutation):
        traj.append(hist.get_prompt(shuffle=True))
    return traj

def get_metadata(root: Path):
    metadatas = {}
    for metadata_path in root.rglob(f"*metadata.json"):
        with open(metadata_path, "r") as f:
            folder = metadata_path.parent.name
            try:
                metadatas[folder] = json.load(f)
            except JSONDecodeError as e:
                print(metadata_path)
                raise e

    return metadatas
