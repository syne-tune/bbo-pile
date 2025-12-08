import json
import pandas as pd

from pathlib import Path
from pyparfor import parfor

from json import JSONDecodeError

from syne_tune.experiments import ExperimentResult
from syne_tune.config_space import config_space_from_json_dict
from open_optformer.history import History, preprocess, Converter


def load_result(name, metric_name, config_space, path):
    usecols = [metric_name, "st_tuner_time", "trial_id", "st_decision"]
    usecols.extend(['config_{}'.format(k) for k in config_space.keys()])
    try:
        return pd.read_csv(path / name / "results.csv.zip", usecols=usecols)
    except Exception:
        return None


def create_history_from_results(name, metadata, path: Path, max_num_trials: int, n_permutation: int = 0, converter: Converter = Converter()) -> [str]:
    config_space = config_space_from_json_dict(json.loads(metadata['config_space']))
    metric_name = metadata["metric_names"][0]
    res = load_result(name, metric_name, config_space, path)

    hist = History.from_syne_tune_experiment(experiment=ExperimentResult(name=name,
                                                                       metadata=metadata,
                                                                       results=res,
                                                                       path=path,
                                                                       tuner=None),
                                             max_num_trials=max_num_trials,
                                             converter=converter)
    traj = []
    traj.append(preprocess(hist.get_prompt()))
    for i in range(n_permutation):
        traj.append(preprocess(hist.get_prompt(shuffle=True)))
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
