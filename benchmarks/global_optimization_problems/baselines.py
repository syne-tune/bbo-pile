from dataclasses import dataclass
from pathlib import Path
from syne_tune.optimizer.baselines import REA, RandomSearch, CQR, TPE, BORE
from syne_tune.optimizer.schedulers.single_objective_scheduler import SingleObjectiveScheduler
from open_optformer.hebo_searcher import HEBOSearcher
from syne_tune.optimizer.schedulers.searchers.fmbo.fmbo_searcher import FMBOSearcher

@dataclass
class MethodArguments:
    config_space: dict
    metric: str
    mode: str
    random_seed: int
    points_to_evaluate: list[dict]
    checkpoint_dir: str
    benchmark_name: str

class Methods:
    BORE = "BORE"
    RS = "RS"
    TPE = "TPE"
    REA = "REA"
    CQR = "CQR"
    HEBO = 'HEBO'
    OPT_CQR = 'OPT-CQR'
    OPT_REA = 'OPT-REA'
    OPT_BORE = 'OPT-BORE'
    OPT_TPE = 'OPT-TPE'
    OPT_HEBO = 'OPT-HEBO'
    OPT_CQR_TS = 'OPT-CQR-TS'
    OPT_CQR_TS_5 = 'OPT-CQR-TS-5'

def _fmbo_scheduler(method_arguments, algorithm: str, n_sample_configurations: int = 1):
    return SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher=FMBOSearcher(
            config_space=method_arguments.config_space,
            checkpoint_dir=Path(method_arguments.checkpoint_dir),
            tokenizer_dir=Path(method_arguments.checkpoint_dir),
            use_vllm=False,
            task_info={'name': method_arguments.benchmark_name,
                       'algorithm': algorithm,
                       'metric_names': "feval"},
            random_seed=method_arguments.random_seed,
            points_to_evaluate=method_arguments.points_to_evaluate,
            n_sample_configurations=n_sample_configurations,
        ),
    )

methods = {
    Methods.RS: lambda method_arguments: RandomSearch(
        config_space=method_arguments.config_space,
        metrics=[method_arguments.metric],
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate
    ),
    Methods.BORE: lambda method_arguments: BORE(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate
    ),
    Methods.CQR: lambda method_arguments: CQR(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate
    ),
    Methods.TPE: lambda method_arguments: TPE(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate
    ),
    Methods.REA: lambda method_arguments: REA(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate
    ),
    Methods.HEBO: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher=HEBOSearcher(
            config_space=method_arguments.config_space,
            do_minimize=method_arguments.mode == "min",
            random_seed=method_arguments.random_seed,
            points_to_evaluate=method_arguments.points_to_evaluate
        ),
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
    ),
    Methods.OPT_CQR: lambda method_arguments: _fmbo_scheduler(method_arguments, "CQR"),
    Methods.OPT_REA: lambda method_arguments: _fmbo_scheduler(method_arguments, "REA"),
    Methods.OPT_BORE: lambda method_arguments: _fmbo_scheduler(method_arguments, "BORE"),
    Methods.OPT_TPE: lambda method_arguments: _fmbo_scheduler(method_arguments, "TPE"),
    Methods.OPT_HEBO: lambda method_arguments: _fmbo_scheduler(method_arguments, "HEBO"),
    Methods.OPT_CQR_TS: lambda method_arguments: _fmbo_scheduler(method_arguments, "CQR", n_sample_configurations=50),
    Methods.OPT_CQR_TS_5: lambda method_arguments: _fmbo_scheduler(method_arguments, "CQR", n_sample_configurations=5),
}
