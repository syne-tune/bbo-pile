import argparse
import json
import logging
import pathlib
import time
from collections import defaultdict

import numpy as np
from syne_tune.config_space import is_log_space
from syne_tune.backend.trial_status import Trial
from syne_tune.blackbox_repository.blackbox_surrogate import add_surrogate
from syne_tune.blackbox_repository import load_blackbox

from syne_tune.optimizer.schedulers.searchers.fmbo.fmbo_searcher import FMBOSearcher
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)
from benchmarks.syne_tune_benchmarks.baselines import MethodArguments, methods

logger = logging.getLogger(__name__)

VALID_METHODS = ["OptformerHF", "OptformerLitGPT", "OptformerVLLM", "OptformerVLLM-TS50", "RS", "CQR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single HPO searcher seed on FCNet-protein and save results."
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=VALID_METHODS,
        help="Which searcher method to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed to use for this run.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials to run.",
    )
    parser.add_argument(
        "--hf-checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("/hf/qwen3_30M_token_2B_lr_5e-3_bsz_4_seed_0"),
    )
    parser.add_argument(
        "--litgpt-checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("/litgpt"),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("/runtime_results"),
        help="Directory to save per-run JSON results.",
    )
    return parser.parse_args()


def get_searcher(
    method_name: str,
    seed: int,
    points_to_evaluate: list,
    *,
    config_space,
    objective: str,
    hf_checkpoint: pathlib.Path,
    litgpt_checkpoint: pathlib.Path,
):
    task_info = {
        "name": "FCNet-protein",
        "algorithm": "CQR",
        "metric_names": objective,
    }

    def fmbo_scheduler(checkpoint_dir: pathlib.Path, use_vllm: bool, n_sample_configurations: int):
        return SingleObjectiveScheduler(
            config_space=config_space,
            metric=objective,
            do_minimize=True,
            random_seed=seed,
            searcher=FMBOSearcher(
                config_space=config_space,
                checkpoint_dir=checkpoint_dir,
                tokenizer_dir=checkpoint_dir,
                use_vllm=use_vllm,
                random_seed=seed,
                task_info=task_info,
                points_to_evaluate=points_to_evaluate,
                n_sample_configurations=n_sample_configurations,
            ),
        )

    if method_name == "OptformerHF":
        return fmbo_scheduler(hf_checkpoint, use_vllm=False, n_sample_configurations=1)
    elif method_name == "OptformerLitGPT":
        return fmbo_scheduler(litgpt_checkpoint, use_vllm=False, n_sample_configurations=1)
    elif method_name == "OptformerVLLM":
        return fmbo_scheduler(hf_checkpoint, use_vllm=True, n_sample_configurations=1)
    elif method_name == "OptformerVLLM-TS50":
        return fmbo_scheduler(hf_checkpoint, use_vllm=True, n_sample_configurations=50)
    elif method_name in ("RS", "CQR"):
        return methods[method_name](
            MethodArguments(
                benchmark_name="fcnet-protein",
                config_space=config_space,
                metric=objective,
                mode="min",
                random_seed=seed,
                num_brackets=1,
                use_surrogates=True,
                points_to_evaluate=points_to_evaluate,
            )
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("eval.log"),
        ],
    )

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output_dir / f"{args.method}_seed{args.seed}.json"
    if output_path.exists():
        logger.info("Result already exists at %s — skipping.", output_path)
        return

    logger.info("Running method=%s seed=%d n_trials=%d", args.method, args.seed, args.n_trials)

    bb = load_blackbox("fcnet", local_files_only=True)["protein_structure"]
    bb = add_surrogate(bb, predict_curves=False)
    config_space = bb.configuration_space
    objective = bb.objectives_names[0]
    logger.info("Objective: %s | Config space size: %d", objective, len(config_space))

    rng = np.random.RandomState(args.seed)
    points_to_evaluate = [
        {
            k: v.sample(random_state=rng) if hasattr(v, "sample") else v
            for k, v in config_space.items()
        }
    ]

    searcher = get_searcher(
        args.method,
        args.seed,
        points_to_evaluate,
        config_space=config_space,
        objective=objective,
        hf_checkpoint=args.hf_checkpoint,
        litgpt_checkpoint=args.litgpt_checkpoint,
    )

    runtimes = []
    configs = defaultdict(list)

    for trial_id in range(args.n_trials):
        t0 = time.time()

        trial_suggestion = searcher.suggest()
        config = trial_suggestion.config

        metric_val = bb(config, fidelity=10)[objective]
        searcher.on_trial_complete(
            Trial(
                trial_id=trial_id,
                config=config,
                creation_time=time.time(),
            ),
            {objective: metric_val},
        )

        trial_runtime = time.time() - t0
        runtimes.append(trial_runtime)

        for hp, val in config.items():
            try:
                if is_log_space(config_space[hp]):
                    configs[hp].append(float(np.log10(val)))
                else:
                    configs[hp].append(float(val))
            except (TypeError, ValueError):
                configs[hp].append(val)

        logger.info("Trial %d/%d — runtime: %.4fs | %s=%.6f", trial_id + 1, args.n_trials, trial_runtime, objective, metric_val)

    result = {
        "method": args.method,
        "seed": args.seed,
        "n_trials": args.n_trials,
        "objective": objective,
        "runtimes": runtimes,
        "configs": dict(configs),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
