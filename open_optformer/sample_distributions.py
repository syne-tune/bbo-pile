"""Sample HP distributions from OptFormer (and reference algorithms) and
write them to a long-format CSV.

  python -m open_optformer.sample_distributions \\
      --benchmark fcnet-protein \\
      --method RS \\
      --n_context_trials 20 \\
      --num_samples 5000 \\
      --seeds 0 1 2 3 4 \\
      --checkpoint /path/to/qwen3_2M_... \\
      --checkpoint /path/to/qwen3_5M_... \\
      --out_csv fcnet_RS_n20.csv

"""
import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from syne_tune.backend.trial_status import Trial
from syne_tune.blackbox_repository import add_surrogate, load_blackbox
from syne_tune.optimizer.baselines import BORE, CQR, RandomSearch, TPE

from open_optformer.optformer_searcher import OptformerScheduler


_REF_FACTORIES = {
    "RS":   lambda cs, metric, seed: RandomSearch(
        config_space=cs, metrics=[metric], do_minimize=True, random_seed=seed,
    ),
    "CQR":  lambda cs, metric, seed: CQR(
        config_space=cs, metric=metric, do_minimize=True, random_seed=seed,
    ),
    "TPE":  lambda cs, metric, seed: TPE(
        config_space=cs, metric=metric, do_minimize=True, random_seed=seed,
    ),
    "BORE": lambda cs, metric, seed: BORE(
        config_space=cs, metric=metric, do_minimize=True, random_seed=seed,
    ),
}


def _load_blackbox(benchmark_name: str):
    """Load the blackbox + surrogate for a benchmark."""
    from benchmarks.syne_tune_benchmarks.fcnet_benchmarks import (
        fcnet_benchmark_definitions,
    )
    from benchmarks.syne_tune_benchmarks.lcbench_benchmarks import (
        lcbench_benchmark_definitions,
    )
    from benchmarks.syne_tune_benchmarks.nas201_benchmarks import (
        nas201_benchmark_definitions,
    )
    from benchmarks.syne_tune_benchmarks.tabrepo_benchmarks import (
        tabrepo_benchmark_definitions,
    )

    defs = {
        **fcnet_benchmark_definitions,
        **lcbench_benchmark_definitions,
        **nas201_benchmark_definitions,
        **tabrepo_benchmark_definitions,
    }
    bd = defs[benchmark_name]
    bb = load_blackbox(bd.blackbox_name)[bd.dataset_name]
    bb = add_surrogate(blackbox=bb, predict_curves=False)
    return bb, bd


def _draw_method_trace(bb, cs, metric, seed, n_trials, mode_sign, method,
                       n_warmup: int = 5):
    """Generate a context trajectory by running `method` for n_trials steps.

    First n_warmup steps are uniform-random; the remaining (n_trials - n_warmup)
    steps come from `method`'s suggest(). All steps are evaluated on the
    blackbox and fed back to the scheduler.

    Returns (configs, signed_ys).
    """
    fidelity_key = list(bb.fidelity_space.keys())[0]
    max_fidelity = max(bb.fidelity_values)
    rng = np.random.RandomState(seed)
    sched = _REF_FACTORIES[method](cs, metric, seed)
    configs, ys = [], []
    for i in range(n_trials):
        if i < n_warmup:
            cfg = {
                k: v.sample(random_state=rng) if hasattr(v, "sample") else v
                for k, v in cs.items()
            }
        else:
            cfg = sched.suggest().config
        y_signed = mode_sign * float(bb(cfg, fidelity={fidelity_key: max_fidelity})[metric])
        sched.on_trial_complete(Trial(i, cfg, 0.0), {metric: y_signed})
        configs.append(cfg)
        ys.append(y_signed)
    return configs, ys


def build_context(
    benchmark_name: str, method: str, seed: int, n_context_trials: int,
    n_warmup: int = 5,
):
    """Build a method-driven context trajectory once. Returns (cs, metric,
    designs, signed_obs); pass these to sample_reference_configs / sample_optformer_configs
    to ensure REF and OptFormer see the IDENTICAL context."""
    if method not in _REF_FACTORIES:
        raise ValueError(f"Unknown method {method!r}.")
    bb, bd = _load_blackbox(benchmark_name)
    cs = bb.configuration_space
    metric = bd.metric
    mode_sign = -1.0 if bd.mode == "max" else 1.0
    designs, obs = _draw_method_trace(
        bb, cs, metric, seed, n_context_trials, mode_sign, method, n_warmup,
    )
    return cs, metric, designs, obs


def sample_reference_configs(
    cs, metric: str, method: str, designs, obs, seed: int,
    num_samples: int = 5000,
) -> List[Dict[str, Any]]:
    """Draw `num_samples` configs from the reference algorithm `method`,
    each from a fresh scheduler primed with the same (designs, obs) context."""
    if method not in _REF_FACTORIES:
        raise ValueError(f"Unknown method {method!r}.")
    out = []
    for j in range(num_samples):
        sched = _REF_FACTORIES[method](cs, metric, seed * num_samples + j)
        for i, (cfg, y) in enumerate(zip(designs, obs)):
            sched.on_trial_complete(Trial(i, cfg, 0.0), {metric: y})
        out.append(sched.suggest().config)
    return out


def sample_optformer_configs(
    checkpoint_dir, benchmark_name: str, method: str,
    cs, metric: str, designs, obs, seed: int, num_samples: int = 5000,
) -> List[Dict[str, Any]]:
    """Sample `num_samples` configs from OptFormer conditioned on (designs, obs)."""
    scheduler = OptformerScheduler(
        config_space=cs, metric=metric, checkpoint_dir=Path(checkpoint_dir),
        task_info={"name": benchmark_name, "algorithm": method, "metric_names": metric},
        do_minimize=True, random_seed=seed, n_sample_configurations=num_samples,
    )
    for i, (cfg, y) in enumerate(zip(designs, obs)):
        scheduler.on_trial_complete(Trial(i, cfg, 0.0), {metric: y})
    configs, _ = scheduler.searcher._sample_n_configs()
    return configs


def _rows(configs, *, series, benchmark, method, n_context_trials, seed):
    """Convert a list of sampled configs into long-format CSV rows."""
    return [
        {
            "series": series,
            "benchmark": benchmark,
            "method": method,
            "n_context_trials": n_context_trials,
            "seed": seed,
            "sample_idx": i,
            **{f"hp_{k}": v for k, v in cfg.items()},
        }
        for i, cfg in enumerate(configs)
    ]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, action="append", required=True,
                   dest="checkpoints",
                   help="HF checkpoint directory. Pass multiple times.")
    p.add_argument("--benchmark", type=str, required=True)
    p.add_argument("--method", type=str, required=True, choices=list(_REF_FACTORIES))
    p.add_argument("--n_context_trials", type=int, default=10,
                   help="Number of context trials (warmup + method-driven).")
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out_csv", type=Path, required=True)
    args = p.parse_args()

    rows = []
    contexts = {}
    for seed in args.seeds:
        print(f"Building {args.method} context seed={seed} ...")
        contexts[seed] = build_context(
            args.benchmark, args.method, seed, args.n_context_trials,
        )

    for seed in args.seeds:
        cs, metric, designs, obs = contexts[seed]
        print(f"Sampling {args.method} reference seed={seed} ...")
        rows.extend(_rows(
            sample_reference_configs(
                cs, metric, args.method, designs, obs, seed, args.num_samples,
            ),
            series="REF", benchmark=args.benchmark, method=args.method,
            n_context_trials=args.n_context_trials, seed=seed,
        ))

    for ckpt in args.checkpoints:
        for seed in args.seeds:
            cs, metric, designs, obs = contexts[seed]
            print(f"Sampling OptFormer {ckpt.name} seed={seed} ...")
            rows.extend(_rows(
                sample_optformer_configs(
                    ckpt, args.benchmark, args.method,
                    cs, metric, designs, obs, seed, args.num_samples,
                ),
                series=ckpt.name, benchmark=args.benchmark, method=args.method,
                n_context_trials=args.n_context_trials, seed=seed,
            ))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(rows)} rows)")

