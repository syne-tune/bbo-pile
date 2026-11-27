import os
from argparse import ArgumentParser
from pathlib import Path

from slurmpilot import SlurmPilot, JobCreationInfo
from slurmpilot.config import load_config
from slurmpilot.util import unify

from tqdm import tqdm

from baselines import (
    Methods,
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_tag", type=str, required=False, default="bench")
    parser.add_argument("--n_workers", type=int, required=False, default=1)
    parser.add_argument("--num_seeds", type=int, required=False, default=1)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--checkpoint_dir", type=str, required=False, default="")
    parser.add_argument("--cluster", type=str, required=True)
    parser.add_argument("--partition", type=str, required=True)
    parser.add_argument(
        "--slurmpilot_folder", type=str, required=False, default="~/slurmpilot/jobs"
    )
    parser.add_argument("--sbatch_arguments", type=str, required=False)
    parser.add_argument("--benchmark_family", type=str, required=False, default='fcnet')
    args, _ = parser.parse_known_args()

    experiment_tag = args.experiment_tag

    num_seeds = args.num_seeds
    cluster = args.cluster
    partition = args.partition
    sbatch_arguments = args.sbatch_arguments
    slurmpilot_folder = args.slurmpilot_folder

    benchmarks_selected = None
    if args.benchmark_family == "hpob":
        from hpob_benchmarks import hpob_benchmark_definitions
        benchmarks_selected = hpob_benchmark_definitions
    elif args.benchmark_family == "tabrepo":
        from tabrepo_benchmarks import tabrepo_benchmark_definitions
        benchmarks_selected = tabrepo_benchmark_definitions
    elif args.benchmark_family == "pd1":
        from pd1_benchmarks import pd1_benchmark_definitions
        benchmarks_selected = pd1_benchmark_definitions
    elif args.benchmark_family == "deepar":
        from deepar_benchmarks import deepar_benchmark_definitions
        benchmarks_selected = deepar_benchmark_definitions
    elif args.benchmark_family == "lcbench":
        from lcbench_benchmarks import lcbench_benchmark_definitions
        benchmarks_selected = lcbench_benchmark_definitions
    elif args.benchmark_family == "fcnet":
        from fcnet_benchmarks import fcnet_benchmark_definitions
        benchmarks_selected = fcnet_benchmark_definitions
    elif args.benchmark_family == "nas201":
        from nas201_benchmarks import nas201_benchmark_definitions
        benchmarks_selected = nas201_benchmark_definitions

    if benchmarks_selected is None:
        raise ValueError(f"Unknown benchmark family: {args.benchmark_family}")
    methods_selected = [
#        Methods.OPT_RS,
#        Methods.OPT_REA,
        Methods.RS,
        Methods.REA,
        Methods.TPE,
        Methods.BORE,
        Methods.CQR,
        Methods.BOTorch
    ]
    print(f"{len(methods_selected)} methods selected: {methods_selected}")

    config = load_config()

    slurm = SlurmPilot(config=config, clusters=[cluster], ssh_engine="ssh")
    max_runtime_minutes = 60 * 2
    python_args = []
    for method in tqdm(methods_selected):
        for benchmark in benchmarks_selected:
            # To avoid memory issues, we run only one seed per job
            for seed in range(num_seeds):
                python_args.append(
                    {
                        "method": method,
                        "n_workers": args.n_workers,
                        "benchmark": benchmark,
                        "checkpoint_dir": args.checkpoint_dir,
                        # run all seeds in [0, seed-1]
                        "seed": seed,
                        "run_all_seeds": 0,
                    }
                )

    jobname = unify(f"synetune/{experiment_tag}", method="date")

    print(f"Going to launch {len(python_args)} jobs, jobname: {jobname}")
    bash_setup_command = "source ~/.bashrc; source /home/aakl689g/original_optformer/bin/activate;"

    if cluster == 'barnard':
        bash_setup_command += " export PYTHONPATH=''"
    jobinfo = JobCreationInfo(
        cluster=cluster,
        partition=partition,
        sbatch_arguments=sbatch_arguments,
        jobname=jobname,
        entrypoint="benchmark_main.py",
        python_args=python_args,
        src_dir=str(Path(__file__).parent),
        python_binary="python",
        python_libraries=[
            str(Path(__file__).parent.parent.parent / "open_optformer"),
        ],
        n_cpus=4,
        n_gpus=0 if cluster == 'barnard' else 1,  # barnard has no GPU
        mem=1024 * 128 if cluster == 'barnard' else 1024 * 16,
        nodes=1,
        max_runtime_minutes=max_runtime_minutes,
#        bash_setup_command="source ~/.bashrc; conda activate optformer",
        bash_setup_command=bash_setup_command,
        env={
            # write tuner files in Slurmpilot folder corresponding to `jobname`
            "CHECKPOINT_DIR": os.environ.get('CHECKPOINT_DIR', args.checkpoint_dir),
            "CURL_CA_BUNDLE": "/etc/ssl/certs/ca-bundle.crt",
            "SYNETUNE_FOLDER": f"{slurmpilot_folder}/{jobname}",
        },
        n_concurrent_jobs=200,  # max number of jobs to run at the same time, setting this number to high will lead to throttling by huggingface
    )
    if not args.dry_run:
        jobid = slurm.schedule_job(jobinfo)
