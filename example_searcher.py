import time

import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from syne_tune.config_space import is_log_space

from syne_tune.util import catchtime
from syne_tune.backend.trial_status import Trial

from syne_tune.blackbox_repository.blackbox_surrogate import add_surrogate
from syne_tune.blackbox_repository import load_blackbox

from syne_tune.optimizer.schedulers.searchers.fmbo.fmbo_searcher import FMBOSearcher
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)


#bb = load_blackbox("fcnet")["imagenet_resnet_batch_size_512"]
bb = load_blackbox("lcbench")["Fashion-MNIST"]
bb = add_surrogate(bb, predict_curves=False)
config_space = bb.configuration_space
objective = bb.objectives_names[0]

points_to_evaluate = [
    {
        k: v.sample(random_state=np.random.RandomState(0)) if hasattr(v, "sample") else v
        for k, v in config_space.items()
    }
    for _ in range(1)
]

print(points_to_evaluate[0])

name = "remove-forward-refactor"

#checkpoint_dir = pathlib.Path("./checkpoint/")
checkpoint_dir = pathlib.Path('/home/aaron/experiments/open_optformer/checkpoints/qwen3_50M_token_2B_lr_1e-4_bsz_64')

searcher = SingleObjectiveScheduler(
    config_space=config_space,
    metric=objective,
    do_minimize=True,
    random_seed=0,
    searcher=FMBOSearcher(
        config_space=config_space,
        checkpoint_dir=checkpoint_dir,
        tokenizer_dir=checkpoint_dir,
        use_vllm=False,
        random_seed=0,
        task_info={'name': 'lcbench_Fashion-MNIST',
                'algorithm': "CQR",
                'metric_names': objective},
        points_to_evaluate=points_to_evaluate
    ),
)


# Store runtimes for each trial
runtimes = {}
n = 100
configs = defaultdict(list)
for trial_id in range(n):
    # Start timer for this trial
    start_time = time.time()

    print(f"Trial: {trial_id}")
    trial_suggestion = searcher.suggest()
    config = trial_suggestion.config
    print("Config: ", config)
    metric = bb(config, fidelity=10)[objective]
    print("Metric: ", metric)
    searcher.on_trial_complete(Trial(trial_id=trial_id, config=config, creation_time=time.time()), {objective: metric})

    # Calculate runtime for this trial
    runtime = time.time() - start_time
    runtimes[trial_id] = runtime
    print(f"Runtime: {runtime:.4f} seconds\n")
    for hp in config:
        if is_log_space(config_space[hp]):
            configs[hp].append(np.log10(config[hp]))
        else:
            configs[hp].append(config[hp])

# Plot the runtimes
plt.figure(figsize=(10, 6))
plt.plot(list(runtimes.keys()), list(runtimes.values()), marker='o', linewidth=2, markersize=8)
plt.xlabel('Trial ID', fontsize=12)
plt.ylabel('Runtime (seconds)', fontsize=12)
plt.title('Runtime per Trial', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add value labels on each point
for i, runtime in runtimes.items():
    plt.text(i, runtime, f'{runtime:.3f}s', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f"fig-{name}.png")
pd.Series(runtimes).to_csv(f"data-{name}.csv", index=False)

# Print summary statistics
print("\n=== Runtime Summary ===")
print(f"Total runtime: {sum(runtimes):.4f} seconds")
print(f"Average runtime: {sum(runtimes) / len(runtimes):.4f} seconds")
print(f"Min runtime: {min(runtimes):.4f} seconds")
print(f"Max runtime: {max(runtimes):.4f} seconds")

for hp, vals in configs.items():
    plt.figure(dpi=200)
    plt.hist(vals)
    plt.title(hp)
    plt.savefig(f"fig-{hp}.png")