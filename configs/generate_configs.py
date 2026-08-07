import os
import math
import yaml
import numpy as np
from pathlib import Path

BASE_PATH_CLUSTER = os.environ['BASE_PATH']
DATASET_NAME = 'all'
VERSION = 'v0.8'
SEED = 0
WANDB_PROJECT = f'open_optformer_qwen3_hp_sweep_{VERSION}'

def generate_configs():

    model_names = {
        'qwen3_2M': 2e6,
        'qwen3_5M': 5e6,
        'qwen3_13M': 13e6,
        'qwen3_30M': 30e6,
        'qwen3_80M': 80e6,
        'qwen3_150M': 150e6,
#        'qwen3_450M': 450e6,
    }

    counter = 0
    for model_name, model_size in model_names.items():
        base_config_path = Path(__file__).parent / f"{model_name}.yaml"

        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)

        token_counts = {
            "100M": 100_000_000,
            "200M": 200_000_000,
            "400M": 400_000_000,
            "600M": 600_000_000,
            "800M": 800_000_000,
            "1B": 1_000_000_000,
            '2B': 2_000_000_000,
            '4B': 4_000_000_000,
        }
        lr_grid = {"5e-3": 5e-3, "1e-2": 1e-2, "2e-2": 2e-2}

#        gas_grid = [1, 2, 4, 8, 16]
#        mbs = 16
        bsz_grid = [4, 8, 16]

        base_path = Path(BASE_PATH_CLUSTER)

        for name, tokens in token_counts.items():

            if tokens < 2 * model_size:
                # Do not run any model with smaller token budget of 2 times the parameter count
                continue

            for lr_name, lr in lr_grid.items():
                for bsz in bsz_grid:
#                    for ws in ws_grid:
#                    bsz = int(gas * mbs)
                    number_of_steps = tokens // (bsz * base_config['train']['max_seq_length'])
                    new_config = base_config.copy()

                    ws = int(number_of_steps * 0.1)  # 10% warm-up
                    print(model_name, name, number_of_steps, bsz, ws, str(ws / number_of_steps * 100) + '%')
                    new_config['optimizer']['init_args']['lr'] = lr

                    new_config['train']['max_tokens'] = tokens
                    new_config['train']['global_batch_size'] = bsz
                    new_config['train']['log_interval'] = math.ceil(number_of_steps / 200)
                    new_config['train']['lr_warmup_steps'] = ws
                    new_config['train']['micro_batch_size'] = bsz
                    new_config['train']['save_interval'] = math.ceil(number_of_steps / 10)  # Save 10 checkpoints per model
                    new_config['eval']['interval'] = math.ceil(number_of_steps / 50)

                    run_name = f"{model_name}_token_{name}_lr_{lr_name}_bsz_{bsz}_ws_{ws}_seed_{SEED}"

                    new_config['log']['run'] = run_name
                    new_config['log']['project'] = WANDB_PROJECT
                    new_config['log']['group'] = model_name
                    new_config['seed'] = SEED
                    new_config['data']['init_args']['data_path'] = str(base_path / 'tokenized_dataset' / VERSION / DATASET_NAME)
                    new_config['tokenizer_dir'] = str(base_path / 'tokenizer' / VERSION )
                    new_config['out_dir'] = str(base_path / 'checkpoints' / VERSION /  run_name)
                    if VERSION == 'v0.3':
                        new_config['model_config']['vocab_size'] = 1106
                        new_config['model_config']['padded_vocab_size'] = 1106
                    elif VERSION == 'v0.4':
                        new_config['model_config']['vocab_size'] = 1072
                        new_config['model_config']['padded_vocab_size'] = 1072
                    elif VERSION in ['v0.5', 'v0.6', 'v0.7', 'v0.8']:
                        new_config['model_config']['vocab_size'] = 1069
                        new_config['model_config']['padded_vocab_size'] = 1069
                    new_filename = f"{run_name}.yaml"
                    new_filepath = base_config_path.parent / new_filename

                    with open(new_filepath, 'w') as f:
                       yaml.dump(new_config, f, sort_keys=False)

                    #print(f"Generated {new_filepath}")
                    counter += 1

    print('num configs: ', counter)
if __name__ == "__main__":
    generate_configs()
