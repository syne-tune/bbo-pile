# BBO-Pile: A Dataset for Pre-training Open Foundation Models for Black-box Optimization

This repository contains code for data generation and training models on BBO-Pile.

> **NeurIPS 2026 Evaluations & Datasets Track submission.** Anonymized for double-blind review.

- [Installation](#installation)
- [Generate BBO-Pile from Scratch](#generate-bbo-pile-from-scratch)
- [Train Models from Scratch](#train-models-from-scratch)
- [Evaluate a Trained Model](#evaluate-a-trained-model)
- [License](#license)

---

## Installation

This project is managed with [uv](https://github.com/astral-sh/uv) (assumed to be on `PATH`). Clone and sync the locked environment:

```bash
git clone <ANONYMIZED_REPO_URL>
cd open_optformer
uv sync
source .venv/bin/activate
```

**Tested on**: Linux x86_64, Python 3.10.12, CUDA 12.x.

---

## Generate BBO-Pile from Scratch


Set core env variables:

```bash
export BASE_PATH=~/experiments/bbo-pile
export VERSION=v0.8
export RESULTS_PATH=~/syne-tune-results
```

**Run an optimizer on a given task:**

```bash
cd benchmarks/syne_tune_benchmarks
python benchmark_main.py --seed 5 --run_all_seeds 1 --n_workers 1 --method RS --benchmark fcnet-protein
```

Available benchmark families are: 
- fcnet 
- nas201
- lcbench
- pd1
- hpob
- tabrepo

**Data Processing**

First to compile the results of all benchmark family into a dataset
```bash
cd generate_training_data
python compile_data.py --path $RESULTS_PATH/fcnet/ --output_path $BASE_PATH/dataset/$VERSION/fcnet/ --remove_names
python compile_data.py --path $RESULTS_PATH/masked_fcnet/ --output_path $BASE_PATH/dataset/$VERSION/masked_fcnet/ --remove_names
python compile_data.py --path $RESULTS_PATH/masked_nas201/ --output_path $BASE_PATH/dataset/$VERSION/masked_nas201/ --remove_names
python compile_data.py --path $RESULTS_PATH/global_optimization_benchmarks/ --output_path $BASE_PATH/dataset/$VERSION/global_optimization_benchmarks/ --remove_names
python compile_data.py --path $RESULTS_PATH/hpob/ --output_path $BASE_PATH/dataset/$VERSION/hpob/ --remove_names
python compile_data.py --path $RESULTS_PATH/lcbench/ --output_path $BASE_PATH/dataset/$VERSION/lcbench/ --remove_names
python compile_data.py --path $RESULTS_PATH/nas201/ --output_path $BASE_PATH/dataset/$VERSION/nas201/ --remove_names
python compile_data.py --path $RESULTS_PATH/pd1/ --output_path $BASE_PATH/dataset/$VERSION/pd1/ --remove_names
python compile_data.py --path $RESULTS_PATH/tabrepo/ --output_path $BASE_PATH/dataset/$VERSION/tabrepo/ --remove_names
``` 

Concat all datasets for training and validation:
```bash
mkdir $BASE_PATH/dataset/$VERSION/all/
python concat_text_files.py $BASE_PATH/dataset/$VERSION/fcnet/train.txt $BASE_PATH/dataset/$VERSION/nas201/train.txt $BASE_PATH/dataset/$VERSION/masked_fcnet/train.txt $BASE_PATH/dataset/$VERSION/masked_nas201/train.txt $BASE_PATH/dataset/$VERSION/hpob/train.txt $BASE_PATH/dataset/$VERSION/tabrepo/train.txt $BASE_PATH/dataset/$VERSION/pd1/train.txt $BASE_PATH/dataset/$VERSION/lcbench/train.txt $BASE_PATH/dataset/$VERSION/global_optimization_benchmarks/train.txt $BASE_PATH/dataset/$VERSION/all/unshuffled_train.txt
python concat_text_files.py $BASE_PATH/dataset/$VERSION/fcnet/valid.txt $BASE_PATH/dataset/$VERSION/nas201/valid.txt $BASE_PATH/dataset/$VERSION/masked_fcnet/valid.txt $BASE_PATH/dataset/$VERSION/masked_nas201/valid.txt $BASE_PATH/dataset/$VERSION/hpob/valid.txt $BASE_PATH/dataset/$VERSION/tabrepo/valid.txt $BASE_PATH/dataset/$VERSION/pd1/valid.txt $BASE_PATH/dataset/$VERSION/lcbench/valid.txt $BASE_PATH/dataset/$VERSION/global_optimization_benchmarks/valid.txt $BASE_PATH/dataset/$VERSION/all/unshuffled_valid.txt
```


Shuffle the data
```bash
cd $BASE_PATH/dataset/$VERSION/all/
shuf --random-source=<(yes 42) unshuffled_train.txt > train.txt
shuf --random-source=<(yes 42) unshuffled_valid.txt > valid.txt
```

---

## Train Models from Scratch

Assumes `$BASE_PATH/dataset/$VERSION/all/{train,valid}.txt` exists — either pulled from [huggingface.co/datasets/synetune/bbo-pile](https://huggingface.co/datasets/synetune/bbo-pile) or produced by [Generate BBO-Pile from Scratch](#generate-bbo-pile-from-scratch).



**Train the tokenizer:**

```bash
python generate_training_data/train_tokenizer.py \
    --input_folder "$BASE_PATH/dataset/$VERSION/all" \
    --output_path "$BASE_PATH/tokenizer/$VERSION" \
    --vocab_size 1069
```

**Pre-tokenize into litdata format** (≥64 GB RAM):

```bash
uv run python generate_training_data/preprocess_data.py \
    --input_path "$BASE_PATH/dataset/$VERSION/all" \
    --output_path "$BASE_PATH/tokenized_dataset/$VERSION/all" \
    --tokenizer_dir "$BASE_PATH/tokenizer/$VERSION"
```

**Launch training**:

Generate per-run YAML configs (model size × token budget × LR × batch size sweep):

Edit ``configs/generate_configs.py`` to customize `WANDB_PROJECT`, `model_names`, `token_counts`, `lr_grid`, or `bsz_grid` in `configs/generate_configs.py`, then run:

```bash
python configs/generate_configs.py
```

Run training with (``ARCH`` in `qwen3_2M` / `qwen3_5M` / `qwen3_13M` / `qwen3_30M` / `qwen3_80M` / `qwen3_150M` / `qwen3_450M`
):

```bash
python open_optformer/training/pretrain.py <ARCH> --config configs/<GENERATED_CONFIG>.yaml
```

Convert to Hugging Face format:

```bash
python -m open_optformer.huggingface_conversion \
    --litgpt_checkpoint "$BASE_PATH/checkpoints/$VERSION/<run_name>/<step>" \
    --output_path "$BASE_PATH/hf_checkpoints/<run_name>"
```

---

## Evaluate a Trained Model

```bash
cd benchmarks/syne_tune_benchmarks
python benchmark_main.py --seed 5 --run_all_seeds 1 --n_workers 1 --method OPT_CQR --checkpoint_dir <PATH_TO_MODEL_CHECKPOINT> --benchmark fcnet-protein
```

---

## License

Apache-2.0 (see [`LICENSE`](LICENSE)).
