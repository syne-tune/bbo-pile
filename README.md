26325846
# Open Optformer

## Installation

Install the required packages by running the following command from the main directory:

    pip install -r requirements.txt

Furthermore, you need to install the following packages, if you want to install the original optformer code:

    pip install -r requirements.txt
    pip install git+https://github.com/google-research/t5x.git@a9b8f1563eac10aa18f4fe384959733a6ae7e4ea --no-deps
    pip install git+https://github.com/google-research/optformer.git@12e2639954b0cd9bf824aab2d040650e6b32089c tensorflow-cpu==2.15.1

## Syne Tune Benchmarks

To run the benchmark locally, first go to the syne_tune_benchmarks folder

    cd benchmarks/syne-tune-benchmarks

and run the following command:

    python benchmark_main.py --seed 5 --run_all_seeds 1 --n_workers 1 --method RS --benchmark fcnet-protein

This will run random search with 5 seeds on the fcnet-protein benchmark.

Alternatively, you can run the following command to submit the job to a cluster, which will run all benchmarks in the fcnet family:

    python launch_slurmpilot.py --num_seeds 5 --n_workers 1 --benchmark_family fcnet --partition <your_partition> --cluster <your_cluster>

Available benchmark families are: 
- fcnet 
- nas201
- lcbench
- pd1
- hpob
- tabrepo

## Model Training

Please make sure to set the environment variable `BASE_PATH` to the path where you want to store the data and models, e.g.:

    export BASE_PATH=~/experiments/syne-tune-benchmarks

And the input path to the results of the benchmarks, e.g.:

    export RESULTS_PATH=~/syne-tune/results/

### Data Processing

First, set $CONVERTER environment variable to either "plain" or "optformer" (Chen et al, NeurIPS 2022). 

From the root of the repository, run the following commands to process the data:

First to compile the results into a dataset

    python benchmarks/syne_tune_benchmarks/generate_training_data/compile_data.py --path $RESULTS_PATH --output_path $BASE_PATH/data/raw --converter $CONVERTER

Now we can train the tokenizer

    python train_tokenizer.py --input_folder $BASE_PATH/data/raw/ --output_path $BASE_PATH/tokenizer --vocab_size 1074 --converter $CONVERTER

And pre-process the dataset to a litdata format, which is required for training the model

    python benchmarks/syne_tune_benchmarks/generate_training_data/preprocess_data.py --input_path $BASE_PATH/data/raw --output_path $BASE_PATH/data/tokenized_data --tokenizer_dir $BASE_PATH/tokenizer

### Pre-training

First, update BASE_PATH_CLUSTER, DATASET_NAME, and WANDB_PROJECT from ``benchmarks/syne_tune_benchmarks/configs/generate_configs.py`` based on your setup. Then, run the script to generate configuration files.

At the end we can start the model training:

    python open_optformer/training/pretrain.py pythia410M --config benchmarks/syne_tune_benchmarks/configs/NAME_OF_YOUR_CONFIG.yaml
