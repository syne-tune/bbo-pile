
# BBO-Pile: A Dataset for Pre-training Open Foundation Models for Black-box Optimization

## Installation

Install the required packages by running the following command from the main directory:

    pip install -r requirements.txt

Furthermore, you need to install the following packages, if you want to install the original optformer code:

    pip install -r requirements.txt
    pip install git+https://github.com/google-research/t5x.git@a9b8f1563eac10aa18f4fe384959733a6ae7e4ea --no-deps
    pip install git+https://github.com/google-research/optformer.git@12e2639954b0cd9bf824aab2d040650e6b32089c tensorflow-cpu==2.15.1

## Syne Tune Benchmarks

To run the benchmark locally, first go to the syne_tune_benchmarks folder

    cd benchmarks/syne_tune_benchmarks

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

If you want to (re-)generate a new version of the data, specify:

    export VERSION=v0.5

Then, switch to the directory:

    cd generate_training_data

Compile the results of all benchmark families directly into a combined dataset (the script recursively discovers all benchmarks, splits the validation sets, and automatically shuffles the training data):

    mkdir -p $BASE_PATH/dataset/$VERSION/all/
    python compile_data.py \
      --path $RESULTS_PATH \
      --output_path $BASE_PATH/dataset/$VERSION/all/ \
      --remove_names \
      --max_seed 30 \
      --num_permutation 5 \
      --sample_shorter_trajectories

The `compile_data.py` script supports several arguments:
- `--path`: The input path where the benchmark results are stored.
- `--output_path`: The directory where the compiled dataset will be saved.
- `--remove_names`: Flag to remove names of benchmarks and hyperparameters, avoiding overfitting.
- `--max_seed`: Maximum number of seeds to include (default: 30).
- `--num_permutation`: Number of times the order of variables in the trajectories is permuted to augment the data (default: 5).
- `--sample_shorter_trajectories`: Flag to additionally include shorter trajectories (first 1, 5, 10, and 20 trials) in the training data.

Now we can train the tokenizer

    python train_tokenizer.py --input_folder $BASE_PATH/dataset/$VERSION/all --output_path $BASE_PATH/tokenizer/$VERSION --vocab_size 1069

And pre-process the dataset to a litdata format, which is required for training the model. This is memory intensive, and you might want to run this on a SLURM cluster.

    python preprocess_data.py \
      --input_path $BASE_PATH/dataset/$VERSION/all \
      --output_path $BASE_PATH/dataset/$VERSION/tokenized_data \
      --tokenizer_dir $BASE_PATH/tokenizer/$VERSION

### Pre-training

First, update BASE_PATH_CLUSTER, DATASET_NAME, and WANDB_PROJECT from ``configs/generate_configs.py`` based on your setup. Then, run the script to generate configuration files:

    cd configs
    python generate_configs.py

At the end we can start the model training:

    python open_optformer/training/pretrain.py pythia410M --config configs/NAME_OF_YOUR_CONFIG.yaml


# Dataset History

## v0.1: 

Initial version of the datasets with 30 seeds of all methods on each dataset. We used the following format:

        benchmark:fcnet-slice
        algorithm:RS
        search-space:
        {name:hp_dropout_2,type:UNI,min_value:0.0,max_value:0.6,linear_scale}
        {name:hp_n_units_2,type:INT,min_value:16,max_value:512,log_scale}
        {name:hp_dropout_1,type:UNI,min_value:0.0,max_value:0.6,linear_scale}
        {name:hp_n_units_1,type:INT,min_value:16,max_value:512,log_scale}
        {name:hp_batch_size,type:INT,min_value:8,max_value:64,log_scale}
        {name:hp_init_lr,type:CAT,categories:[0.0005,0.001,0.005,0.01,0.05,0.1]}
        {name:hp_activation_fn_1,type:CAT,categories:['tanh','relu']}
        {name:hp_activation_fn_2,type:CAT,categories:['tanh','relu']}
        {name:hp_lr_schedule,type:CAT,categories:['cosine','const']}
        history
        500,0,0,400,667,<3>,<1>,<0>,<1>*7|500,1000,500,0,333,<1>,<0>,<0>,<0>*2|


## v0.2: 

We changed the format to a single line per trajectories, i.e remove all '\n' except the last one. That allowed us to reshuffles rows to break any order in the dataset

## v0.3: 

We removed hyperparameter and benchmark names to avoid overfitting of the model. Example trajectories looked like this:

    algorithm:BORE,search-space:{type:INT,min_value:8,max_value:64,log_scale}{type:UNI,min_value:0.0,max_value:0.6,linear_scale}{type:UNI,min_value:0.0,max_value:0.6,linear_scale}{type:INT,min_value:16,max_value:512,log_scale}{type:INT,min_value:16,max_value:512,log_scale}{type:CAT,categories:['tanh','relu']}{type:CAT,categories:['tanh','relu']}{type:CAT,categories:[0.0005,0.001,0.005,0.01,0.05,0.1]}{type:CAT,categories:['cosine','const']},history:0,500,500,400,0,<1>,<1>,<4>,<1>*0|0,500,0,1000,800,<1>,<0>,<5>,<0>*0|

## v0.4: 

We found a bug in our data collection script that didn't encode whether a blackbox problem is maximized or minimized. We now map every problem to a minimization problem.
Also, we removed names of categorical variables to avoid additional overfitting:

    algorithm:REA,search-space:{type:INT,min_value:8,max_value:64,log_scale}{type:INT,min_value:16,max_value:512,log_scale}{type:UNI,min_value:0.0,max_value:0.6,linear_scale}{type:UNI,min_value:0.0,max_value:0.6,linear_scale}{type:INT,min_value:16,max_value:512,log_scale}{type:CAT,categories:[0,1]}{type:CAT,categories:[0,1]}{type:CAT,categories:[0,1]}{type:CAT,categories:[0,1,2,3,4,5]},history:0,0,0,500,0,<1>,<1>,<1>,<3>*159|1000,800,500,0,1000,<1>,<0>,<0>,<1>*67|

## v0.5: 

There was a bug in the quantization: values were mapped to [0, 1000], but our tokenizer only encodes integers from 0 to 999. Parameters with a value of 1000 were therefore mapped to two tokens, which interfered with the sampling process.

## v0.6: 

We add a data augmentation step and ran all optimizers on different sub-spaces of the original FCNET and NAS201 search space, but masking out one or two hyperparameters.

## v0.7:

We changed the permutation numbers of each dataset to better balance the relative distribution of the different benchmark families.

## v0.8:

Adjust the number of permutations for each family, sample shorter sequences with T_max in [1, 5, 10, 20] trials to account for a different distributions during the optimization process.