import logging
import json
import os
import random
import itertools

from pathlib import Path
from argparse import ArgumentParser
from syne_tune.util import catchtime
from open_optformer.history import OptformerConverter, Converter
from load_data import get_metadata, create_history_from_results


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="path where to find the results",
    )
    parser.add_argument(
        "--max_seed",
        type=int,
        required=False,
        default=30,
    )
    parser.add_argument(
        "--num_permutation",
        type=int,
        required=False,
        default=5,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="path to store the results",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="ratio of data used for training",
    )
    parser.add_argument(
        "--converter",
        type=str,
        required=True,
        choices=["plain", "optformer"],
        help="Converter formatting for numeric tokens",
    )

    methods = [
        "REA",
        "TPE",
        "BORE",
        "CQR",
        "RS",
        "HEBO"
    ]

    args, _ = parser.parse_known_args()

    print(args.__dict__)
    assert Path(args.path).exists()
    max_seed = args.max_seed
    max_num_trials = 100

    path = Path(args.path)
    output_path = Path(args.output_path)
    os.makedirs(output_path, exist_ok=True)
    experiment_filter = None

    validation_tasks = json.load(open('validation_tasks.json'))
    validation_tasks = list(itertools.chain.from_iterable(validation_tasks.values()))

    converter = OptformerConverter(q=args.quantization_levels) if args.converter == "optformer" else Converter(q=args.quantization_levels)

    with catchtime("load benchmark results"):

        with catchtime("Load metadata"):
            metadatas = get_metadata(root=path)

        # todo strict metadata filtering as the one above may fail
        methods = set(methods) if methods is not None else None
        metadatas = {
            k: v
            for k, v in metadatas.items()
            if (max_seed is None or v["seed"] < max_seed)
               and (methods is None or v["algorithm"] in methods)
        }
        if experiment_filter:
            metadatas = {k: v for k, v in metadatas.items() if experiment_filter(v)}
        print(f"loaded {len(metadatas)} experiment metadata")
        # metadatas = {k: v for k, v in metadatas.items() if "yahpo" not in v["benchmark"]}

        with catchtime("Load results dataframes"):
            # load results in parallel

            hist_train = []
            hist_valid = []
            for name, metadata in metadatas.items():
#                try:
                    benchmark_name = metadata['benchmark']
                    if benchmark_name in validation_tasks:
                        hist_valid.extend(create_history_from_results(name, metadata, path, max_num_trials, args.num_permutation, converter))
                    else:
                        hist_train.extend(create_history_from_results(name, metadata, path, max_num_trials, args.num_permutation, converter))
#                except Exception as e:
#                    print(f"Error processing {name}: {e}")
#                    continue
            #        hist = parfor(
            #            lambda name, metadata: create_history_from_results(name, metadata, path),
            #            inputs=list(metadatas.items()),
            #            engine=engine,
            #        )
            random.shuffle(hist_train)
            for split in ['train', 'valid']:
                file_name = f"{split}.txt"
                if split == 'train':
                    hist_split = hist_train
                else:
                    hist_split = hist_valid
                with open(str(output_path / file_name), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(hist_split))