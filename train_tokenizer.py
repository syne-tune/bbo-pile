import argparse
import os
from pathlib import Path

import sentencepiece as spm
from open_optformer.history import Converter, OptformerConverter


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input_folder",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        type=str,
    )
    parser.add_argument(
        "--model_prefix",
        default='tokenizer',
        type=str,
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=1040
    )
    parser.add_argument(
        "--max_sentence_length",
        type=int,
        default=7000000,
    )
    parser.add_argument(
        "--quantization_levels",
        type=int,
        default=1000,
        help="Number of quantization levels (q parameter)",
    )
    parser.add_argument(
        "--converter",
        type=str,
        required=True,
        choices=["plain", "optformer"],
        help="Converter formatting for numeric tokens",
    )
    args, _ = parser.parse_known_args()

    # Instantiate converter for user-defined symbols
    if args.converter == "optformer":
        converter = OptformerConverter(q=args.quantization_levels)
    else:
        converter = Converter(q=args.quantization_levels)
    user_defined_symbols = converter.get_user_defined_symbols()

    input_folder = Path(args.input_folder)
    os.makedirs(args.output_path, exist_ok=True)
    spm.SentencePieceTrainer.Train(
        input=f'{input_folder / "train.txt"},{input_folder / "valid.txt"}',
        vocab_size=args.vocab_size,
        model_prefix=str(Path(args.output_path) / args.model_prefix),
        character_coverage=1.0,
        max_sentence_length=args.max_sentence_length,
        user_defined_symbols=user_defined_symbols,
    )
