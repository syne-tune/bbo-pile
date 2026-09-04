"""
This is a standalone example that just samples a string using vLLM with regex-guided decoding.

Usage:
    python sample_vllm.py --model_path /path/to/model
"""

import argparse
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.config.structured_outputs import StructuredOutputsConfig
from transformers import AutoTokenizer
from syne_tune.config_space import Float, Integer, Categorical


class ConfigGrammar:
    """
    Generates a regex pattern to constrain LLM output to valid configurations.

    The output format is: {cont_values},{cat_values}*{output}|

    Example with 2 continuous and 1 categorical hyperparameter:
        "500,400,<0>*123|"
    """

    def __init__(
        self,
        tokenizer,
        config_space,
        n_continuous: int,
        n_categorical: int,
        hp_cat_names: list[str],
        num_numeric_tokens: int = 1000,
        num_categorical_tokens: int = 15,
    ):
        self.tokenizer = tokenizer
        self.n_continuous = n_continuous
        self.n_categorical = n_categorical
        self.hp_cat_names = hp_cat_names
        self.config_space = config_space
        self.num_numeric_tokens = num_numeric_tokens
        self.num_categorical_tokens = num_categorical_tokens

    def _get_continuous_tokens(self) -> list[str]:
        return [
            self.tokenizer.convert_ids_to_tokens(i)
            for i in range(self.num_numeric_tokens)
        ]

    def _get_categorical_tokens(self) -> list[str]:
        return [f'<{i}>' for i in range(self.num_categorical_tokens)]

    def _get_separator_tokens(self) -> dict[str, str]:
        token_to_id = self.tokenizer.convert_tokens_to_ids
        return {
            'comma': self.tokenizer.convert_ids_to_tokens(token_to_id(',')),
            'star': self.tokenizer.convert_ids_to_tokens(token_to_id('*')),
            'pipe': self.tokenizer.convert_ids_to_tokens(token_to_id('|')),
        }

    def _escape_regex(self, s: str) -> str:
        import re
        return re.escape(s)

    def _build_continuous_pattern(self) -> str:
        tokens = self._get_continuous_tokens()
        escaped = [self._escape_regex(t) for t in tokens]
        return '(' + '|'.join(escaped) + ')'

    def _build_categorical_pattern(self, max_categories: int = None) -> str:
        if max_categories is None:
            tokens = self._get_categorical_tokens()
        else:
            tokens = [f'<{i}>' for i in range(max_categories)]
        escaped = [self._escape_regex(t) for t in tokens]
        return '(' + '|'.join(escaped) + ')'

    def build_regex(self) -> str:
        cont_pattern = self._build_continuous_pattern()
        separators = self._get_separator_tokens()

        comma = self._escape_regex(separators['comma'])
        star = self._escape_regex(separators['star'])
        pipe = self._escape_regex(separators['pipe'])

        patterns = []

        # Continuous hyperparameters first
        for _ in range(self.n_continuous):
            patterns.append(cont_pattern)

        # Categorical hyperparameters after
        for hp_cat in self.hp_cat_names:
            n_categories = len(self.config_space[hp_cat].categories)
            patterns.append(self._build_categorical_pattern(n_categories))

        if patterns:
            hp_pattern = comma.join(patterns)
            regex = hp_pattern + star + cont_pattern + pipe
        else:
            regex = star + cont_pattern + pipe

        return regex


def main():
    parser = argparse.ArgumentParser(description="Sample from a model using vLLM with regex-guided decoding")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    args = parser.parse_args()

    # Context string as specified
    context_string = "benchmark:test,algorithm:test,search-space:{name:x,type:UNI,min_value:0,max_value:1,linear_scale}{name:y,type:INT,min_value:0,max_value:10,linear_scale}{name:z,type:CAT,categories:[0,1,2]},history:500,500,<0>*0|600,600,<1>*1000|"

    # Define config space matching the context string:
    # x: continuous (UNI), 0-1
    # y: continuous (INT), 0-10
    # z: categorical, categories [0, 1, 2]
    config_space = {
        "x": Float(0, 1),
        "y": Integer(0, 10),
        "z": Categorical([0, 1, 2]),
    }

    # Separate continuous and categorical HPs
    hp_cont_names = ["x", "y"]  # Float and Integer are continuous
    hp_cat_names = ["z"]        # Categorical

    print(f"Loading model from: {args.model_path}")

    # Load model and tokenizer
    model = LLM(model=args.model_path, structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Build the regex grammar
    grammar = ConfigGrammar(
        tokenizer=tokenizer,
        config_space=config_space,
        n_continuous=len(hp_cont_names),
        n_categorical=len(hp_cat_names),
        hp_cat_names=hp_cat_names,
        num_numeric_tokens=1000,
        num_categorical_tokens=15,
    )
    regex_pattern = grammar.build_regex()

    print(f"\nContext string:\n{context_string}")
    print(f"\nConfig space:")
    print(f"  - x: Float(0, 1) [continuous]")
    print(f"  - y: Integer(0, 10) [continuous]")
    print(f"  - z: Categorical([0, 1, 2]) [categorical]")
    print(f"\nRegex pattern (truncated): {regex_pattern[:200]}...")

    # Calculate max tokens: 2 tokens per HP (value + comma) + 1 for star + 1 for output + 1 for pipe
    max_new_tokens = (len(hp_cont_names) + len(hp_cat_names)) * 2 + 1

    # Set up sampling parameters with guided decoding
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        n=args.num_samples,
        structured_outputs=StructuredOutputsParams(regex=regex_pattern),
    )

    print(f"\nGenerating {args.num_samples} samples...")

    # Generate samples
    outputs = model.generate([context_string], sampling_params)

    print("\n" + "=" * 60)
    print("COMPLETIONS:")
    print("=" * 60)

    for i, output in enumerate(outputs[0].outputs):
        completion_text = output.text
        token_ids = list(output.token_ids)
        print(f"\nSample {i + 1}:")
        print(f"  Completion: {completion_text}")
        print(f"  Token IDs:  {token_ids}")

        # Decode the completion
        try:
            # Parse the completion format: "val1,val2,<cat>*output|"
            # Remove trailing pipe if present
            clean = completion_text.rstrip('|')
            hp_part, output_part = clean.split('*')
            hp_values = hp_part.split(',')

            print(f"  Parsed:")
            print(f"    - x (continuous):    {hp_values[0]}")
            print(f"    - y (continuous):    {hp_values[1]}")
            print(f"    - z (categorical):   {hp_values[2]}")
            print(f"    - predicted output:  {output_part}")
        except Exception as e:
            print(f"  Parse error: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
