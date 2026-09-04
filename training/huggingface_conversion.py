"""
Convert LitGPT Qwen3 checkpoint to HuggingFace format.
"""
import sys
import torch
import yaml
from pathlib import Path
from typing import Union
import os
import json


def load_litgpt(path: Union[str, Path]):
    """Load a LitGPT model from checkpoint directory."""
    from litgpt.config import Config
    from litgpt.model import GPT

    path = Path(path)
    config = Config.from_file(str(path / 'model_config.yaml'))
    model = GPT(config)

    state_dict = torch.load(
        str(path / 'lit_model.pth'),
        weights_only=True,
        map_location='cpu'
    )
    if 'model' in state_dict:
        state_dict = state_dict['model']

    model.load_state_dict(state_dict)
    model.eval()
    return model


def convert_to_huggingface(path: Union[str, Path], output_dir: Union[str, Path] = "qwen3-hf"):
    """Convert LitGPT checkpoint to HuggingFace Qwen3 model and save it."""
    from transformers import Qwen3ForCausalLM, Qwen3Config

    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = Path(path)

    # Load LitGPT config
    with open(path / 'model_config.yaml', 'r') as f:
        litgpt_config = yaml.safe_load(f)

    # Load LitGPT weights
    state_dict = torch.load(
        str(path / 'lit_model.pth'),
        weights_only=True,
        map_location='cpu'
    )
    if 'model' in state_dict:
        state_dict = state_dict['model']

    # Create HuggingFace config
    hf_config = Qwen3Config(
        vocab_size=litgpt_config['vocab_size'],
        hidden_size=litgpt_config['n_embd'],
        intermediate_size=litgpt_config['intermediate_size'],
        num_hidden_layers=litgpt_config['n_layer'],
        num_attention_heads=litgpt_config['n_head'],
        num_key_value_heads=litgpt_config['n_query_groups'],
        head_dim=litgpt_config['head_size'],
        max_position_embeddings=litgpt_config['block_size'],
        rms_norm_eps=litgpt_config['norm_eps'],
        rope_theta=litgpt_config['rope_base'],
        tie_word_embeddings=True,  # From hyperparameters.yaml
        attention_bias=litgpt_config['attn_bias'],
    )

    # Convert weights
    hf_state_dict = {}

    # Embeddings
    hf_state_dict['model.embed_tokens.weight'] = state_dict['transformer.wte.weight']

    # Final layer norm
    hf_state_dict['model.norm.weight'] = state_dict['transformer.ln_f.weight']

    # LM head (tied with embeddings, but we include it anyway)
    hf_state_dict['lm_head.weight'] = state_dict['lm_head.weight']

    # Per-layer weights
    n_layer = litgpt_config['n_layer']
    n_head = litgpt_config['n_head']
    n_query_groups = litgpt_config['n_query_groups']
    head_size = litgpt_config['head_size']

    for i in range(n_layer):
        # Layer norms
        hf_state_dict[f'model.layers.{i}.input_layernorm.weight'] = state_dict[f'transformer.h.{i}.norm_1.weight']
        hf_state_dict[f'model.layers.{i}.post_attention_layernorm.weight'] = state_dict[f'transformer.h.{i}.norm_2.weight']

        # QKV split
        # LitGPT fuses QKV with shape [q_size + k_size + v_size, hidden_size]
        # where q_size = n_head * head_size, k_size = v_size = n_query_groups * head_size
        # LitGPT uses CONTIGUOUS [Q, K, V] layout (not interleaved)
        qkv_weight = state_dict[f'transformer.h.{i}.attn.qkv.weight']

        q_size = n_head * head_size
        k_size = n_query_groups * head_size
        v_size = n_query_groups * head_size

        # Simple contiguous split
        q_weight = qkv_weight[:q_size, :]
        k_weight = qkv_weight[q_size:q_size + k_size, :]
        v_weight = qkv_weight[q_size + k_size:, :]

        hf_state_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = q_weight
        hf_state_dict[f'model.layers.{i}.self_attn.k_proj.weight'] = k_weight
        hf_state_dict[f'model.layers.{i}.self_attn.v_proj.weight'] = v_weight

        # Output projection
        hf_state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = state_dict[f'transformer.h.{i}.attn.proj.weight']

        # QK norms (Qwen3 has these)
        hf_state_dict[f'model.layers.{i}.self_attn.q_norm.weight'] = state_dict[f'transformer.h.{i}.attn.norm_q.weight']
        hf_state_dict[f'model.layers.{i}.self_attn.k_norm.weight'] = state_dict[f'transformer.h.{i}.attn.norm_k.weight']

        # MLP
        # LitGPT: fc_1 is gate, fc_2 is up, proj is down
        hf_state_dict[f'model.layers.{i}.mlp.gate_proj.weight'] = state_dict[f'transformer.h.{i}.mlp.fc_1.weight']
        hf_state_dict[f'model.layers.{i}.mlp.up_proj.weight'] = state_dict[f'transformer.h.{i}.mlp.fc_2.weight']
        hf_state_dict[f'model.layers.{i}.mlp.down_proj.weight'] = state_dict[f'transformer.h.{i}.mlp.proj.weight']

    # Create model and load weights
    model = Qwen3ForCausalLM(hf_config)
    info = model.load_state_dict(hf_state_dict, strict=False)

    if info.missing_keys:
        print("WARNING: Missing keys:", info.missing_keys)
    if info.unexpected_keys:
        print("WARNING: Unexpected keys:", info.unexpected_keys)

    model.eval()

    # Save model + config
    print(f"Saving HuggingFace model to {output_dir}")
    model.save_pretrained(output_dir)

    # Convert the SentencePiece tokenizer to a HuggingFace PreTrainedTokenizerFast.
    # Build the tokenizer directly from the SentencePiece vocab using the
    # `tokenizers` library with Metaspace pre-tokenizer/decoder to handle
    # the ▁ prefix correctly (matching SentencePiece's add_dummy_prefix).
    import sentencepiece as spm
    from tokenizers import Tokenizer as HFTokenizer, pre_tokenizers, decoders
    from tokenizers.models import Unigram
    from transformers import PreTrainedTokenizerFast

    print(f"Loading SentencePiece model from: {path}")
    sp = spm.SentencePieceProcessor(model_file=str(path / "tokenizer.model"))

    vocab = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())]
    tokenizer_obj = HFTokenizer(Unigram(vocab, unk_id=sp.unk_id(), byte_fallback=False))
    tokenizer_obj.pre_tokenizer = pre_tokenizers.Metaspace(replacement='\u2581', prepend_scheme='always')
    tokenizer_obj.decoder = decoders.Metaspace(replacement='\u2581', prepend_scheme='always')

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        model_input_names=["input_ids", "attention_mask"],
    )
    hf_tokenizer.save_pretrained(output_dir)
    print(f"Converted tokenizer saved to: {output_dir}")

    print("Save complete.")

    return model



def do_inference_litgpt(context: list[int], litgpt_model) -> torch.Tensor:
    """Run inference with LitGPT model, returns logits for next token."""
    with torch.no_grad():
        input_tensor = torch.tensor([context], dtype=torch.long)
        logits = litgpt_model(input_tensor)
        # Return logits for the last position
        return logits[0, -1, :]


def do_inference_huggingface(context: list[int], hf_model) -> torch.Tensor:
    """Run inference with HuggingFace model, returns logits for next token."""
    with torch.no_grad():
        input_tensor = torch.tensor([context], dtype=torch.long)
        outputs = hf_model(input_tensor)
        # Return logits for the last position
        return outputs.logits[0, -1, :]


if __name__ == '__main__':
    import random
    import sys
    import torch
    import sentencepiece as spm
    import json
    from transformers import AutoTokenizer

    # Check for correct number of arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py <litgpt_checkpoint_path> <hf_output_path>")
        sys.exit(1)

    litgpt_path = sys.argv[1]
    hf_path = sys.argv[2]

    print(f"Loading LitGPT model from: {litgpt_path}")
    litgpt_model = load_litgpt(litgpt_path)

    print(f"Converting to HuggingFace at: {hf_path}")
    hf_model = convert_to_huggingface(litgpt_path, hf_path)

    # Generate random context
    vocab_size = litgpt_model.config.vocab_size
    context_length = 10
    random.seed(42)
    torch.manual_seed(42)
    random_context = [random.randint(0, vocab_size - 1) for _ in range(context_length)]
    print(f"Random context: {random_context}")

    print("Running LitGPT inference...")
    litgpt_logits = do_inference_litgpt(random_context, litgpt_model)

    print("Running HuggingFace inference...")
    hf_logits = do_inference_huggingface(random_context, hf_model)

    # Compare logits
    print(f"LitGPT logits shape: {litgpt_logits.shape}")
    print(f"HuggingFace logits shape: {hf_logits.shape}")
    print(f"LitGPT logits (first 10): {litgpt_logits[:10]}")
    print(f"HuggingFace logits (first 10): {hf_logits[:10]}")

    # Check if predictions match
    max_diff = torch.max(torch.abs(litgpt_logits - hf_logits)).item()
    print(f"Max absolute difference: {max_diff}")

    # Check argmax matches
    litgpt_pred = torch.argmax(litgpt_logits).item()
    hf_pred = torch.argmax(hf_logits).item()
    print(f"LitGPT prediction: {litgpt_pred}")
    print(f"HuggingFace prediction: {hf_pred}")

    # Assert close enough
    assert torch.allclose(litgpt_logits, hf_logits, atol=1e-4), \
        f"Logits don't match! Max diff: {max_diff}"

    print("\nSUCCESS: Model predictions match!")

    # --- TOKENIZER COMPARISON ---
    print("\n--- Verifying Tokenizer Parity ---")
    # Path to the original .model file within the checkpoint folder
    sp_model_path = f"{litgpt_path}/tokenizer.model"
    sp_processor = spm.SentencePieceProcessor(model_file=sp_model_path)

    # Load your newly converted Hugging Face Tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_path)

    # Test string with your categorical and special tokens
    test_text = "benchmark:test,algorithm:test,search-space:{name:x,type:UNI,min_value:0,max_value:1,linear_scale}{name:y,type:INT,min_value:0,max_value:10,linear_scale}{name:z,type:CAT,categories:[0,1,2]},history:500,500,<0>*0|599,599,<1>*999|"
    # Check ID 1035
    try:
        print(f"Token ID 1035 represents: '{sp_processor.decode([1035])}'")
    except:
        print("Token ID 1035 out of bounds for SentencePiece processor.")

    litgpt_ids = sp_processor.encode(test_text)
    hf_ids = hf_tokenizer.encode(test_text)

    print(f"LitGPT IDs: {litgpt_ids}")
    print(f"HF IDs:     {hf_ids}")

    if litgpt_ids == hf_ids:
        print("✅ Success! Token IDs match perfectly.")
    else:
        print("❌ Warning: ID mismatch detected.")

    # --- VOCAB SIZE CHECK ---
    print("\n--- Verifying Vocab Size ---")
    tokenizer_vocab_size = len(hf_tokenizer)

    # Load model config from the converted path
    hf_config_path = f"{hf_path}/config.json"
    with open(hf_config_path, "r") as f:
        config = json.load(f)
    model_vocab_size = config.get("vocab_size")

    print(f"Tokenizer Vocab Size: {tokenizer_vocab_size}")
    print(f"Model Config Vocab Size: {model_vocab_size}")

    if tokenizer_vocab_size == model_vocab_size:
        print("✅ Match! Vocab sizes are consistent.")
    elif tokenizer_vocab_size < model_vocab_size:
        print("⚠️ Warning: Model config vocab_size is larger than tokenizer.")
    else:
        print("❌ Error: Tokenizer vocab is larger than model config!")

    # --- TEXT GENERATION COMPARISON ---
    print("\n--- Comparing Text Generation ---")

    # Use the same test text as input prompt
    prompt = test_text
    max_new_tokens = 100
    temperature = 0.0  # Deterministic generation

    print(f"Input prompt: {prompt[:100]}...")
    print(f"Generating {max_new_tokens} tokens with temperature={temperature}")

    # Generate with LitGPT
    print("\nGenerating with LitGPT...")
    torch.manual_seed(42)
    litgpt_input_ids = torch.tensor([litgpt_ids], dtype=torch.long)
    litgpt_model.eval()

    with torch.no_grad():
        litgpt_generated_ids = litgpt_input_ids.clone()
        for _ in range(max_new_tokens):
            logits = litgpt_model(litgpt_generated_ids)
            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            litgpt_generated_ids = torch.cat([litgpt_generated_ids, next_token], dim=1)

    litgpt_output_ids = litgpt_generated_ids[0].tolist()
    litgpt_generated_text = sp_processor.decode(litgpt_output_ids)

    # Generate with HuggingFace
    print("Generating with HuggingFace...")
    torch.manual_seed(42)
    hf_input_ids = torch.tensor([hf_ids], dtype=torch.long)
    hf_model.eval()

    with torch.no_grad():
        hf_generated_ids = hf_input_ids.clone()
        for _ in range(max_new_tokens):
            outputs = hf_model(hf_generated_ids)
            next_token_logits = outputs.logits[:, -1, :]

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            hf_generated_ids = torch.cat([hf_generated_ids, next_token], dim=1)

    hf_output_ids = hf_generated_ids[0].tolist()
    hf_generated_text = hf_tokenizer.decode(hf_output_ids, skip_special_tokens=False)

    # Compare outputs
    print("\n--- Generation Results ---")
    print(f"\nLitGPT generated IDs: {litgpt_output_ids[len(litgpt_ids):]}")
    print(f"HF generated IDs:     {hf_output_ids[len(hf_ids):]}")

    print(f"\nLitGPT full output:\n{litgpt_generated_text}")
    print(f"\nHF full output:\n{hf_generated_text}")

    # Check if generated token IDs match
    litgpt_gen_only = litgpt_output_ids[len(litgpt_ids):]
    hf_gen_only = hf_output_ids[len(hf_ids):]

    if litgpt_gen_only == hf_gen_only:
        print("\n✅ SUCCESS: Generated token IDs match perfectly!")
        print("(Note: Display differences like ⁇ vs <unk> are just formatting, the actual tokens are identical)")
    else:
        print("\n❌ WARNING: Generated token IDs differ!")
        first_diff = next((i for i, (a, b) in enumerate(zip(litgpt_gen_only, hf_gen_only)) if a != b), None)
        if first_diff is not None:
            print(f"First difference at position {first_diff}:")
            print(f"  LitGPT: {litgpt_gen_only[first_diff]}")
            print(f"  HF: {hf_gen_only[first_diff]}")
        else:
            print(f"Sequences are different lengths: LitGPT={len(litgpt_gen_only)}, HF={len(hf_gen_only)}")

    # Also compare the generated text with normalized unknown tokens
    litgpt_normalized = litgpt_generated_text.replace('⁇', '<unk>')
    if litgpt_normalized == hf_generated_text:
        print("✅ Generated text also matches (after normalizing <unk> tokens)")
    # --- TOKENIZER MAPPING VERIFICATION ---
    print("\n--- Verifying Tokenizer Mapping ---")

    # Special tokens that are expected to differ in representation
    SPECIAL_TOKEN_IDS = {0, 1, 2}  # <unk>, <s>, </s>

    print("Checking if token ID -> string mapping is identical...")

    mismatches = []
    special_token_mismatches = []
    sample_size = min(1000, vocab_size)

    for token_id in range(sample_size):
        try:
            sp_decoded = sp_processor.decode([token_id])
            hf_decoded = hf_tokenizer.decode([token_id], skip_special_tokens=False)

            # Normalize the unknown token representations
            sp_normalized = sp_decoded.replace('⁇', '<unk>')

            if sp_normalized != hf_decoded:
                mismatch_info = {
                    'id': token_id,
                    'sp': sp_decoded,
                    'hf': hf_decoded,
                    'sp_normalized': sp_normalized
                }

                if token_id in SPECIAL_TOKEN_IDS:
                    special_token_mismatches.append(mismatch_info)
                else:
                    mismatches.append(mismatch_info)
        except Exception as e:
            mismatches.append({
                'id': token_id,
                'error': str(e)
            })

    # Report special token differences (expected)
    if special_token_mismatches:
        print(f"ℹ️  Found {len(special_token_mismatches)} special token representation differences (expected):")
        for mismatch in special_token_mismatches:
            print(f"  Token {mismatch['id']}: SP='{mismatch['sp']}' vs HF='{mismatch['hf']}'")

    # Report actual content mismatches (unexpected)
    if not mismatches:
        print(f"✅ SUCCESS: All {sample_size} non-special token mappings match perfectly!")
    else:
        print(f"❌ WARNING: Found {len(mismatches)} mismatches in first {sample_size} tokens:")
        for mismatch in mismatches[:10]:
            if 'error' in mismatch:
                print(f"  Token {mismatch['id']}: ERROR - {mismatch['error']}")
            else:
                print(f"  Token {mismatch['id']}:")
                print(f"    SentencePiece: '{mismatch['sp']}'")
                print(f"    HuggingFace:   '{mismatch['hf']}'")