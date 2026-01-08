import time
import torch
import yaml
import argparse
import tempfile
import os
from pathlib import Path
from litgpt import GPT
from litgpt.config import Config
from litgpt.generate.base import generate
from transformers import AutoConfig, AutoModelForCausalLM
from llama_cpp import Llama
import numpy as np
import gguf

# Optional vLLM import (only available on GPU machines)
# To use vLLM on a GPU machine:
#   1. Install: pip install vllm
#   2. Ensure CUDA is available (check with: nvidia-smi)
#   3. Run this script - vLLM will be automatically used
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

context_length = 500
number_output_tokens = 20

# Detect available device
def get_device():
    """Detect and return the best available device for PyTorch models."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS has compatibility issues with some operations (e.g., LitGPT's mask cache)
        # Use CPU for PyTorch models, but LlamaCPP can still use Metal
        device = "cpu"
        print("Apple Metal GPU (MPS) detected but using CPU for PyTorch models")
        print("(LlamaCPP will use Metal GPU acceleration)")
    else:
        device = "cpu"
        print("Using CPU")
    return device

device = get_device()
mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def load_config():
    """Load model config from YAML file."""
    with open("model_config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


def create_random_litgpt_model(config):
    """Create a LitGPT model with random weights."""
    model = GPT(config)
    model.eval()
    return model


def generate_litgpt(litgpt_model, context: list[int], number_samples: int):
    """Run inference with LitGPT and return runtime.

    Note: LitGPT's generate function doesn't support batch generation,
    so we need to loop through samples individually.
    """
    # Set up KV cache for generation
    litgpt_model.set_kv_cache(batch_size=1, device=device)

    litgpt_model.to(device)

    input_ids = torch.tensor(context, dtype=torch.long, device=device)

    times = []
    start_time_total = time.perf_counter()

    for i in range(number_samples):
        start_time = time.perf_counter()

        output = generate(
            litgpt_model,
            input_ids,
            max_returned_tokens=len(context) + number_output_tokens,
            temperature=0.0,  # Greedy decoding
            eos_id=None  # Don't stop early
        )

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        times.append(elapsed)

    end_time_total = time.perf_counter()
    total_time = end_time_total - start_time_total
    avg_time = total_time / number_samples

    # Clear KV cache
    litgpt_model.clear_kv_cache()

    print(f"\nLitGPT Results ({device.upper()}):")
    print(f"  Generated {number_samples} samples in {total_time:.4f}s")
    print(f"  Average time per sample: {avg_time:.4f}s")
    print(f"  Tokens/second (per sample): {number_output_tokens / avg_time:.2f}")

    return times


def create_random_huggingface_model(config):
    """Create a HuggingFace model with random weights based on LitGPT config."""
    hf_config = AutoConfig.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        vocab_size=config.vocab_size,
        hidden_size=config.n_embd,
        num_hidden_layers=config.n_layer,
        num_attention_heads=config.n_head,
        num_key_value_heads=config.n_query_groups,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.block_size,
        rms_norm_eps=config.norm_eps,
        rope_theta=config.rope_base,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_config(hf_config)
    model.eval()
    return model


def generate_huggingface(huggingface_model, context: list[int], number_samples: int):
    """Run inference with HuggingFace using batched generation."""
    huggingface_model.to(device)

    # Batch the inputs: replicate context for all samples
    #input_ids = torch.tensor([context] * number_samples, dtype=torch.long, device=device)
    input_ids = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    start_time = time.perf_counter()

    with torch.no_grad():
        output = huggingface_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=number_output_tokens,
            do_sample=True,  # Greedy decoding
            pad_token_id=huggingface_model.config.eos_token_id,
            num_return_sequences=number_samples,
        )

    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time = total_time / number_samples

    print(f"\nHuggingFace Results ({device.upper()}):")
    print(f"  Generated {number_samples} samples in {total_time:.4f}s (batched)")
    print(f"  Average time per sample: {avg_time:.4f}s")
    print(f"  Tokens/second (per sample): {number_output_tokens / avg_time:.2f}")

    return [avg_time] * number_samples


def convert_hf_to_gguf(huggingface_model, config):
    """
    Create a GGUF format file for llama.cpp with random weights.
    """
    gguf_path = "model_random.gguf"

    print(f"Creating GGUF file with random weights: {gguf_path}")

    # Use "llama" architecture for compatibility
    # Qwen models are based on LLaMA architecture, and using "llama" ensures
    # the tensor names and metadata are compatible with llama.cpp
    # For production use with real Qwen weights, you'd use the specific qwen/qwen2/qwen3 arch
    arch = "llama"
    print(f"Using GGUF architecture: {arch} (compatible with Qwen-based models)")

    # Create GGUF writer
    writer = gguf.GGUFWriter(gguf_path, arch)

    # Add architecture metadata
    writer.add_name(config.name)
    writer.add_block_count(config.n_layer)
    writer.add_context_length(config.block_size)
    writer.add_embedding_length(config.n_embd)
    writer.add_head_count(config.n_head)
    writer.add_head_count_kv(config.n_query_groups)
    writer.add_layer_norm_rms_eps(config.norm_eps)

    rope_dim = int(config.n_embd // config.n_head * config.rotary_percentage)
    writer.add_rope_dimension_count(rope_dim)
    writer.add_rope_freq_base(config.rope_base)
    writer.add_file_type(gguf.GGMLQuantizationType.F32)
    writer.add_feed_forward_length(config.intermediate_size)

    # Add tokenizer metadata (use llama tokenizer which doesn't require merges)
    writer.add_tokenizer_model("llama")
    writer.add_vocab_size(config.vocab_size)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_pad_token_id(0)

    # Add minimal vocabulary
    tokens = []
    scores = []
    token_types = []

    for i in range(config.vocab_size):
        tokens.append(f"<token_{i}>".encode('utf-8'))
        scores.append(-float(i))  # Llama tokenizer uses negative scores
        token_types.append(gguf.TokenType.NORMAL)

    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)

    # Add tensors with random weights
    np.random.seed(42)

    # Token embeddings
    writer.add_tensor("token_embd.weight",
                     np.random.randn(config.vocab_size, config.n_embd).astype(np.float32))

    # Add transformer layers
    kv_dim = config.n_embd // config.n_head * config.n_query_groups

    for i in range(config.n_layer):
        # Attention weights (llama.cpp expects transposed)
        writer.add_tensor(f"blk.{i}.attn_q.weight",
                         np.random.randn(config.n_embd, config.n_embd).astype(np.float32))
        writer.add_tensor(f"blk.{i}.attn_k.weight",
                         np.random.randn(kv_dim, config.n_embd).astype(np.float32))
        writer.add_tensor(f"blk.{i}.attn_v.weight",
                         np.random.randn(kv_dim, config.n_embd).astype(np.float32))
        writer.add_tensor(f"blk.{i}.attn_output.weight",
                         np.random.randn(config.n_embd, config.n_embd).astype(np.float32))

        # MLP weights (llama.cpp expects transposed)
        writer.add_tensor(f"blk.{i}.ffn_gate.weight",
                         np.random.randn(config.intermediate_size, config.n_embd).astype(np.float32))
        writer.add_tensor(f"blk.{i}.ffn_up.weight",
                         np.random.randn(config.intermediate_size, config.n_embd).astype(np.float32))
        writer.add_tensor(f"blk.{i}.ffn_down.weight",
                         np.random.randn(config.n_embd, config.intermediate_size).astype(np.float32))

        # Layer norms
        writer.add_tensor(f"blk.{i}.attn_norm.weight",
                         np.ones(config.n_embd, dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight",
                         np.ones(config.n_embd, dtype=np.float32))

    # Output layer
    writer.add_tensor("output_norm.weight",
                     np.ones(config.n_embd, dtype=np.float32))
    writer.add_tensor("output.weight",
                     np.random.randn(config.vocab_size, config.n_embd).astype(np.float32))

    # Write the file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"GGUF file created successfully")
    print(f"Note: llama.cpp may have strict compatibility requirements.")
    print(f"If loading fails, the file structure is correct but may need adjustments for llama.cpp.")

    return gguf_path, None


def create_llama_cpp_model(gguf_path, config):
    """Load a GGUF model with llama.cpp."""
    if gguf_path is None:
        return None

    # Determine GPU offloading
    # n_gpu_layers: number of layers to offload to GPU (-1 for all layers)
    if device == "cuda":
        n_gpu_layers = -1  # Offload all layers to CUDA GPU
        print("LlamaCPP: Using CUDA GPU acceleration")
    #elif mps_available:
    #    n_gpu_layers = 1  # Metal has limited support, use conservative setting
    #    print("LlamaCPP: Using Metal GPU acceleration")
    else:
        n_gpu_layers = 0  # CPU only
        print("LlamaCPP: Using CPU only")

    model = Llama(
        model_path=gguf_path,
        n_ctx=context_length + number_output_tokens,
        n_threads=os.cpu_count() if n_gpu_layers == 0 else None,
        n_gpu_layers=n_gpu_layers,
        verbose=False
    )
    print(f"LlamaCPP model loaded successfully")
    return model


def generate_llama_cpp(llama_cpp_model, context: list[int], number_samples: int):
    """Run inference with llama.cpp using the create_completion API.

    Note: LlamaCPP doesn't support batch generation like HuggingFace,
    so we loop through samples individually using the completion API.
    """
    if llama_cpp_model is None:
        print("Skipped: LlamaCPP requires a pre-converted GGUF file")
        print("The benchmark structure is ready - provide a .gguf file to enable this benchmark")
        return []

    times = []
    start_time_total = time.perf_counter()

    for i in range(number_samples):
        start_time = time.perf_counter()

        # Use the create_completion API for proper generation
        _ = llama_cpp_model.create_completion(
            prompt=context,  # Pass token IDs directly
            max_tokens=number_output_tokens,
            temperature=0.0,  # Greedy decoding
            echo=False  # Don't echo the prompt
        )

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        times.append(elapsed)

    end_time_total = time.perf_counter()
    total_time = end_time_total - start_time_total
    avg_time = total_time / number_samples

    #accel = "CUDA GPU" if device == "cuda" else "Metal GPU" if mps_available else "CPU"
    accel = "CUDA GPU" if device == "cuda" else "CPU"
    print(f"\nLlamaCPP Results ({accel}):")
    print(f"  Generated {number_samples} samples in {total_time:.4f}s")
    print(f"  Average time per sample: {avg_time:.4f}s")
    print(f"  Tokens/second (per sample): {number_output_tokens / avg_time:.2f}")

    return times


def create_vllm_model(huggingface_model, config):
    """
    Create a vLLM model from HuggingFace model.

    vLLM requires the model to be saved on disk, so we save the HF model
    to a temporary directory and load it with vLLM.
    """
    if not VLLM_AVAILABLE:
        print("vLLM is not available (requires GPU and CUDA)")
        return None, None

    # Save HuggingFace model to temp directory
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "model")

    print(f"Saving HuggingFace model for vLLM: {model_path}")
    huggingface_model.save_pretrained(model_path)

    # Create a minimal tokenizer config (vLLM needs this)
    tokenizer_config = {
        "vocab_size": config.vocab_size,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    import json
    with open(os.path.join(model_path, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)

    try:
        # Initialize vLLM model
        print("Loading model with vLLM...")
        vllm_model = LLM(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=context_length + number_output_tokens,
        )
        print("vLLM model loaded successfully")
        return vllm_model, temp_dir
    except Exception as e:
        print(f"Failed to load vLLM model: {e}")
        return None, temp_dir


def generate_vllm(vllm_model, context: list[int], number_samples: int):
    """
    Run inference with vLLM using batched generation.

    vLLM is optimized for high-throughput batch inference on GPUs.
    It processes all samples in a single batched call.
    """
    if vllm_model is None:
        print("Skipped: vLLM requires GPU and CUDA")
        print("To use vLLM:")
        print("  1. Install on a GPU machine: pip install vllm")
        print("  2. Ensure CUDA is available")
        return []

    # vLLM expects string prompts or token IDs
    # We'll use token IDs by converting to a prompt-like format
    # Create dummy prompts for each sample (vLLM will tokenize internally)
    prompts = [context] * number_samples

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding
        max_tokens=number_output_tokens,
        min_tokens=number_output_tokens,  # Force exactly N tokens
    )

    start_time = time.perf_counter()

    # vLLM processes all prompts in a single batched call
    outputs = vllm_model.generate(
        prompt_token_ids=prompts,
        sampling_params=sampling_params,
        use_tqdm=False
    )

    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time = total_time / number_samples

    print(f"\nvLLM Results:")
    print(f"  Generated {number_samples} samples in {total_time:.4f}s (batched on GPU)")
    print(f"  Average time per sample: {avg_time:.4f}s")
    print(f"  Tokens/second (per sample): {number_output_tokens / avg_time:.2f}")
    print(f"  Total throughput: {number_samples * number_output_tokens / total_time:.2f} tokens/second")

    return [avg_time] * number_samples


def generate_random_context(vocab_size, context_length):
    """Generate random token IDs for context."""
    return torch.randint(0, vocab_size, (context_length,)).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark inference speed across different frameworks")
    parser.add_argument("--samples", type=int, default=2, help="Number of samples to run (default: 2)")
    args = parser.parse_args()

    number_samples = args.samples

    print("Loading configuration...")
    config = load_config()
    print(f"Model config: {config.name}, vocab_size={config.vocab_size}, n_layer={config.n_layer}")
    print(f"Number of samples: {number_samples}")

    print("\nCreating random LitGPT model...")
    litgpt_model = create_random_litgpt_model(config)

    print("\nCreating random HuggingFace model...")
    huggingface_model = create_random_huggingface_model(config)

    print("\nConverting to GGUF for llama.cpp...")
    try:
        gguf_path, temp_dir = convert_hf_to_gguf(huggingface_model, config)
        if gguf_path is not None:
            llama_cpp_model = create_llama_cpp_model(gguf_path, config)
        else:
            llama_cpp_model = None
    except Exception as e:
        print(f"Warning: GGUF conversion or loading failed: {e}")
        print("LlamaCPP benchmarking will be skipped.")
        llama_cpp_model = None

    print("\nPreparing vLLM model...")
    try:
        vllm_model, vllm_temp_dir = create_vllm_model(huggingface_model, config)
    except Exception as e:
        print(f"Warning: vLLM model creation failed: {e}")
        print("vLLM benchmarking will be skipped.")
        vllm_model = None

    print("\nGenerating random context...")
    random_context = generate_random_context(config.vocab_size, context_length)
    print(f"Context length: {len(random_context)} tokens")

    print("\n" + "="*60)
    print("Benchmarking LitGPT")
    print("="*60)
    generate_litgpt(litgpt_model, random_context, number_samples)

    print("\n" + "="*60)
    print("Benchmarking HuggingFace")
    print("="*60)
    generate_huggingface(huggingface_model, random_context, number_samples)

    print("\n" + "="*60)
    print("Benchmarking LlamaCPP")
    print("="*60)
    generate_llama_cpp(llama_cpp_model, random_context, number_samples)

    print("\n" + "="*60)
    print("Benchmarking vLLM")
    print("="*60)
    generate_vllm(vllm_model, random_context, number_samples)
