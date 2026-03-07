#!/usr/bin/env python3
"""Convert HuggingFace models to NeuralForge llama2.c format.

Downloads a model from HuggingFace Hub, converts weights from safetensors
to llama2.c binary format, and converts the tokenizer to tokenizer.bin.

Supports Llama-architecture models including:
  - TinyLlama 1.1B
  - SmolLM 135M / 360M / 1.7B
  - Llama 2 / 3 (any size)
  - Any LlamaForCausalLM-compatible model

Handles grouped-query attention (GQA) by expanding KV heads to match query heads.

Usage:
    pip install safetensors transformers torch sentencepiece
    python3 convert_hf.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output models/
    python3 convert_hf.py --model HuggingFaceTB/SmolLM-135M --output models/
    python3 convert_hf.py --model meta-llama/Llama-2-7b --output models/ --token YOUR_TOKEN

Output:
    models/model.bin       — weights in llama2.c format (7-int header + float32 weights)
    models/tokenizer.bin   — tokenizer in llama2.c format
    models/model_card.json — model metadata for NeuralForge app display
"""

import argparse
import json
import os
import struct
import sys
import time

def check_dependencies():
    """Check that required Python packages are installed."""
    missing = []
    try:
        import safetensors
    except ImportError:
        missing.append("safetensors")
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    if missing:
        print(f"Error: Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def load_model_config(model_id, token=None):
    """Load model configuration from HuggingFace."""
    from transformers import AutoConfig
    print(f"Loading config for {model_id}...")
    config = AutoConfig.from_pretrained(model_id, token=token, trust_remote_code=True)

    # Validate architecture
    arch = getattr(config, "architectures", ["Unknown"])[0]
    supported = ["LlamaForCausalLM", "MistralForCausalLM", "GemmaForCausalLM",
                 "Qwen2ForCausalLM", "PhiForCausalLM"]
    if arch not in supported:
        print(f"Warning: Architecture '{arch}' may not be compatible.")
        print(f"Supported: {', '.join(supported)}")
        print("Attempting conversion anyway (Llama-style weight layout assumed)...")

    dim = config.hidden_size
    hidden_dim = config.intermediate_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    vocab_size = config.vocab_size
    seq_len = getattr(config, "max_position_embeddings", 2048)

    print(f"  Architecture: {arch}")
    print(f"  dim={dim}, hidden={hidden_dim}, layers={n_layers}")
    print(f"  heads={n_heads}, kv_heads={n_kv_heads}, vocab={vocab_size}, seq={seq_len}")

    # Compute param count
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim
    params_per_layer = (dim * dim  # Wq
                       + kv_dim * dim  # Wk
                       + kv_dim * dim  # Wv
                       + dim * dim  # Wo
                       + hidden_dim * dim  # W1 (gate)
                       + dim * hidden_dim  # W2 (down)
                       + hidden_dim * dim  # W3 (up)
                       + 2 * dim)  # 2x RMSNorm
    total_params = n_layers * params_per_layer + vocab_size * dim + dim
    print(f"  Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    if n_kv_heads != n_heads:
        ratio = n_heads // n_kv_heads
        expanded = total_params + n_layers * 2 * (dim * dim - kv_dim * dim)
        print(f"  GQA detected: {n_kv_heads} KV heads → expanding to {n_heads} (×{ratio})")
        print(f"  Expanded parameters: {expanded:,} ({expanded / 1e6:.1f}M)")

    return {
        "dim": dim,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "architecture": arch,
        "head_dim": head_dim,
    }


def load_weights(model_id, token=None):
    """Load model weights from HuggingFace safetensors."""
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download, list_repo_files
    import numpy as np

    print(f"\nDownloading weights from {model_id}...")

    # Find safetensors files in repo
    try:
        files = list_repo_files(model_id, token=token)
    except Exception as e:
        print(f"Error listing repo files: {e}")
        sys.exit(1)

    st_files = [f for f in files if f.endswith(".safetensors")]
    if not st_files:
        print("Error: No safetensors files found in repository.")
        print("Available files:", [f for f in files if f.endswith(('.bin', '.safetensors', '.pt'))])
        sys.exit(1)

    print(f"  Found {len(st_files)} safetensors file(s)")

    # Download and load all safetensors files
    weights = {}
    for st_file in st_files:
        print(f"  Downloading {st_file}...")
        path = hf_hub_download(model_id, st_file, token=token)
        with safe_open(path, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
                if weights[key].dtype != np.float32:
                    weights[key] = weights[key].astype(np.float32)

    print(f"  Loaded {len(weights)} tensors")
    return weights


def map_tensor_name(hf_name):
    """Map HuggingFace tensor name to (layer, component) tuple.

    Returns (layer_idx, component_name) or (None, component_name) for non-layer tensors.
    """
    # Token embedding
    if hf_name in ("model.embed_tokens.weight", "lm_head.weight"):
        return (None, hf_name.split(".")[-2])

    # Final layer norm
    if hf_name in ("model.norm.weight",):
        return (None, "final_norm")

    # Per-layer weights
    # Format: model.layers.{L}.{component}.weight
    parts = hf_name.split(".")
    if len(parts) >= 4 and parts[1] == "layers":
        layer = int(parts[2])
        rest = ".".join(parts[3:])

        mapping = {
            "input_layernorm.weight": "rms_att",
            "self_attn.q_proj.weight": "wq",
            "self_attn.k_proj.weight": "wk",
            "self_attn.v_proj.weight": "wv",
            "self_attn.o_proj.weight": "wo",
            "post_attention_layernorm.weight": "rms_ffn",
            "mlp.gate_proj.weight": "w1",
            "mlp.down_proj.weight": "w2",
            "mlp.up_proj.weight": "w3",
        }
        if rest in mapping:
            return (layer, mapping[rest])

    return (None, None)


def expand_kv_heads(weight, n_heads, n_kv_heads, dim):
    """Expand GQA KV weight from [kv_dim, dim] to [dim, dim] by repeating heads."""
    import numpy as np

    if n_kv_heads == n_heads:
        return weight  # No expansion needed

    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim
    ratio = n_heads // n_kv_heads

    assert weight.shape[0] == kv_dim, f"Expected KV weight shape[0]={kv_dim}, got {weight.shape[0]}"

    # Reshape to [n_kv_heads, head_dim, dim], repeat, reshape back
    w = weight.reshape(n_kv_heads, head_dim, dim)
    w = np.repeat(w, ratio, axis=0)  # [n_heads, head_dim, dim]
    return w.reshape(dim, dim)


def write_llama2c(weights, config, output_path):
    """Write weights in llama2.c format: 7-int header + float32 weights."""
    import numpy as np

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    vocab_size = config["vocab_size"]
    seq_len = config["seq_len"]

    print(f"\nWriting llama2.c model to {output_path}...")

    # Map all weights
    layer_weights = {}  # layer_idx -> {component: tensor}
    global_weights = {}  # component -> tensor

    for name, tensor in weights.items():
        layer, component = map_tensor_name(name)
        if component is None:
            continue
        if layer is not None:
            if layer not in layer_weights:
                layer_weights[layer] = {}
            layer_weights[layer][component] = tensor
        else:
            global_weights[component] = tensor

    # Validate we have all layers
    for L in range(n_layers):
        if L not in layer_weights:
            print(f"Error: Missing weights for layer {L}")
            sys.exit(1)
        required = ["rms_att", "wq", "wk", "wv", "wo", "rms_ffn", "w1", "w2", "w3"]
        for comp in required:
            if comp not in layer_weights[L]:
                print(f"Error: Missing {comp} for layer {L}")
                sys.exit(1)

    if "embed_tokens" not in global_weights:
        print("Error: Missing token embedding weights")
        sys.exit(1)

    # Handle shared embedding (lm_head.weight == embed_tokens.weight)
    shared_embed = "lm_head" not in global_weights
    if shared_embed:
        print("  Using shared embedding (lm_head = embed_tokens)")

    with open(output_path, "wb") as f:
        # Write header: 7 ints
        # vocab_size negative = shared embedding (no separate classifier)
        vs = -vocab_size if shared_embed else vocab_size
        header = struct.pack("iiiiiii", dim, hidden_dim, n_layers, n_heads, n_kv_heads, vs, seq_len)
        f.write(header)

        # Weight order in llama2.c format:
        # 1. Token embedding [vocab_size, dim]
        embed = global_weights["embed_tokens"]
        assert embed.shape == (vocab_size, dim), f"Embed shape mismatch: {embed.shape}"
        f.write(embed.tobytes())
        print(f"  Wrote embed_tokens: {embed.shape}")

        # 2. Per-layer: rms_att [dim]
        for L in range(n_layers):
            w = layer_weights[L]["rms_att"]
            assert w.shape == (dim,), f"Layer {L} rms_att shape: {w.shape}"
            f.write(w.tobytes())

        # 3. Per-layer: Wq [dim, dim]
        for L in range(n_layers):
            w = layer_weights[L]["wq"]
            assert w.shape == (dim, dim), f"Layer {L} wq shape: {w.shape}"
            f.write(w.tobytes())

        # 4. Per-layer: Wk [kv_dim, dim] → expanded to [dim, dim] if GQA
        for L in range(n_layers):
            w = layer_weights[L]["wk"]
            w = expand_kv_heads(w, n_heads, n_kv_heads, dim)
            f.write(w.tobytes())
        if n_kv_heads != n_heads:
            print(f"  Expanded Wk: [{n_kv_heads * dim // n_heads}, {dim}] → [{dim}, {dim}]")

        # 5. Per-layer: Wv [kv_dim, dim] → expanded to [dim, dim] if GQA
        for L in range(n_layers):
            w = layer_weights[L]["wv"]
            w = expand_kv_heads(w, n_heads, n_kv_heads, dim)
            f.write(w.tobytes())
        if n_kv_heads != n_heads:
            print(f"  Expanded Wv: [{n_kv_heads * dim // n_heads}, {dim}] → [{dim}, {dim}]")

        # 6. Per-layer: Wo [dim, dim]
        for L in range(n_layers):
            w = layer_weights[L]["wo"]
            assert w.shape == (dim, dim), f"Layer {L} wo shape: {w.shape}"
            f.write(w.tobytes())

        # 7. Per-layer: rms_ffn [dim]
        for L in range(n_layers):
            w = layer_weights[L]["rms_ffn"]
            assert w.shape == (dim,), f"Layer {L} rms_ffn shape: {w.shape}"
            f.write(w.tobytes())

        # 8. Per-layer: W1/gate_proj [hidden_dim, dim]
        for L in range(n_layers):
            w = layer_weights[L]["w1"]
            assert w.shape == (hidden_dim, dim), f"Layer {L} w1 shape: {w.shape}"
            f.write(w.tobytes())

        # 9. Per-layer: W2/down_proj [dim, hidden_dim]
        for L in range(n_layers):
            w = layer_weights[L]["w2"]
            assert w.shape == (dim, hidden_dim), f"Layer {L} w2 shape: {w.shape}"
            f.write(w.tobytes())

        # 10. Per-layer: W3/up_proj [hidden_dim, dim]
        for L in range(n_layers):
            w = layer_weights[L]["w3"]
            assert w.shape == (hidden_dim, dim), f"Layer {L} w3 shape: {w.shape}"
            f.write(w.tobytes())

        # 11. Final RMS norm [dim]
        if "final_norm" in global_weights:
            w = global_weights["final_norm"]
            assert w.shape == (dim,), f"Final norm shape: {w.shape}"
            f.write(w.tobytes())
        else:
            print("  Warning: No final norm found, writing ones")
            f.write(np.ones(dim, dtype=np.float32).tobytes())

        # 12. Classifier weights (if not shared)
        # In llama2.c format, these come after freq_cis (which we skip)
        # For shared embedding models, classifier = embedding (already written)

    file_size = os.path.getsize(output_path)
    print(f"  Model written: {file_size:,} bytes ({file_size / 1e9:.2f} GB)")
    return file_size


def convert_tokenizer(model_id, output_path, vocab_size, token=None):
    """Convert HuggingFace tokenizer to llama2.c tokenizer.bin format.

    Format: max_token_length(i32) then for each token:
        score(f32), len(i32), bytes(len bytes)
    """
    from transformers import AutoTokenizer
    import numpy as np

    print(f"\nConverting tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: Could not load tokenizer: {e}")
        print("  Skipping tokenizer conversion.")
        return False

    # Get vocabulary
    vocab = tokenizer.get_vocab()
    actual_vocab_size = len(vocab)
    print(f"  Tokenizer vocab size: {actual_vocab_size}")

    if actual_vocab_size != vocab_size:
        print(f"  Warning: Tokenizer vocab ({actual_vocab_size}) != model vocab ({vocab_size})")
        # Use model's vocab_size, pad or truncate as needed

    # Build token list sorted by token ID
    id_to_token = {}
    for token_str, token_id in vocab.items():
        if token_id < vocab_size:
            id_to_token[token_id] = token_str

    # Compute max token length and scores
    max_token_length = 0
    tokens = []
    for i in range(vocab_size):
        if i in id_to_token:
            t = id_to_token[i]
            # Convert token string to bytes
            # Handle special tokens and byte-level tokens
            try:
                token_bytes = t.encode("utf-8")
            except UnicodeEncodeError:
                token_bytes = t.encode("utf-8", errors="replace")
            max_token_length = max(max_token_length, len(token_bytes))
            # Use negative index as score (BPE merge priority: lower = higher priority)
            score = -float(i)
            tokens.append((score, token_bytes))
        else:
            # Missing token — placeholder
            tokens.append((0.0, b""))

    print(f"  Max token length: {max_token_length}")

    with open(output_path, "wb") as f:
        f.write(struct.pack("i", max_token_length))
        for score, token_bytes in tokens:
            f.write(struct.pack("f", score))
            f.write(struct.pack("i", len(token_bytes)))
            f.write(token_bytes)

    file_size = os.path.getsize(output_path)
    print(f"  Tokenizer written: {file_size:,} bytes")
    return True


def write_model_card(model_id, config, output_path, model_size, has_tokenizer):
    """Write model card JSON for NeuralForge app display."""
    card = {
        "name": model_id.split("/")[-1],
        "source": f"https://huggingface.co/{model_id}",
        "repo_id": model_id,
        "architecture": config["architecture"],
        "dim": config["dim"],
        "hidden_dim": config["hidden_dim"],
        "n_layers": config["n_layers"],
        "n_heads": config["n_heads"],
        "n_kv_heads": config["n_kv_heads"],
        "vocab_size": config["vocab_size"],
        "seq_len": config["seq_len"],
        "params_millions": round(config.get("total_params", 0) / 1e6, 1),
        "file_size_bytes": model_size,
        "has_tokenizer": has_tokenizer,
        "gqa_expanded": config["n_kv_heads"] != config["n_heads"],
        "converted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "format": "llama2c_f32",
    }

    with open(output_path, "w") as f:
        json.dump(card, f, indent=2)
    print(f"\nModel card written to {output_path}")
    return card


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to NeuralForge llama2.c format"
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    parser.add_argument("--output", required=True,
                        help="Output directory for converted files")
    parser.add_argument("--token", default=None,
                        help="HuggingFace API token (for gated models like Llama 2)")
    parser.add_argument("--skip-tokenizer", action="store_true",
                        help="Skip tokenizer conversion")
    parser.add_argument("--model-name", default=None,
                        help="Override model filename (default: model.bin)")
    args = parser.parse_args()

    check_dependencies()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load config
    config = load_model_config(args.model, token=args.token)

    # Compute total params for card
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    # After GQA expansion, all weights are full-size
    params_per_layer = (4 * dim * dim  # Wq, Wk, Wv, Wo (expanded)
                       + 3 * hidden_dim * dim  # W1, W2, W3
                       + 2 * dim)  # 2x RMSNorm
    config["total_params"] = n_layers * params_per_layer + config["vocab_size"] * dim + dim

    # Load weights
    weights = load_weights(args.model, token=args.token)

    # Write model
    model_name = args.model_name or "model.bin"
    model_path = os.path.join(args.output, model_name)
    model_size = write_llama2c(weights, config, model_path)

    # Convert tokenizer
    has_tokenizer = False
    if not args.skip_tokenizer:
        tok_path = os.path.join(args.output, "tokenizer.bin")
        has_tokenizer = convert_tokenizer(args.model, tok_path, config["vocab_size"],
                                           token=args.token)

    # Write model card
    card_path = os.path.join(args.output, "model_card.json")
    card = write_model_card(args.model, config, card_path, model_size, has_tokenizer)

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Model:     {model_path}")
    if has_tokenizer:
        print(f"  Tokenizer: {os.path.join(args.output, 'tokenizer.bin')}")
    print(f"  Card:      {card_path}")
    print(f"\nUsage with NeuralForge:")
    print(f"  neuralforge info --model {model_path}")
    print(f"  neuralforge generate --model {model_path} --tokenizer {os.path.join(args.output, 'tokenizer.bin')} --prompt \"Hello\"")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
