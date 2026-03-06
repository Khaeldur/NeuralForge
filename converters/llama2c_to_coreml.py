#!/usr/bin/env python3
"""Convert llama2.c or NeuralForge checkpoint to CoreML model.

Usage:
    python3 llama2c_to_coreml.py --llama2c model.bin --output NeuralForge.mlpackage
    python3 llama2c_to_coreml.py --ckpt checkpoint.bin --output NeuralForge.mlpackage

Requires: pip install coremltools numpy
"""

import argparse
import struct
import numpy as np
import sys
import os

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil import register_torch_op
    from coremltools.models.neural_network import NeuralNetworkBuilder
    from coremltools.models import datatypes
except ImportError:
    print("Error: coremltools not installed. Run: pip install coremltools", file=sys.stderr)
    sys.exit(1)


def load_llama2c(path):
    """Load weights from llama2.c format."""
    with open(path, "rb") as f:
        cfg = struct.unpack("7i", f.read(28))
        dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq = cfg
        vocab = abs(vocab)
        print(f"  Model: dim={dim} hidden={hidden} layers={n_layers} heads={n_heads} vocab={vocab} seq={seq}")

        hd = dim // n_heads
        weights = {}

        weights["embed"] = np.frombuffer(f.read(vocab * dim * 4), dtype=np.float32).reshape(vocab, dim)
        for L in range(n_layers):
            weights[f"rms_att.{L}"] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
        for L in range(n_layers):
            weights[f"Wq.{L}"] = np.frombuffer(f.read(dim * dim * 4), dtype=np.float32).reshape(dim, dim)
        for L in range(n_layers):
            weights[f"Wk.{L}"] = np.frombuffer(f.read(dim * dim * 4), dtype=np.float32).reshape(dim, dim)
        for L in range(n_layers):
            weights[f"Wv.{L}"] = np.frombuffer(f.read(dim * dim * 4), dtype=np.float32).reshape(dim, dim)
        for L in range(n_layers):
            weights[f"Wo.{L}"] = np.frombuffer(f.read(dim * dim * 4), dtype=np.float32).reshape(dim, dim)
        for L in range(n_layers):
            weights[f"rms_ffn.{L}"] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
        for L in range(n_layers):
            weights[f"W1.{L}"] = np.frombuffer(f.read(hidden * dim * 4), dtype=np.float32).reshape(hidden, dim)
        for L in range(n_layers):
            weights[f"W2.{L}"] = np.frombuffer(f.read(dim * hidden * 4), dtype=np.float32).reshape(dim, hidden)
        for L in range(n_layers):
            weights[f"W3.{L}"] = np.frombuffer(f.read(hidden * dim * 4), dtype=np.float32).reshape(hidden, dim)
        weights["rms_final"] = np.frombuffer(f.read(dim * 4), dtype=np.float32)

    return weights, {"dim": dim, "hidden": hidden, "n_layers": n_layers,
                     "n_heads": n_heads, "hd": hd, "vocab": vocab, "seq": seq}


def load_checkpoint(path):
    """Load weights from NeuralForge checkpoint format."""
    with open(path, "rb") as f:
        hdr_fmt = "<IIiiiiiiiifffddddiii"
        hdr_size = struct.calcsize(hdr_fmt)
        hdr_data = f.read(hdr_size)
        hdr = struct.unpack(hdr_fmt, hdr_data[:hdr_size])

        magic, version, step = hdr[0], hdr[1], hdr[2]
        if magic != 0x424C5A54 or version != 2:
            raise ValueError(f"Invalid checkpoint: magic={magic:#x} version={version}")

        n_layers = hdr[4]
        vocab = hdr[5]
        dim = hdr[6]
        hidden = hdr[7]
        n_heads = hdr[8]
        seq = hdr[9]
        loss = hdr[11]
        hd = dim // n_heads

        print(f"  Checkpoint: step={step}, loss={loss:.4f}")
        print(f"  Model: dim={dim} hidden={hidden} layers={n_layers} heads={n_heads} vocab={vocab} seq={seq}")

        weights = {}
        for L in range(n_layers):
            weights[f"Wq.{L}"] = np.frombuffer(f.read(dim*dim*4), dtype=np.float32).reshape(dim, dim)
            weights[f"Wk.{L}"] = np.frombuffer(f.read(dim*dim*4), dtype=np.float32).reshape(dim, dim)
            weights[f"Wv.{L}"] = np.frombuffer(f.read(dim*dim*4), dtype=np.float32).reshape(dim, dim)
            weights[f"Wo.{L}"] = np.frombuffer(f.read(dim*dim*4), dtype=np.float32).reshape(dim, dim)
            weights[f"W1.{L}"] = np.frombuffer(f.read(hidden*dim*4), dtype=np.float32).reshape(hidden, dim)
            weights[f"W2.{L}"] = np.frombuffer(f.read(dim*hidden*4), dtype=np.float32).reshape(dim, hidden)
            weights[f"W3.{L}"] = np.frombuffer(f.read(hidden*dim*4), dtype=np.float32).reshape(hidden, dim)
            weights[f"rms_att.{L}"] = np.frombuffer(f.read(dim*4), dtype=np.float32)
            weights[f"rms_ffn.{L}"] = np.frombuffer(f.read(dim*4), dtype=np.float32)

            # Skip Adam state
            adam_skip = (dim*dim*2 + dim*dim*2 + dim*dim*2 + dim*dim*2 +
                         hidden*dim*2 + dim*hidden*2 + hidden*dim*2 +
                         dim*2 + dim*2) * 4
            f.read(adam_skip)

        weights["rms_final"] = np.frombuffer(f.read(dim*4), dtype=np.float32)
        f.read(dim * 2 * 4)  # skip Adam state
        weights["embed"] = np.frombuffer(f.read(vocab*dim*4), dtype=np.float32).reshape(vocab, dim)

    return weights, {"dim": dim, "hidden": hidden, "n_layers": n_layers,
                     "n_heads": n_heads, "hd": hd, "vocab": vocab, "seq": seq}


def precompute_rope(seq, hd):
    """Precompute RoPE (Rotary Position Embedding) sin/cos tables."""
    freqs = 1.0 / (10000.0 ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
    t = np.arange(seq, dtype=np.float32)
    freqs = np.outer(t, freqs)  # (seq, hd/2)
    cos_table = np.cos(freqs).astype(np.float32)
    sin_table = np.sin(freqs).astype(np.float32)
    return cos_table, sin_table


def rmsnorm(x, weight, eps=1e-5):
    """RMSNorm for verification."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def convert_to_coreml(weights, config, output_path):
    """Build a CoreML model from weights using coremltools MIL."""
    dim = config["dim"]
    hidden = config["hidden"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    hd = config["hd"]
    vocab = config["vocab"]
    seq = config["seq"]

    # Precompute RoPE tables
    cos_table, sin_table = precompute_rope(seq, hd)

    # Build CoreML model using the Neural Network builder
    # Input: token_ids (seq_len,) integer array
    # Output: logits (vocab,) float array for next token prediction

    input_features = [("input_ids", datatypes.Array(seq))]
    output_features = [("logits", datatypes.Array(vocab))]

    builder = NeuralNetworkBuilder(
        input_features, output_features,
        disable_rank5_shape_mapping=True
    )

    # Token embedding lookup
    builder.add_embedding(
        name="token_embd",
        W=weights["embed"],  # (vocab, dim)
        b=None,
        input_dim=vocab,
        output_channels=dim,
        has_bias=False,
        input_name="input_ids",
        output_name="embedded"
    )

    x_name = "embedded"

    for L in range(n_layers):
        prefix = f"layer_{L}"

        # Attention RMSNorm
        builder.add_mvn(
            name=f"{prefix}_rms_att_norm",
            input_name=x_name,
            output_name=f"{prefix}_rms_att_normed",
            across_channels=True,
            normalize_variance=True,
            epsilon=1e-5
        )

        # Scale by RMSNorm weight
        builder.add_scale(
            name=f"{prefix}_rms_att_scale",
            W=weights[f"rms_att.{L}"],
            b=None,
            has_bias=False,
            input_name=f"{prefix}_rms_att_normed",
            output_name=f"{prefix}_att_input",
            shape_scale=[dim]
        )

        # Q, K, V projections
        for proj_name, weight_key in [("q", "Wq"), ("k", "Wk"), ("v", "Wv")]:
            builder.add_inner_product(
                name=f"{prefix}_attn_{proj_name}",
                W=weights[f"{weight_key}.{L}"].T,  # CoreML expects (out, in)
                b=None,
                input_channels=dim,
                output_channels=dim,
                has_bias=False,
                input_name=f"{prefix}_att_input",
                output_name=f"{prefix}_{proj_name}"
            )

        # Attention output projection
        builder.add_inner_product(
            name=f"{prefix}_attn_output",
            W=weights[f"Wo.{L}"].T,
            b=None,
            input_channels=dim,
            output_channels=dim,
            has_bias=False,
            input_name=f"{prefix}_att_input",  # Simplified: skip RoPE+SDPA for CoreML
            output_name=f"{prefix}_attn_out"
        )

        # Residual add
        builder.add_add_broadcastable(
            name=f"{prefix}_residual1",
            input_names=[x_name, f"{prefix}_attn_out"],
            output_name=f"{prefix}_post_attn"
        )

        # FFN RMSNorm
        builder.add_mvn(
            name=f"{prefix}_rms_ffn_norm",
            input_name=f"{prefix}_post_attn",
            output_name=f"{prefix}_rms_ffn_normed",
            across_channels=True,
            normalize_variance=True,
            epsilon=1e-5
        )

        builder.add_scale(
            name=f"{prefix}_rms_ffn_scale",
            W=weights[f"rms_ffn.{L}"],
            b=None,
            has_bias=False,
            input_name=f"{prefix}_rms_ffn_normed",
            output_name=f"{prefix}_ffn_input",
            shape_scale=[dim]
        )

        # SwiGLU FFN: out = W2 @ (silu(W1 @ x) * W3 @ x)
        # W1 (gate)
        builder.add_inner_product(
            name=f"{prefix}_ffn_gate",
            W=weights[f"W1.{L}"].T if weights[f"W1.{L}"].shape[0] == hidden else weights[f"W1.{L}"],
            b=None,
            input_channels=dim,
            output_channels=hidden,
            has_bias=False,
            input_name=f"{prefix}_ffn_input",
            output_name=f"{prefix}_gate_proj"
        )

        # SiLU activation on gate
        builder.add_activation(
            name=f"{prefix}_silu",
            non_linearity="SIGMOID",
            input_name=f"{prefix}_gate_proj",
            output_name=f"{prefix}_gate_sigmoid"
        )
        builder.add_multiply_broadcastable(
            name=f"{prefix}_silu_mul",
            input_names=[f"{prefix}_gate_proj", f"{prefix}_gate_sigmoid"],
            output_name=f"{prefix}_gate_silu"
        )

        # W3 (up)
        builder.add_inner_product(
            name=f"{prefix}_ffn_up",
            W=weights[f"W3.{L}"].T if weights[f"W3.{L}"].shape[0] == hidden else weights[f"W3.{L}"],
            b=None,
            input_channels=dim,
            output_channels=hidden,
            has_bias=False,
            input_name=f"{prefix}_ffn_input",
            output_name=f"{prefix}_up_proj"
        )

        # gate * up
        builder.add_multiply_broadcastable(
            name=f"{prefix}_ffn_mul",
            input_names=[f"{prefix}_gate_silu", f"{prefix}_up_proj"],
            output_name=f"{prefix}_ffn_hidden"
        )

        # W2 (down)
        builder.add_inner_product(
            name=f"{prefix}_ffn_down",
            W=weights[f"W2.{L}"].T if weights[f"W2.{L}"].shape[0] == dim else weights[f"W2.{L}"],
            b=None,
            input_channels=hidden,
            output_channels=dim,
            has_bias=False,
            input_name=f"{prefix}_ffn_hidden",
            output_name=f"{prefix}_ffn_out"
        )

        # Residual add
        builder.add_add_broadcastable(
            name=f"{prefix}_residual2",
            input_names=[f"{prefix}_post_attn", f"{prefix}_ffn_out"],
            output_name=f"{prefix}_output"
        )

        x_name = f"{prefix}_output"

    # Final RMSNorm
    builder.add_mvn(
        name="final_rms_norm",
        input_name=x_name,
        output_name="final_normed",
        across_channels=True,
        normalize_variance=True,
        epsilon=1e-5
    )

    builder.add_scale(
        name="final_rms_scale",
        W=weights["rms_final"],
        b=None,
        has_bias=False,
        input_name="final_normed",
        output_name="final_scaled",
        shape_scale=[dim]
    )

    # Output projection (shared with embedding)
    builder.add_inner_product(
        name="output_proj",
        W=weights["embed"],  # Shared embedding (vocab, dim) — already (out, in) for CoreML
        b=None,
        input_channels=dim,
        output_channels=vocab,
        has_bias=False,
        input_name="final_scaled",
        output_name="logits"
    )

    # Build and save
    model = builder.spec

    # Set metadata
    model.description.metadata.author = "NeuralForge"
    model.description.metadata.shortDescription = (
        f"LLaMA-style transformer ({n_layers}L, {dim}D, {n_heads}H) "
        f"trained with NeuralForge on Apple Neural Engine"
    )

    coreml_model = ct.models.MLModel(model)
    coreml_model.save(output_path)

    print(f"\nSaved CoreML model to {output_path}")
    print(f"  Architecture: {n_layers} layers, dim={dim}, heads={n_heads}")
    print(f"  Vocabulary: {vocab}, Context: {seq}")
    print(f"\nNote: This is a simplified export (no RoPE/SDPA in CoreML graph).")
    print(f"For full inference fidelity, use the llama2c format with a runtime")
    print(f"that applies RoPE and attention natively.")


def main():
    parser = argparse.ArgumentParser(description="Export NeuralForge model to CoreML")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ckpt", help="NeuralForge checkpoint path")
    group.add_argument("--llama2c", help="llama2.c model path")
    parser.add_argument("--output", "-o", required=True, help="Output .mlpackage path")
    args = parser.parse_args()

    if args.ckpt:
        print(f"Loading checkpoint: {args.ckpt}")
        weights, config = load_checkpoint(args.ckpt)
    else:
        print(f"Loading llama2.c model: {args.llama2c}")
        weights, config = load_llama2c(args.llama2c)

    convert_to_coreml(weights, config, args.output)


if __name__ == "__main__":
    main()
