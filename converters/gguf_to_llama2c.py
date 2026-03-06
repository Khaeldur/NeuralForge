#!/usr/bin/env python3
"""Convert a GGUF model file to llama2.c format.

Usage:
    python3 gguf_to_llama2c.py --gguf model.gguf --output model.bin
    python3 gguf_to_llama2c.py --gguf model.gguf --output model.bin --f16

Requires: pip install numpy
"""

import argparse
import struct
import numpy as np
import sys
import os

# GGUF constants
GGUF_MAGIC = 0x46475547  # "GGUF"
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def read_string(f):
    """Read a GGUF string (uint64 length + bytes)."""
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def read_value(f, vtype):
    """Read a GGUF typed value."""
    if vtype == GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif vtype == GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif vtype == GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif vtype == GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif vtype == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif vtype == GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif vtype == GGUF_TYPE_BOOL:
        return struct.unpack("<B", f.read(1))[0] != 0
    elif vtype == GGUF_TYPE_STRING:
        return read_string(f)
    elif vtype == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif vtype == GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif vtype == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    elif vtype == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        return [read_value(f, elem_type) for _ in range(count)]
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def read_gguf(path):
    """Read a GGUF file and return metadata + tensors."""
    with open(path, "rb") as f:
        # Header
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic: 0x{magic:08X})")

        version = struct.unpack("<I", f.read(4))[0]
        if version < 2 or version > 3:
            raise ValueError(f"Unsupported GGUF version: {version}")

        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        print(f"GGUF v{version}: {n_tensors} tensors, {n_kv} metadata entries")

        # Read metadata KV pairs
        metadata = {}
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            value = read_value(f, vtype)
            metadata[key] = value

        # Read tensor info headers
        tensor_infos = []
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack("<Q", f.read(8))[0])
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensor_infos.append({
                "name": name,
                "dims": dims,
                "dtype": dtype,
                "offset": offset,
            })

        # Align to 32 bytes for tensor data start
        alignment = metadata.get("general.alignment", 32)
        pos = f.tell()
        aligned_pos = ((pos + alignment - 1) // alignment) * alignment
        f.seek(aligned_pos)
        data_start = f.tell()

        # Read tensor data
        tensors = {}
        for info in tensor_infos:
            f.seek(data_start + info["offset"])
            n_elements = 1
            for d in info["dims"]:
                n_elements *= d

            if info["dtype"] == GGML_TYPE_F32:
                data = np.frombuffer(f.read(n_elements * 4), dtype=np.float32).copy()
            elif info["dtype"] == GGML_TYPE_F16:
                data = np.frombuffer(f.read(n_elements * 2), dtype=np.float16).copy()
                data = data.astype(np.float32)
            else:
                raise ValueError(f"Unsupported tensor type: {info['dtype']} for {info['name']}")

            tensors[info["name"]] = data.reshape(info["dims"])

        return metadata, tensors


def write_llama2c(output_path, metadata, tensors):
    """Write tensors in llama2.c format."""
    # Extract model dimensions from metadata
    dim = int(metadata.get("llama.embedding_length", 768))
    hidden_dim = int(metadata.get("llama.feed_forward_length", 2048))
    n_layers = int(metadata.get("llama.block_count", 12))
    n_heads = int(metadata.get("llama.attention.head_count", 12))
    vocab_size = int(metadata.get("llama.vocab_size", 32000))
    seq_len = int(metadata.get("llama.context_length", 256))

    print(f"Model: dim={dim} hidden={hidden_dim} layers={n_layers} "
          f"heads={n_heads} vocab={vocab_size} seq={seq_len}")

    # llama2.c config header: 7 ints
    # dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
    n_kv_heads = int(metadata.get("llama.attention.head_count_kv", n_heads))

    with open(output_path, "wb") as f:
        # Write config (negative vocab_size indicates shared weights in llama2.c)
        f.write(struct.pack("<7i", dim, hidden_dim, n_layers, n_heads,
                            n_kv_heads, -vocab_size, seq_len))

        # Write weights in llama2.c order:
        # 1. token_embd
        get_tensor(tensors, "token_embd.weight", (vocab_size, dim)).tofile(f)

        # 2. Per-layer: rms_att_norm
        for L in range(n_layers):
            get_tensor(tensors, f"blk.{L}.attn_norm.weight", (dim,)).tofile(f)

        # 3. Per-layer: Wq
        for L in range(n_layers):
            get_tensor(tensors, f"blk.{L}.attn_q.weight", (dim, dim)).tofile(f)

        # 4. Per-layer: Wk
        for L in range(n_layers):
            get_tensor(tensors, f"blk.{L}.attn_k.weight", (dim, dim)).tofile(f)

        # 5. Per-layer: Wv
        for L in range(n_layers):
            get_tensor(tensors, f"blk.{L}.attn_v.weight", (dim, dim)).tofile(f)

        # 6. Per-layer: Wo
        for L in range(n_layers):
            get_tensor(tensors, f"blk.{L}.attn_output.weight", (dim, dim)).tofile(f)

        # 7. Per-layer: rms_ffn_norm
        for L in range(n_layers):
            get_tensor(tensors, f"blk.{L}.ffn_norm.weight", (dim,)).tofile(f)

        # 8. Per-layer: W1 (gate)
        for L in range(n_layers):
            get_tensor(tensors, f"blk.{L}.ffn_gate.weight", (hidden_dim, dim)).tofile(f)

        # 9. Per-layer: W2 (down)
        for L in range(n_layers):
            get_tensor(tensors, f"blk.{L}.ffn_down.weight", (dim, hidden_dim)).tofile(f)

        # 10. Per-layer: W3 (up)
        for L in range(n_layers):
            get_tensor(tensors, f"blk.{L}.ffn_up.weight", (hidden_dim, dim)).tofile(f)

        # 11. Final RMS norm
        get_tensor(tensors, "output_norm.weight", (dim,)).tofile(f)

        # 12. RoPE freqs (skip — llama2.c computes these)
        # Write dim/2 zeros as placeholder for freq_cis_real and freq_cis_imag
        head_size = dim // n_heads
        np.zeros(seq_len * head_size // 2, dtype=np.float32).tofile(f)  # freq_cis_real
        np.zeros(seq_len * head_size // 2, dtype=np.float32).tofile(f)  # freq_cis_imag

        # 13. Output/classifier weights (shared with token_embd for negative vocab)
        # Already included via negative vocab_size flag

    file_size = os.path.getsize(output_path)
    print(f"Wrote {output_path} ({file_size / 1e6:.1f} MB)")


def get_tensor(tensors, name, expected_shape):
    """Get a tensor by name, validating shape."""
    if name not in tensors:
        print(f"  Warning: tensor '{name}' not found, using zeros {expected_shape}")
        return np.zeros(expected_shape, dtype=np.float32)
    t = tensors[name].astype(np.float32).flatten()
    expected_size = 1
    for d in expected_shape:
        expected_size *= d
    if t.size != expected_size:
        print(f"  Warning: tensor '{name}' size {t.size} != expected {expected_size}")
        if t.size > expected_size:
            t = t[:expected_size]
        else:
            t = np.pad(t, (0, expected_size - t.size))
    return t


def main():
    parser = argparse.ArgumentParser(description="Convert GGUF to llama2.c format")
    parser.add_argument("--gguf", required=True, help="Input GGUF file")
    parser.add_argument("--output", required=True, help="Output llama2.c file")
    args = parser.parse_args()

    if not os.path.exists(args.gguf):
        print(f"Error: {args.gguf} not found")
        sys.exit(1)

    print(f"Reading {args.gguf}...")
    metadata, tensors = read_gguf(args.gguf)

    print(f"\nMetadata:")
    for k, v in sorted(metadata.items()):
        if not isinstance(v, list):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: [{len(v)} items]")

    print(f"\nTensors: {len(tensors)}")
    for name, t in sorted(tensors.items()):
        print(f"  {name}: {t.shape} ({t.dtype})")

    print(f"\nWriting llama2.c format to {args.output}...")
    write_llama2c(args.output, metadata, tensors)
    print("Done!")


if __name__ == "__main__":
    main()
