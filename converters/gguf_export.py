#!/usr/bin/env python3
"""Export NeuralForge checkpoint to GGUF format for llama.cpp inference.

Usage:
    python3 gguf_export.py --ckpt checkpoint.bin --output model.gguf
    python3 gguf_export.py --llama2c model.bin --output model.gguf

Requires: pip install numpy
"""

import argparse
import struct
import numpy as np
import sys
import os

# Model constants (Stories 110M)
DIM = 768
HIDDEN = 2048
HEADS = 12
SEQ = 256
NLAYERS = 12
VOCAB = 32000
HD = DIM // HEADS  # 64

WQ_SZ = DIM * DIM
WO_SZ = DIM * DIM
W1_SZ = HIDDEN * DIM
W2_SZ = DIM * HIDDEN
W3_SZ = HIDDEN * DIM

# GGUF constants
GGUF_MAGIC = 0x46475547  # "GGUF"
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

# GGUF tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


class GGUFWriter:
    def __init__(self, f):
        self.f = f
        self.kv_data = bytearray()
        self.tensor_infos = bytearray()
        self.tensor_data = bytearray()
        self.n_kv = 0
        self.n_tensors = 0
        self.alignment = 32

    def write_string(self, buf, s):
        b = s.encode("utf-8")
        buf += struct.pack("<Q", len(b))
        buf += b

    def add_kv_string(self, key, value):
        self.write_string(self.kv_data, key)
        self.kv_data += struct.pack("<I", GGUF_TYPE_STRING)
        self.write_string(self.kv_data, value)
        self.n_kv += 1

    def add_kv_uint32(self, key, value):
        self.write_string(self.kv_data, key)
        self.kv_data += struct.pack("<II", GGUF_TYPE_UINT32, value)
        self.n_kv += 1

    def add_kv_float32(self, key, value):
        self.write_string(self.kv_data, key)
        self.kv_data += struct.pack("<If", GGUF_TYPE_FLOAT32, value)
        self.n_kv += 1

    def add_tensor(self, name, data, dtype=GGML_TYPE_F32):
        """Add a tensor. data should be a numpy float32 array."""
        self.write_string(self.tensor_infos, name)

        # Number of dimensions
        shape = data.shape
        n_dims = len(shape)
        self.tensor_infos += struct.pack("<I", n_dims)

        # Dimensions (GGUF uses row-major, dimensions reversed from numpy)
        for d in reversed(shape):
            self.tensor_infos += struct.pack("<Q", d)

        # Type
        self.tensor_infos += struct.pack("<I", dtype)

        # Offset into tensor data block
        # Align to self.alignment bytes
        offset = len(self.tensor_data)
        pad = (self.alignment - (offset % self.alignment)) % self.alignment
        self.tensor_data += b"\x00" * pad
        offset += pad

        self.tensor_infos += struct.pack("<Q", offset)

        # Write tensor data
        if dtype == GGML_TYPE_F16:
            self.tensor_data += data.astype(np.float16).tobytes()
        else:
            self.tensor_data += data.astype(np.float32).tobytes()

        self.n_tensors += 1

    def write(self):
        # Header
        self.f.write(struct.pack("<II", GGUF_MAGIC, GGUF_VERSION))
        self.f.write(struct.pack("<QQ", self.n_tensors, self.n_kv))

        # KV pairs
        self.f.write(bytes(self.kv_data))

        # Tensor infos
        self.f.write(bytes(self.tensor_infos))

        # Align before tensor data
        pos = self.f.tell()
        pad = (self.alignment - (pos % self.alignment)) % self.alignment
        self.f.write(b"\x00" * pad)

        # Tensor data
        self.f.write(bytes(self.tensor_data))


def load_llama2c(path):
    """Load weights from llama2.c format."""
    with open(path, "rb") as f:
        # Config header: 7 ints
        cfg = struct.unpack("7i", f.read(28))
        dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq = cfg
        vocab = abs(vocab)
        print(f"  Model: dim={dim} hidden={hidden} layers={n_layers} heads={n_heads} vocab={vocab} seq={seq}")

        weights = {}
        weights["token_embd.weight"] = np.frombuffer(f.read(vocab * dim * 4), dtype=np.float32).reshape(vocab, dim)

        for L in range(n_layers):
            weights[f"blk.{L}.attn_norm.weight"] = np.frombuffer(f.read(dim * 4), dtype=np.float32)

        for L in range(n_layers):
            weights[f"blk.{L}.attn_q.weight"] = np.frombuffer(f.read(dim * dim * 4), dtype=np.float32).reshape(dim, dim)
        for L in range(n_layers):
            weights[f"blk.{L}.attn_k.weight"] = np.frombuffer(f.read(dim * dim * 4), dtype=np.float32).reshape(dim, dim)
        for L in range(n_layers):
            weights[f"blk.{L}.attn_v.weight"] = np.frombuffer(f.read(dim * dim * 4), dtype=np.float32).reshape(dim, dim)
        for L in range(n_layers):
            weights[f"blk.{L}.attn_output.weight"] = np.frombuffer(f.read(dim * dim * 4), dtype=np.float32).reshape(dim, dim)

        for L in range(n_layers):
            weights[f"blk.{L}.ffn_norm.weight"] = np.frombuffer(f.read(dim * 4), dtype=np.float32)

        for L in range(n_layers):
            weights[f"blk.{L}.ffn_gate.weight"] = np.frombuffer(f.read(hidden * dim * 4), dtype=np.float32).reshape(hidden, dim)
        for L in range(n_layers):
            weights[f"blk.{L}.ffn_down.weight"] = np.frombuffer(f.read(dim * hidden * 4), dtype=np.float32).reshape(dim, hidden)
        for L in range(n_layers):
            weights[f"blk.{L}.ffn_up.weight"] = np.frombuffer(f.read(hidden * dim * 4), dtype=np.float32).reshape(hidden, dim)

        weights["output_norm.weight"] = np.frombuffer(f.read(dim * 4), dtype=np.float32)

    return weights, {"dim": dim, "hidden": hidden, "n_layers": n_layers,
                     "n_heads": n_heads, "vocab": vocab, "seq": seq}


def load_checkpoint(path):
    """Load weights from NeuralForge checkpoint format."""
    with open(path, "rb") as f:
        # CkptHdr: magic(u32) version(u32) step(i32) total_steps(i32)
        #          n_layers(i32) vocab_size(i32) dim(i32) hidden_dim(i32)
        #          n_heads(i32) seq_len(i32) lr(f32) loss(f32)
        #          cum_compile(f64) cum_train(f64) cum_wall(f64)
        #          cum_steps(i32) cum_batches(i32) adam_t(i32)
        hdr_fmt = "<IIiiiiiiiifffddddiii"
        # Pad to correct size - may need adjustment
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

        print(f"  Checkpoint: step={step}, loss={loss:.4f}")
        print(f"  Model: dim={dim} hidden={hidden} layers={n_layers} heads={n_heads} vocab={vocab} seq={seq}")

        weights = {}

        for L in range(n_layers):
            # Weights
            weights[f"blk.{L}.attn_q.weight"] = np.frombuffer(f.read(dim*dim*4), dtype=np.float32).reshape(dim, dim)
            weights[f"blk.{L}.attn_k.weight"] = np.frombuffer(f.read(dim*dim*4), dtype=np.float32).reshape(dim, dim)
            weights[f"blk.{L}.attn_v.weight"] = np.frombuffer(f.read(dim*dim*4), dtype=np.float32).reshape(dim, dim)
            weights[f"blk.{L}.attn_output.weight"] = np.frombuffer(f.read(dim*dim*4), dtype=np.float32).reshape(dim, dim)
            weights[f"blk.{L}.ffn_gate.weight"] = np.frombuffer(f.read(hidden*dim*4), dtype=np.float32).reshape(hidden, dim)
            weights[f"blk.{L}.ffn_down.weight"] = np.frombuffer(f.read(dim*hidden*4), dtype=np.float32).reshape(dim, hidden)
            weights[f"blk.{L}.ffn_up.weight"] = np.frombuffer(f.read(hidden*dim*4), dtype=np.float32).reshape(hidden, dim)
            weights[f"blk.{L}.attn_norm.weight"] = np.frombuffer(f.read(dim*4), dtype=np.float32)
            weights[f"blk.{L}.ffn_norm.weight"] = np.frombuffer(f.read(dim*4), dtype=np.float32)

            # Skip Adam state (m and v for each weight)
            adam_skip = (dim*dim*2 + dim*dim*2 + dim*dim*2 + dim*dim*2 +  # Wq,Wk,Wv,Wo m&v
                         hidden*dim*2 + dim*hidden*2 + hidden*dim*2 +       # W1,W2,W3 m&v
                         dim*2 + dim*2) * 4                                  # rms_att, rms_ffn m&v
            f.read(adam_skip)

        weights["output_norm.weight"] = np.frombuffer(f.read(dim*4), dtype=np.float32)
        f.read(dim * 2 * 4)  # skip Adam state for rms_final

        weights["token_embd.weight"] = np.frombuffer(f.read(vocab*dim*4), dtype=np.float32).reshape(vocab, dim)

    return weights, {"dim": dim, "hidden": hidden, "n_layers": n_layers,
                     "n_heads": n_heads, "vocab": vocab, "seq": seq}


def convert_to_gguf(weights, config, output_path, use_f16=False):
    """Write weights to GGUF format."""
    with open(output_path, "wb") as f:
        writer = GGUFWriter(f)

        # Metadata
        writer.add_kv_string("general.architecture", "llama")
        writer.add_kv_string("general.name", "NeuralForge Stories 110M")
        writer.add_kv_uint32("llama.embedding_length", config["dim"])
        writer.add_kv_uint32("llama.feed_forward_length", config["hidden"])
        writer.add_kv_uint32("llama.block_count", config["n_layers"])
        writer.add_kv_uint32("llama.attention.head_count", config["n_heads"])
        writer.add_kv_uint32("llama.attention.head_count_kv", config["n_heads"])
        writer.add_kv_uint32("llama.context_length", config["seq"])
        writer.add_kv_uint32("llama.vocab_size", config["vocab"])
        writer.add_kv_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)
        writer.add_kv_string("general.file_type", "f16" if use_f16 else "f32")
        writer.add_kv_string("tokenizer.ggml.model", "llama")

        # Tensors
        dtype = GGML_TYPE_F16 if use_f16 else GGML_TYPE_F32
        for name, data in sorted(weights.items()):
            writer.add_tensor(name, data, dtype)

        writer.write()

    size = os.path.getsize(output_path)
    print(f"\nWrote {output_path} ({size / 1e6:.1f} MB, {len(weights)} tensors)")


def main():
    parser = argparse.ArgumentParser(description="Export NeuralForge model to GGUF")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ckpt", help="NeuralForge checkpoint path")
    group.add_argument("--llama2c", help="llama2.c model path")
    parser.add_argument("--output", "-o", required=True, help="Output GGUF path")
    parser.add_argument("--f16", action="store_true", help="Store weights as float16")
    args = parser.parse_args()

    if args.ckpt:
        print(f"Loading checkpoint: {args.ckpt}")
        weights, config = load_checkpoint(args.ckpt)
    else:
        print(f"Loading llama2.c model: {args.llama2c}")
        weights, config = load_llama2c(args.llama2c)

    convert_to_gguf(weights, config, args.output, use_f16=args.f16)


if __name__ == "__main__":
    main()
