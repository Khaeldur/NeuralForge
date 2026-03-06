#!/bin/bash
# Download Stories 110M model weights + training data for NeuralForge
set -e

MODELS_DIR="${1:-$(dirname "$0")/../models}"
mkdir -p "$MODELS_DIR"

echo "=== NeuralForge Model & Data Downloader ==="
echo ""

# Stories 110M model (from Karpathy's llama2.c)
MODEL_URL="https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin"
MODEL_FILE="$MODELS_DIR/stories110M.bin"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading stories110M.bin (438MB)..."
    curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"
    echo "  Saved to $MODEL_FILE"
else
    echo "  stories110M.bin already exists"
fi

# TinyStories tokenized training data
DATA_URL="https://huggingface.co/datasets/enio/TinyStories/resolve/main/TinyStories_tok32000/data00.bin"
DATA_FILE="$MODELS_DIR/tinystories_data00.bin"
if [ ! -f "$DATA_FILE" ]; then
    echo "Downloading tinystories_data00.bin (~190MB)..."
    curl -L --progress-bar -o "$DATA_FILE" "$DATA_URL"
    echo "  Saved to $DATA_FILE"
else
    echo "  tinystories_data00.bin already exists"
fi

# Copy tokenizer from vendor
TOKENIZER_SRC="$(dirname "$0")/../vendor/ANE/assets/models/tokenizer.bin"
TOKENIZER_DST="$MODELS_DIR/tokenizer.bin"
if [ -f "$TOKENIZER_SRC" ] && [ ! -f "$TOKENIZER_DST" ]; then
    cp "$TOKENIZER_SRC" "$TOKENIZER_DST"
    echo "  Copied tokenizer.bin"
fi

echo ""
echo "=== Downloads complete ==="
ls -lh "$MODELS_DIR"
echo ""
echo "Quick start:"
echo "  cd cli && make"
echo "  ./neuralforge info --model $MODEL_FILE"
echo "  ./neuralforge train --model $MODEL_FILE --data $DATA_FILE --steps 100"
