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

# TinyStories tokenized training data (from enio/TinyStories on HuggingFace)
TAR_URL="https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok32000/TinyStories_tok32000.tar.gz?download=true"
DATA_FILE="$MODELS_DIR/tinystories_data00.bin"
TAR_FILE="$MODELS_DIR/TinyStories_tok32000.tar.gz"
if [ ! -f "$DATA_FILE" ]; then
    echo "Downloading pretokenized TinyStories (32K vocab, ~993MB tar.gz)..."
    curl -L --progress-bar -o "$TAR_FILE" "$TAR_URL"
    # Verify it's a valid gzip file
    if ! file "$TAR_FILE" | grep -q "gzip"; then
        echo "Error: Downloaded file is not a valid gzip archive."
        rm -f "$TAR_FILE"
        exit 1
    fi
    echo "  Extracting data00.bin from archive..."
    DATA_ENTRY=$(tar tzf "$TAR_FILE" 2>/dev/null | grep 'data00\.bin' | head -1)
    if [ -z "$DATA_ENTRY" ]; then
        echo "Error: data00.bin not found in archive."
        exit 1
    fi
    tar xzf "$TAR_FILE" -C "$MODELS_DIR" "$DATA_ENTRY"
    EXTRACTED="$MODELS_DIR/$DATA_ENTRY"
    if [ "$EXTRACTED" != "$DATA_FILE" ]; then
        mv "$EXTRACTED" "$DATA_FILE"
        rmdir "$(dirname "$EXTRACTED")" 2>/dev/null || true
    fi
    rm -f "$TAR_FILE"
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
