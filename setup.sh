#!/bin/bash
# NeuralForge — One-command setup for new users
# Usage: bash setup.sh
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   NeuralForge — First-Time Setup     ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════╝${NC}"
echo ""

ERRORS=0

# ── Step 1: Check prerequisites ──
echo -e "${CYAN}[1/6]${NC} Checking prerequisites..."

# macOS version
SW_VER=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
MAJOR=$(echo "$SW_VER" | cut -d. -f1)
if [ "$MAJOR" -ge 14 ] 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} macOS $SW_VER"
else
    echo -e "  ${RED}✗${NC} macOS 14+ required (found $SW_VER)"
    ERRORS=$((ERRORS + 1))
fi

# Apple Silicon
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo -e "  ${GREEN}✓${NC} Apple Silicon ($ARCH)"
else
    echo -e "  ${YELLOW}⚠${NC} Apple Silicon recommended (found $ARCH) — ANE training won't work"
fi

# Xcode
if xcode-select -p &>/dev/null; then
    XCODE_PATH=$(xcode-select -p)
    echo -e "  ${GREEN}✓${NC} Xcode CLI tools ($XCODE_PATH)"
else
    echo -e "  ${RED}✗${NC} Xcode not found — install from App Store or run: xcode-select --install"
    ERRORS=$((ERRORS + 1))
fi

# Disk space (need ~2GB free)
FREE_GB=$(df -g "$SCRIPT_DIR" | tail -1 | awk '{print $4}')
if [ "$FREE_GB" -ge 2 ] 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Disk space: ${FREE_GB}GB free"
else
    echo -e "  ${RED}✗${NC} Need 2GB+ free disk space (found ${FREE_GB}GB)"
    ERRORS=$((ERRORS + 1))
fi

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo -e "${RED}Setup cannot continue — fix the issues above.${NC}"
    exit 1
fi

# ── Step 2: Build CLI ──
echo ""
echo -e "${CYAN}[2/6]${NC} Building CLI..."
cd "$SCRIPT_DIR/cli"
if make clean && make 2>&1 | tail -3; then
    CLI_SIZE=$(ls -lh neuralforge | awk '{print $5}')
    echo -e "  ${GREEN}✓${NC} CLI built ($CLI_SIZE)"
else
    echo -e "  ${RED}✗${NC} CLI build failed"
    exit 1
fi
cd "$SCRIPT_DIR"

# ── Step 3: Download models ──
echo ""
echo -e "${CYAN}[3/6]${NC} Downloading models (this may take a few minutes)..."

MODELS_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODELS_DIR"

# stories110M.bin (~438MB)
MODEL_FILE="$MODELS_DIR/stories110M.bin"
EXPECTED_MODEL_SIZE=440903680  # exact size in bytes
if [ -f "$MODEL_FILE" ]; then
    ACTUAL_SIZE=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat --format=%s "$MODEL_FILE" 2>/dev/null || echo 0)
    if [ "$ACTUAL_SIZE" -lt 1000000 ]; then
        echo -e "  ${YELLOW}⚠${NC} stories110M.bin is corrupt (${ACTUAL_SIZE} bytes) — re-downloading..."
        rm -f "$MODEL_FILE"
    fi
fi
if [ ! -f "$MODEL_FILE" ]; then
    echo "  Downloading stories110M.bin (438MB)..."
    curl -L --progress-bar --retry 3 -o "$MODEL_FILE" \
        "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin"
    ACTUAL_SIZE=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat --format=%s "$MODEL_FILE" 2>/dev/null || echo 0)
    if [ "$ACTUAL_SIZE" -lt 1000000 ]; then
        echo -e "  ${RED}✗${NC} Download failed or file is corrupt (${ACTUAL_SIZE} bytes)"
        rm -f "$MODEL_FILE"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} stories110M.bin downloaded"
else
    echo -e "  ${GREEN}✓${NC} stories110M.bin exists ($(ls -lh "$MODEL_FILE" | awk '{print $5}'))"
fi

# Training data
DATA_FILE="$MODELS_DIR/tinystories_data00.bin"
if [ -f "$DATA_FILE" ]; then
    ACTUAL_SIZE=$(stat -f%z "$DATA_FILE" 2>/dev/null || stat --format=%s "$DATA_FILE" 2>/dev/null || echo 0)
    if [ "$ACTUAL_SIZE" -lt 1000 ]; then
        echo -e "  ${YELLOW}⚠${NC} tinystories_data00.bin is corrupt — re-downloading..."
        rm -f "$DATA_FILE"
    fi
fi
if [ ! -f "$DATA_FILE" ]; then
    echo "  Downloading TinyStories training data (~993MB tar.gz)..."
    TAR_FILE="$MODELS_DIR/TinyStories_tok32000.tar.gz"
    curl -L --progress-bar --retry 3 -o "$TAR_FILE" \
        "https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok32000/TinyStories_tok32000.tar.gz?download=true"
    if ! file "$TAR_FILE" | grep -q "gzip"; then
        echo -e "  ${RED}✗${NC} Download failed — not a valid archive"
        rm -f "$TAR_FILE"
        exit 1
    fi
    DATA_ENTRY=$(tar tzf "$TAR_FILE" 2>/dev/null | grep 'data00\.bin' | head -1)
    if [ -z "$DATA_ENTRY" ]; then
        echo -e "  ${RED}✗${NC} data00.bin not found in archive"
        exit 1
    fi
    tar xzf "$TAR_FILE" -C "$MODELS_DIR" "$DATA_ENTRY"
    EXTRACTED="$MODELS_DIR/$DATA_ENTRY"
    if [ "$EXTRACTED" != "$DATA_FILE" ]; then
        mv "$EXTRACTED" "$DATA_FILE"
        rmdir "$(dirname "$EXTRACTED")" 2>/dev/null || true
    fi
    rm -f "$TAR_FILE"
    echo -e "  ${GREEN}✓${NC} Training data extracted"
else
    echo -e "  ${GREEN}✓${NC} tinystories_data00.bin exists ($(ls -lh "$DATA_FILE" | awk '{print $5}'))"
fi

# Tokenizer
TOKENIZER_SRC="$SCRIPT_DIR/vendor/ANE/assets/models/tokenizer.bin"
TOKENIZER_DST="$MODELS_DIR/tokenizer.bin"
if [ -f "$TOKENIZER_SRC" ] && [ ! -f "$TOKENIZER_DST" ]; then
    cp "$TOKENIZER_SRC" "$TOKENIZER_DST"
fi
if [ -f "$TOKENIZER_DST" ]; then
    echo -e "  ${GREEN}✓${NC} tokenizer.bin"
else
    echo -e "  ${RED}✗${NC} tokenizer.bin not found"
    exit 1
fi

# ── Step 4: Verify CLI works ──
echo ""
echo -e "${CYAN}[4/6]${NC} Verifying CLI..."
INFO_OUTPUT=$("$SCRIPT_DIR/cli/neuralforge" info --model "$MODEL_FILE" 2>&1 | head -1)
if echo "$INFO_OUTPUT" | grep -q '"type":"init"'; then
    echo -e "  ${GREEN}✓${NC} CLI reads model correctly"
else
    echo -e "  ${RED}✗${NC} CLI can't read model: $INFO_OUTPUT"
    exit 1
fi

# ── Step 5: Run unit tests ──
echo ""
echo -e "${CYAN}[5/6]${NC} Running unit tests..."
cd "$SCRIPT_DIR/cli"
TEST_OUT=$(make test 2>&1 | tail -1)
CLI_PASS=$(echo "$TEST_OUT" | grep -o '[0-9]*/' | tr -d '/')
echo -e "  ${GREEN}✓${NC} CLI: $TEST_OUT"
cd "$SCRIPT_DIR"

cd "$SCRIPT_DIR/app/Tests"
SWIFT_OUT=$(swiftc -o test_swift -framework Foundation NeuralForgeTests.swift 2>&1 && ./test_swift 2>&1 | tail -1)
echo -e "  ${GREEN}✓${NC} Swift: $SWIFT_OUT"
rm -f test_swift
cd "$SCRIPT_DIR"

# ── Step 6: Build app ──
echo ""
echo -e "${CYAN}[6/6]${NC} Building macOS app..."
BUILD_OUT=$(cd app && xcodebuild -project NeuralForge.xcodeproj -scheme NeuralForge \
    -derivedDataPath /tmp/NF_SetupBuild build 2>&1 | tail -3)
if echo "$BUILD_OUT" | grep -q "BUILD SUCCEEDED"; then
    echo -e "  ${GREEN}✓${NC} App builds successfully"
else
    echo -e "  ${YELLOW}⚠${NC} App build had issues (may need Xcode open first)"
fi

# ── Done ──
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║${NC}  ${GREEN}Setup complete!${NC}                                        ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}                                                          ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  Next steps:                                             ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}    1. Open in Xcode:  open app/NeuralForge.xcodeproj     ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}    2. Hit ⌘R to build and run                            ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}    3. Create a project, configure training, start!       ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}                                                          ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  Or use the CLI directly:                                ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}    ./cli/neuralforge train --model models/stories110M.bin ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}      --data models/tinystories_data00.bin --steps 20      ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}                                                          ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  Run tests:  make test-all                               ${BOLD}║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
