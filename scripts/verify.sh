#!/bin/bash
# NeuralForge — System Health Check
# Run this to verify everything is working before starting development

set -e
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

check() {
    if eval "$2" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} $1"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗${NC} $1"
        FAIL=$((FAIL + 1))
    fi
}

warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

echo "=== NeuralForge Health Check ==="
echo ""

# 1. Check prerequisites
echo "Prerequisites:"
check "Xcode CLI tools" "xcode-select -p"
check "Apple Silicon" "sysctl -n machdep.cpu.brand_string | grep -qi 'apple\|m[1-4]'"
check "Python 3" "python3 --version"
check "numpy" "python3 -c 'import numpy'"

echo ""

# 2. Check model files
echo "Model files:"
if [ -f models/stories110M.bin ]; then
    SIZE=$(stat -f%z models/stories110M.bin 2>/dev/null || stat -c%s models/stories110M.bin 2>/dev/null)
    if [ "$SIZE" -gt 1000 ]; then
        check "stories110M.bin ($(du -h models/stories110M.bin | cut -f1))" "true"
    else
        echo -e "  ${RED}✗${NC} stories110M.bin is corrupt ($SIZE bytes — probably Git LFS placeholder)"
        FAIL=$((FAIL + 1))
    fi
else
    echo -e "  ${RED}✗${NC} stories110M.bin not found — run: bash scripts/download_model.sh"
    FAIL=$((FAIL + 1))
fi

if [ -f models/tokenizer.bin ]; then
    SIZE=$(stat -f%z models/tokenizer.bin 2>/dev/null || stat -c%s models/tokenizer.bin 2>/dev/null)
    if [ "$SIZE" -gt 1000 ]; then
        check "tokenizer.bin ($(du -h models/tokenizer.bin | cut -f1))" "true"
    else
        echo -e "  ${RED}✗${NC} tokenizer.bin is corrupt ($SIZE bytes)"
        FAIL=$((FAIL + 1))
    fi
else
    echo -e "  ${RED}✗${NC} tokenizer.bin not found"
    FAIL=$((FAIL + 1))
fi

if [ -f models/tinystories_data00.bin ]; then
    SIZE=$(stat -f%z models/tinystories_data00.bin 2>/dev/null || stat -c%s models/tinystories_data00.bin 2>/dev/null)
    if [ "$SIZE" -gt 1000 ]; then
        check "tinystories_data00.bin ($(du -h models/tinystories_data00.bin | cut -f1))" "true"
    else
        warn "tinystories_data00.bin is small ($SIZE bytes) — may need regeneration"
    fi
else
    warn "tinystories_data00.bin not found — training will need data"
fi

echo ""

# 3. Build and test
echo "CLI:"
check "CLI builds" "cd cli && make clean > /dev/null 2>&1 && make"
check "CLI tests pass" "cd cli && make test"

echo ""
echo "Swift:"
check "Swift tests pass" "cd app/Tests && swiftc -o test_swift -framework Foundation NeuralForgeTests.swift && ./test_swift"

echo ""
echo "App:"
check "Xcode project builds" "cd app && xcodebuild -project NeuralForge.xcodeproj -scheme NeuralForge build 2>&1 | grep -q 'BUILD SUCCEEDED'"

echo ""

# 4. Quick sanity test (only if CLI built and model exists)
if [ -f cli/neuralforge ] && [ -f models/stories110M.bin ] && [ -f models/tinystories_data00.bin ]; then
    echo "Sanity:"
    SIZE=$(stat -f%z models/tinystories_data00.bin 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 1000 ]; then
        check "Training runs (2 steps)" "cli/neuralforge train --model models/stories110M.bin --data models/tinystories_data00.bin --steps 2 2>/dev/null | grep -q 'step'"
    else
        warn "Skipping training sanity — data file too small"
    fi
fi

echo ""
echo "==============================="
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"

if [ $FAIL -gt 0 ]; then
    echo -e "${RED}Some checks failed. Fix issues above before starting work.${NC}"
    exit 1
else
    echo -e "${GREEN}All systems go! Ready to develop.${NC}"
    exit 0
fi
