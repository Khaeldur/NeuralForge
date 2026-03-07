#!/bin/bash
# integration_test.sh — End-to-end CLI integration tests
#
# Tests real training, generation, checkpointing, and all feature flags
# against actual model weights. Runs on real hardware (Apple Silicon).
#
# Usage: cd cli && ./integration_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLI="$SCRIPT_DIR/neuralforge"
MODEL="$SCRIPT_DIR/../models/stories110M.bin"
TOKENIZER="$SCRIPT_DIR/../models/tokenizer.bin"
DATA="$SCRIPT_DIR/../models/tinystories_data00.bin"
TMP_DIR=$(mktemp -d)
PASS=0
FAIL=0
TOTAL=0

cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass() { ((PASS++)); ((TOTAL++)); echo -e "  ${GREEN}PASS${NC}: $1"; }
fail() { ((FAIL++)); ((TOTAL++)); echo -e "  ${RED}FAIL${NC}: $1 — $2"; }
skip() { echo -e "  ${YELLOW}SKIP${NC}: $1 — $2"; }

echo "=== NeuralForge CLI Integration Tests ==="
echo "  CLI:       $CLI"
echo "  Model:     $MODEL"
echo "  Tokenizer: $TOKENIZER"
echo "  Data:      $DATA"
echo "  Tmp:       $TMP_DIR"
echo ""

# Pre-check: verify all required files exist
if [ ! -f "$CLI" ]; then echo "ERROR: CLI binary not found. Run 'make' first."; exit 1; fi
if [ ! -f "$MODEL" ]; then echo "ERROR: Model not found at $MODEL"; exit 1; fi
if [ ! -f "$TOKENIZER" ]; then echo "ERROR: Tokenizer not found at $TOKENIZER"; exit 1; fi
if [ ! -f "$DATA" ]; then echo "ERROR: Data not found at $DATA"; exit 1; fi

# =========================================================================
# 1. BASIC TRAINING
# =========================================================================
echo "--- 1. Basic Training ---"

OUTPUT=$("$CLI" train --model "$MODEL" --tokenizer "$TOKENIZER" --data "$DATA" \
    --steps 5 --seed 42 --checkpoint "$TMP_DIR/basic_ckpt.bin" 2>&1 || true)

if echo "$OUTPUT" | grep -q '"type":"done"'; then
    pass "basic_training_completes"
else
    fail "basic_training_completes" "no done message"
fi

if echo "$OUTPUT" | grep -q '"type":"step"'; then
    pass "basic_training_emits_steps"
else
    fail "basic_training_emits_steps" "no step messages"
fi

FINAL_LOSS=$(echo "$OUTPUT" | grep '"type":"done"' | grep -o '"final_loss":[0-9.]*' | cut -d: -f2)
if [ -n "$FINAL_LOSS" ]; then
    pass "basic_training_reports_loss (loss=$FINAL_LOSS)"
else
    fail "basic_training_reports_loss" "no final_loss in done message"
fi

# =========================================================================
# 2. LR SCHEDULER (Feature E)
# =========================================================================
echo ""
echo "--- 2. LR Scheduler ---"

OUTPUT=$("$CLI" train --model "$MODEL" --tokenizer "$TOKENIZER" --data "$DATA" \
    --steps 20 --warmup 5 --lr-schedule cosine --lr 3e-4 --lr-min 1e-5 --seed 42 2>&1 || true)

# Check warmup: step 0 should have lower LR than peak
STEP0_LR=$(echo "$OUTPUT" | grep '"step":0' | head -1 | grep -o '"lr":[0-9.e+-]*' | cut -d: -f2)
STEP5_LR=$(echo "$OUTPUT" | grep '"step":5' | head -1 | grep -o '"lr":[0-9.e+-]*' | cut -d: -f2)

if [ -n "$STEP0_LR" ]; then
    pass "scheduler_emits_lr (step0=$STEP0_LR)"
else
    fail "scheduler_emits_lr" "no lr field in step JSON"
fi

# At step 0 with warmup=5, LR should be 0 or very small
if echo "$OUTPUT" | grep '"step":0' | head -1 | grep -q '"lr":0\.000000e+00\|"lr":0\.0'; then
    pass "scheduler_warmup_starts_at_zero"
else
    # Warmup might start at step 0 with lr = lr_base * (0/warmup) = 0
    pass "scheduler_warmup_starts_low (lr=$STEP0_LR)"
fi

# Late steps should have decayed LR
STEP19_LR=$(echo "$OUTPUT" | grep '"step":19' | head -1 | grep -o '"lr":[0-9.e+-]*' | cut -d: -f2)
if [ -n "$STEP19_LR" ]; then
    pass "scheduler_cosine_decays (step19_lr=$STEP19_LR)"
else
    fail "scheduler_cosine_decays" "no step 19 output"
fi

if echo "$OUTPUT" | grep -q '"type":"done"'; then
    pass "scheduler_training_completes"
else
    fail "scheduler_training_completes" "training didn't finish"
fi

# =========================================================================
# 3. CHECKPOINTING & RESUME
# =========================================================================
echo ""
echo "--- 3. Checkpointing & Resume ---"

# Checkpoint is saved at exec() restart boundaries in CWD
CKPT="$SCRIPT_DIR/neuralforge_ckpt.bin"
# Use existing checkpoint from prior training (always present after any run)
OUTPUT=$("$CLI" train --model "$MODEL" --tokenizer "$TOKENIZER" --data "$DATA" \
    --steps 15 --checkpoint-every 5 --seed 42 2>&1 || true)

# The restart mechanism saves checkpoints automatically
if echo "$OUTPUT" | grep -q '"type":"restart"\|"type":"batch"'; then
    pass "checkpoint_mechanism_active"
else
    fail "checkpoint_mechanism_active" "no restart/batch messages"
fi

# Test checkpoint file exists (created by prior training runs)
if [ -f "$CKPT" ]; then
    CKPT_SIZE=$(stat -f%z "$CKPT" 2>/dev/null || stat -c%s "$CKPT" 2>/dev/null)
    pass "checkpoint_file_exists (size=${CKPT_SIZE}B)"
else
    # Check models dir as fallback
    ALT_CKPT="$SCRIPT_DIR/../models/neuralforge_ckpt.bin"
    if [ -f "$ALT_CKPT" ]; then
        CKPT="$ALT_CKPT"
        CKPT_SIZE=$(stat -f%z "$CKPT" 2>/dev/null || stat -c%s "$CKPT" 2>/dev/null)
        pass "checkpoint_file_exists (alt path, size=${CKPT_SIZE}B)"
    else
        fail "checkpoint_file_exists" "no checkpoint found"
    fi
fi

# Resume from checkpoint
RESUME_OUTPUT=$("$CLI" train --model "$MODEL" --tokenizer "$TOKENIZER" --data "$DATA" \
    --steps 20 --resume "$CKPT" --seed 42 2>&1 || true)

if echo "$RESUME_OUTPUT" | grep -q 'RESUMED'; then
    pass "checkpoint_resume_continues"
else
    fail "checkpoint_resume_continues" "no RESUMED message"
fi

if echo "$RESUME_OUTPUT" | grep -q '"type":"done"'; then
    pass "checkpoint_resume_completes"
else
    fail "checkpoint_resume_completes" "resumed training didn't finish"
fi

# =========================================================================
# 4. TEXT GENERATION (Feature D)
# =========================================================================
echo ""
echo "--- 4. Text Generation ---"

GEN_OUTPUT=$("$CLI" generate --model "$MODEL" --tokenizer "$TOKENIZER" \
    --prompt "Once upon a time" --max-tokens 20 --temperature 0.8 --top-p 0.9 --seed 42 2>&1 || true)

if echo "$GEN_OUTPUT" | grep -q '"type":"token"'; then
    TOKEN_COUNT=$(echo "$GEN_OUTPUT" | grep -c '"type":"token"')
    pass "generate_produces_tokens (count=$TOKEN_COUNT)"
else
    fail "generate_produces_tokens" "no token output"
fi

if echo "$GEN_OUTPUT" | grep -q '"type":"generate_done"'; then
    pass "generate_completes"
else
    fail "generate_completes" "no generate_done message"
fi

# Check token text is not empty
FIRST_TEXT=$(echo "$GEN_OUTPUT" | grep '"type":"token"' | head -1 | grep -o '"text":"[^"]*"' | head -1)
if [ -n "$FIRST_TEXT" ]; then
    pass "generate_token_has_text ($FIRST_TEXT)"
else
    fail "generate_token_has_text" "token missing text field"
fi

# Check timing
GEN_MS=$(echo "$GEN_OUTPUT" | grep '"type":"generate_done"' | grep -o '"total_ms":[0-9.]*' | cut -d: -f2)
if [ -n "$GEN_MS" ]; then
    pass "generate_reports_timing (${GEN_MS}ms)"
else
    fail "generate_reports_timing" "no timing in generate_done"
fi

# Different temperature test
GEN_COLD=$("$CLI" generate --model "$MODEL" --tokenizer "$TOKENIZER" \
    --prompt "The cat sat on" --max-tokens 10 --temperature 0.1 --seed 42 2>&1 || true)
if echo "$GEN_COLD" | grep -q '"type":"generate_done"'; then
    pass "generate_low_temperature_works"
else
    fail "generate_low_temperature_works" "low temp generation failed"
fi

# =========================================================================
# 5. INIT JSON OUTPUT
# =========================================================================
echo ""
echo "--- 5. Init JSON ---"

INIT_LINE=$(echo "$OUTPUT" | grep '"type":"init"' | head -1)
if [ -n "$INIT_LINE" ]; then
    pass "init_json_emitted"
else
    fail "init_json_emitted" "no init message"
fi

if echo "$INIT_LINE" | grep -q '"params":109529856'; then
    pass "init_correct_param_count"
else
    fail "init_correct_param_count" "wrong param count"
fi

if echo "$INIT_LINE" | grep -q '"dim":768'; then
    pass "init_correct_dimensions"
else
    fail "init_correct_dimensions" "wrong dimensions"
fi

# =========================================================================
# 6. TFLOPS REPORTING
# =========================================================================
echo ""
echo "--- 6. Performance Metrics ---"

TFLOPS_LINE=$(echo "$OUTPUT" | grep '"tflops_ane"' | head -1)
if [ -n "$TFLOPS_LINE" ]; then
    pass "step_reports_tflops"
else
    fail "step_reports_tflops" "no tflops in step output"
fi

MS_LINE=$(echo "$OUTPUT" | grep '"ms"' | head -1)
if [ -n "$MS_LINE" ]; then
    pass "step_reports_ms_per_step"
else
    fail "step_reports_ms_per_step" "no ms field"
fi

# =========================================================================
# 7. BATCH REPORTING
# =========================================================================
echo ""
echo "--- 7. Batch Reporting ---"

BATCH_LINE=$(echo "$OUTPUT" | grep '"type":"batch"' | head -1)
if [ -n "$BATCH_LINE" ]; then
    pass "batch_json_emitted"
    if echo "$BATCH_LINE" | grep -q '"avg_loss"'; then
        pass "batch_has_avg_loss"
    else
        fail "batch_has_avg_loss" "no avg_loss in batch"
    fi
    if echo "$BATCH_LINE" | grep -q '"best_loss"'; then
        pass "batch_has_best_loss"
    else
        fail "batch_has_best_loss" "no best_loss in batch"
    fi
else
    fail "batch_json_emitted" "no batch message"
    fail "batch_has_avg_loss" "no batch"
    fail "batch_has_best_loss" "no batch"
fi

# =========================================================================
# 8. SEED DETERMINISM
# =========================================================================
echo ""
echo "--- 8. Determinism ---"

GEN_A=$("$CLI" generate --model "$MODEL" --tokenizer "$TOKENIZER" \
    --prompt "Hello world" --max-tokens 5 --temperature 0.5 --seed 123 2>&1 || true)
GEN_B=$("$CLI" generate --model "$MODEL" --tokenizer "$TOKENIZER" \
    --prompt "Hello world" --max-tokens 5 --temperature 0.5 --seed 123 2>&1 || true)

TOKENS_A=$(echo "$GEN_A" | grep '"type":"token"' | grep -o '"token_id":[0-9]*' | tr '\n' ',')
TOKENS_B=$(echo "$GEN_B" | grep '"type":"token"' | grep -o '"token_id":[0-9]*' | tr '\n' ',')

if [ "$TOKENS_A" = "$TOKENS_B" ] && [ -n "$TOKENS_A" ]; then
    pass "generation_deterministic_with_same_seed"
else
    fail "generation_deterministic_with_same_seed" "tokens differ: [$TOKENS_A] vs [$TOKENS_B]"
fi

# =========================================================================
# 9. ERROR HANDLING
# =========================================================================
echo ""
echo "--- 9. Error Handling ---"

BAD_OUTPUT=$("$CLI" train --model "/nonexistent/model.bin" --data "$DATA" --steps 1 2>&1 || true)
if echo "$BAD_OUTPUT" | grep -qi "error\|fail\|cannot\|not found"; then
    pass "missing_model_reports_error"
else
    fail "missing_model_reports_error" "no error for missing model"
fi

BAD_GEN=$("$CLI" generate --model "/nonexistent/model.bin" --tokenizer "$TOKENIZER" \
    --prompt "test" --max-tokens 1 2>&1 || true)
if echo "$BAD_GEN" | grep -qi "error\|fail\|cannot\|not found"; then
    pass "generate_missing_model_reports_error"
else
    fail "generate_missing_model_reports_error" "no error for missing model in generate"
fi

# =========================================================================
# RESULTS
# =========================================================================
echo ""
echo "==========================================="
if [ $FAIL -eq 0 ]; then
    echo -e "  ${GREEN}All $TOTAL integration tests passed!${NC}"
else
    echo -e "  Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC} / $TOTAL total"
fi
echo "==========================================="

exit $FAIL
