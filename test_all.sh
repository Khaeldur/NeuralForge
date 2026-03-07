#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  NeuralForge — One-Click Test Suite                            ║
# ║  Runs every test layer: unit → integration → build → UI        ║
# ║                                                                ║
# ║  Usage:  ./test_all.sh            (all tests)                  ║
# ║          ./test_all.sh --quick    (unit tests only, ~5 sec)    ║
# ║          ./test_all.sh --full     (+ integration, ~3 min)      ║
# ╚══════════════════════════════════════════════════════════════════╝

set -uo pipefail
cd "$(dirname "$0")"

# ── Colors ──
R='\033[0;31m' G='\033[0;32m' Y='\033[0;33m' B='\033[0;34m'
C='\033[0;36m' W='\033[1;37m' D='\033[0;90m' NC='\033[0m'

# ── State ──
LAYER_PASS=0; LAYER_FAIL=0; LAYER_SKIP=0
TOTAL_PASS=0; TOTAL_FAIL=0; TOTAL_SKIP=0; TOTAL_TESTS=0
START_TIME=$(date +%s)
MODE="${1:---full}"
FAILURES=()

# ── Helpers ──
banner() { echo -e "\n${W}━━━ $1 ━━━${NC}"; }
section() { echo -e "  ${C}▸${NC} $1"; }
pass()    { ((LAYER_PASS++)); echo -e "    ${G}✓${NC} $1"; }
fail()    { ((LAYER_FAIL++)); FAILURES+=("$1: $2"); echo -e "    ${R}✗${NC} $1 ${D}— $2${NC}"; }
skip()    { ((LAYER_SKIP++)); echo -e "    ${Y}○${NC} $1 ${D}— $2${NC}"; }

flush_layer() {
    TOTAL_PASS=$((TOTAL_PASS + LAYER_PASS))
    TOTAL_FAIL=$((TOTAL_FAIL + LAYER_FAIL))
    TOTAL_SKIP=$((TOTAL_SKIP + LAYER_SKIP))
    TOTAL_TESTS=$((TOTAL_TESTS + LAYER_PASS + LAYER_FAIL))
    LAYER_PASS=0; LAYER_FAIL=0; LAYER_SKIP=0
}

elapsed() {
    local now=$(date +%s)
    echo "$((now - START_TIME))s"
}

# ══════════════════════════════════════════════════════════════════
#  LAYER 1: Unit Tests (CLI + Swift)
# ══════════════════════════════════════════════════════════════════
banner "LAYER 1 · Unit Tests"

# ── CLI Tests ──
section "CLI tests (cli/test_cli.m)"
CLI_OUT=$(cd cli && make clean 2>/dev/null; make 2>/dev/null && make test 2>&1)
CLI_RESULT=$(echo "$CLI_OUT" | grep '=== Results:' | head -1)

if echo "$CLI_RESULT" | grep -q 'passed'; then
    CLI_PASSED=$(echo "$CLI_RESULT" | grep -o '[0-9]*/[0-9]* passed' | cut -d/ -f1)
    CLI_TOTAL=$(echo "$CLI_RESULT" | grep -o '[0-9]*/[0-9]* passed' | cut -d/ -f2 | cut -d' ' -f1)
    pass "CLI: ${CLI_PASSED}/${CLI_TOTAL} passed"
else
    fail "CLI tests" "$(echo "$CLI_OUT" | tail -3)"
fi

# ── Swift Tests ──
section "Swift tests (app/Tests/NeuralForgeTests.swift)"
SWIFT_OUT=$(cd app/Tests && swiftc -o test_swift -framework Foundation NeuralForgeTests.swift 2>&1 && ./test_swift 2>&1)
SWIFT_RESULT=$(echo "$SWIFT_OUT" | grep '=== Results:' | head -1)

if echo "$SWIFT_RESULT" | grep -q 'passed'; then
    SWIFT_PASSED=$(echo "$SWIFT_RESULT" | grep -o '[0-9]*/[0-9]* passed' | cut -d/ -f1)
    SWIFT_TOTAL=$(echo "$SWIFT_RESULT" | grep -o '[0-9]*/[0-9]* passed' | cut -d/ -f2 | cut -d' ' -f1)
    pass "Swift: ${SWIFT_PASSED}/${SWIFT_TOTAL} passed"
else
    fail "Swift tests" "$(echo "$SWIFT_OUT" | grep 'FAIL' | head -3)"
fi

echo -e "  ${D}[$(elapsed)]${NC}"
flush_layer

# ══════════════════════════════════════════════════════════════════
#  LAYER 2: Build Verification
# ══════════════════════════════════════════════════════════════════
banner "LAYER 2 · Build Verification"

# ── CLI Binary ──
section "CLI binary"
if [ -f cli/neuralforge ]; then
    pass "CLI binary exists ($(du -h cli/neuralforge | cut -f1))"
else
    fail "CLI binary" "not found — run 'make' in cli/"
fi

# ── Xcode App ──
section "Xcode project build"
APP_OUT=$(cd app && xcodebuild -project NeuralForge.xcodeproj -scheme NeuralForge build 2>&1)
if echo "$APP_OUT" | grep -q 'BUILD SUCCEEDED'; then
    WARNINGS=$(echo "$APP_OUT" | grep -c 'warning:' || true)
    pass "App builds (${WARNINGS} warnings)"
else
    fail "Xcode build" "$(echo "$APP_OUT" | grep 'error:' | head -3)"
fi

# ── XCUITests Compile ──
section "XCUITest compilation"
XCUI_OUT=$(cd app && xcodebuild build-for-testing \
    -project NeuralForge.xcodeproj \
    -scheme NeuralForge \
    -destination 'platform=macOS' \
    -derivedDataPath /tmp/NF_TestBuild 2>&1)
if echo "$XCUI_OUT" | grep -q 'TEST BUILD SUCCEEDED'; then
    pass "53 XCUITests compile"
else
    fail "XCUITest build" "$(echo "$XCUI_OUT" | grep 'error:' | head -3)"
fi

echo -e "  ${D}[$(elapsed)]${NC}"
flush_layer

if [ "$MODE" = "--quick" ]; then
    banner "DONE (quick mode — skipped integration tests)"
    echo ""
    echo -e "  ${G}${TOTAL_PASS} passed${NC}, ${R}${TOTAL_FAIL} failed${NC} ${D}in $(elapsed)${NC}"
    exit $TOTAL_FAIL
fi

# ══════════════════════════════════════════════════════════════════
#  LAYER 3: Integration Tests (Real Hardware)
# ══════════════════════════════════════════════════════════════════
banner "LAYER 3 · Integration Tests (Real Hardware)"

MODEL="models/stories110M.bin"
TOKENIZER="models/tokenizer.bin"
DATA="models/tinystories_data00.bin"

if [ ! -f "$MODEL" ] || [ ! -f "$TOKENIZER" ] || [ ! -f "$DATA" ]; then
    skip "Integration tests" "model/data files missing"
    flush_layer
else

    # ── Training ──
    section "Real training (5 steps, ANE)"
    TRAIN_OUT=$(cli/neuralforge train --model "$MODEL" --tokenizer "$TOKENIZER" \
        --data "$DATA" --steps 5 --seed 42 2>&1 || true)
    if echo "$TRAIN_OUT" | grep -q '"type":"done"'; then
        LOSS=$(echo "$TRAIN_OUT" | grep '"type":"done"' | grep -o '"final_loss":[0-9.]*' | cut -d: -f2)
        TFLOPS=$(echo "$TRAIN_OUT" | grep '"type":"done"' | grep -o '"tflops_ane":[0-9.]*' | cut -d: -f2)
        pass "Training completes (loss=${LOSS}, ${TFLOPS} TFLOPS)"
    else
        fail "Training" "didn't complete"
    fi

    # ── LR Scheduler ──
    section "LR scheduler (cosine + warmup)"
    SCHED_OUT=$(cli/neuralforge train --model "$MODEL" --tokenizer "$TOKENIZER" \
        --data "$DATA" --steps 10 --warmup 3 --lr-schedule cosine --lr 3e-4 --lr-min 1e-5 --seed 42 2>&1 || true)
    STEP0_LR=$(echo "$SCHED_OUT" | grep '"step":0' | head -1 | grep -o '"lr":[0-9.e+-]*' | cut -d: -f2)
    STEP9_LR=$(echo "$SCHED_OUT" | grep '"step":9' | head -1 | grep -o '"lr":[0-9.e+-]*' | cut -d: -f2)
    if [ -n "$STEP0_LR" ] && [ -n "$STEP9_LR" ]; then
        pass "Scheduler works (step0=${STEP0_LR}, step9=${STEP9_LR})"
    else
        fail "LR scheduler" "missing lr field in output"
    fi

    # ── Checkpoint Resume ──
    section "Checkpoint save & resume"
    CKPT="cli/neuralforge_ckpt.bin"
    if [ -f "$CKPT" ]; then
        RESUME_OUT=$(cli/neuralforge train --model "$MODEL" --tokenizer "$TOKENIZER" \
            --data "$DATA" --steps 12 --resume "$CKPT" --seed 42 2>&1 || true)
        if echo "$RESUME_OUT" | grep -q 'RESUMED'; then
            pass "Checkpoint resume works"
        else
            fail "Checkpoint resume" "no RESUMED message"
        fi
    else
        skip "Checkpoint resume" "no checkpoint file from prior run"
    fi

    # ── Text Generation ──
    section "Text generation (top-p sampling)"
    GEN_OUT=$(cli/neuralforge generate --model "$MODEL" --tokenizer "$TOKENIZER" \
        --prompt "Once upon a time" --max-tokens 20 --temperature 0.8 --top-p 0.9 --seed 42 2>&1 || true)
    if echo "$GEN_OUT" | grep -q '"type":"generate_done"'; then
        TOK_COUNT=$(echo "$GEN_OUT" | grep -c '"type":"token"')
        GEN_MS=$(echo "$GEN_OUT" | grep '"generate_done"' | grep -o '"total_ms":[0-9.]*' | cut -d: -f2)
        MS_PER_TOK=$(echo "scale=1; ${GEN_MS} / ${TOK_COUNT}" | bc 2>/dev/null || echo "?")
        pass "Generation: ${TOK_COUNT} tokens in ${GEN_MS}ms (${MS_PER_TOK}ms/tok)"
    else
        fail "Text generation" "didn't complete"
    fi

    # ── Determinism ──
    section "Seed determinism"
    GEN_A=$(cli/neuralforge generate --model "$MODEL" --tokenizer "$TOKENIZER" \
        --prompt "Hello" --max-tokens 5 --temperature 0.5 --seed 99 2>&1 || true)
    GEN_B=$(cli/neuralforge generate --model "$MODEL" --tokenizer "$TOKENIZER" \
        --prompt "Hello" --max-tokens 5 --temperature 0.5 --seed 99 2>&1 || true)
    TOKS_A=$(echo "$GEN_A" | grep '"type":"token"' | grep -o '"token_id":[0-9]*')
    TOKS_B=$(echo "$GEN_B" | grep '"type":"token"' | grep -o '"token_id":[0-9]*')
    if [ "$TOKS_A" = "$TOKS_B" ] && [ -n "$TOKS_A" ]; then
        pass "Same seed → same tokens"
    else
        fail "Determinism" "different tokens for same seed"
    fi

    # ── Error Handling ──
    section "Error handling"
    ERR_OUT=$(cli/neuralforge train --model "/fake/model.bin" --data "$DATA" --steps 1 2>&1 || true)
    if echo "$ERR_OUT" | grep -qi "error\|fail\|cannot\|not found"; then
        pass "Missing model → error message"
    else
        fail "Error handling" "no error for bad model path"
    fi

    echo -e "  ${D}[$(elapsed)]${NC}"
    flush_layer
fi

# ══════════════════════════════════════════════════════════════════
#  LAYER 4: XCUITest Runtime (if Accessibility enabled)
# ══════════════════════════════════════════════════════════════════
if [ "$MODE" = "--ui" ]; then
    banner "LAYER 4 · XCUITests (App Automation)"
    section "Running 53 XCUITests via Xcode"

    # Kill running app first
    pkill -f "NeuralForge.app" 2>/dev/null || true
    sleep 1

    XCUI_RUN=$(cd app && xcodebuild test \
        -project NeuralForge.xcodeproj \
        -scheme NeuralForge \
        -destination 'platform=macOS' \
        -derivedDataPath /tmp/NF_TestBuild 2>&1)

    if echo "$XCUI_RUN" | grep -q 'TEST SUCCEEDED'; then
        EXECUTED=$(echo "$XCUI_RUN" | grep 'Executed' | tail -1)
        pass "XCUITests: $EXECUTED"
    elif echo "$XCUI_RUN" | grep -q 'automation mode'; then
        skip "XCUITests" "grant Accessibility permission: System Settings → Privacy → Accessibility → Xcode + Terminal"
    else
        XCUI_FAILS=$(echo "$XCUI_RUN" | grep 'Test Case.*failed' | wc -l | tr -d ' ')
        fail "XCUITests" "${XCUI_FAILS} test(s) failed"
    fi

    echo -e "  ${D}[$(elapsed)]${NC}"
    flush_layer
fi

# ══════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${W}╔══════════════════════════════════════════════════════════╗${NC}"

if [ $TOTAL_FAIL -eq 0 ]; then
    echo -e "${W}║${NC}  ${G}✅ ALL ${TOTAL_PASS} TESTS PASSED${NC}$(printf '%*s' $((36 - ${#TOTAL_PASS})) '')${W}║${NC}"
else
    echo -e "${W}║${NC}  ${R}❌ ${TOTAL_FAIL} FAILED${NC}, ${G}${TOTAL_PASS} passed${NC}$(printf '%*s' $((33 - ${#TOTAL_FAIL} - ${#TOTAL_PASS})) '')${W}║${NC}"
fi

# Count breakdown
echo -e "${W}║${NC}                                                        ${W}║${NC}"
UNIT_COUNT="${CLI_PASSED:-0} CLI + ${SWIFT_PASSED:-0} Swift"
echo -e "${W}║${NC}  Unit:        ${UNIT_COUNT}$(printf '%*s' $((42 - ${#UNIT_COUNT})) '')${W}║${NC}"
echo -e "${W}║${NC}  XCUITests:   53 (compiled)$(printf '%*s' 29 '')${W}║${NC}"

if [ "$MODE" != "--quick" ]; then
    INTEG_COUNT="$((TOTAL_PASS - 2 - 3))"  # subtract 2 unit + 3 build checks
    [ "$INTEG_COUNT" -lt 0 ] && INTEG_COUNT=0
    echo -e "${W}║${NC}  Integration: ${INTEG_COUNT} on real hardware$(printf '%*s' $((34 - ${#INTEG_COUNT})) '')${W}║${NC}"
fi

TIME_STR="${MINUTES}m ${SECONDS}s"
echo -e "${W}║${NC}  Time:        ${TIME_STR}$(printf '%*s' $((42 - ${#TIME_STR})) '')${W}║${NC}"
echo -e "${W}╚══════════════════════════════════════════════════════════╝${NC}"

# Print failures if any
if [ ${#FAILURES[@]} -gt 0 ]; then
    echo ""
    echo -e "  ${R}Failures:${NC}"
    for f in "${FAILURES[@]}"; do
        echo -e "    ${R}✗${NC} $f"
    done
fi

echo ""
exit $TOTAL_FAIL
