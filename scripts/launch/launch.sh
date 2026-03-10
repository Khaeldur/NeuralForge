#!/bin/bash
# NeuralForge Launch Automation — Master Script
# Usage: bash scripts/launch/launch.sh [phase]
#
# Phases:
#   prep     — Record GIF, capture screenshots, generate posts, update README
#   posts    — Open all post files for copy-pasting
#   submit   — Open browser tabs for each platform (you paste + submit)
#   metrics  — Track current GitHub metrics
#   all      — Run prep → posts → submit
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PHASE="${1:-all}"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   🚀 NeuralForge Launch Automation           ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
echo ""

do_prep() {
    echo -e "${BOLD}━━━ PHASE 1: PREP ━━━${NC}"
    echo ""

    # Step 1: Generate post copy
    echo -e "${CYAN}[1/4]${NC} Generating platform-specific posts..."
    bash "$SCRIPT_DIR/generate_posts.sh"
    echo ""

    # Step 2: Record demo GIF
    echo -e "${CYAN}[2/4]${NC} Record demo GIF?"
    if [ -f "$ROOT/assets/demo.gif" ]; then
        echo -e "  ${GREEN}✓${NC} demo.gif already exists ($(du -h "$ROOT/assets/demo.gif" | awk '{print $1}'))"
        echo -n "  Re-record? [y/N]: "
        read -r response
        [ "$response" = "y" ] && bash "$SCRIPT_DIR/record_demo.sh"
    else
        echo -n "  Record now? [Y/n]: "
        read -r response
        [ "$response" != "n" ] && bash "$SCRIPT_DIR/record_demo.sh"
    fi
    echo ""

    # Step 3: Capture screenshots
    echo -e "${CYAN}[3/4]${NC} Capture screenshots?"
    SCREEN_COUNT=$(ls "$ROOT/assets/screenshots"/*.png 2>/dev/null | wc -l | tr -d ' ')
    if [ "$SCREEN_COUNT" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} $SCREEN_COUNT screenshots already exist"
        echo -n "  Retake? [y/N]: "
        read -r response
        [ "$response" = "y" ] && bash "$SCRIPT_DIR/capture_screenshots.sh"
    else
        echo -n "  Capture now? [Y/n]: "
        read -r response
        [ "$response" != "n" ] && bash "$SCRIPT_DIR/capture_screenshots.sh"
    fi
    echo ""

    # Step 4: Update README
    echo -e "${CYAN}[4/4]${NC} Updating README with visual assets..."
    if [ -f "$ROOT/assets/demo.gif" ] || [ "$SCREEN_COUNT" -gt 0 ]; then
        bash "$SCRIPT_DIR/update_readme.sh"
    else
        echo -e "  ${YELLOW}⚠${NC} No assets yet — skipping README update"
    fi
    echo ""
}

do_posts() {
    echo -e "${BOLD}━━━ PHASE 2: REVIEW POSTS ━━━${NC}"
    echo ""
    POSTS_DIR="$ROOT/assets/posts"
    if [ ! -d "$POSTS_DIR" ]; then
        echo "  No posts generated yet. Run: bash scripts/launch/launch.sh prep"
        return
    fi

    echo "  Your ready-to-paste posts:"
    echo ""
    for f in "$POSTS_DIR"/*.txt; do
        NAME=$(basename "$f" .txt)
        LINES=$(wc -l < "$f" | tr -d ' ')
        echo -e "  ${GREEN}✓${NC} $NAME ($LINES lines)"
    done
    echo ""
    echo -n "  Open all in your text editor? [Y/n]: "
    read -r response
    if [ "$response" != "n" ]; then
        open "$POSTS_DIR"/*.txt 2>/dev/null || xdg-open "$POSTS_DIR" 2>/dev/null
    fi
    echo ""
}

do_submit() {
    echo -e "${BOLD}━━━ PHASE 3: SUBMIT (Opening Browser Tabs) ━━━${NC}"
    echo ""
    echo "  Opening submission pages..."
    echo ""

    URLS=(
        "https://news.ycombinator.com/submit:Hacker News — Submit"
        "https://www.reddit.com/r/LocalLLaMA/submit:Reddit r/LocalLLaMA"
        "https://www.reddit.com/r/MachineLearning/submit:Reddit r/MachineLearning"
        "https://www.reddit.com/r/swift/submit:Reddit r/swift"
        "https://www.reddit.com/r/apple/submit:Reddit r/apple"
        "https://twitter.com/compose/tweet:Twitter/X — Compose"
        "https://dev.to/new:Dev.to — New Article"
        "https://www.producthunt.com/posts/new:Product Hunt"
    )

    echo "  Which platforms to open?"
    echo ""
    for i in "${!URLS[@]}"; do
        URL="${URLS[$i]%%:*}"
        LABEL="${URLS[$i]##*:}"
        echo "    $((i+1)). $LABEL"
    done
    echo "    a. All of them"
    echo "    q. Skip"
    echo ""
    echo -n "  Enter numbers (comma-separated) or 'a': "
    read -r selection

    if [ "$selection" = "q" ]; then
        return
    fi

    if [ "$selection" = "a" ]; then
        for entry in "${URLS[@]}"; do
            URL="${entry%%:*}"
            LABEL="${entry##*:}"
            open "$URL" 2>/dev/null
            echo -e "  ${GREEN}✓${NC} Opened $LABEL"
            sleep 0.5
        done
    else
        IFS=',' read -ra INDICES <<< "$selection"
        for idx in "${INDICES[@]}"; do
            idx=$(echo "$idx" | tr -d ' ')
            i=$((idx - 1))
            if [ "$i" -ge 0 ] && [ "$i" -lt "${#URLS[@]}" ]; then
                URL="${URLS[$i]%%:*}"
                LABEL="${URLS[$i]##*:}"
                open "$URL" 2>/dev/null
                echo -e "  ${GREEN}✓${NC} Opened $LABEL"
            fi
        done
    fi
    echo ""

    # Also open the posts folder for copy-pasting
    echo "  Opening posts folder for copy-pasting..."
    open "$ROOT/assets/posts" 2>/dev/null || true
    echo ""
}

do_metrics() {
    echo -e "${BOLD}━━━ METRICS ━━━${NC}"
    echo ""
    bash "$SCRIPT_DIR/track_metrics.sh"
}

# ── Execute requested phase ──
case "$PHASE" in
    prep)    do_prep ;;
    posts)   do_posts ;;
    submit)  do_submit ;;
    metrics) do_metrics ;;
    all)
        do_prep
        do_posts
        do_submit
        echo -e "${BOLD}━━━ LAUNCH PREP COMPLETE ━━━${NC}"
        echo ""
        echo "  Next steps:"
        echo "    1. Paste posts into each platform"
        echo "    2. Stay online 2+ hours to reply to comments"
        echo "    3. Track metrics: bash scripts/launch/launch.sh metrics"
        echo ""
        ;;
    *)
        echo "Usage: bash scripts/launch/launch.sh [prep|posts|submit|metrics|all]"
        echo ""
        echo "  prep    — Record GIF, screenshots, generate posts, update README"
        echo "  posts   — Open post files for review"
        echo "  submit  — Open browser tabs for each platform"
        echo "  metrics — Show current GitHub metrics"
        echo "  all     — Full launch sequence"
        ;;
esac
