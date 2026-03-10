#!/bin/bash
# Record a demo GIF of NeuralForge for README/social media
# Usage: bash scripts/launch/record_demo.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ASSETS_DIR="$ROOT/assets"
mkdir -p "$ASSETS_DIR"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Check for ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg required: brew install ffmpeg"
    exit 1
fi

echo -e "${CYAN}=== NeuralForge Demo Recording ===${NC}"
echo ""
echo "This will record your screen and convert to GIF."
echo ""
echo -e "${YELLOW}Before starting, prepare the app:${NC}"
echo "  1. Open NeuralForge"
echo "  2. Create a project with model + data configured"
echo "  3. Position the window nicely"
echo ""
echo "The demo should show: Create project → Configure → Start training → See loss curve"
echo ""

# Step 1: Record screen
MOV_FILE="$ASSETS_DIR/demo_raw.mov"
echo -e "${CYAN}[1/3]${NC} Screen recording"
echo "  Two options:"
echo "    a) Full screen recording (auto — 30 seconds)"
echo "    b) Window recording (manual — you stop it)"
echo -n "  Choose [a/b]: "
read -r choice

if [ "$choice" = "a" ]; then
    echo ""
    echo -e "  ${YELLOW}Recording starts in 3 seconds... (30 second capture)${NC}"
    sleep 3
    # Record 30 seconds of the screen
    screencapture -v -V 30 "$MOV_FILE" 2>/dev/null &
    CAPTURE_PID=$!
    echo "  Recording... (30 seconds)"
    echo "  DO YOUR DEMO NOW: click through the app"
    wait $CAPTURE_PID 2>/dev/null
else
    echo ""
    echo -e "  ${YELLOW}Click on the NeuralForge window when prompted${NC}"
    echo -n "  Press Enter to start recording: "
    read -r
    screencapture -v "$MOV_FILE"
fi

if [ ! -f "$MOV_FILE" ]; then
    echo "  Recording failed or was cancelled"
    exit 1
fi

echo -e "  ${GREEN}✓${NC} Recorded $(du -h "$MOV_FILE" | awk '{print $1}')"

# Step 2: Convert to GIF
GIF_FILE="$ASSETS_DIR/demo.gif"
echo ""
echo -e "${CYAN}[2/3]${NC} Converting to GIF..."

# Two-pass for quality: generate palette first, then use it
PALETTE="/tmp/nf_palette.png"
ffmpeg -y -i "$MOV_FILE" \
    -vf "fps=10,scale=800:-1:flags=lanczos,palettegen=stats_mode=diff" \
    "$PALETTE" 2>/dev/null

ffmpeg -y -i "$MOV_FILE" -i "$PALETTE" \
    -lavfi "fps=10,scale=800:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" \
    -loop 0 "$GIF_FILE" 2>/dev/null

rm -f "$PALETTE"
echo -e "  ${GREEN}✓${NC} GIF: $(du -h "$GIF_FILE" | awk '{print $1}')"

# Step 3: Also create a smaller version for social media
GIF_SMALL="$ASSETS_DIR/demo_small.gif"
echo ""
echo -e "${CYAN}[3/3]${NC} Creating social-media-sized version..."

PALETTE="/tmp/nf_palette2.png"
ffmpeg -y -i "$MOV_FILE" \
    -vf "fps=8,scale=480:-1:flags=lanczos,palettegen=stats_mode=diff" \
    "$PALETTE" 2>/dev/null

ffmpeg -y -i "$MOV_FILE" -i "$PALETTE" \
    -lavfi "fps=8,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" \
    -t 15 -loop 0 "$GIF_SMALL" 2>/dev/null

rm -f "$PALETTE"
echo -e "  ${GREEN}✓${NC} Small GIF: $(du -h "$GIF_SMALL" | awk '{print $1}')"

# Cleanup
rm -f "$MOV_FILE"

echo ""
echo -e "${GREEN}=== Demo assets ready ===${NC}"
echo "  Full GIF:  $GIF_FILE"
echo "  Small GIF: $GIF_SMALL"
echo ""
echo "Next: run 'bash scripts/launch/update_readme.sh' to embed in README"
