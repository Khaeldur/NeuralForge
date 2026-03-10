#!/bin/bash
# Capture NeuralForge app screenshots for marketing
# Usage: bash scripts/launch/capture_screenshots.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ASSETS_DIR="$ROOT/assets/screenshots"
mkdir -p "$ASSETS_DIR"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${CYAN}=== NeuralForge Screenshot Capture ===${NC}"
echo ""

# Step 1: Build and launch the app
echo -e "${CYAN}[1/4]${NC} Building and launching app..."
APP_PATH=$(find /tmp/NF_SetupBuild /tmp/NF_TestBuild /tmp/NF_DerivedData ~/Library/Developer/Xcode/DerivedData \
    -name "NeuralForge.app" -type d 2>/dev/null | head -1)

if [ -z "$APP_PATH" ]; then
    echo "  Building app first..."
    cd "$ROOT/app"
    xcodebuild -project NeuralForge.xcodeproj -scheme NeuralForge \
        -derivedDataPath /tmp/NF_Screenshots build 2>&1 | tail -3
    APP_PATH=$(find /tmp/NF_Screenshots -name "NeuralForge.app" -type d | head -1)
fi

if [ -z "$APP_PATH" ]; then
    echo "  ERROR: Could not find or build NeuralForge.app"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} App found at $APP_PATH"

# Step 2: Launch app
echo -e "${CYAN}[2/4]${NC} Launching NeuralForge..."
open "$APP_PATH"
sleep 3

# Step 3: Capture window screenshots
echo -e "${CYAN}[3/4]${NC} Capturing screenshots..."
echo -e "  ${YELLOW}→ Position the app window, then press Enter to capture each screenshot${NC}"
echo ""

SCREENSHOTS=(
    "01_dashboard:Dashboard — show the app with a training loss curve"
    "02_config:Config — show training configuration with LoRA settings"
    "03_generate:Generate — show text generation with output"
    "04_export:Export — show export format selection"
    "05_sidebar:Sidebar — show the full sidebar navigation"
)

for entry in "${SCREENSHOTS[@]}"; do
    NAME="${entry%%:*}"
    DESC="${entry##*:}"
    echo -e "  ${CYAN}[$NAME]${NC} $DESC"
    echo -n "  Press Enter when ready (or 's' to skip): "
    read -r response
    if [ "$response" = "s" ]; then
        echo "  Skipped"
        continue
    fi
    # Capture the frontmost window
    screencapture -o -w "$ASSETS_DIR/${NAME}.png"
    # Resize to max 1200px wide for web
    sips --resampleWidth 1200 "$ASSETS_DIR/${NAME}.png" >/dev/null 2>&1
    echo -e "  ${GREEN}✓${NC} Saved $ASSETS_DIR/${NAME}.png"
done

# Step 4: Also capture a CLI training screenshot from terminal
echo ""
echo -e "${CYAN}[4/4]${NC} Terminal screenshot..."
echo "  Run this in a separate terminal for a CLI screenshot:"
echo "    cd $ROOT/cli && ./neuralforge train --model ../models/stories110M.bin \\"
echo "      --data ../models/tinystories_data00.bin --steps 10"
echo -n "  Press Enter when terminal is showing training output (or 's' to skip): "
read -r response
if [ "$response" != "s" ]; then
    screencapture -o -w "$ASSETS_DIR/06_terminal.png"
    sips --resampleWidth 1200 "$ASSETS_DIR/06_terminal.png" >/dev/null 2>&1
    echo -e "  ${GREEN}✓${NC} Saved terminal screenshot"
fi

echo ""
echo -e "${GREEN}=== Screenshots saved to $ASSETS_DIR ===${NC}"
ls -lh "$ASSETS_DIR"/*.png 2>/dev/null
