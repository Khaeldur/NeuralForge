#!/bin/bash
# Auto-update README.md with demo GIF and screenshots
# Usage: bash scripts/launch/update_readme.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
README="$ROOT/README.md"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${CYAN}=== Updating README with Visual Assets ===${NC}"

# Check for assets
GIF="$ROOT/assets/demo.gif"
SCREENS="$ROOT/assets/screenshots"

HAS_GIF=false
HAS_SCREENS=false
[ -f "$GIF" ] && HAS_GIF=true
[ -d "$SCREENS" ] && [ "$(ls "$SCREENS"/*.png 2>/dev/null | wc -l)" -gt 0 ] && HAS_SCREENS=true

if ! $HAS_GIF && ! $HAS_SCREENS; then
    echo -e "  ${YELLOW}No assets found. Run these first:${NC}"
    echo "    bash scripts/launch/record_demo.sh"
    echo "    bash scripts/launch/capture_screenshots.sh"
    exit 1
fi

# Backup current README
cp "$README" "$README.bak"

# Build the visual section
VISUAL_SECTION=""
if $HAS_GIF; then
    VISUAL_SECTION+="\n<p align=\"center\">\n  <img src=\"assets/demo.gif\" alt=\"NeuralForge Demo\" width=\"700\">\n</p>\n"
    echo -e "  ${GREEN}✓${NC} Demo GIF will be embedded"
fi

if $HAS_SCREENS; then
    VISUAL_SECTION+="\n## Screenshots\n\n"
    VISUAL_SECTION+="<table>\n<tr>\n"
    COL=0
    for img in "$SCREENS"/*.png; do
        NAME=$(basename "$img" .png)
        LABEL=$(echo "$NAME" | sed 's/^[0-9]*_//' | tr '_' ' ' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) substr($i,2)}1')
        VISUAL_SECTION+="<td align=\"center\"><img src=\"assets/screenshots/$NAME.png\" width=\"350\"><br><sub>$LABEL</sub></td>\n"
        COL=$((COL + 1))
        if [ "$COL" -eq 2 ]; then
            VISUAL_SECTION+="</tr>\n<tr>\n"
            COL=0
        fi
    done
    VISUAL_SECTION+="</tr>\n</table>\n"
    echo -e "  ${GREEN}✓${NC} $(ls "$SCREENS"/*.png | wc -l | tr -d ' ') screenshots will be embedded"
fi

# Check if README already has a visual section marker
if grep -q "<!-- LAUNCH_VISUALS_START -->" "$README"; then
    # Replace existing visual section
    python3 -c "
import re
with open('$README', 'r') as f:
    content = f.read()
pattern = r'<!-- LAUNCH_VISUALS_START -->.*?<!-- LAUNCH_VISUALS_END -->'
replacement = '<!-- LAUNCH_VISUALS_START -->\n${VISUAL_SECTION}\n<!-- LAUNCH_VISUALS_END -->'
content = re.sub(pattern, replacement, content, flags=re.DOTALL)
with open('$README', 'w') as f:
    f.write(content)
"
    echo -e "  ${GREEN}✓${NC} Updated existing visual section"
else
    # Insert after the first heading (# NeuralForge)
    python3 -c "
with open('$README', 'r') as f:
    lines = f.readlines()
insert_idx = None
for i, line in enumerate(lines):
    if line.startswith('# ') and i == 0:
        # Find the next blank line after the title block
        for j in range(i+1, len(lines)):
            if lines[j].strip() == '':
                insert_idx = j + 1
                break
        break
if insert_idx is None:
    insert_idx = 1
marker_start = '<!-- LAUNCH_VISUALS_START -->\n'
marker_end = '<!-- LAUNCH_VISUALS_END -->\n\n'
visual = '''${VISUAL_SECTION}'''
lines.insert(insert_idx, marker_start + visual + marker_end)
with open('$README', 'w') as f:
    f.writelines(lines)
"
    echo -e "  ${GREEN}✓${NC} Inserted visual section into README"
fi

echo ""
echo -e "${GREEN}=== README updated ===${NC}"
echo "  Review: $README"
echo "  Backup: $README.bak"
