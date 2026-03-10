#!/bin/bash
# Track GitHub metrics over time — run daily or on-demand
# Usage: bash scripts/launch/track_metrics.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
METRICS_FILE="$ROOT/assets/metrics.csv"
mkdir -p "$(dirname "$METRICS_FILE")"

REPO="Khaeldur/NeuralForge"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M)

# Fetch metrics from GitHub API
REPO_DATA=$(gh api "repos/$REPO" 2>/dev/null)
TRAFFIC=$(gh api "repos/$REPO/traffic/views" 2>/dev/null || echo '{"count":0,"uniques":0}')
CLONES=$(gh api "repos/$REPO/traffic/clones" 2>/dev/null || echo '{"count":0,"uniques":0}')
REFERRERS=$(gh api "repos/$REPO/traffic/popular/referrers" 2>/dev/null || echo '[]')

STARS=$(echo "$REPO_DATA" | python3 -c "import sys,json; print(json.load(sys.stdin).get('stargazers_count',0))" 2>/dev/null || echo 0)
FORKS=$(echo "$REPO_DATA" | python3 -c "import sys,json; print(json.load(sys.stdin).get('forks_count',0))" 2>/dev/null || echo 0)
WATCHERS=$(echo "$REPO_DATA" | python3 -c "import sys,json; print(json.load(sys.stdin).get('subscribers_count',0))" 2>/dev/null || echo 0)
ISSUES=$(echo "$REPO_DATA" | python3 -c "import sys,json; print(json.load(sys.stdin).get('open_issues_count',0))" 2>/dev/null || echo 0)
VIEWS=$(echo "$TRAFFIC" | python3 -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null || echo 0)
UNIQUE_VIEWS=$(echo "$TRAFFIC" | python3 -c "import sys,json; print(json.load(sys.stdin).get('uniques',0))" 2>/dev/null || echo 0)
CLONE_COUNT=$(echo "$CLONES" | python3 -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null || echo 0)
UNIQUE_CLONES=$(echo "$CLONES" | python3 -c "import sys,json; print(json.load(sys.stdin).get('uniques',0))" 2>/dev/null || echo 0)
TOP_REFERRER=$(echo "$REFERRERS" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r[0]['referrer'] if r else 'none')" 2>/dev/null || echo "none")

# Create CSV header if new file
if [ ! -f "$METRICS_FILE" ]; then
    echo "date,time,stars,forks,watchers,issues,views,unique_views,clones,unique_clones,top_referrer" > "$METRICS_FILE"
fi

# Append row
echo "$DATE,$TIME,$STARS,$FORKS,$WATCHERS,$ISSUES,$VIEWS,$UNIQUE_VIEWS,$CLONE_COUNT,$UNIQUE_CLONES,$TOP_REFERRER" >> "$METRICS_FILE"

# Display
echo "╔══════════════════════════════════════╗"
echo "║   NeuralForge Metrics — $DATE   ║"
echo "╠══════════════════════════════════════╣"
printf "║  ⭐ Stars:         %-17s║\n" "$STARS"
printf "║  🍴 Forks:         %-17s║\n" "$FORKS"
printf "║  👀 Watchers:      %-17s║\n" "$WATCHERS"
printf "║  🐛 Open Issues:   %-17s║\n" "$ISSUES"
printf "║  📊 Views (14d):   %-17s║\n" "$VIEWS ($UNIQUE_VIEWS unique)"
printf "║  📦 Clones (14d):  %-17s║\n" "$CLONE_COUNT ($UNIQUE_CLONES unique)"
printf "║  🔗 Top Referrer:  %-17s║\n" "$TOP_REFERRER"
echo "╚══════════════════════════════════════╝"
echo ""
echo "History saved to: $METRICS_FILE"
