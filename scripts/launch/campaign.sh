#!/bin/bash
# NeuralForge Marketing Campaign — Automated Posting Pipeline
# Usage: bash scripts/launch/campaign.sh [platform]
#
# This script handles the actual posting for each platform.
# Called by scheduled tasks at optimal times, or manually.
#
# Platforms: hn | reddit-localllama | reddit-ml | reddit-swift | reddit-apple |
#            twitter | devto | producthunt | awesome-lists | all-status
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
POSTS_DIR="$ROOT/assets/posts"
CAMPAIGN_LOG="$ROOT/assets/campaign_log.csv"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# Initialize campaign log
if [ ! -f "$CAMPAIGN_LOG" ]; then
    echo "timestamp,platform,status,url,notes" > "$CAMPAIGN_LOG"
fi

log_post() {
    local platform="$1" status="$2" url="$3" notes="$4"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$platform,$status,$url,$notes" >> "$CAMPAIGN_LOG"
}

copy_to_clipboard() {
    local file="$1"
    if [ -f "$file" ]; then
        pbcopy < "$file"
        echo -e "  ${GREEN}✓${NC} Copied to clipboard: $(basename "$file")"
    else
        echo -e "  ${RED}✗${NC} File not found: $file"
        return 1
    fi
}

# ── Platform: Hacker News ──
do_hn() {
    echo -e "${BOLD}🟠 Hacker News — Show HN${NC}"
    echo ""

    # Copy title to clipboard first
    copy_to_clipboard "$POSTS_DIR/01_hackernews_title.txt"
    echo ""

    # Open HN submit page
    open "https://news.ycombinator.com/submit"
    echo -e "  ${CYAN}→ Submit page opened${NC}"
    echo -e "  ${YELLOW}ACTION:${NC} Paste title, add URL: https://github.com/Khaeldur/NeuralForge"
    echo ""

    # Wait for user to submit, then post first comment
    echo -n "  Press ENTER after submitting title (to copy first comment)..."
    read -r
    copy_to_clipboard "$POSTS_DIR/01_hackernews_comment.txt"
    echo -e "  ${YELLOW}ACTION:${NC} Find your post, click 'add comment', paste comment"
    echo ""

    log_post "hackernews" "submitted" "https://news.ycombinator.com" "Show HN posted"
    echo -e "  ${GREEN}✓ HN submitted!${NC} Stay online 2+ hours to reply."
}

# ── Platform: Reddit ──
do_reddit() {
    local subreddit="$1" post_file="$2" label="$3"
    echo -e "${BOLD}🔴 Reddit — r/$subreddit${NC}"
    echo ""

    # Check if we have a Reddit CLI tool
    if command -v reddit-cli &>/dev/null; then
        # Auto-post via CLI if available
        TITLE=$(head -1 "$post_file" | sed 's/^TITLE: //')
        BODY=$(sed -n '/^BODY:/,$ p' "$post_file" | tail -n +3)
        echo -e "  ${CYAN}Auto-posting via reddit-cli...${NC}"
        reddit-cli submit --subreddit "$subreddit" --title "$TITLE" --body "$BODY" 2>&1 || true
    else
        # Manual: copy + open browser
        copy_to_clipboard "$post_file"
        open "https://www.reddit.com/r/$subreddit/submit"
        echo -e "  ${CYAN}→ Submit page opened for r/$subreddit${NC}"
        echo -e "  ${YELLOW}ACTION:${NC} Paste content. Title is on line 1, body starts at 'BODY:'"
    fi

    echo ""
    log_post "reddit-$subreddit" "submitted" "https://reddit.com/r/$subreddit" "$label"
    echo -e "  ${GREEN}✓ r/$subreddit ready!${NC}"
}

# ── Platform: Twitter/X ──
do_twitter() {
    echo -e "${BOLD}🐦 Twitter/X — Thread${NC}"
    echo ""

    # Split the thread into individual tweets
    local tweet_num=1
    while IFS= read -r line; do
        if [[ "$line" == "TWEET $tweet_num:" ]]; then
            # Read until next --- separator
            local tweet=""
            while IFS= read -r tline; do
                [[ "$tline" == "---" ]] && break
                tweet+="$tline"$'\n'
            done
            tweet=$(echo "$tweet" | sed '/^$/d' | head -c 280)

            echo -e "  ${CYAN}Tweet $tweet_num:${NC}"
            echo "$tweet" | head -3
            echo "  ..."
            echo ""
            tweet_num=$((tweet_num + 1))
        fi
    done < "$POSTS_DIR/06_twitter_thread.txt"

    # Copy first tweet to clipboard
    python3 -c "
content = open('$POSTS_DIR/06_twitter_thread.txt').read()
tweets = content.split('---')
first = tweets[0].replace('TWEET 1:', '').strip()
# Remove [ATTACH:...] directives
import re
first = re.sub(r'\[ATTACH:.*?\]', '', first).strip()
print(first)
" | pbcopy

    echo -e "  ${GREEN}✓${NC} First tweet copied to clipboard"
    open "https://twitter.com/compose/tweet"
    echo -e "  ${CYAN}→ Compose page opened${NC}"
    echo -e "  ${YELLOW}ACTION:${NC} Paste first tweet, post, then reply with each subsequent tweet"
    echo ""

    log_post "twitter" "submitted" "https://twitter.com" "6-tweet thread"
    echo -e "  ${GREEN}✓ Twitter thread ready!${NC}"
}

# ── Platform: Dev.to ──
do_devto() {
    echo -e "${BOLD}📝 Dev.to — Technical Article${NC}"
    echo ""

    # Check for Dev.to API key
    DEVTO_KEY="${DEVTO_API_KEY:-}"
    if [ -n "$DEVTO_KEY" ]; then
        echo -e "  ${CYAN}Auto-posting via Dev.to API...${NC}"
        ARTICLE_BODY=$(cat "$POSTS_DIR/07_devto_article.txt")

        # Extract title (first line after removing any header markers)
        TITLE=$(head -1 "$POSTS_DIR/07_devto_article.txt" | sed 's/^#* *//')

        curl -s -X POST "https://dev.to/api/articles" \
            -H "Content-Type: application/json" \
            -H "api-key: $DEVTO_KEY" \
            -d "$(python3 -c "
import json, sys
body = open('$POSTS_DIR/07_devto_article.txt').read()
article = {
    'article': {
        'title': 'NeuralForge: Fine-Tune LLMs on Apple Neural Engine (Open Source)',
        'published': True,
        'body_markdown': body,
        'tags': ['machinelearning', 'macos', 'opensource', 'swift'],
        'series': None
    }
}
print(json.dumps(article))
")" | python3 -c "import sys,json; r=json.load(sys.stdin); print(f\"  Published: {r.get('url','error')}\")"

        log_post "devto" "auto-posted" "" "Via API"
    else
        copy_to_clipboard "$POSTS_DIR/07_devto_article.txt"
        open "https://dev.to/new"
        echo -e "  ${CYAN}→ New article page opened${NC}"
        echo -e "  ${YELLOW}ACTION:${NC} Paste article content"
        echo ""
        echo -e "  ${YELLOW}TIP:${NC} Set DEVTO_API_KEY env var for auto-posting next time"
        echo -e "  Get your key: https://dev.to/settings/extensions → Generate API Key"

        log_post "devto" "manual" "https://dev.to/new" "Manual paste"
    fi

    echo ""
    echo -e "  ${GREEN}✓ Dev.to article ready!${NC}"
}

# ── Platform: Product Hunt ──
do_producthunt() {
    echo -e "${BOLD}🐱 Product Hunt — Launch${NC}"
    echo ""

    copy_to_clipboard "$POSTS_DIR/08_producthunt.txt"
    open "https://www.producthunt.com/posts/new"
    echo -e "  ${CYAN}→ New product page opened${NC}"
    echo -e "  ${YELLOW}ACTION:${NC} Fill in the product details from clipboard"
    echo -e "  ${YELLOW}TIMING:${NC} Best to submit at 12:01 AM PT (10:01 AM EET) on a Tuesday"
    echo ""

    log_post "producthunt" "submitted" "https://producthunt.com" "Launch page"
    echo -e "  ${GREEN}✓ Product Hunt ready!${NC}"
}

# ── Platform: Awesome Lists ──
do_awesome() {
    echo -e "${BOLD}📋 Awesome Lists — GitHub PRs${NC}"
    echo ""

    LISTS=(
        "jaywcjlove/awesome-mac"
        "serhii-londar/open-source-mac-os-apps"
        "kelvins/awesome-mlops"
    )

    for repo in "${LISTS[@]}"; do
        echo -e "  ${CYAN}→${NC} Opening fork page for $repo"
        open "https://github.com/$repo/fork"
        sleep 1
    done

    copy_to_clipboard "$POSTS_DIR/09_awesome_lists.txt"
    echo ""
    echo -e "  ${YELLOW}ACTION:${NC} Fork each repo → edit README → add NeuralForge entry → PR"
    echo -e "  Post content copied to clipboard for reference."
    echo ""

    log_post "awesome-lists" "started" "" "3 repos opened"
    echo -e "  ${GREEN}✓ Awesome list PRs initiated!${NC}"
}

# ── Status Dashboard ──
do_status() {
    echo ""
    echo -e "${BOLD}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║   📊 NeuralForge Campaign Status               ║${NC}"
    echo -e "${BOLD}╚════════════════════════════════════════════════╝${NC}"
    echo ""

    if [ -f "$CAMPAIGN_LOG" ]; then
        echo -e "  ${BOLD}Posts:${NC}"
        tail -n +2 "$CAMPAIGN_LOG" | while IFS=',' read -r ts platform status url notes; do
            local icon="⬜"
            case "$status" in
                submitted|auto-posted) icon="✅" ;;
                manual) icon="🟡" ;;
                started) icon="🔵" ;;
                *) icon="⬜" ;;
            esac
            printf "  %s %-22s %s  %s\n" "$icon" "$platform" "$status" "$ts"
        done
        echo ""
    fi

    # Show upcoming scheduled posts
    echo -e "  ${BOLD}Upcoming Schedule (EET):${NC}"
    echo ""
    echo "  📅 Tue Mar 11  15:00  — Hacker News (Show HN)"
    echo "  📅 Tue Mar 11  17:00  — Reddit r/LocalLLaMA"
    echo "  📅 Wed Mar 12  15:00  — Reddit r/MachineLearning"
    echo "  📅 Wed Mar 12  18:00  — Reddit r/swift"
    echo "  📅 Thu Mar 13  15:00  — Reddit r/apple"
    echo "  📅 Thu Mar 13  18:00  — Twitter/X thread"
    echo "  📅 Fri Mar 14  16:00  — Dev.to article"
    echo "  📅 Tue Mar 17  10:01  — Product Hunt launch"
    echo "  📅 Wed Mar 18  15:00  — Awesome list PRs"
    echo ""

    # Show GitHub metrics
    if [ -f "$ROOT/assets/metrics.csv" ]; then
        echo -e "  ${BOLD}Latest Metrics:${NC}"
        tail -1 "$ROOT/assets/metrics.csv" | awk -F',' '{printf "  ⭐ Stars: %s  🍴 Forks: %s  👀 Views: %s  📦 Clones: %s\n", $3, $4, $7, $9}'
    fi
    echo ""
}

# ── Help ──
do_help() {
    echo ""
    echo -e "${BOLD}NeuralForge Campaign Commands:${NC}"
    echo ""
    echo "  bash scripts/launch/campaign.sh hn              — Post to Hacker News"
    echo "  bash scripts/launch/campaign.sh reddit-localllama — Post to r/LocalLLaMA"
    echo "  bash scripts/launch/campaign.sh reddit-ml       — Post to r/MachineLearning"
    echo "  bash scripts/launch/campaign.sh reddit-swift    — Post to r/swift"
    echo "  bash scripts/launch/campaign.sh reddit-apple    — Post to r/apple"
    echo "  bash scripts/launch/campaign.sh twitter         — Post Twitter thread"
    echo "  bash scripts/launch/campaign.sh devto           — Post Dev.to article"
    echo "  bash scripts/launch/campaign.sh producthunt     — Product Hunt launch"
    echo "  bash scripts/launch/campaign.sh awesome-lists   — Awesome list PRs"
    echo "  bash scripts/launch/campaign.sh status          — Campaign status dashboard"
    echo ""
    echo "  Environment variables:"
    echo "    DEVTO_API_KEY — Dev.to API key for auto-posting"
    echo ""
}

# ── Execute ──
PLATFORM="${1:-help}"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   🚀 NeuralForge Campaign — $PLATFORM${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
echo ""

case "$PLATFORM" in
    hn)                do_hn ;;
    reddit-localllama) do_reddit "LocalLLaMA" "$POSTS_DIR/02_reddit_localllama.txt" "Primary Reddit post" ;;
    reddit-ml)         do_reddit "MachineLearning" "$POSTS_DIR/03_reddit_machinelearning.txt" "Academic angle" ;;
    reddit-swift)      do_reddit "swift" "$POSTS_DIR/04_reddit_swift.txt" "Swift dev community" ;;
    reddit-apple)      do_reddit "apple" "$POSTS_DIR/05_reddit_apple.txt" "Apple consumer community" ;;
    twitter)           do_twitter ;;
    devto)             do_devto ;;
    producthunt)       do_producthunt ;;
    awesome-lists)     do_awesome ;;
    status)            do_status ;;
    help|*)            do_help ;;
esac
