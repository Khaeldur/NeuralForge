#!/bin/bash
# Generate all marketing posts as ready-to-paste text files
# Usage: bash scripts/launch/generate_posts.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
POSTS_DIR="$ROOT/assets/posts"
mkdir -p "$POSTS_DIR"

REPO_URL="https://github.com/Khaeldur/NeuralForge"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}=== Generating Launch Posts ===${NC}"
echo ""

# ── 1. Hacker News ──
cat > "$POSTS_DIR/01_hackernews_title.txt" <<'HNEOF'
Show HN: NeuralForge – Fine-tune LLMs on your Mac using Apple Neural Engine
HNEOF

cat > "$POSTS_DIR/01_hackernews_comment.txt" <<HNEOF
Hi HN, I built NeuralForge because I wanted to fine-tune small LLMs on my MacBook without renting cloud GPUs or setting up CUDA.

It uses Apple's Neural Engine directly (not Metal, not CPU) to hit ~1.2 TFLOPS on a consumer Mac. The app wraps a C/Obj-C training engine in a SwiftUI GUI with live loss curves, LoRA support, and one-click export to GGUF/CoreML.

What actually works today:
- Real training on ANE (110M parameter llama2.c models)
- LoRA fine-tuning (rank 4-64 on attention weights)
- Cosine LR schedule with warmup
- Checkpoint save/resume (survives crashes)
- Text generation at 66ms/token with top-p sampling
- Export to GGUF, llama2c, CoreML formats
- 648 automated tests (unit + integration on real hardware)

Current limitations:
- Only llama2.c format models (110M tested, larger planned)
- macOS 14+ on Apple Silicon only
- Some UI features are still stubs (being honest)

Setup: git clone + bash setup.sh (one command)

The hardest part was ANE itself. There's basically zero documentation on using it for training — it's designed for inference only. I had to reverse-engineer the MIL compiler, figure out the 119-kernel compilation limit per process, and build an exec() restart mechanism that transparently re-launches the training process to get fresh kernel budget.

MIT licensed: ${REPO_URL}

Happy to answer questions about ANE internals.
HNEOF
echo -e "  ${GREEN}✓${NC} Hacker News (title + comment)"

# ── 2. Reddit r/LocalLLaMA ──
cat > "$POSTS_DIR/02_reddit_localllama.txt" <<REOF
TITLE: I built a macOS app for fine-tuning LLMs on Apple Neural Engine — no cloud, no CUDA [open source]

BODY:

I've been working on NeuralForge, an open-source macOS app that lets you fine-tune LLMs directly on your Mac using Apple's Neural Engine.

**Why?** Every fine-tuning tool (Axolotl, Unsloth, LLaMA-Factory) assumes you have an NVIDIA GPU. If you're on a Mac, your options are MLX Python scripts or renting cloud GPUs. I wanted a native app that just works.

**What it does:**
- Trains on Apple Neural Engine (~1.2 TFLOPS on M4)
- LoRA fine-tuning (rank 4-64)
- Cosine LR scheduling with warmup
- Checkpoint save & resume
- Live dashboard with loss curves + metrics
- Text generation at 66ms/token
- Export to GGUF, CoreML, llama2c
- 100% on-device, data never leaves your Mac

**What it doesn't do yet:**
- Only llama2.c format models (110M tested)
- macOS 14+ / Apple Silicon required
- Some advanced features (cloud sync, distributed) are WIP
- Not a replacement for Unsloth on a 4090 — this is for people who only have a Mac

**Setup:**
\`\`\`bash
git clone ${REPO_URL}.git
cd NeuralForge
bash setup.sh   # builds CLI, downloads models, runs 648 tests
open app/NeuralForge.xcodeproj  # ⌘R to run
\`\`\`

GitHub: ${REPO_URL}

What models/formats would you want to see supported? Interested in feedback from the community.
REOF
echo -e "  ${GREEN}✓${NC} Reddit r/LocalLLaMA"

# ── 3. Reddit r/MachineLearning ──
cat > "$POSTS_DIR/03_reddit_machinelearning.txt" <<REOF
TITLE: [P] NeuralForge: LLM fine-tuning on Apple Neural Engine with zero official documentation

BODY:

I built an open-source training engine that runs LLM fine-tuning on Apple's Neural Engine (ANE). As far as I know, this is the first tool to use ANE for training rather than just inference.

**Technical details:**
- ANE programming via MIL (Machine Intermediate Language) compiler
- Hit the 119-kernel compilation limit — solved with transparent exec() restart
- Forward + backward pass kernels compiled to ANE
- ~1.2 TFLOPS sustained throughput on consumer Apple Silicon
- LoRA adapters on attention weights with correct gradient backpropagation
- Cosine annealing LR scheduler with linear warmup
- Full checkpoint system (model + Adam optimizer states)

**Why this was hard:** Apple provides zero documentation for training on ANE. The entire Neural Engine SDK is designed for inference only. I had to reverse-engineer the MIL compilation pipeline, figure out memory layout for gradient tensors on IOSurface, and work around the kernel compilation budget.

**Results:**
- 110M parameter model, batch size 4: ~71ms/step
- Training data: TinyStories (32K vocab, pretokenized)
- Loss converges from ~11.0 to predictable curves
- Generation: 66ms/token with nucleus sampling

Includes a SwiftUI macOS app with live training dashboard, and exports to GGUF/CoreML.

GitHub: ${REPO_URL}

Paper-like writeup of the ANE training approach coming soon. Happy to discuss the technical details.
REOF
echo -e "  ${GREEN}✓${NC} Reddit r/MachineLearning"

# ── 4. Reddit r/swift ──
cat > "$POSTS_DIR/04_reddit_swift.txt" <<REOF
TITLE: Open-source SwiftUI app that fine-tunes LLMs on Apple Neural Engine

BODY:

I built NeuralForge — a macOS app for fine-tuning large language models on Apple Silicon. The architecture might be interesting to Swift developers:

**App Architecture:**
- SwiftUI frontend with HSplitView sidebar navigation
- Spawns a C/Obj-C CLI subprocess for the heavy lifting
- Real-time NDJSON protocol between CLI and app (one JSON object per stdout line)
- @Published state management drives live-updating loss curves
- Project persistence to ~/Library/Application Support
- 53 XCUITests + 416 Swift unit tests

**The interesting part** is the CLI ↔ App communication. Training runs in a separate process (so the app stays responsive), and every training step emits a JSON line:
\`\`\`json
{"type":"step","step":42,"loss":5.23,"ms":71.2,"tflops":1.19}
\`\`\`
The SwiftUI dashboard reads this stream and updates Charts in real-time.

**Stack:** SwiftUI + Charts, Obj-C CLI, ANE via MIL, XCUITest

GitHub: ${REPO_URL}

MIT licensed, macOS 14+ / Apple Silicon required.
REOF
echo -e "  ${GREEN}✓${NC} Reddit r/swift"

# ── 5. Reddit r/apple ──
cat > "$POSTS_DIR/05_reddit_apple.txt" <<REOF
TITLE: Your Mac can train AI models — I built an open-source app for it

BODY:

Everyone talks about Mac for AI inference (Ollama, LM Studio), but what about training? I built NeuralForge, a free app that fine-tunes language models using your Mac's Neural Engine.

**What is this?** Think of it as teaching an AI to write in a specific style — using only your Mac's built-in hardware, no cloud required.

**How it works:**
- Uses the Apple Neural Engine (the same chip that powers Siri, photo features, etc.)
- Gets ~1.2 TFLOPS — similar to entry-level cloud GPUs
- Your data never leaves your machine
- Export trained models to CoreML for use in your own iOS/macOS apps

**Who is this for?**
- Developers wanting to build AI features into their apps
- Anyone curious about how AI training actually works
- Privacy-conscious users who don't want their data in the cloud

Free, open source: ${REPO_URL}

Requires macOS 14+ on Apple Silicon (M1/M2/M3/M4).
REOF
echo -e "  ${GREEN}✓${NC} Reddit r/apple"

# ── 6. Twitter/X Thread ──
cat > "$POSTS_DIR/06_twitter_thread.txt" <<TEOF
TWEET 1:
I built an open-source macOS app that fine-tunes LLMs on Apple Neural Engine.

No cloud. No CUDA. No GPU rental.

1.2 TFLOPS on a MacBook.

🧵👇

[ATTACH: demo.gif]

---

TWEET 2:
Why does this exist?

Every fine-tuning tool assumes NVIDIA GPUs.

If you're on a Mac:
- Python scripts (MLX) → works but no GUI
- Rent cloud GPUs → \$\$\$
- Give up → most people

NeuralForge: download, open, train.

---

TWEET 3:
What actually works:

✅ Real ANE training (110M params)
✅ LoRA fine-tuning
✅ Live loss curves + metrics
✅ Cosine LR schedule with warmup
✅ Checkpoint save/resume
✅ Text generation (66ms/tok)
✅ Export → GGUF, CoreML, llama2c
✅ 648 automated tests

---

TWEET 4:
The hardest part: Apple Neural Engine.

Zero documentation for training.
Designed for inference only.

I had to:
→ Reverse-engineer the MIL compiler
→ Work around a 119-kernel limit per process
→ Build a transparent exec() restart mechanism

---

TWEET 5:
Who is this for?

→ Mac devs wanting on-device AI
→ Privacy-first teams (data stays local)
→ Students learning LLM fine-tuning
→ iOS devs wanting custom CoreML models
→ Anyone who can't justify cloud GPU costs

---

TWEET 6:
It's fully open source (MIT):

${REPO_URL}

One command from clone to training:
git clone → bash setup.sh → ⌘R

Star it if you think Macs deserve first-class AI tools ⭐

TAG: @_karpathy @awnihannun
TEOF
echo -e "  ${GREEN}✓${NC} Twitter/X thread (6 tweets)"

# ── 7. Dev.to / Medium Article Outline ──
cat > "$POSTS_DIR/07_devto_article.txt" <<DEOF
TITLE: How I Built an LLM Training Engine on Apple Neural Engine (With Zero Documentation)

TAGS: machinelearning, apple, swift, opensource

---

INTRO:
Every LLM fine-tuning tool assumes you have an NVIDIA GPU. I only have a MacBook. So I built NeuralForge — an open-source macOS app that trains language models on Apple's Neural Engine.

This is the story of how I reverse-engineered ANE for training, hit a 119-kernel brick wall, and built a system that gets 1.2 TFLOPS on consumer hardware.

SECTIONS:
1. The Problem: Mac Users Can't Fine-Tune LLMs
2. Why Apple Neural Engine (Not Metal, Not CPU)
3. Reverse-Engineering MIL: The ANE Programming Language
4. The 119-Kernel Wall (and the exec() Hack)
5. Building Forward + Backward Passes for ANE
6. Results: 1.2 TFLOPS, 66ms/tok Generation
7. The SwiftUI App: NDJSON Real-Time Protocol
8. What I Learned About Apple Silicon Internals
9. What's Next: Larger Models, More Formats

CLOSING:
Link to GitHub repo, invite contributions, mention specific areas where help is needed.

GitHub: ${REPO_URL}
DEOF
echo -e "  ${GREEN}✓${NC} Dev.to article outline"

# ── 8. Product Hunt ──
cat > "$POSTS_DIR/08_producthunt.txt" <<PHEOF
NAME: NeuralForge

TAGLINE: Fine-tune LLMs on your Mac — no cloud, no CUDA

DESCRIPTION:
NeuralForge is an open-source macOS app for fine-tuning large language models on Apple Silicon. It uses Apple's Neural Engine to achieve ~1.2 TFLOPS — no cloud GPUs or NVIDIA hardware needed.

Features:
• Train on Apple Neural Engine (M1/M2/M3/M4)
• LoRA fine-tuning for memory efficiency
• Live dashboard with loss curves
• Export to GGUF, CoreML, llama2c
• 100% on-device — your data never leaves your Mac
• One-command setup: git clone + bash setup.sh

WHO IS IT FOR:
Mac developers who want on-device AI, privacy-conscious teams, students learning ML, and anyone who doesn't want to pay for cloud GPUs.

MAKER COMMENT:
I built NeuralForge because I couldn't find a single tool that let me fine-tune an LLM on my MacBook. Everything assumes NVIDIA GPUs. Apple's Neural Engine is powerful hardware sitting unused for training — so I reverse-engineered it.

The core training engine is C/Objective-C for maximum performance. The GUI is SwiftUI. The whole thing is MIT licensed.

What would make this most useful for you?

TOPICS: Artificial Intelligence, Developer Tools, Open Source, Mac
PHEOF
echo -e "  ${GREEN}✓${NC} Product Hunt listing"

# ── 9. Awesome Lists PRs ──
cat > "$POSTS_DIR/09_awesome_lists.txt" <<ALEOF
=== awesome-mac (https://github.com/jaywcjlove/awesome-mac) ===

Add to: ## Machine Learning / AI section

- [NeuralForge](${REPO_URL}) - On-device LLM fine-tuning for Apple Silicon using Neural Engine. [![Open-Source Software][oss icon]](${REPO_URL}) ![Freeware][freeware icon]

PR Title: Add NeuralForge — LLM fine-tuning app for Apple Silicon

---

=== open-source-mac-os-apps (https://github.com/serhii-londar/open-source-mac-os-apps) ===

Add to: ## Machine Learning section

- [NeuralForge](${REPO_URL}) - Fine-tune LLMs on Apple Neural Engine with live dashboard and LoRA support. ![swift_icon]

PR Title: Add NeuralForge — on-device LLM fine-tuning

---

=== awesome-mlops (https://github.com/visenger/awesome-mlops) ===

Add to: ## Training section

- [NeuralForge](${REPO_URL}) - macOS-native LLM fine-tuning on Apple Neural Engine. MIT license.

PR Title: Add NeuralForge — Apple Silicon LLM training
ALEOF
echo -e "  ${GREEN}✓${NC} Awesome lists PR templates"

echo ""
echo -e "${GREEN}=== All posts generated in $POSTS_DIR ===${NC}"
echo ""
ls -1 "$POSTS_DIR"
echo ""
echo "Copy-paste from these files when posting. Each file is ready to go."
