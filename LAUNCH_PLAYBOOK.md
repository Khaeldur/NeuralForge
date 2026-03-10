# NeuralForge — Launch Playbook

## Your Unfair Advantage

No one else has this: **a macOS-native GUI for fine-tuning LLMs on Apple Neural Engine**.

The competitive landscape:
- **Axolotl / Unsloth / LLaMA-Factory** → Linux/CUDA only, CLI only
- **LM Studio / Ollama** → inference only, no training
- **MLX fine-tuning** → Python scripts, no GUI
- **Cloud (SageMaker, Lambda Labs)** → expensive, complex, data leaves your machine

**NeuralForge = the only app where you download, open, train, and export — all on your Mac.**

---

## Week 1: Prep (Before Posting Anywhere)

### 1. Record a 30-Second Demo GIF
The single most important marketing asset. Show:
- Open app → create project → start training → loss curve moving → generate text

Tools: QuickTime screen record → convert to GIF with `ffmpeg`:
```bash
ffmpeg -i demo.mov -vf "fps=12,scale=800:-1" -loop 0 demo.gif
```

### 2. Take 4-5 Polished Screenshots
- Dashboard with live loss curve (training in progress)
- Config page with LoRA settings
- Text generation with output
- Export format selection
- Terminal showing CLI training with TFLOPS

### 3. Write a One-Liner
Use this everywhere:
> **NeuralForge — Fine-tune LLMs on your Mac. No cloud. No CUDA. No GPU rental.**

### 4. Update README with Visuals
Add the GIF and screenshots to README.md. This is your landing page.

Structure:
```
# NeuralForge
> Fine-tune LLMs on your Mac using Apple Neural Engine

[30-second demo GIF here]

## Features
- Train 110M+ parameter models on Apple Silicon
- 1.2 TFLOPS on Neural Engine — no GPU needed
- LoRA fine-tuning (2MB adapters vs 400MB full model)
- Export to GGUF, CoreML, llama2c
- Live loss curves, LR scheduling, checkpoints
- 100% on-device — your data never leaves your Mac

## Quick Start
bash setup.sh
open app/NeuralForge.xcodeproj  # ⌘R to run

[Screenshots here]
```

---

## Week 2: Launch Sequence

### Day 1 (Tuesday/Wednesday) — Hacker News

**Best time: 8:00 AM US Eastern (5:00 AM Pacific)**

Post as "Show HN":
```
Show HN: NeuralForge – Fine-tune LLMs on your Mac using Apple Neural Engine
```
URL: https://github.com/Khaeldur/NeuralForge

**First comment** (post immediately after submitting):
```
Hi HN, I built NeuralForge because I wanted to fine-tune small LLMs
on my MacBook without renting cloud GPUs or setting up CUDA.

It uses Apple's Neural Engine directly (not Metal, not CPU) to hit
~1.2 TFLOPS on a consumer Mac. The app wraps a C/Obj-C training
engine in a SwiftUI GUI with live loss curves, LoRA support, and
one-click export to GGUF/CoreML.

What actually works today:
- Real training (110M params, cosine LR schedule, checkpoint resume)
- Text generation (66ms/token, top-p sampling)
- LoRA adapters
- Export to GGUF, llama2c, CoreML

What's next: larger model support, data pipeline improvements.

The whole thing is MIT licensed. Happy to answer questions about
ANE internals — there's almost no documentation on using it for
training, so this was a lot of reverse engineering.
```

**HN Tips:**
- Stay online for 2+ hours and reply to EVERY comment
- Technical depth wins — explain ANE internals when asked
- Don't ask for upvotes (against rules, gets flagged)
- If it doesn't gain traction, you can repost in a week

### Day 1-2 — Reddit (Same Day or Next)

**r/LocalLLaMA** (380K+ members — your core audience)
```
Title: I built a macOS app for fine-tuning LLMs on Apple Neural Engine — no cloud, no CUDA

Body:
I've been working on NeuralForge, an open-source macOS app that
lets you fine-tune LLMs directly on your Mac using Apple's Neural
Engine.

**Why?** Every fine-tuning tool assumes you have an NVIDIA GPU.
If you're on a Mac, your options are basically MLX Python scripts
or renting cloud GPUs. I wanted a native app that just works.

**What it does:**
- Trains on Apple Neural Engine (~1.2 TFLOPS on M4)
- LoRA fine-tuning (rank 4-64)
- Live dashboard with loss curves
- Export to GGUF, CoreML, llama2c
- 100% on-device, your data stays on your Mac

**Current limitations (being honest):**
- Only supports llama2.c format models (110M tested)
- macOS 14+ / Apple Silicon required
- GUI covers core training flow, some advanced features still WIP

GitHub: https://github.com/Khaeldur/NeuralForge
Setup: `git clone && bash setup.sh` (downloads models, builds, runs tests)

Happy to answer questions or take feedback on what features matter most.
```

**Other subreddits** (post over 2-3 days, don't spam):
- r/MachineLearning — more academic, emphasize the ANE reverse-engineering
- r/swift — emphasize SwiftUI + native macOS development
- r/apple — "fine-tune AI on your Mac" angle
- r/selfhosted — privacy/on-device angle
- r/MLOps — tool announcement

### Day 3-4 — Twitter/X

**Thread format** (6-8 tweets):
```
Tweet 1:
I built an open-source macOS app that fine-tunes LLMs
on Apple Neural Engine.

No cloud. No CUDA. No GPU rental.

1.2 TFLOPS on a MacBook.

Thread 🧵

[attach demo GIF]

Tweet 2:
Why does this exist?

Every fine-tuning tool assumes NVIDIA GPUs.
If you're on a Mac, your options are:
- Python scripts (MLX)
- Rent cloud GPUs ($$$)
- Give up

NeuralForge: download, open, train.

Tweet 3:
What actually works:
✅ Real ANE training (110M params)
✅ LoRA fine-tuning
✅ Live loss curves + LR scheduling
✅ Checkpoint save/resume
✅ Text generation (66ms/token)
✅ Export → GGUF, CoreML, llama2c
✅ 648 automated tests

Tweet 4:
The hardest part was Apple Neural Engine.

There's basically zero documentation for using ANE for training.
It's designed for inference only.

I had to reverse-engineer the MIL compiler, figure out the
119-kernel limit, and build an exec() restart mechanism.

Tweet 5:
Who is this for?
- Mac developers wanting on-device AI
- Privacy-conscious teams (data never leaves your Mac)
- Students learning LLM fine-tuning
- Anyone who doesn't want to pay for cloud GPUs

Tweet 6:
It's fully open source (MIT):
https://github.com/Khaeldur/NeuralForge

bash setup.sh ← one command from clone to training

Star it if you think Macs should be first-class for AI. ⭐
```

**Tag/mention:** @_karpathy (his llama2.c is the model format), @awnihannun (MLX lead at Apple), @AppleMLResearch

### Day 5-7 — Long-Form Content

**Dev.to / Hashnode / Medium article:**
```
Title: "How I Built an LLM Training Engine on Apple Neural Engine
       (With Zero Documentation)"
```
Focus on the technical story:
- ANE has no training API — how you reverse-engineered it
- The 119-kernel compilation limit and exec() restart hack
- Getting 1.2 TFLOPS on consumer hardware
- Building a real-time NDJSON protocol between CLI and SwiftUI

This is the content that gets shared by developers. The technical depth is your moat.

---

## Week 3-4: Amplification

### Product Hunt Launch

**Prep checklist:**
- [ ] Create Product Hunt maker profile
- [ ] Upload 5 screenshots + demo GIF
- [ ] Write tagline: "Fine-tune LLMs on your Mac — no cloud, no CUDA"
- [ ] Schedule launch for Tuesday 12:01 AM PT
- [ ] Prepare first comment explaining what makes it unique
- [ ] Be online all day to respond to comments

**Key:** Product Hunt rewards engagement. Reply to every comment within 2 hours.

### GitHub Amplification
- Submit to [awesome-mac](https://github.com/jaywcjlove/awesome-mac)
- Submit to [open-source-mac-os-apps](https://github.com/serhii-londar/open-source-mac-os-apps)
- Submit to [awesome-mlops](https://github.com/visenger/awesome-mlops)
- Add GitHub topics (already done ✅)

### Community Engagement
- Answer Apple Silicon questions on r/LocalLLaMA linking to NeuralForge
- Comment on "can I fine-tune on Mac?" threads with your solution
- Join Discord servers: MLX Community, LocalLLaMA, Apple Developers

---

## Messaging Cheat Sheet

| Audience | Hook |
|----------|------|
| **r/LocalLLaMA** | "Fine-tune on Mac without CUDA — here's how" |
| **Hacker News** | "Using Apple Neural Engine for LLM training (zero docs existed)" |
| **r/swift** | "SwiftUI app that runs real ML training on ANE" |
| **Twitter AI crowd** | "1.2 TFLOPS on a MacBook. No cloud needed." |
| **Product Hunt** | "The missing fine-tuning app for Mac" |
| **r/apple** | "Your Mac can train AI models now" |
| **r/selfhosted** | "100% local LLM fine-tuning — zero data leaves your machine" |
| **Dev.to** | "I reverse-engineered Apple Neural Engine for training" |

---

## Metrics to Track

After launch week:
- GitHub stars (target: 100 in first week, 500 in first month)
- Clones / unique visitors (GitHub Insights)
- Issues opened (shows real users)
- Forks (shows developer interest)
- HN points + comment count
- Reddit upvotes

---

## What NOT to Do

1. **Don't oversell** — be honest about limitations (only 110M models, macOS only)
2. **Don't post everywhere same day** — spread over 5-7 days
3. **Don't ask for upvotes** — gets flagged on HN, banned on Reddit
4. **Don't hide the stub features** — if someone asks about cloud sync, say "planned, not done yet"
5. **Don't compare to Unsloth/Axolotl** — different category (Mac vs Linux/CUDA)
6. **Don't ignore comments** — every comment reply = more visibility
