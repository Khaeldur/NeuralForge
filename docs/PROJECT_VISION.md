# NeuralForge — Project Vision & Architecture

## What Is NeuralForge?

NeuralForge is an on-device LLM fine-tuning platform for macOS, powered by Apple's Neural Engine (ANE). It lets anyone fine-tune transformer models directly on their Mac — no cloud, no GPU rental, no data leaving the device.

Built on [maderix/ANE](https://github.com/maderix/ANE), which reverse-engineers Apple's private `AppleNeuralEngine.framework` for direct hardware access.

---

## Enterprise Vision

**Target:** On-device private fine-tuning platform for regulated industries (healthcare, legal, finance, defense).

**Value Proposition:**
- Data never leaves the device — HIPAA, SOX, ITAR, GDPR compliant by design
- Zero cloud costs — runs on hardware the company already owns
- No GPU shortage dependency — Apple Silicon is consumer hardware
- Simple enough for domain experts (doctors, lawyers) — not just ML engineers

**Market Opportunity:** $5-50M ARR for a fine-tuning platform that solves the data sovereignty problem enterprises face with cloud-based ML.

---

## Architecture: One App, Multiple Processes

NeuralForge uses a thin-app / heavy-CLI architecture. The SwiftUI app is a lightweight dashboard (~3MB) that spawns CLI subprocesses for all compute work. This keeps the app fast and responsive while enabling powerful features.

```
┌─────────────────────────────────────────┐
│            NeuralForge.app              │
│         (SwiftUI dashboard)             │
│                                         │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐  │
│  │Dashboard│ │Configure│ │ Generate │  │
│  │  View   │ │  View   │ │  View    │  │
│  └─────────┘ └─────────┘ └──────────┘  │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐  │
│  │ Ingest  │ │ Models  │ │ AI Assist│  │
│  │  View   │ │  View   │ │  View    │  │
│  └────┬────┘ └────┬────┘ └────┬─────┘  │
│       │           │           │         │
│   CLIRunner (spawns processes, reads NDJSON)
└───────┼───────────┼───────────┼─────────┘
        ▼           ▼           ▼
   neuralforge   neuralforge   Claude
     ingest       download      API
   (subprocess)  (subprocess)  (HTTPS)
```

### Design Principles

1. **App = Dashboard, CLI = Engine** — The app does zero compute. All training, inference, tokenization, and data processing run as CLI subprocesses.
2. **NDJSON Protocol** — CLI communicates via newline-delimited JSON on stdout. The app reads line-by-line and updates `@Published` SwiftUI state.
3. **Process Isolation** — Heavy work runs out-of-process. If the CLI crashes, the app survives. If the app closes, the CLI can be run standalone.
4. **External Storage** — Models/data stored in `~/Documents/NeuralForge/`, not bundled in the app. Users download what they need.

### Feature Weight Budget

| Component | Current Size | After Enterprise | Notes |
|-----------|-------------|-----------------|-------|
| NeuralForge.app | ~3 MB | ~4 MB | Just more SwiftUI views |
| neuralforge CLI | ~500 KB | ~800 KB | More subcommands |
| Sync daemon | — | ~100 KB | Optional LaunchAgent |
| Models | ~400 MB | 2-8 GB | User downloads, stored externally |

### Where Each Feature Lives

| Feature | Location | Why |
|---------|----------|-----|
| Document ingestion (PDF/DOCX) | `neuralforge ingest` CLI subcommand | Heavy parsing runs as subprocess |
| Fast tokenizer | Inside CLI binary (C code) | Replaces existing BPE, same binary |
| Audit log | CLIRunner.swift (append JSON) | ~20 lines, writes to `~/Library/Logs/` |
| Model downloads | `neuralforge download` CLI subcommand | Streams download progress via NDJSON |
| Team sync | LaunchAgent daemon | Separate tiny process, watches checkpoints |
| LLM assistant | In-app URLSession calls | Lightweight HTTPS to Claude API |

---

## Gap Analysis (vs. Enterprise PDF Vision)

### What's Done (v2.0)

| Capability | Status | Notes |
|-----------|--------|-------|
| ANE training engine | Done | Forward/backward on Neural Engine |
| Multi-model dimensions | Done | Runtime ModelConfig, not hardcoded |
| LoRA fine-tuning | Done | Rank 4-64, configurable targets |
| LR scheduler (cosine + warmup) | Done | Configurable in app |
| Data pipeline (multi-shard, shuffle) | Done | Train/val split, shuffle |
| Live dashboard (EMA, TFLOPS, val loss) | Done | Real-time charts |
| Text generation / inference | Done | Temperature, top-p sampling |
| Checkpoint save/resume | Done | Survives exec() restarts |
| GGUF + CoreML export | Done | Python converters |
| NDJSON app-CLI protocol | Done | Structured JSON streaming |
| exec() restart cycle | Done | Handles ANE kernel budget |
| Compile timer UX | Done | Orange banner with seconds counter |
| 109 CLI tests + 119 Swift tests | Done | Comprehensive coverage |

### What's Missing (Enterprise Gaps)

| Capability | Priority | Effort | Blocker? |
|-----------|----------|--------|----------|
| Fast tokenizer (BPE hangs >10KB) | P0 | 2-3 days | Yes — blocks document ingestion |
| Document ingestion (PDF/DOCX/TXT) | P1 | 3-4 days | Needs fast tokenizer first |
| Audit log (compliance) | P1 | 1 day | No |
| Real base models (TinyLlama, Phi-3) | P2 | 2-3 days | No |
| LLM assistant (Claude API) | P2 | 3-5 days | No |
| Team sync service | P3 | 5-7 days | No |
| Distributed ANE compute | P4 | weeks | Research phase |

### Estimated Timeline to Enterprise Beta

- **Weeks 1-2:** Fast tokenizer + document ingestion + audit log
- **Week 3:** Real base models + model download UI
- **Weeks 4-5:** LLM assistant integration
- **Week 6+:** Team sync, polish, security hardening

---

## Competitive Position

### Speed Comparison

| Platform | Hardware | TFLOPS | Cost |
|----------|----------|--------|------|
| NeuralForge | M4 Mac | ~2 | $0 (own hardware) |
| Google Colab (T4) | T4 GPU | ~8 | Free tier / $10/mo |
| Lambda Labs | A100 | ~300 | $1.10/hr |
| Cloud H100 | H100 | ~500 | $2-4/hr |

### Where NeuralForge Wins

NeuralForge is **not** competing on speed. It wins on:

1. **Privacy** — Data physically cannot leave the device. No trust required.
2. **Cost** — $0/month after hardware purchase. No cloud bills.
3. **Simplicity** — One app, one click to start training. No SSH, no Docker, no CUDA.
4. **Compliance** — On-device = HIPAA/SOX/ITAR compliant by architecture.
5. **Accessibility** — Runs on any Apple Silicon Mac. 800M+ devices in the wild.

---

## Long-Term Product Roadmap

### Phase 1: Core Platform (Done)
Single-model fine-tuning with live dashboard, LoRA, scheduling, data pipeline.

### Phase 2: Enterprise Ready (Next)
Document ingestion, audit logs, real base models, fast tokenizer.

### Phase 3: Intelligence Layer
LLM assistant, auto-hyperparameter tuning, output evaluation, dataset analysis.

### Phase 4: Collaboration
Team sync, shared model registry, centralized audit dashboard.

### Phase 5: Scale
Distributed ANE compute across Mac fleets, automatic model routing, A/B testing.
