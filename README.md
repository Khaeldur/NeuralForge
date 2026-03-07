# NeuralForge

On-device AI fine-tuning for macOS, powered by Apple's Neural Engine.

NeuralForge lets you fine-tune transformer models directly on your Mac using the Apple Neural Engine (ANE). Your data never leaves your device. Built on top of [maderix/ANE](https://github.com/maderix/ANE), which reverse-engineers the private `AppleNeuralEngine.framework` for direct access to the neural hardware.

## Features

- **On-device training** — Fine-tune LLMs on Apple Silicon using the Neural Engine
- **Native macOS app** — SwiftUI dashboard with live loss charts, project management, and menu bar integration
- **LoRA support** — Memory-efficient fine-tuning with configurable rank and target layers
- **Text generation** — Interactive inference with temperature and top-p sampling
- **Data pipeline** — Multi-shard loading, train/val split, shuffle, and tokenization
- **LR scheduler** — Cosine annealing with warmup
- **Export** — GGUF (llama.cpp), CoreML, and llama2c formats
- **Distributed training** — Multi-Mac cluster via Bonjour with gradient aggregation
- **Cloud sync** — S3 and iCloud checkpoint backup
- **Enterprise audit** — Audit logging, compliance reports, and web dashboard
- **Quantization** — INT8 and INT4 weight quantization with calibration

## Architecture

```
NeuralForge/
├── cli/          # C/Obj-C CLI binary (training engine)
├── app/          # SwiftUI macOS app (39 source files)
│   ├── NeuralForge/
│   │   ├── Models/       # Project, TrainingProgress
│   │   ├── Views/        # 19 views (dashboard, config, export, etc.)
│   │   └── Services/     # 15 services (CLI runner, sync, cluster, etc.)
│   ├── NeuralForgeUITests/  # XCUITest end-to-end UI tests
│   └── Tests/            # 356 unit tests
├── converters/   # Python export scripts (GGUF, CoreML)
├── vendor/       # Vendored ANE framework (MIT)
├── scripts/      # Helper scripts
├── models/       # Model weights + tokenizer
└── docs/         # Architecture, roadmap, dev guide
```

**CLI** handles all heavy lifting: ANE kernel compilation, forward/backward passes, Adam optimizer, checkpointing. Communicates with the app via NDJSON on stdout.

**App** is a native SwiftUI macOS application that spawns the CLI as a subprocess, parses JSON progress, and renders a live training dashboard with EMA-smoothed loss charts.

## Requirements

- macOS 14+ with Apple Silicon (M1/M2/M3/M4)
- Xcode 15+ (for building)
- Python 3 with `numpy` (for converters)

## Quick Start

### 1. Build the CLI

```bash
cd cli
make
```

### 2. Download model weights

```bash
bash scripts/download_model.sh
```

This downloads:
- `stories110M.bin` — 110M parameter LLaMA model (llama2.c format)
- `tokenizer.bin` — BPE tokenizer (32K vocab)
- TinyStories tokenized data

### 3. Run training

```bash
./cli/neuralforge train \
  --model models/stories110M.bin \
  --data models/tinystories_data00.bin \
  --steps 100 \
  --warmup 10 \
  --lr-schedule cosine
```

### 4. Generate text

```bash
./cli/neuralforge generate \
  --model models/stories110M.bin \
  --prompt "Once upon a time" \
  --max-tokens 100 \
  --temperature 0.8
```

### 5. Build the macOS app

```bash
cd app
xcodebuild -project NeuralForge.xcodeproj -scheme NeuralForge build
```

Or open `app/NeuralForge.xcodeproj` in Xcode and press Run.

## CLI Commands

```
neuralforge train      [options]   Train a model
neuralforge generate   [options]   Generate text from a model
neuralforge tokenize   [options]   Tokenize text to binary tokens
neuralforge export     [options]   Export checkpoint to model format
neuralforge info       [options]   Show model info
neuralforge benchmark  [options]   Benchmark ANE forward pass speed
neuralforge help                   Show this help
```

### Training

```bash
neuralforge train --model stories110M.bin --data tokens.bin --steps 10000
neuralforge train --resume --ckpt checkpoint.bin --data tokens.bin
neuralforge train --lr 1e-4 --accum 5 --no-ane-extras
neuralforge train --warmup 100 --lr-schedule cosine --lr-min 1e-5
neuralforge train --val-data val_tokens.bin --val-every 100 --shuffle
neuralforge train --config config.json --steps 5000
neuralforge train --beta1 0.85 --beta2 0.995 --eps 1e-7 --grad-clip 0.5
```

Output is NDJSON — one JSON object per line:
```json
{"type":"init","params":110000000,"layers":12,"dim":768,...}
{"type":"step","step":1,"total":10000,"loss":5.23,"lr":0.0001,"ms":42.0,"tflops_ane":1.5,...}
{"type":"val","step":100,"val_loss":4.1}
{"type":"checkpoint","path":"checkpoint.bin","step":100,"loss":3.2}
{"type":"done","total_steps":10000,"final_loss":1.8,...}
```

### Generate

```bash
neuralforge generate --model stories110M.bin --prompt "The wizard" --temperature 0.9 --top-p 0.95 --max-tokens 200
```

### Tokenize

```bash
neuralforge tokenize --input my_data.txt --output tokens.bin --tokenizer tokenizer.bin
```

### Export

```bash
# Export to GGUF format (for llama.cpp)
neuralforge export --ckpt checkpoint.bin --format gguf --output model.gguf

# Export to llama2.c format (full weights)
neuralforge export --ckpt checkpoint.bin --format llama2c --output model.bin
```

## macOS App

The SwiftUI app provides a full GUI for the entire workflow:

- **Onboarding wizard** — Guided first-run setup with CLI path detection and HuggingFace token
- **Project management** — Create, configure, and manage multiple training projects
- **Live dashboard** — EMA-smoothed loss charts with validation overlay, TFLOPS monitor
- **Training config** — Learning rate, scheduler, LoRA rank, batch size, and more
- **Text generation** — Interactive inference with streaming output
- **Data import** — Drag & drop text files, tokenize directly in the app
- **Export** — One-click export to GGUF, CoreML, or llama2c
- **Model cards** — Auto-generated HuggingFace-style model cards
- **AI assistant** — Claude API integration for training guidance
- **Sync dashboard** — Local and cloud checkpoint sync status
- **Compute cluster** — Bonjour-discovered multi-Mac distributed training
- **Audit & compliance** — Audit trail, compliance reports, web dashboard
- **Benchmarks** — ANE performance profiling and perplexity evaluation
- **Training history** — Searchable log of all past training runs
- **Settings** — CLI path, API keys, default training parameters
- **Menu bar** — Live training progress in the macOS menu bar

## Python Converters

### GGUF Export (for llama.cpp)

```bash
pip install numpy
python3 converters/gguf_export.py --ckpt checkpoint.bin --output model.gguf
python3 converters/gguf_export.py --llama2c model.bin --output model.gguf --f16
```

### GGUF to llama2.c (reverse conversion)

```bash
python3 converters/gguf_to_llama2c.py --gguf model.gguf --output model.bin
```

### CoreML Export

```bash
pip install coremltools numpy
python3 converters/llama2c_to_coreml.py --llama2c model.bin --output Model.mlpackage
```

## How It Works

### ANE Training Pipeline

1. **Kernel Compilation**: MIL (Model Intermediate Language) programs are generated and compiled to ANE kernels
2. **Forward Pass**: 12 transformer layers run on ANE (attention, FFN, normalization)
3. **Loss Computation**: Cross-entropy loss on CPU
4. **Backward Pass**: Gradient computation split between ANE and CPU (via Accelerate/cblas)
5. **Adam Update**: Optimizer step on CPU
6. **Gradient Accumulation**: 10 micro-batches per optimizer step (default)

### exec() Restart

ANE has a ~119 kernel compilation limit per process. When approaching this limit, the CLI:
1. Saves a checkpoint
2. Emits `{"type":"restart",...}`
3. Calls `execl()` with `--resume` flag
4. The new process loads the checkpoint and continues

Since `exec()` replaces the process image but preserves the PID and file descriptors, the parent app's stdout pipe stays open — the restart is invisible to the SwiftUI app.

### App ↔ CLI Protocol

The app spawns the CLI via `Foundation.Process`, reads stdout line-by-line, and parses each line as a JSON `CLIMessage`. This drives `@Published` properties on `TrainingState`, which SwiftUI observes for live dashboard updates.

Stopping training sends SIGINT → the CLI catches it, saves a checkpoint, and exits gracefully.

## Running Tests

```bash
# CLI tests (152 tests)
cd cli && make test

# Swift unit tests (356 tests)
cd app/Tests && swiftc -o test_swift -framework Foundation NeuralForgeTests.swift && ./test_swift

# XCUITests (UI automation, requires Xcode)
xcodebuild test -project app/NeuralForge.xcodeproj -scheme NeuralForge -destination 'platform=macOS'

# Full build verification
cd app && xcodebuild -project NeuralForge.xcodeproj -scheme NeuralForge build
```

Total: **508 tests** (152 CLI + 356 Swift), 0 warnings, 39 source files.

## Performance

Measured on Apple M4 with Stories 110M (12-layer, dim=768, seq=256):

| Metric | Value |
|--------|-------|
| **Forward pass (ANE)** | 15.0 ms/step |
| **Forward TFLOPS** | 2.89 |
| **Training step (fwd+bwd)** | ~71 ms/step (steady state) |
| **Training TFLOPS (ANE)** | 1.48 |
| **Training TFLOPS (total)** | 2.44 |
| **Kernel compilation** | ~5.5s per batch (86 kernels) |
| **Checkpoint save** | 1.3 GB (weights + Adam states) |

The `--no-ane-extras` flag moves classifier/softmax/rmsnorm_bwd to CPU, which can be faster on some hardware:

| Config | Forward ms/step | TFLOPS |
|--------|----------------|--------|
| With ANE extras | 15.0 | 2.89 |
| Without ANE extras | 11.7 | 3.71 |

## Model Details

Default model: Stories 110M (LLaMA architecture)
- **Dimensions**: 768
- **Hidden**: 2048 (SwiGLU FFN)
- **Heads**: 12
- **Layers**: 12
- **Sequence Length**: 256
- **Vocabulary**: 32,000 (BPE)
- **Parameters**: ~110M

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | Technical architecture, data flow, ANE pipeline |
| [Development](docs/DEVELOPMENT.md) | Build, test, debug, and contribute |
| [Roadmap](docs/ROADMAP.md) | Feature status tracker and version history |
| [Project Vision](docs/PROJECT_VISION.md) | Enterprise vision, gap analysis, competitive position |
| [Data Ingestion Plan](docs/cron_job_data_ingestion_plan.md) | Automated data pipeline via launchd |
| [LLM Integration Plan](docs/llm_agent_integration_plan.md) | Claude API assistant integration |

## License

NeuralForge code is MIT. Vendored ANE code from [maderix/ANE](https://github.com/maderix/ANE) is also MIT.
