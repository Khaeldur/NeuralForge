# NeuralForge — Project Instructions

## Startup Rules (ALWAYS follow these)

1. **Read the vision first.** At the start of every session, read `docs/PROJECT_VISION.md` and `docs/ROADMAP.md` to understand current state and priorities.
2. **Check what's done.** Run `cd cli && make test` and `cd app/Tests && swiftc -o test_swift -framework Foundation NeuralForgeTests.swift && ./test_swift` to verify the system is healthy before making changes.
3. **Work from the roadmap.** Pick the highest-priority incomplete item from `docs/ROADMAP.md`. Don't skip ahead.
4. **Checkpoint after every feature.** After completing any feature or significant change:
   - Run all tests (CLI + Swift)
   - Build the Xcode project
   - Update `docs/ROADMAP.md` to mark items as complete
   - Commit with a descriptive message
5. **Update docs on changes.** If you add a new CLI command, JSON message type, or app view, update `CLAUDE.md` and `docs/ARCHITECTURE.md`.
6. **Never break existing tests.** All 112 CLI + 119 Swift tests must pass after every change.

## Automated Work Checklist

When working autonomously on features, follow this sequence for each task:

```
1. Read docs/ROADMAP.md → identify next task
2. Read relevant source files → understand current code
3. Implement the change
4. Run CLI tests → must pass
5. Run Swift tests → must pass
6. Build Xcode project → must succeed
7. Update docs/ROADMAP.md → mark done, add notes
8. If feature is user-facing, verify with real run (5-step training or generation test)
9. Move to next task
```

## Project Overview

NeuralForge is an on-device LLM fine-tuning platform for macOS using Apple's Neural Engine (ANE). It has two main components:

1. **CLI** (`cli/`) — C/Objective-C binary that runs training, inference, tokenization on ANE
2. **App** (`app/`) — SwiftUI macOS app that spawns the CLI and displays a live training dashboard

The app communicates with the CLI via NDJSON (one JSON object per stdout line).

## Repository Structure

```
NeuralForge/
├── cli/                    # C/Obj-C training engine
│   ├── main.m              # CLI entry point, training loop, all commands
│   ├── config.h            # NFConfig struct + arg parsing
│   ├── progress.h          # NDJSON emission helpers
│   ├── tokenizer.h         # BPE tokenizer (encode/decode)
│   └── test_cli.m          # 109 CLI tests
├── app/                    # SwiftUI macOS app
│   ├── NeuralForge/
│   │   ├── NeuralForgeApp.swift      # App entry point
│   │   ├── Models/
│   │   │   ├── Project.swift         # NFProject + TrainingConfig
│   │   │   └── TrainingProgress.swift # CLIMessage parsing + TrainingState
│   │   ├── Services/
│   │   │   ├── CLIRunner.swift       # Process management, NDJSON reading
│   │   │   ├── ProjectManager.swift  # Project CRUD + persistence
│   │   │   └── DocumentImporter.swift # File import utilities
│   │   └── Views/
│   │       ├── MainView.swift        # Top-level navigation
│   │       ├── ProjectListView.swift # Project sidebar
│   │       ├── ProjectDetailView.swift # Tab container
│   │       ├── TrainingConfigView.swift # Training config form
│   │       ├── DashboardView.swift   # Live training dashboard
│   │       ├── GenerateView.swift    # Text generation UI
│   │       ├── ExportView.swift      # Model export
│   │       └── DataImportView.swift  # Data import
│   └── Tests/
│       └── NeuralForgeTests.swift    # 119 Swift tests
├── converters/             # Python export scripts
│   ├── gguf_export.py      # Checkpoint → GGUF
│   ├── gguf_to_llama2c.py  # GGUF → llama2.c
│   └── llama2c_to_coreml.py # llama2.c → CoreML
├── vendor/ANE/             # Vendored ANE framework (MIT)
│   └── training/
│       ├── stories_config.h   # ModelConfig, weight structs
│       ├── stories_mil.h      # MIL kernel generators (ANE programs)
│       ├── stories_cpu_ops.h  # CPU ops (rmsnorm, embed, LoRA)
│       ├── stories_io.h       # IOSurface I/O, blob building
│       ├── ane_classifier.h   # Classifier forward/backward
│       └── ane_rmsnorm_bwd.h  # RMSNorm backward pass
├── models/                 # Model weights + tokenizer (not in git)
│   ├── stories110M.bin     # 110M param LLaMA model
│   ├── tokenizer.bin       # BPE tokenizer (32K vocab)
│   └── tinystories_data00.bin # Tokenized training data
├── docs/                   # Documentation
└── scripts/                # Helper scripts
```

## Build & Test Commands

```bash
# CLI: build
cd cli && make clean && make

# CLI: run tests (109 tests)
cd cli && make test

# Swift: run tests (119 tests)
cd app/Tests && swiftc -o test_swift -framework Foundation NeuralForgeTests.swift && ./test_swift

# App: build via Xcode
cd app && xcodebuild -project NeuralForge.xcodeproj -scheme NeuralForge build

# Quick training sanity check
./cli/neuralforge train --model models/stories110M.bin --data models/tinystories_data00.bin --steps 5

# Text generation test
./cli/neuralforge generate --model models/stories110M.bin --prompt "Once upon a time" --max-tokens 50
```

## Key Technical Details

### ANE Training Pipeline
- ANE has a ~119 kernel compilation limit per process
- When approaching this limit, CLI does `exec()` restart: saves checkpoint, re-launches itself with `--resume`
- The exec() preserves the PID and stdout pipe, so the app doesn't notice
- First compile takes ~20-30 seconds (orange banner shows timer in app)
- Steady-state training: ~71ms/step on M4

### NDJSON Protocol (CLI → App)
All CLI output is JSON, one object per line:
- `{"type":"init","params":110000000,"layers":12,"dim":768,...}` — Model loaded
- `{"type":"step","step":1,"loss":5.23,"ms":42.0,...}` — Training step
- `{"type":"batch","batch":1,"avg_loss":4.8,...}` — Batch summary
- `{"type":"checkpoint","path":"...","step":100}` — Checkpoint saved
- `{"type":"restart","step":100,"compiles":86}` — exec() restart
- `{"type":"val","step":50,"val_loss":3.2}` — Validation loss
- `{"type":"token","token_id":1234,"text":"hello"}` — Generation token
- `{"type":"done","final_loss":1.8,...}` — Training complete
- `{"type":"error","message":"...","code":1}` — Error

### Model Format
Uses llama2.c format: 7-int header (dim, hidden, layers, heads, kv_heads, vocab, seq) followed by float32 weights. Checkpoint adds Adam states (m, v vectors).

### LoRA
Low-rank adapters on attention weights. Rank 4-64, configurable targets (Q/K/V/O). LoRA weights are tiny (~2MB for rank 8) vs full model (~400MB).

## Code Style

- **CLI (C/Obj-C):** Functions prefixed with `nf_`. Structs prefixed with `NF`. All in header files (single-compilation-unit pattern). Use `strlcpy` not `strcpy`. Validate all numeric inputs with `nf_safe_atoi`/`nf_safe_atof`.
- **Swift:** Standard SwiftUI patterns. `@Published` properties on `ObservableObject` classes. Views are structs. Use SF Symbols for icons.
- **Tests:** CLI tests use `assert()` in `test_cli.m`. Swift tests use custom `assert`/`assertEqual` functions in `NeuralForgeTests.swift`.

## Known Issues

- **Training data files may be corrupt** — Git LFS placeholder files (15 bytes) instead of actual data. Verify with `ls -la` before training.
- **screencapture requires permissions** — Cannot take screenshots programmatically without accessibility permissions.

## Recently Fixed

- **BPE tokenizer** — Was O(n^2), hung on >10KB. Replaced with O(n log n) priority queue algorithm. Now tokenizes 1MB in <1 second.

## Security Rules

- All CLI string inputs are bounds-checked (`strlcpy` with `PATH_MAX`)
- Numeric args use range-validated parsers (`nf_safe_atoi`, `nf_safe_atof`)
- NDJSON output escapes special characters to prevent injection
- Checkpoint files validated on load (magic bytes, version, size checks)
- No network access in CLI — everything runs locally
