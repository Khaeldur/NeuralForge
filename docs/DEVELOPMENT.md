# NeuralForge — Development Guide

## Prerequisites

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- Xcode 15+ (for SwiftUI app and build tools)
- Python 3 with `numpy`, `coremltools` (for converters — `pip install -r converters/requirements.txt`)
- ~2GB free disk space (for model files + build artifacts)

## Repository Setup

```bash
git clone <repo-url>
cd NeuralForge
bash setup.sh        # one command: builds CLI, downloads models, runs tests, builds app
```

Or do it manually:

### Download Model Files

```bash
bash scripts/download_model.sh
```

This downloads to `models/`:
- `stories110M.bin` — 110M parameter LLaMA model (~400MB)
- `tokenizer.bin` — BPE tokenizer (32K vocab, ~500KB)
- `tinystories_data00.bin` — Tokenized training data (~40MB)

**Verify downloads** (Git LFS can sometimes leave placeholder files):
```bash
ls -la models/
# stories110M.bin should be ~400MB, NOT 15 bytes
# If a file is tiny, re-download manually
```

---

## Building

### CLI Binary

```bash
cd cli
make clean && make
```

Produces `./neuralforge` binary. The Makefile links against:
- `AppleNeuralEngine.framework` (private, loaded at runtime)
- `IOSurface.framework`
- `CoreML.framework`
- `Accelerate.framework` (cblas)
- `Foundation.framework`

### macOS App

**Via Xcode:**
```bash
open app/NeuralForge.xcodeproj
# Press Cmd+R to build and run
```

**Via command line:**
```bash
cd app
xcodebuild -project NeuralForge.xcodeproj -scheme NeuralForge build
```

The app expects the CLI binary at `cli/neuralforge` relative to the project root. CLIRunner resolves this path at runtime.

---

## Running Tests

### CLI Tests (109 tests)

```bash
cd cli
make test
```

Compiles and runs `test_cli.m`. Tests cover:
- Config parsing and defaults
- Safe numeric parsers (overflow, NaN, edge cases)
- JSON protocol emission and escaping
- Tokenizer encode/decode round-trips
- LR scheduler computation (cosine, warmup)
- LoRA config validation
- Data pipeline config
- Security (path traversal, injection, bounds)
- Stability (large values, empty inputs)
- ModelConfig initialization

### Swift Tests (119 tests)

```bash
cd app/Tests
swiftc -o test_swift -framework Foundation NeuralForgeTests.swift
./test_swift
```

Tests cover:
- CLIMessage JSON parsing (all message types)
- TrainingState updates from messages
- NFProject and TrainingConfig serialization
- CLIRunner arg building
- Optimizer config validation
- LR scheduler config
- LoRA config
- Data pipeline config
- Security (path validation, encoding)
- Stability (edge cases, empty states)

### Full Verification Suite

Run everything in one go:
```bash
# One-click full suite (recommended)
make test-all

# Or individually:
cd cli && make clean && make && make test       # CLI tests
cd app/Tests && swiftc -o test_swift -framework Foundation NeuralForgeTests.swift && ./test_swift  # Swift tests
cd app && xcodebuild -project NeuralForge.xcodeproj -scheme NeuralForge build  # Xcode build

# Sanity check: train 5 steps
./cli/neuralforge train \
  --model models/stories110M.bin \
  --data models/tinystories_data00.bin \
  --steps 5

# Sanity check: generate text
./cli/neuralforge generate \
  --model models/stories110M.bin \
  --prompt "Once upon a time" \
  --max-tokens 20
```

---

## CLI Usage

### Training

```bash
# Basic training
./neuralforge train --model stories110M.bin --data tokens.bin --steps 1000

# With LR schedule + warmup
./neuralforge train --model stories110M.bin --data tokens.bin \
  --steps 5000 --warmup 100 --lr-schedule cosine --lr-min 1e-5

# With validation
./neuralforge train --model stories110M.bin --data tokens.bin \
  --val-data val_tokens.bin --val-every 50

# With LoRA
./neuralforge train --model stories110M.bin --data tokens.bin \
  --lora-rank 8 --lora-alpha 16

# Resume from checkpoint
./neuralforge train --resume --ckpt checkpoint.bin --data tokens.bin

# Full kitchen sink
./neuralforge train --model stories110M.bin --data tokens.bin \
  --steps 10000 --lr 1e-4 --accum 5 \
  --warmup 200 --lr-schedule cosine --lr-min 1e-6 \
  --val-data val.bin --val-every 100 --shuffle \
  --lora-rank 16 --lora-alpha 32 \
  --checkpoint-every 500 --seed 123
```

### Text Generation

```bash
./neuralforge generate --model stories110M.bin \
  --prompt "The little robot" \
  --temperature 0.8 --top-p 0.9 --max-tokens 200
```

### Tokenize

```bash
./neuralforge tokenize --input text.txt --output tokens.bin --tokenizer tokenizer.bin
```

**Warning:** The current BPE tokenizer is O(n^2) and will hang on files >10KB. For large files, use Python:
```python
# Generate token data with SentencePiece or similar
import struct, random
tokens = [random.randint(3, 31999) for _ in range(50000)]
with open('tokens.bin', 'wb') as f:
    for t in tokens:
        f.write(struct.pack('<H', t))
```

### Export

```bash
# To GGUF
python3 converters/gguf_export.py --ckpt checkpoint.bin --output model.gguf

# To CoreML
python3 converters/llama2c_to_coreml.py --llama2c model.bin --output Model.mlpackage
```

---

## Adding New Features

### Adding a New CLI Subcommand

1. Add config fields to `NFConfig` in `config.h`
2. Add arg parsing in `nf_config_from_args()`
3. Add `nf_cmd_yourcommand()` function in `main.m`
4. Add dispatch in `main()` command switch
5. Add NDJSON emission in `progress.h` if needed
6. Add tests in `test_cli.m`

### Adding a New App View

1. Create `YourView.swift` in `app/NeuralForge/Views/`
2. Add tab case in `ProjectDetailView.swift`
3. If it needs CLI interaction, add method to `CLIRunner.swift`
4. If it needs new JSON messages, add case to `CLIMessage` in `TrainingProgress.swift`
5. Add tests in `app/Tests/NeuralForgeTests.swift`

### Adding a New JSON Message Type

1. Add struct in `CLIMessage` (e.g., `struct YourMsg: Decodable { ... }`)
2. Add case to `CLIMessage` enum (e.g., `.your(YourMsg)`)
3. Add decode case in `CLIMessage.init(from:)`
4. Add handling in `TrainingState.handle(_:)`
5. Add `nf_emit_your()` in `progress.h`
6. Call it from the appropriate CLI command

---

## Debugging

### CLI

```bash
# Run with verbose output (not JSON)
./neuralforge train --model stories110M.bin --data tokens.bin --steps 5 2>&1

# Stderr goes to console, stdout is JSON
# Check stderr for ANE compilation errors, memory issues
```

### App

1. Open `app/NeuralForge.xcodeproj` in Xcode
2. Set breakpoints in `CLIRunner.swift` (line reading) or `TrainingState.handle()`
3. Press Cmd+R to build and run with debugger
4. Console shows CLI stderr output

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| "token data too small for sequence length" | Data file is tiny/corrupt | Check `ls -la` on data file, regenerate if needed |
| Training hangs after "Compiling" | Normal — ANE compile takes 20-30s | Wait for compile to finish |
| SIGTERM in Xcode | Something killed the app externally | Press Cmd+R to relaunch |
| Loss is NaN | Learning rate too high or corrupt data | Lower LR, verify data file |
| exec() restarts every ~10 steps | Normal ANE kernel budget behavior | Expected — dashboard shows orange banner |
