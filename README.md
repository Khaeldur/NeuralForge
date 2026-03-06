# NeuralForge

On-device AI fine-tuning for macOS, powered by Apple's Neural Engine.

NeuralForge lets you fine-tune transformer models directly on your Mac using the Apple Neural Engine (ANE). Your data never leaves your device. Built on top of [maderix/ANE](https://github.com/maderix/ANE), which reverse-engineers the private `AppleNeuralEngine.framework` for direct access to the neural hardware.

## Architecture

```
NeuralForge/
├── cli/          # C/Obj-C CLI binary (training engine)
├── app/          # SwiftUI macOS app (dashboard + project management)
├── converters/   # Python export scripts (GGUF, CoreML)
├── vendor/       # Vendored ANE framework (MIT)
├── scripts/      # Helper scripts
└── models/       # Model weights + tokenizer
```

**CLI** handles all heavy lifting: ANE kernel compilation, forward/backward passes, Adam optimizer, checkpointing. Communicates with the app via NDJSON on stdout.

**App** is a native SwiftUI macOS application that spawns the CLI as a subprocess, parses JSON progress, and renders a live training dashboard.

## Requirements

- macOS 13+ with Apple Silicon (M1/M2/M3/M4)
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
  --steps 100
```

### 4. Build the macOS app

```bash
cd app
xcodebuild -project NeuralForge.xcodeproj -scheme NeuralForge build
```

Or open `app/NeuralForge.xcodeproj` in Xcode and press Run.

## CLI Commands

```
neuralforge train      [options]   Train a model
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
neuralforge train --config config.json --steps 5000  # JSON config + CLI overrides
neuralforge train --beta1 0.85 --beta2 0.995 --eps 1e-7 --grad-clip 0.5
```

Output is NDJSON — one JSON object per line:
```json
{"type":"init","params":110000000,"layers":12,"dim":768,...}
{"type":"step","step":1,"total":10000,"loss":5.23,"ms":42.0,"tflops_ane":1.5,...}
{"type":"batch","batch":1,"step":10,"avg_loss":4.8,...}
{"type":"checkpoint","path":"checkpoint.bin","step":100,"loss":3.2}
{"type":"done","total_steps":10000,"final_loss":1.8,...}
```

### Tokenize

```bash
neuralforge tokenize --input my_data.txt --output tokens.bin --tokenizer tokenizer.bin
```

### Export

```bash
# Export to llama2.c format (full weights)
neuralforge export --ckpt checkpoint.bin --format llama2c --output model.bin

# Export to GGUF format (for llama.cpp)
neuralforge export --ckpt checkpoint.bin --format gguf --output model.gguf
```

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
# CLI tests (43 tests: config, JSON protocol, tokenizer, format, security, stability)
cd cli && make test

# Swift tests (32 tests: JSON parsing, project model, optimizer config, security, stability)
cd app/Tests && swiftc -o test_swift -framework Foundation NeuralForgeTests.swift && ./test_swift
```

## Model Details

Default model: Stories 110M (LLaMA architecture)
- **Dimensions**: 768
- **Hidden**: 2048 (SwiGLU FFN)
- **Heads**: 12
- **Layers**: 12
- **Sequence Length**: 256
- **Vocabulary**: 32,000 (BPE)
- **Parameters**: ~110M

## License

NeuralForge code is MIT. Vendored ANE code from [maderix/ANE](https://github.com/maderix/ANE) is also MIT.
