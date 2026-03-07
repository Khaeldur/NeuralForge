# NeuralForge — Technical Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                   NeuralForge.app                        │
│                (SwiftUI + AppKit)                         │
│                                                          │
│  ProjectManager ──▶ NFProject (Codable, persisted)       │
│  CLIRunner ──▶ Foundation.Process ──▶ stdout NDJSON      │
│  TrainingState ──▶ @Published ──▶ SwiftUI Views          │
│                                                          │
│  Views: Dashboard │ Config │ Generate │ Export │ Import   │
└──────────────────────┬───────────────────────────────────┘
                       │ stdin/stdout pipe
                       ▼
┌─────────────────────────────────────────────────────────┐
│              neuralforge CLI (C / Obj-C)                 │
│                                                          │
│  Commands: train │ generate │ tokenize │ export │ info   │
│                                                          │
│  ┌─────────────────────────────────────────────────┐     │
│  │            Training Loop                         │     │
│  │                                                  │     │
│  │  1. Load model weights → malloc + mmap           │     │
│  │  2. Compile MIL → ANE kernels (20-30s first time)│     │
│  │  3. Forward pass (ANE) → loss (CPU)              │     │
│  │  4. Backward pass (ANE + CPU cblas)              │     │
│  │  5. Adam update (CPU)                            │     │
│  │  6. Repeat 2-5 with gradient accumulation        │     │
│  │  7. Emit step JSON, checkpoint periodically      │     │
│  │  8. exec() restart when kernel budget reached    │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────┬───────────────────────────────────┘
                       │ AppleNeuralEngine.framework (private)
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Apple Neural Engine (ANE)                    │
│                                                          │
│  IOSurface FP16 buffers ──▶ ANE hardware compute         │
│  MIL programs compiled to ANE kernels                    │
│  ~119 kernel budget per process lifetime                 │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### CLI (`cli/`)

Single-compilation-unit C/Obj-C binary. All code lives in header files included by `main.m`.

| File | Purpose | Key Functions |
|------|---------|---------------|
| `main.m` | Entry point, command dispatch, training loop | `nf_cmd_train()`, `nf_cmd_generate()`, `nf_cmd_tokenize()`, `nf_cmd_export()` |
| `config.h` | NFConfig struct, arg parsing, safe numeric parsers | `nf_config_from_args()`, `nf_safe_atoi()`, `nf_safe_atof()` |
| `progress.h` | NDJSON emission to stdout | `nf_emit_init()`, `nf_emit_step()`, `nf_emit_val()`, `nf_emit_done()` |
| `tokenizer.h` | BPE tokenizer encode/decode | `nf_tokenizer_load()`, `nf_tokenizer_encode()`, `nf_tokenizer_decode()` |
| `test_cli.m` | 109 unit tests | Config, JSON, tokenizer, format, security, stability tests |

### ANE Vendor Code (`vendor/ANE/training/`)

Reverse-engineered Apple Neural Engine interface.

| File | Purpose |
|------|---------|
| `stories_config.h` | ModelConfig struct, LayerWeights/Acts/Grads/Adam structs |
| `stories_mil.h` | MIL kernel generators: SDPA fwd/bwd, FFN fwd/bwd, QKV bias |
| `stories_cpu_ops.h` | CPU fallback ops: rmsnorm, embed, cross-entropy, LoRA fwd/bwd |
| `stories_io.h` | IOSurface creation, blob building, weight I/O |
| `ane_classifier.h` | Classifier (embed→logits) forward/backward on ANE |
| `ane_rmsnorm_bwd.h` | RMSNorm backward pass on ANE |

### App (`app/`)

Standard SwiftUI macOS app with MVVM-ish architecture.

**Models:**
- `NFProject` — Codable struct with model/data paths + `TrainingConfig`
- `TrainingConfig` — All training hyperparameters (LR, schedule, LoRA, etc.)
- `CLIMessage` — Decodable enum for all NDJSON message types
- `TrainingState` — `@MainActor ObservableObject` with `@Published` properties

**Services:**
- `CLIRunner` — `ObservableObject` that manages `Foundation.Process`, reads stdout, parses JSON, updates `TrainingState`
- `ProjectManager` — CRUD for NFProject, persists to Application Support

**Views:**
- `MainView` → `ProjectListView` (sidebar) + `ProjectDetailView` (detail)
- `ProjectDetailView` → tabs: Config | Dashboard | Generate | Export | Import
- `DashboardView` — Live loss chart (Swift Charts), metric cards, compile timer, TFLOPS chart
- `TrainingConfigView` — Form for all training parameters
- `GenerateView` — Prompt input + streaming token output

## Data Flow

### Training

```
User clicks "Start" in DashboardView
  → CLIRunner.startTraining(project:)
    → Foundation.Process(neuralforge train --model X --data Y --steps N ...)
      → CLI loads model, compiles ANE kernels
        → Emits {"type":"init",...} on stdout
      → Training loop runs
        → Emits {"type":"step",...} every step
        → Emits {"type":"batch",...} every accum_steps
        → Emits {"type":"checkpoint",...} every checkpoint_every
        → Emits {"type":"val",...} every val_every
        → Emits {"type":"restart",...} on exec() restart
      → Training completes
        → Emits {"type":"done",...}
    → CLIRunner reads each line, decodes CLIMessage
      → TrainingState.handle(msg) updates @Published properties
        → SwiftUI views re-render automatically
```

### Text Generation

```
User enters prompt in GenerateView, clicks "Generate"
  → CLIRunner.generate(project:, prompt:, ...)
    → Process(neuralforge generate --model X --prompt "..." --max-tokens N)
      → CLI loads model, compiles forward-only kernels
      → Autoregressive loop:
        → Forward pass → sample token
        → Emits {"type":"token","text":"word"} per token
      → Emits {"type":"generate_done","tokens":N,"total_ms":M}
    → CLIRunner streams tokens to GenerateView
      → Text appears word-by-word
```

## exec() Restart Mechanism

ANE has a ~119 kernel compilation limit per process. The training loop compiles ~86 kernels per batch (forward + backward for 12 layers). After approaching the limit:

1. CLI saves checkpoint to disk
2. Emits `{"type":"restart","step":N,"compiles":86}`
3. Calls `execl(argv[0], ..., "--resume", "--ckpt", ckpt_path, ...)`
4. New process image loads, resumes from checkpoint
5. PID stays the same, stdout pipe stays open
6. App sees a brief pause (recompilation) then training continues

This is transparent to the user — the dashboard shows an orange "Recompiling ANE kernels..." banner with a timer.

## Model Architecture (LLaMA)

Default model: Stories 110M

| Parameter | Value |
|-----------|-------|
| Dimensions | 768 |
| Hidden (FFN) | 2048 (SwiGLU) |
| Attention Heads | 12 |
| Head Dimension | 64 |
| Layers | 12 |
| Sequence Length | 256 |
| Vocabulary | 32,000 (BPE) |
| Total Parameters | ~110M |

With `ModelConfig` (Feature B), these are all runtime values read from the model file header — not compile-time constants.

## LoRA Architecture

```
Base weight W [out_dim, in_dim]  ← FROZEN during LoRA training

LoRA adapter:
  A [rank, in_dim]   ← trained (initialized with kaiming uniform)
  B [out_dim, rank]   ← trained (initialized to zero)

Forward:  output = x @ W^T + (x @ A^T @ B^T) * (alpha / rank)
```

LoRA params for rank=8, all attention targets: 4 * 2 * 768 * 8 * 12 = 589K (0.5% of base model).

## Performance (M4)

| Metric | Value |
|--------|-------|
| Forward pass (ANE) | 15.0 ms/step |
| Forward TFLOPS | 2.89 |
| Training step (fwd+bwd) | ~71 ms/step |
| Training TFLOPS (ANE) | 1.48 |
| Training TFLOPS (total) | 2.44 |
| Kernel compilation | ~5.5s per batch |
| Checkpoint size | ~1.3 GB |
