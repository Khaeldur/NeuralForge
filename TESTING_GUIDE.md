# NeuralForge — Testing Guide

## Quick Start

```bash
# One command to test everything
make test-all          # ~3 min — unit + build + integration

# Other modes
make test-quick        # ~5 sec — unit tests only
make test-integration  # ~3 min — real training on hardware
make test-ui           # needs Accessibility permission
```

---

## What You Can Actually Test (Working Features)

### 1. Core Training Loop
Open the app (⌘R in Xcode), then:
1. Click **"+"** to create a new project
2. Go to **Config** tab in sidebar
3. Set:
   - Model: `models/stories110M.bin`
   - Tokenizer: `models/tokenizer.bin`
   - Data: `models/tinystories_data00.bin`
   - Steps: 20, LR: 3e-4, Batch: 4
4. Click **Start Training**
5. Watch the loss curve update live on Dashboard

**What to verify:**
- Loss should decrease from ~11.0 to ~10.x over 20 steps
- TFLOPS counter should show ~1.0-1.2
- Step counter advances
- Stop button works, training halts

### 2. Text Generation
1. After training (or with base model), go to **Generate** tab
2. Type a prompt: "Once upon a time"
3. Set temperature: 0.8, top-p: 0.9, max tokens: 50
4. Click Generate

**What to verify:**
- Text appears in output area
- Coherent TinyStories-style output
- Different temperatures give different results
- Temperature 0.0 gives deterministic output

### 3. LR Scheduler
1. In Config tab, enable LR Scheduler
2. Set: Cosine Annealing, Warmup Steps: 5
3. Train for 20 steps

**What to verify (in terminal output):**
- Steps 0-4: LR ramps up (warmup)
- Steps 5-19: LR decays following cosine curve

### 4. Checkpoint Save & Resume
1. Train for 10 steps → stop
2. Note the last loss value
3. Start training again

**What to verify:**
- Training resumes from step 10
- Loss continues from where it left off
- `neuralforge_ckpt.bin` exists (1.3GB)

### 5. Model Export
1. After training, go to **Export** tab
2. Select format: GGUF / llama2c / CoreML
3. Set output path
4. Click Export

**What to verify:**
- Output file created at specified path
- File size matches model dimensions

### 6. Training Profiles
1. Go to Config tab
2. Configure hyperparameters
3. Save as profile (e.g., "Fast Debug")
4. Change settings, then load the saved profile

**What to verify:**
- Profile saves all config values
- Loading restores them exactly
- Built-in presets (Conservative, Aggressive, etc.) work

### 7. Document Import
1. Go to **Data** tab
2. Click import, select a PDF or DOCX file

**What to verify:**
- File appears in import list
- Text extraction shows content preview

### 8. Project Management
1. Create 3+ projects with different names
2. Switch between them
3. Delete one

**What to verify:**
- Each project saves its own config
- Switching projects loads correct config
- Delete removes project from list

### 9. Settings
1. Open Settings (⌘,)
2. Set CLI binary path
3. Change training defaults

**What to verify:**
- Settings persist after app restart
- CLI path is validated

---

## CLI Testing (Terminal)

### Basic Training
```bash
cd cli
./neuralforge train \
  --model ../models/stories110M.bin \
  --tokenizer ../models/tokenizer.bin \
  --data ../models/tinystories_data00.bin \
  --steps 10 --lr 3e-4 --batch 4
```

### Text Generation
```bash
./neuralforge generate \
  --model ../models/stories110M.bin \
  --tokenizer ../models/tokenizer.bin \
  --prompt "Once upon a time" \
  --tokens 50 --temperature 0.8 --topp 0.9
```

### Determinism Check
```bash
# Run twice with same seed — output should be identical
./neuralforge generate --model ../models/stories110M.bin \
  --tokenizer ../models/tokenizer.bin \
  --prompt "Hello" --tokens 20 --seed 42

./neuralforge generate --model ../models/stories110M.bin \
  --tokenizer ../models/tokenizer.bin \
  --prompt "Hello" --tokens 20 --seed 42
```

### Model Info
```bash
./neuralforge info --model ../models/stories110M.bin
```

---

## Automated Test Layers

### Layer 1: Unit Tests (568 tests, ~5s)
```bash
make test              # or make test-cli && make test-swift
```
- **152 CLI tests**: training math, tokenizer, export formats, LoRA gradients
- **416 Swift tests**: JSON parsing, state transitions, project CRUD, config serialization

### Layer 2: Build Verification (~10s)
```bash
make build && make build-app
```
- CLI binary compiles
- Xcode app builds with 0 warnings
- 53 XCUITests compile

### Layer 3: Integration Tests (27 tests, ~3min)
```bash
make test-integration
```
- Real 110M model training on Apple Neural Engine
- LR scheduler cosine decay verification
- Checkpoint save → resume cycle
- Text generation with top-p sampling
- Seed determinism (same seed = same tokens)
- Error handling (missing model → proper error)

### Layer 4: XCUITests (53 tests)
```bash
make test-ui           # needs System Settings → Accessibility
```
- Sidebar navigation (all 14 tabs)
- Project create/switch/delete
- Config interactions
- Keyboard shortcuts

---

## What to Look For (Known Limitations)

### These Features Have UI But Don't Work Yet
| Feature | What You'll See | What's Actually Happening |
|---------|----------------|--------------------------|
| Cloud Sync | Settings tab with API key fields | Keys saved but sync never triggers |
| Webhooks | Webhook config panel | Config saved but no events fire |
| Compute Cluster | Cluster discovery view | Bonjour coded but never searches |
| Claude Assistant | Chat interface | API key field exists but chat doesn't send |
| Training History | Empty history list | Runs are never persisted |
| Audit Dashboard | Empty audit viewer | No audit events are logged |
| Compliance Reports | Empty report view | No reports generated |
| Data Ingest/Shard | Ingest UI | Sharding logic not wired |
| MLX Backend | Backend selector | Detection works, execution doesn't |
| Benchmarks | Benchmark view | Shows hardcoded sample data |

### Real vs Demo
- **Dashboard loss chart**: REAL — plots actual training loss
- **TFLOPS counter**: REAL — measured from ANE kernel timing
- **Benchmark results**: DEMO — hardcoded sample data
- **Cluster nodes**: DEMO — no actual discovery
- **Sync status**: DEMO — always shows idle
