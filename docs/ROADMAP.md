# NeuralForge — Roadmap & Status Tracker

## Version History

### v1.0 — Core Training Engine (Complete)
- [x] ANE forward/backward pass pipeline
- [x] Adam optimizer with gradient accumulation
- [x] Checkpoint save/resume (survives exec() restarts)
- [x] NDJSON app-CLI communication protocol
- [x] SwiftUI macOS app with live dashboard
- [x] BPE tokenizer (encode + decode)
- [x] CLI commands: train, tokenize, export, info, benchmark
- [x] GGUF export (for llama.cpp)
- [x] CoreML export
- [x] Basic test suite (43 CLI + 32 Swift)

### v2.0 — Feature Platform (Complete)
Six major features added:

- [x] **E: LR Scheduler** — Cosine annealing with linear warmup
  - `--warmup N`, `--lr-min`, `--lr-schedule cosine`
  - LR displayed in dashboard, included in step JSON

- [x] **F: Data Pipeline** — Multi-shard, shuffle, train/val split
  - `--val-data`, `--val-every N`, `--shuffle`
  - Validation loss tracked and charted

- [x] **C: Live Charts** — EMA smoothing, TFLOPS chart, val loss overlay
  - EMA toggle (alpha=0.98), chart window picker (All/500/1K/2K)
  - TFLOPS over time, validation loss dashed overlay

- [x] **D: Text Generation** — Autoregressive inference with sampling
  - `neuralforge generate --prompt "..." --temperature 0.8 --top-p 0.9`
  - Streaming token output via NDJSON
  - GenerateView in app with parameter controls

- [x] **A: LoRA Fine-Tuning** — Low-rank adaptation
  - Rank 4-64, configurable alpha, target selection (Q/K/V/O)
  - Tiny checkpoints (~2MB), merge-on-export support

- [x] **B: Multi-Model Support** — Runtime dimensions
  - ModelConfig replaces compile-time #defines
  - All MIL generators and CPU ops parameterized

Additional v2.0 work:
- [x] Security audit (input validation, bounds checking, NDJSON escaping)
- [x] Compile timer UX (orange banner with seconds counter)
- [x] Expanded test suite (109 CLI + 119 Swift tests)

---

## v2.1 — Enterprise Foundations (Complete)

### P0: Fast Tokenizer ✅
- [x] Replace O(n^2) BPE with priority queue (max-heap) algorithm — O(n log n)
- [x] Target: tokenize 1MB text in <1 second — achieved: **936ms for 1MB**
- [x] Maintain compatibility with existing tokenizer.bin format — all 112 tests pass
- [x] Speed tests added: 10K (8.7ms), 100K (86ms), 1M (936ms)
- [x] CLI `tokenize` command now works on large files (previously hung on >10KB)
- **Status:** Complete.

### P1: Document Ingestion Pipeline ✅
- [x] `neuralforge ingest` CLI subcommand with full pipeline
- [x] PDF text extraction (via PDFKit/Quartz)
- [x] DOCX text extraction (via macOS textutil)
- [x] Plain text (.txt, .md, .csv, .json, code files) support
- [x] Code file support (.py, .js, .c, .m, .h, .swift, .rs, .go, .java, .ts, .tsx, .jsx)
- [x] Manifest file for shard tracking (JSON with version, timestamps, processed files)
- [x] Incremental mode (--incremental) — skips unchanged files based on mtime
- [x] Configurable shard size (--max-shard-mb, default 50MB)
- [x] App UI: IngestView with source/output folder pickers, shard size picker, incremental toggle
- [x] CLIRunner.ingest() with streaming per-file progress via NDJSON
- [x] Audit log integration for ingest events
- [x] 16 CLI tests + 12 Swift tests covering extraction, scanning, manifests, JSON parsing
- **Status:** Complete.

### P1: Audit Log ✅
- [x] Append-only JSONL log file (`~/Library/Logs/NeuralForge/audit.jsonl`)
- [x] Log: training start/stop, config used, checkpoint saves, exports, generation
- [x] Log: who ran what, when, with which data (user, timestamp, model, data paths)
- [x] Tamper detection — SHA-256 hash chain (prev_hash → hash per entry)
- [x] Hash chain verification function (`nf_audit_verify`) with tamper location detection
- [x] Convenience functions: `nf_audit_training_start/stop`, `checkpoint`, `export`, `generate`
- [x] 11 CLI tests + 11 Swift tests covering chain integrity, tamper detection, format validation
- **Status:** Complete.

### P2: Real Base Models ✅
- [x] Model registry with 5 built-in models (SmolLM 135M/360M/1.7B, TinyLlama 1.1B/1.1B-base)
- [x] `neuralforge models` CLI command (text + JSON output)
- [x] `neuralforge download` CLI command with streaming NDJSON progress
- [x] Python converter: HuggingFace safetensors → llama2.c format (convert_hf.py)
- [x] GQA → MHA KV head expansion for cross-architecture compatibility
- [x] Tokenizer conversion (HF tokenizer.json → tokenizer.bin)
- [x] Model card JSON metadata (model_card.json per download)
- [x] ModelCardView in app: browse models, download with progress, architecture details
- [x] 13 CLI tests + 8 Swift tests covering registry, search, JSON emission, download events
- **Status:** Complete.

### P2: LLM Assistant Integration ✅
- [x] Claude API client (NFIntelligence.swift) — async URLSession, rate limiting (5 req/min), retry
- [x] API key management — macOS Keychain storage (save/load/delete), settings UI
- [x] Training assistant chat view (AssistantView.swift) — full chat UI with message history
- [x] System prompt with live training context (model info, loss curve, config, TFLOPS)
- [x] Auto hyperparameter suggestions — analyzes setup, returns structured JSON, one-click apply
- [x] Generated text evaluation — fluency/coherence/creativity scoring with grammar analysis
- [x] Privacy-first design: only metadata sent, never weights or training data, 100% optional
- [x] 12 Swift tests covering message types, JSON parsing, context building, rate limiting
- **Status:** Complete.

---

## v3.0 — Collaboration (Complete)

### Centralized Audit Dashboard ✅
- [x] AuditLogReader.swift — JSONL parser with SHA-256 hash chain verification
- [x] AuditEntry model with computed properties (eventIcon, eventColor, summary, date)
- [x] AuditStats aggregation (entry counts, users, training time, best loss)
- [x] AuditVerification with chain integrity status and tamper location detection
- [x] AuditDashboardView.swift — full compliance audit log viewer
- [x] Stats bar (entries, trainings, checkpoints, exports, generations, train time, best loss)
- [x] Filter bar with event type picker and text search
- [x] Scrollable entry list with icons, seq numbers, event types, timestamps, truncated hashes
- [x] Hash chain verification sheet with visual pass/fail indicators
- [x] Entry detail sheet showing all audit fields + hash chain info
- [x] CSV export via NSSavePanel
- [x] 12 Swift tests covering entry parsing, chain format, verification, stats, CSV export
- **Status:** Complete.

### Team Sync Service ✅
- [x] SyncService.swift — checkpoint sync engine with configurable shared directory
- [x] Automatic checkpoint detection and sync across all projects
- [x] Shared model registry — browse synced checkpoints and models
- [x] LaunchAgent plist generation with configurable interval (5-120 min)
- [x] LaunchAgent install/uninstall via launchctl
- [x] Restore checkpoints from shared directory to any project
- [x] Sync status tracking (idle, syncing, success, error)
- [x] Pending sync detection (unsynced checkpoints)
- [x] SyncDashboardView.swift — full sync UI with setup, status, shared browser
- [x] Sync history with file sizes and step numbers
- [x] Project name sanitization for safe directory names
- [x] Sync tab added to ProjectDetailView
- [x] 12 Swift tests covering config, codable, paths, plist, sanitization
- **Status:** Complete. CloudKit/S3 backend deferred to future release.

### Cloud Sync Backend (Future)
- [ ] CloudKit or S3 backend for remote sync
- [ ] Conflict resolution for concurrent training

### Compliance Report Generation ✅
- [x] ComplianceReportGenerator.swift — generates structured reports from audit data
- [x] Three compliance frameworks: General Audit, HIPAA, SOX
- [x] HIPAA sections: §164.312(a-e) — audit controls, access control, integrity, authentication, transmission
- [x] SOX sections: §302 management assessment, §404 internal controls, separation of duties, config changes
- [x] Date range filtering for report period selection
- [x] Hash chain integrity verification integrated into all report types
- [x] Multi-user access tracking with threshold-based warnings
- [x] ComplianceReportView.swift — full report UI with framework picker, preview, export
- [x] Report status badges: Compliant (green), Needs Review (orange), Non-Compliant (red)
- [x] Section severity indicators: INFO, PASS, REVIEW, CRITICAL
- [x] Text export (plain text with formatted sections)
- [x] PDF export (via NSPrintOperation)
- [x] Reports tab added to ProjectDetailView
- [x] 12 Swift tests covering frameworks, statuses, sections, filtering, thresholds
- **Status:** Complete.

### Multi-User Audit Aggregation
- [ ] Web dashboard for audit log aggregation across machines
- [ ] Multi-user compliance reporting with merged logs

### Distributed ANE Compute ✅
- [x] ComputeClusterService.swift — Bonjour-based multi-Mac ANE compute cluster
- [x] DeviceCapabilities model — chip detection, ANE TFLOPS estimation, memory, CPU/GPU cores
- [x] IOPlatformUUID-based stable device identification
- [x] Chip family database: M1/M2/M3/M4 (base/Pro/Max/Ultra) with TFLOPS + GPU core estimates
- [x] NWListener-based service advertisement with TXT record metadata
- [x] NWBrowser-based automatic device discovery on local network
- [x] ClusterNode model with status tracking (discovered/available/training/syncing/error/offline)
- [x] TFLOPS-weighted shard distribution algorithm for data parallelism
- [x] Cluster metrics aggregation (total TFLOPS, total memory, node count)
- [x] ComputeClusterView.swift — full cluster dashboard UI
- [x] Local device info card with chip, memory, ANE TFLOPS, CPU cores
- [x] Discovered nodes list with status badges and capability display
- [x] Shard distribution visualization with proportional bars
- [x] Cluster tab added to ProjectDetailView
- [x] 12 Swift tests covering service type, status, TFLOPS/memory formatting, GPU/ANE estimates, shard distribution, device model
- **Status:** Complete. Gradient aggregation protocol deferred to future release.

---

## v4.0 — Productivity & Insights (Complete)

### Settings & Preferences ✅
- [x] Dedicated SettingsView.swift with 4-tab layout (General, Training, API Keys, About)
- [x] CLI binary path management with browse, auto-detect, and status indicator
- [x] Default training hyperparameters (steps, LR, accumulation, checkpoint, grad clip, seed)
- [x] Default scheduler settings (warmup, LR schedule, shuffle, LoRA rank)
- [x] API key management — Claude API + HuggingFace token via macOS Keychain
- [x] Export format defaults (GGUF, llama2c, CoreML)
- [x] Auto-save interval and max history entries settings
- [x] About tab with version info, CLI status, and feature summary
- [x] 12 Swift tests covering defaults, ranges, identifiers, options
- **Status:** Complete.

### Training Run History & Experiment Tracker ✅
- [x] TrainingHistoryService.swift — persist completed training runs to JSON
- [x] TrainingRun model with full metadata: project, model, config snapshot, results, timestamps
- [x] TrainingRunConfig snapshot preserving all hyperparameters at time of run
- [x] LossPoint model for serializable loss curve data
- [x] Loss curve downsampling for efficient storage (500 train + 200 val points max)
- [x] Auto-save on training completion via `recordCompletedRun()`
- [x] Run queries: by project, best run, recent, search (name/notes/model/LoRA)
- [x] CRUD operations: add, delete, batch delete, update notes, clear
- [x] TrainingHistoryView.swift — browse runs with sortable table (date, steps, loss, duration, LR, LoRA, TFLOPS)
- [x] Search bar with text filtering
- [x] Sort orders: newest, oldest, best loss, longest
- [x] Run detail sheet with loss chart, config, model info, notes
- [x] ComparisonSheet — side-by-side run comparison with overlaid loss curves
- [x] CSV export via NSSavePanel
- [x] Best run trophy indicator
- [x] History tab added to ProjectDetailView
- [x] 12 Swift tests covering model, formatting, codable, downsample, search, export
- **Status:** Complete.

### Model Evaluation Benchmarks ✅
- [x] BenchmarkService.swift — perplexity evaluation engine with persistent results
- [x] BenchmarkResult model: perplexity, avg loss, tokens, eval time, checkpoint info
- [x] Perplexity scoring via CLI `evaluatePerplexity` command
- [x] CLIRunner.evaluatePerplexity() — streaming batch evaluation with NDJSON progress
- [x] Checkpoint-to-checkpoint comparison with trend detection (improving/stable/degrading)
- [x] BenchmarkStats aggregation (best/worst/avg perplexity, trend analysis)
- [x] Automated quality regression detection with configurable thresholds
- [x] RegressionAlert model with warning (>0.5) and critical (>2.0) severity levels
- [x] BenchmarkView.swift — full evaluation UI with stats bar, perplexity chart, results table
- [x] Evaluation controls: data path picker, run button, streaming progress
- [x] BenchmarkDetailSheet with metrics, checkpoint info, metadata
- [x] CSV export for benchmark results
- [x] Best result star indicator
- [x] Benchmarks tab added to ProjectDetailView
- [x] 12 Swift tests covering perplexity math, formatting, trends, regression, stats, export
- **Status:** Complete.

---

## v5.0 — Polish & Production Readiness (Complete)

### Onboarding Wizard ✅
- [x] OnboardingView.swift — 4-page first-run wizard (Welcome, Setup, Goal, Ready)
- [x] CLI binary auto-detection from common paths + manual browse
- [x] HuggingFace token input with macOS Keychain storage
- [x] Training goal selection (Experiment / Fine-tune / Production) with per-goal defaults
- [x] Goal-based default hyperparameters (steps: 1K/5K/10K, LR: 3e-4/2e-4/1e-4)
- [x] First project creation on completion
- [x] @AppStorage("onboardingComplete") conditional routing in app entry point
- [x] Dynamic window sizing (620×520 onboarding, 1200×800 main)
- [x] NFKeychain extension for arbitrary service/account key pairs
- [x] 12 Swift tests covering goals, defaults, page nav, path validation
- **Status:** Complete.

### Menu Bar Integration ✅
- [x] MenuBarManager.swift — @MainActor singleton for training status
- [x] Real-time tracking: step, total, loss, best loss, TFLOPS, ms/step
- [x] Progress percentage and ETA calculation with timer-based elapsed tracking
- [x] Dynamic menu bar icon (bolt.fill when training, cpu when idle)
- [x] MenuBarView with training metrics grid (loss, TFLOPS, ms/step, elapsed, ETA)
- [x] Idle state display with "Open NeuralForge" and "Quit" actions
- [x] MenuBarExtra scene with .window style in app entry point
- [x] 10 Swift tests covering progress, ETA, formatting, icons, loss tracking
- **Status:** Complete.

### Quantization Service ✅
- [x] QuantizationService.swift — GGUF quantization and CoreML conversion pipeline
- [x] 8 quantization types: F16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0, Q3_K_M, Q2_K
- [x] Per-type metadata: bits/weight, quality rating (1-10), descriptions
- [x] Size estimation: estimateSize(modelParams:quantType:) and formatSize()
- [x] QuantizationJob model with status tracking (pending/running/success/failed)
- [x] CoreMLConfig with compute unit selection (All/CPU+GPU/CPU) and precision (F16/F32)
- [x] QuantizationService @MainActor singleton: quantizeGGUF(), convertCoreML()
- [x] ExportView updated with quantization picker, size estimates, export history
- [x] 18 Swift tests covering types, ordering, size estimation, formatting, jobs
- **Status:** Complete.

### Generate → Evaluate Pipeline ✅
- [x] Full eval pipeline in AssistantView.requestEvaluation()
- [x] Auto-detect tokenizer from model directory (tokenizer.bin, tokenizer.model)
- [x] Generate 3 text samples with diverse prompts via CLIRunner.generate()
- [x] Evaluate samples via Claude API (NFIntelligence.evaluateGeneratedText)
- [x] Display eval report and collected samples in AssistantView
- [x] 9 Swift tests covering prompts, tokenizer detection, path construction, params
- **Status:** Complete.

### Bug Fixes (15) ✅
- [x] Fixed 3 HIGH bugs: orphan CLIRunner, swallowed taps, broken selection checkboxes
- [x] Fixed 5 MEDIUM bugs: @StateObject → @ObservedObject for singletons, CSV export filtering, tokenizer auto-detection, SyncDashboardView status enum matching
- [x] Fixed 4 LOW bugs: deprecated onChange, sync config save, unused env object, eval stub
- **Status:** Complete. All 15 bugs verified fixed, BUILD SUCCEEDED.

### App Entry Point ✅
- [x] @AppStorage("onboardingComplete") conditional routing
- [x] Dynamic defaultSize based on onboarding state
- [x] MenuBarExtra scene with MenuBarView and dynamic status icon/title
- [x] EnvironmentObject injection for projectManager + cliRunner on all views
- [x] 8 Swift tests covering routing, window sizing, env objects
- **Status:** Complete.

---

## v5.1: Deferred Platform Features ✅

### CloudKit/S3 Remote Sync ✅
- [x] CloudSyncProvider protocol (upload, download, list, delete, testConnection)
- [x] S3SyncProvider with AWS Signature V4 (HMAC-SHA256), presigned URLs
- [x] CloudKitSyncProvider with CKContainer/CKDatabase/CKAsset (iCloud private DB)
- [x] CloudSyncConfig (Codable) with S3/CloudKit settings, Keychain credential storage
- [x] CloudSyncManager (@MainActor singleton): upload/download/list/sync/testConnection
- [x] 12 Swift tests covering config, errors, URL construction, credential handling
- **Status:** Complete.

### Gradient Aggregation Protocol ✅
- [x] GradientMessage wire protocol (Codable): assignWork, gradientReady, aggregated, heartbeat, syncCheckpoint
- [x] AggregationConfig: AllReduce, ParameterServer, GossipProtocol strategies
- [x] StragglerPolicy: Wait, Skip, Timeout modes
- [x] GradientAggregator (@MainActor singleton): coordinator/worker modes, all-reduce averaging, ring-reduce
- [x] GradientMetrics (ObservableObject): rounds, throughput, straggler/failure counts, rolling averages
- [x] Gradient compression (threshold-based sparsification) and checksum verification
- [x] 15 Swift tests covering strategies, metrics, compression, ring topology
- **Status:** Complete.

### Multi-User Audit Web Dashboard ✅
- [x] WebDashboardConfig (Codable): port, bind address, auth token, refresh interval
- [x] AuditAggregator: local log scanning, multi-machine sync directory scanning, entry merging
- [x] AuditAPIHandler: HTTP request parsing, 6 REST routes (/, /api/entries, /api/stats, /api/verify, /api/machines, /health)
- [x] Full HTML dashboard with dark theme, stats cards, filter bar, audit entry table, auto-refresh
- [x] AuditWebServer (@MainActor singleton): NWListener-based HTTP server, CORS support, bearer token auth
- [x] Thread-safe connection handling with ObjectIdentifier-based tracking
- [x] 18 Swift tests covering config, URL generation, request parsing, response serialization, auth
- **Status:** Complete.

---

## Test Coverage

| Component | Tests | Last Verified |
|-----------|-------|---------------|
| CLI (test_cli.m) | 152 | 2026-03-07 |
| Swift (NeuralForgeTests.swift) | 356 | 2026-03-07 |
| Xcode build (39 source files) | SUCCEEDED | 2026-03-07 |
| Real training (50 steps) | PASSED | 2025-03-07 |
| Real generation (100 tokens) | PASSED | 2025-03-07 |
| CLI tokenize (45KB file) | PASSED | 2025-03-07 |

---

## Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| ~~BPE tokenizer O(n^2), hangs on >10KB~~ | ~~High~~ | **Fixed** — replaced with O(n log n) heap |
| Training data may be Git LFS placeholder (15 bytes) | Medium | Workaround: regenerate with Python |
| ~~Tokenizer hangs CLI `tokenize` command on large files~~ | ~~High~~ | **Fixed** — same fix, 1MB in <1s |
| First ANE compile takes 20-30s (no visible progress) | Low | Fixed — compile timer added |
