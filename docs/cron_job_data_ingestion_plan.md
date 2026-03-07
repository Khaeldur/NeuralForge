# NeuralForge — Cron Job Data Ingestion Plan

## Assessment: Is Automated Data Ingestion Needed?

**Short answer: Not for the core product, but beneficial as an optional power-user feature.**

NeuralForge is a desktop macOS app for on-device LLM fine-tuning. Its current data pipeline is manual and intentionally simple:

```
User provides text → CLI tokenizes → .bin → User launches training
```

However, there are legitimate use cases where automated/scheduled data ingestion adds value:

1. **Continuous fine-tuning** — A developer training on their own codebase, docs, or chat logs that grow daily
2. **Dataset curation pipelines** — Scraping/collecting text from sources, tokenizing, and preparing for the next training run
3. **Multi-shard management** — Splitting large corpora into shards and rotating them for training diversity

---

## Architecture

### Option A: Lightweight launchd Plist (Recommended)

macOS native scheduling via `launchd` — no external dependencies, respects system sleep/wake, integrates with macOS power management.

```
┌─────────────────────────────────────────────────┐
│  launchd (macOS scheduler)                      │
│  Triggers: every N hours / on file change       │
├─────────────────────────────────────────────────┤
│  neuralforge ingest                             │
│  ├── Scan source directories for new .txt files │
│  ├── Tokenize new files → .bin shards           │
│  ├── Update manifest.json (shard registry)      │
│  └── Emit JSON status to log file               │
├─────────────────────────────────────────────────┤
│  NeuralForge app (reads manifest on next train) │
│  └── Auto-discovers new shards from manifest    │
└─────────────────────────────────────────────────┘
```

### Implementation

#### 1. New CLI Command: `neuralforge ingest`

Add to `cli/main.m`:

```c
// nf_cmd_ingest() — Scan source dirs, tokenize new files, update manifest
// Args:
//   --source <dir>         Directory to scan for .txt files
//   --output <dir>         Directory for output .bin shards
//   --tokenizer <path>     Path to tokenizer.bin
//   --max-shard-mb <N>     Max shard size in MB (default: 50)
//   --manifest <path>      Path to manifest.json
//   --incremental          Only process files newer than last run
```

Logic:
1. Read `manifest.json` to get last-run timestamp and known files
2. Scan `--source` directory for `.txt` files modified after last run
3. For each new/modified file:
   - Tokenize with `nf_tokenizer_encode()`
   - Append tokens to current shard until `--max-shard-mb` reached
   - Start new shard when size limit hit
4. Update `manifest.json` with new shard list and timestamp
5. Emit NDJSON status: `{"type":"ingest","new_files":5,"new_tokens":150000,"shards":3}`

#### 2. Manifest File Format

```json
{
  "version": 1,
  "last_run": "2024-01-15T10:30:00Z",
  "tokenizer": "tokenizer.bin",
  "vocab_size": 32000,
  "shards": [
    {
      "path": "shard_000.bin",
      "tokens": 5000000,
      "bytes": 10000000,
      "created": "2024-01-14T08:00:00Z",
      "source_files": ["doc1.txt", "doc2.txt"]
    },
    {
      "path": "shard_001.bin",
      "tokens": 4500000,
      "bytes": 9000000,
      "created": "2024-01-15T10:30:00Z",
      "source_files": ["doc3.txt"]
    }
  ],
  "total_tokens": 9500000,
  "processed_files": {
    "doc1.txt": {"mtime": "2024-01-14T07:00:00Z", "tokens": 2500000},
    "doc2.txt": {"mtime": "2024-01-14T07:30:00Z", "tokens": 2500000},
    "doc3.txt": {"mtime": "2024-01-15T09:00:00Z", "tokens": 4500000}
  }
}
```

#### 3. launchd Plist

File: `~/Library/LaunchAgents/com.neuralforge.ingest.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.neuralforge.ingest</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/neuralforge</string>
        <string>ingest</string>
        <string>--source</string>
        <string>/Users/USERNAME/Documents/training_text</string>
        <string>--output</string>
        <string>/Users/USERNAME/Documents/NeuralForge/data</string>
        <string>--tokenizer</string>
        <string>/Users/USERNAME/Documents/NeuralForge/models/tokenizer.bin</string>
        <string>--manifest</string>
        <string>/Users/USERNAME/Documents/NeuralForge/data/manifest.json</string>
        <string>--incremental</string>
    </array>
    <key>StartInterval</key>
    <integer>3600</integer>  <!-- Every hour -->
    <key>StandardOutPath</key>
    <string>/Users/USERNAME/Documents/NeuralForge/logs/ingest.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/USERNAME/Documents/NeuralForge/logs/ingest_err.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
```

Install: `launchctl load ~/Library/LaunchAgents/com.neuralforge.ingest.plist`

#### 4. App Integration

The NeuralForge SwiftUI app would add:

- **Data Source picker** in project settings: select source directory for text files
- **Auto-discover shards**: Read `manifest.json` to show available training data
- **Ingest status**: Show last ingest time, new files count, total tokens
- **Schedule toggle**: Install/uninstall the launchd plist from the app

---

## Alternative: WatchPaths (File System Trigger)

Instead of periodic scheduling, use macOS file system events:

```xml
<key>WatchPaths</key>
<array>
    <string>/Users/USERNAME/Documents/training_text</string>
</array>
```

This triggers the ingest command whenever files change in the watched directory — more responsive than polling.

---

## Scope Estimate

| Component | Lines | Effort |
|-----------|-------|--------|
| `nf_cmd_ingest()` in main.m | ~150 | Medium |
| Manifest JSON read/write | ~80 | Low |
| launchd plist template | ~30 | Low |
| App UI (data source picker) | ~100 | Medium |
| **Total** | **~360** | **1-2 days** |

---

## Recommendation

**Implement as Phase 2 feature.** The current manual pipeline (drag-and-drop `.txt` → tokenize → train) covers 90% of use cases. The cron job adds value for power users with continuously growing datasets, but is not blocking for the v2.0 release. If implemented, the `neuralforge ingest` CLI command is the critical piece — the launchd plist and app UI are straightforward follow-ups.
