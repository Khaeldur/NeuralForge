// CLIRunner.swift — Manages neuralforge CLI subprocess + JSON parsing

import Foundation
import AppKit

@MainActor
class CLIRunner: ObservableObject {
    @Published var isRunning = false
    @Published var state = TrainingState()

    private var process: Process?
    private var outputPipe: Pipe?
    private var stderrPipe: Pipe?
    private(set) var cliBinaryPath: String

    init() {
        // Search order: bundle → env var → sibling to app → well-known paths
        cliBinaryPath = CLIRunner.findCLIBinary()
    }

    private static func findCLIBinary() -> String {
        // 1. Bundled inside .app
        if let bundled = Bundle.main.url(forResource: "neuralforge", withExtension: nil)?.path,
           FileManager.default.isExecutableFile(atPath: bundled) {
            return bundled
        }

        // 2. Environment variable
        if let env = ProcessInfo.processInfo.environment["NEURALFORGE_CLI"],
           FileManager.default.isExecutableFile(atPath: env) {
            return env
        }

        // 3. Relative to app bundle (dev builds: app is in DerivedData, CLI is in project)
        if let bundlePath = Bundle.main.bundlePath as NSString? {
            // Walk up to find NeuralForge project root
            var dir = bundlePath as String
            for _ in 0..<10 {
                let candidate = (dir as NSString).appendingPathComponent("cli/neuralforge")
                if FileManager.default.isExecutableFile(atPath: candidate) {
                    return candidate
                }
                dir = (dir as NSString).deletingLastPathComponent
            }
        }

        // 4. UserDefaults (saved from settings)
        if let saved = UserDefaults.standard.string(forKey: "CLIBinaryPath"),
           FileManager.default.isExecutableFile(atPath: saved) {
            return saved
        }

        // 5. Common locations
        let commonPaths = [
            NSHomeDirectory() + "/Desktop/sl/NeuralForge/cli/neuralforge",
            "/usr/local/bin/neuralforge",
            "/opt/homebrew/bin/neuralforge",
        ]
        for path in commonPaths {
            if FileManager.default.isExecutableFile(atPath: path) {
                return path
            }
        }

        return "neuralforge"  // Last resort — rely on PATH
    }

    var cliFound: Bool {
        FileManager.default.isExecutableFile(atPath: cliBinaryPath)
    }

    func setCLIPath(_ path: String) {
        cliBinaryPath = path
        UserDefaults.standard.set(path, forKey: "CLIBinaryPath")
    }

    // MARK: – Generic CLI execution

    private func runCLI(args: [String], onLine: @escaping (String) -> Void,
                        onComplete: @escaping (Int32) -> Void) {
        guard FileManager.default.isExecutableFile(atPath: cliBinaryPath) else {
            state.phase = .error
            state.errorMessage = "CLI binary not found at \(cliBinaryPath). Set path in Settings."
            return
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: cliBinaryPath)
        proc.arguments = args

        let stdout = Pipe()
        let stderr = Pipe()
        proc.standardOutput = stdout
        proc.standardError = stderr

        stdout.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty,
                  let text = String(data: data, encoding: .utf8) else { return }
            for line in text.components(separatedBy: "\n") where !line.isEmpty {
                onLine(line)
            }
        }

        stderr.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if let str = String(data: data, encoding: .utf8), !str.isEmpty {
                print("[neuralforge] \(str)", terminator: "")
            }
        }

        proc.terminationHandler = { proc in
            Task { @MainActor in
                stdout.fileHandleForReading.readabilityHandler = nil
                stderr.fileHandleForReading.readabilityHandler = nil
                onComplete(proc.terminationStatus)
            }
        }

        do {
            try proc.run()
        } catch {
            state.phase = .error
            state.errorMessage = "Failed to launch CLI: \(error.localizedDescription)"
        }
    }

    // MARK: – Training

    func startTraining(project: NFProject) {
        guard !isRunning else { return }

        state.reset()

        var args = ["train"]
        if !project.modelPath.isEmpty { args += ["--model", project.modelPath] }
        if !project.dataPath.isEmpty { args += ["--data", project.dataPath] }

        let cfg = project.config
        args += ["--steps", "\(cfg.totalSteps)"]
        args += ["--lr", String(format: "%.2e", cfg.learningRate)]
        args += ["--accum", "\(cfg.accumSteps)"]
        args += ["--ckpt-every", "\(cfg.checkpointEvery)"]
        args += ["--seed", "\(cfg.seed)"]
        args += ["--beta1", String(cfg.beta1)]
        args += ["--beta2", String(cfg.beta2)]
        args += ["--eps", String(format: "%.0e", cfg.eps)]
        args += ["--grad-clip", String(cfg.gradClipNorm)]

        // LR Scheduler
        if cfg.warmupSteps > 0 {
            args += ["--warmup", "\(cfg.warmupSteps)"]
        }
        if cfg.lrSchedule != "none" {
            args += ["--lr-schedule", cfg.lrSchedule]
            args += ["--lr-min", String(format: "%.2e", cfg.lrMin)]
        }

        // Data Pipeline
        if !cfg.valDataPath.isEmpty {
            args += ["--val-data", cfg.valDataPath]
        }
        if cfg.valEvery > 0 {
            args += ["--val-every", "\(cfg.valEvery)"]
            args += ["--val-batches", "\(cfg.valBatches)"]
        }
        if cfg.shuffle {
            args += ["--shuffle"]
        }

        // LoRA
        if cfg.loraRank > 0 {
            args += ["--lora-rank", "\(cfg.loraRank)"]
            args += ["--lora-alpha", String(format: "%.1f", cfg.loraAlpha)]
            args += ["--lora-targets", "\(cfg.loraTargets)"]
        }

        let ckptDir = project.directory.appendingPathComponent("checkpoints")
        try? FileManager.default.createDirectory(at: ckptDir, withIntermediateDirectories: true)
        let ckptPath = ckptDir.appendingPathComponent("checkpoint.bin").path
        args += ["--ckpt", ckptPath]

        if !cfg.useANEExtras { args += ["--no-ane-extras"] }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: cliBinaryPath)
        proc.arguments = args

        let stdout = Pipe()
        let stderr = Pipe()
        proc.standardOutput = stdout
        proc.standardError = stderr
        self.outputPipe = stdout
        self.stderrPipe = stderr
        self.process = proc

        let decoder = JSONDecoder()
        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty,
                  let line = String(data: data, encoding: .utf8) else { return }

            for jsonLine in line.components(separatedBy: "\n") where !jsonLine.isEmpty {
                guard let jsonData = jsonLine.data(using: .utf8) else { continue }
                do {
                    let msg = try decoder.decode(CLIMessage.self, from: jsonData)
                    Task { @MainActor [weak self] in
                        self?.state.handle(msg)
                    }
                } catch {
                    print("[NF] JSON parse error: \(error) for: \(jsonLine)")
                }
            }
        }

        stderr.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if let str = String(data: data, encoding: .utf8), !str.isEmpty {
                print("[neuralforge] \(str)", terminator: "")
            }
        }

        proc.terminationHandler = { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.isRunning = false
                stdout.fileHandleForReading.readabilityHandler = nil
                stderr.fileHandleForReading.readabilityHandler = nil
                if self?.state.phase == .training || self?.state.phase == .compiling {
                    self?.state.phase = .idle
                }
            }
        }

        do {
            try proc.run()
            isRunning = true
            state.phase = .compiling
        } catch {
            state.phase = .error
            state.errorMessage = "Failed to launch CLI: \(error.localizedDescription)"
        }
    }

    func stopTraining() {
        guard isRunning, let proc = process, proc.isRunning else { return }
        proc.interrupt()  // SIGINT → graceful checkpoint + exit
    }

    func killTraining() {
        guard let proc = process, proc.isRunning else { return }
        proc.terminate()  // SIGTERM
        isRunning = false
    }

    // MARK: – Tokenize

    func tokenize(inputPath: String, outputPath: String, tokenizerPath: String,
                  onComplete: @escaping (Bool, Int?) -> Void) {
        var tokenCount: Int?

        runCLI(args: ["tokenize",
                      "--input", inputPath,
                      "--output", outputPath,
                      "--tokenizer", tokenizerPath],
               onLine: { line in
            // Parse JSON response
            if let data = line.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let tokens = json["tokens"] as? Int {
                tokenCount = tokens
            }
        }, onComplete: { status in
            onComplete(status == 0, tokenCount)
        })
    }

    // MARK: – Export

    func exportModel(checkpointPath: String, format: String, outputPath: String,
                     onComplete: @escaping (Bool, String?) -> Void) {
        var errorMsg: String?

        runCLI(args: ["export",
                      "--ckpt", checkpointPath,
                      "--format", format,
                      "--output", outputPath],
               onLine: { line in
            if let data = line.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                if let msg = json["message"] as? String {
                    errorMsg = msg
                }
            }
        }, onComplete: { status in
            onComplete(status == 0, errorMsg)
        })
    }

    // MARK: – Benchmark

    func benchmark(modelPath: String, steps: Int = 100,
                   onComplete: @escaping (Double?, Double?) -> Void) {
        var avgMs: Double?
        var tflops: Double?

        runCLI(args: ["benchmark", "--model", modelPath, "--steps", "\(steps)"],
               onLine: { line in
            if let data = line.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                avgMs = json["avg_ms"] as? Double
                tflops = json["tflops_ane"] as? Double
            }
        }, onComplete: { _ in
            onComplete(avgMs, tflops)
        })
    }

    // MARK: – Evaluate Perplexity

    func evaluatePerplexity(checkpointPath: String, evalDataPath: String,
                            onBatch: @escaping (Int, Double, Int) -> Void,
                            onComplete: @escaping (Bool, Double?, Double?, Int?) -> Void) {
        var avgLoss: Double?
        var perplexity: Double?
        var totalTokens: Int?

        runCLI(args: ["benchmark", "--model", checkpointPath,
                      "--data", evalDataPath, "--eval-perplexity"],
               onLine: { line in
            if let data = line.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let type = json["type"] as? String {
                if type == "eval_batch",
                   let batch = json["batch"] as? Int,
                   let loss = json["loss"] as? Double,
                   let tokens = json["tokens"] as? Int {
                    Task { @MainActor in
                        onBatch(batch, loss, tokens)
                    }
                } else if type == "eval_done" {
                    avgLoss = json["avg_loss"] as? Double
                    perplexity = json["perplexity"] as? Double
                    totalTokens = json["total_tokens"] as? Int
                }
            }
        }, onComplete: { status in
            onComplete(status == 0, avgLoss, perplexity, totalTokens)
        })
    }

    // MARK: – CoreML Export (via Python converter)

    /// Find the converters/ directory relative to the CLI binary
    var convertersDir: String? {
        let cliDir = (cliBinaryPath as NSString).deletingLastPathComponent
        let converters = (cliDir as NSString).appendingPathComponent("../converters")
        let resolved = (converters as NSString).standardizingPath
        if FileManager.default.fileExists(atPath: resolved) { return resolved }
        return nil
    }

    func exportCoreML(checkpointPath: String, outputPath: String,
                      onComplete: @escaping (Bool, String?) -> Void) {
        guard let convDir = convertersDir else {
            onComplete(false, "Cannot find converters/ directory relative to CLI binary")
            return
        }

        let scriptPath = (convDir as NSString).appendingPathComponent("llama2c_to_coreml.py")
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            onComplete(false, "CoreML converter not found at \(scriptPath)")
            return
        }

        // Step 1: Export checkpoint to llama2c temp file
        let tempBin = FileManager.default.temporaryDirectory
            .appendingPathComponent("nf_coreml_\(UUID().uuidString).bin").path

        exportModel(checkpointPath: checkpointPath, format: "llama2c", outputPath: tempBin) { [weak self] success, error in
            guard success else {
                onComplete(false, error ?? "Failed to export llama2c intermediate format")
                return
            }

            // Step 2: Run Python converter
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
            proc.arguments = [scriptPath, "--llama2c", tempBin, "--output", outputPath]

            let stderr = Pipe()
            proc.standardError = stderr
            proc.standardOutput = FileHandle.nullDevice

            proc.terminationHandler = { process in
                // Cleanup temp file
                try? FileManager.default.removeItem(atPath: tempBin)

                Task { @MainActor in
                    if process.terminationStatus == 0 {
                        onComplete(true, nil)
                    } else {
                        let errData = stderr.fileHandleForReading.readDataToEndOfFile()
                        let errStr = String(data: errData, encoding: .utf8) ?? "Python converter failed"
                        onComplete(false, errStr)
                    }
                }
            }

            do {
                try proc.run()
            } catch {
                try? FileManager.default.removeItem(atPath: tempBin)
                Task { @MainActor in
                    onComplete(false, "Failed to launch Python: \(error.localizedDescription)")
                }
            }
        }
    }

    // MARK: – Generate

    func generate(modelPath: String, tokenizerPath: String, prompt: String,
                  temperature: Double = 0.8, topP: Double = 0.9,
                  maxTokens: Int = 256, seed: Int = 42,
                  onToken: @escaping (String) -> Void,
                  onComplete: @escaping (Bool, Int?, Double?) -> Void) {
        var totalTokens: Int?
        var totalMs: Double?

        runCLI(args: ["generate",
                      "--model", modelPath,
                      "--tokenizer", tokenizerPath,
                      "--prompt", prompt,
                      "--temperature", String(format: "%.2f", temperature),
                      "--top-p", String(format: "%.2f", topP),
                      "--max-tokens", "\(maxTokens)",
                      "--seed", "\(seed)"],
               onLine: { line in
            if let data = line.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let type = json["type"] as? String {
                if type == "token", let text = json["text"] as? String {
                    Task { @MainActor in
                        onToken(text)
                    }
                } else if type == "generate_done" {
                    totalTokens = json["tokens"] as? Int
                    totalMs = json["total_ms"] as? Double
                }
            }
        }, onComplete: { status in
            onComplete(status == 0, totalTokens, totalMs)
        })
    }

    // MARK: – Ingest

    func ingest(sourcePath: String, outputPath: String, tokenizerPath: String,
                maxShardMB: Int = 50, incremental: Bool = true,
                onFile: @escaping (String, Int) -> Void,
                onComplete: @escaping (Bool, Int, Int, Int) -> Void) {
        var newFiles = 0, totalTokens = 0, shards = 0

        var args = ["ingest",
                    "--source", sourcePath,
                    "--output", outputPath,
                    "--tokenizer", tokenizerPath,
                    "--max-shard-mb", "\(maxShardMB)"]
        if incremental { args += ["--incremental"] }

        runCLI(args: args, onLine: { line in
            if let data = line.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let type = json["type"] as? String {
                if type == "ingest_file",
                   let file = json["file"] as? String,
                   let tokens = json["tokens"] as? Int {
                    Task { @MainActor in
                        onFile(file, tokens)
                    }
                } else if type == "ingest_done" {
                    newFiles = json["new_files"] as? Int ?? 0
                    totalTokens = json["total_tokens"] as? Int ?? 0
                    shards = json["shards"] as? Int ?? 0
                }
            }
        }, onComplete: { status in
            onComplete(status == 0, newFiles, totalTokens, shards)
        })
    }

    // MARK: – Models

    func listModels(onModel: @escaping ([String: Any]) -> Void,
                    onComplete: @escaping () -> Void) {
        runCLI(args: ["models", "--json"], onLine: { line in
            if let data = line.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let type = json["type"] as? String,
               type == "model_info" {
                Task { @MainActor in
                    onModel(json)
                }
            }
        }, onComplete: { _ in
            onComplete()
        })
    }

    // MARK: – Download

    func downloadModel(modelName: String, outputPath: String, hfToken: String? = nil,
                       onProgress: @escaping (String, Int) -> Void,
                       onComplete: @escaping (Bool, String?, String?) -> Void) {
        var modelPath: String? = nil
        var tokenizerPath: String? = nil
        var success = false

        var args = ["download", "--model", modelName, "--output", outputPath]
        if let token = hfToken, !token.isEmpty {
            args += ["--token", token]
        }

        runCLI(args: args, onLine: { line in
            if let data = line.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let type = json["type"] as? String {
                if type == "download_progress",
                   let status = json["status"] as? String,
                   let percent = json["percent"] as? Int {
                    Task { @MainActor in
                        onProgress(status, percent)
                    }
                } else if type == "download_done" {
                    success = json["success"] as? Bool ?? false
                    modelPath = json["model_path"] as? String
                    tokenizerPath = json["tokenizer_path"] as? String
                }
            }
        }, onComplete: { _ in
            onComplete(success, modelPath, tokenizerPath)
        })
    }

    // MARK: – Info

    func getModelInfo(modelPath: String) async -> CLIMessage.InitMsg? {
        guard FileManager.default.isExecutableFile(atPath: cliBinaryPath) else { return nil }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: cliBinaryPath)
        proc.arguments = ["info", "--model", modelPath]

        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice

        do {
            try proc.run()
            proc.waitUntilExit()

            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            guard let line = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                  let jsonData = line.data(using: .utf8) else { return nil }

            struct InfoResp: Decodable {
                let type: String
                let key: String?
                let value: ModelInfo?

                struct ModelInfo: Decodable {
                    let dim, hidden_dim, n_layers, n_heads, vocab_size, seq_len: Int
                    let params_millions: Double
                }
            }

            let resp = try JSONDecoder().decode(InfoResp.self, from: jsonData)
            guard let info = resp.value else { return nil }

            return CLIMessage.InitMsg(
                params: Int(info.params_millions * 1e6),
                layers: info.n_layers,
                dim: info.dim,
                hidden: info.hidden_dim,
                heads: info.n_heads,
                seq: info.seq_len,
                vocab: info.vocab_size,
                timestamp: nil
            )
        } catch {
            return nil
        }
    }
}
