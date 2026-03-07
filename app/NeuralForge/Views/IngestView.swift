// IngestView.swift — Document ingestion pipeline UI
// Scans a source directory, extracts text from documents, tokenizes, and writes
// token shards with manifest tracking via the CLI `ingest` command.

import SwiftUI

struct IngestView: View {
    @Binding var project: NFProject
    @EnvironmentObject var cliRunner: CLIRunner
    @State private var sourcePath: String = ""
    @State private var outputPath: String = ""
    @State private var maxShardMB: Int = 50
    @State private var incremental: Bool = true
    @State private var isRunning = false
    @State private var statusMessage: String?
    @State private var errorMessage: String?
    @State private var processedFiles: [(name: String, tokens: Int)] = []
    @State private var manifest: IngestManifest?

    private var defaultOutputPath: String {
        project.directory.appendingPathComponent("data/shards").path
    }

    var body: some View {
        VStack(spacing: 20) {
            GroupBox("Source Directory") {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Select a folder containing documents to ingest. Supported formats: TXT, PDF, DOCX, MD, CSV, and common code files.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    HStack {
                        TextField("Source folder path", text: $sourcePath)
                            .textFieldStyle(.roundedBorder)
                        Button("Browse") {
                            let panel = NSOpenPanel()
                            panel.canChooseFiles = false
                            panel.canChooseDirectories = true
                            panel.allowsMultipleSelection = false
                            if panel.runModal() == .OK, let url = panel.url {
                                sourcePath = url.path
                            }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                .padding(8)
            }

            GroupBox("Output Settings") {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        TextField("Output folder", text: $outputPath)
                            .textFieldStyle(.roundedBorder)
                        Button("Browse") {
                            let panel = NSOpenPanel()
                            panel.canChooseFiles = false
                            panel.canChooseDirectories = true
                            panel.allowsMultipleSelection = false
                            if panel.runModal() == .OK, let url = panel.url {
                                outputPath = url.path
                            }
                        }
                        .buttonStyle(.bordered)
                    }

                    HStack(spacing: 20) {
                        HStack {
                            Text("Max shard size:")
                            Picker("", selection: $maxShardMB) {
                                Text("10 MB").tag(10)
                                Text("25 MB").tag(25)
                                Text("50 MB").tag(50)
                                Text("100 MB").tag(100)
                                Text("250 MB").tag(250)
                            }
                            .frame(width: 100)
                        }

                        Toggle("Incremental", isOn: $incremental)
                            .help("Only process new/modified files since last run")
                    }
                }
                .padding(8)
            }

            // Run button
            HStack {
                Button(action: runIngest) {
                    if isRunning {
                        ProgressView()
                            .controlSize(.small)
                        Text("Ingesting...")
                    } else {
                        Label("Run Ingest", systemImage: "arrow.triangle.2.circlepath")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isRunning || sourcePath.isEmpty)

                Spacer()

                if let m = manifest {
                    VStack(alignment: .trailing) {
                        Text("\(m.totalTokens.formatted()) tokens in \(m.shardCount) shard(s)")
                            .font(.caption.monospaced())
                        Text("\(m.processedFileCount) files processed")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            // Progress / Status
            if !processedFiles.isEmpty {
                GroupBox("Processed Files") {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 4) {
                            ForEach(processedFiles, id: \.name) { file in
                                HStack {
                                    Image(systemName: "doc.text")
                                        .foregroundStyle(.secondary)
                                    Text(file.name)
                                        .font(.body.monospaced())
                                    Spacer()
                                    Text("\(file.tokens.formatted()) tokens")
                                        .foregroundStyle(.secondary)
                                        .font(.caption.monospaced())
                                }
                            }
                        }
                        .padding(4)
                    }
                    .frame(maxHeight: 200)
                }
            }

            if let msg = statusMessage {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Text(msg)
                    Spacer()
                }
                .padding(10)
                .background(.green.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            }

            if let err = errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(err)
                        .foregroundStyle(.red)
                    Spacer()
                }
                .padding(10)
                .background(.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            }

            Spacer()
        }
        .onAppear {
            if outputPath.isEmpty { outputPath = defaultOutputPath }
            loadManifest()
        }
    }

    private func runIngest() {
        isRunning = true
        statusMessage = nil
        errorMessage = nil
        processedFiles = []

        let tokenizerPath = findTokenizerPath()

        cliRunner.ingest(
            sourcePath: sourcePath,
            outputPath: outputPath,
            tokenizerPath: tokenizerPath,
            maxShardMB: maxShardMB,
            incremental: incremental,
            onFile: { name, tokens in
                Task { @MainActor in
                    processedFiles.append((name: name, tokens: tokens))
                }
            },
            onComplete: { success, newFiles, totalTokens, shards in
                Task { @MainActor in
                    isRunning = false
                    if success {
                        statusMessage = "Ingested \(newFiles) file(s) — \(totalTokens.formatted()) total tokens in \(shards) shard(s)"
                        loadManifest()
                    } else {
                        errorMessage = "Ingestion failed. Check source path and tokenizer."
                    }
                }
            }
        )
    }

    private func findTokenizerPath() -> String {
        let cliDir = (cliRunner.cliBinaryPath as NSString).deletingLastPathComponent
        let candidates = [
            cliDir + "/tokenizer.bin",
            cliDir + "/../models/tokenizer.bin",
            cliDir + "/../vendor/ANE/assets/models/tokenizer.bin",
            NSHomeDirectory() + "/Desktop/sl/NeuralForge/vendor/ANE/assets/models/tokenizer.bin",
            NSHomeDirectory() + "/Desktop/sl/NeuralForge/models/tokenizer.bin",
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) } ?? "tokenizer.bin"
    }

    private func loadManifest() {
        let manifestPath = outputPath.isEmpty ? defaultOutputPath : outputPath
        let path = (manifestPath as NSString).appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { manifest = nil; return }

        manifest = IngestManifest(
            totalTokens: json["total_tokens"] as? Int ?? 0,
            shardCount: (json["shards"] as? [[String: Any]])?.count ?? 0,
            processedFileCount: (json["processed_files"] as? [String: Any])?.count ?? 0,
            lastRun: json["last_run"] as? String
        )
    }
}

struct IngestManifest {
    let totalTokens: Int
    let shardCount: Int
    let processedFileCount: Int
    let lastRun: String?
}
