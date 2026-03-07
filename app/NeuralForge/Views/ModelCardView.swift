// ModelCardView.swift — Browse and download pre-trained models
import SwiftUI

struct ModelInfo: Identifiable {
    let id = UUID()
    let name: String
    let displayName: String
    let repoId: String
    let architecture: String
    let dim: Int
    let hiddenDim: Int
    let nLayers: Int
    let nHeads: Int
    let nKVHeads: Int
    let vocabSize: Int
    let seqLen: Int
    let paramsMillions: Double
    let description: String
    let gated: Bool

    init(from json: [String: Any]) {
        name = json["name"] as? String ?? ""
        displayName = json["display_name"] as? String ?? name
        repoId = json["repo_id"] as? String ?? ""
        architecture = json["architecture"] as? String ?? "Unknown"
        dim = json["dim"] as? Int ?? 0
        hiddenDim = json["hidden_dim"] as? Int ?? 0
        nLayers = json["n_layers"] as? Int ?? 0
        nHeads = json["n_heads"] as? Int ?? 0
        nKVHeads = json["n_kv_heads"] as? Int ?? 0
        vocabSize = json["vocab_size"] as? Int ?? 0
        seqLen = json["seq_len"] as? Int ?? 0
        paramsMillions = json["params_millions"] as? Double ?? 0
        description = json["description"] as? String ?? ""
        gated = json["gated"] as? Bool ?? false
    }

    var paramsFormatted: String {
        if paramsMillions >= 1000 {
            return String(format: "%.1fB", paramsMillions / 1000)
        }
        return String(format: "%.0fM", paramsMillions)
    }

    var hasGQA: Bool { nKVHeads != nHeads }
}

struct ModelCardView: View {
    @Binding var project: NFProject
    @EnvironmentObject var cliRunner: CLIRunner
    @State private var models: [ModelInfo] = []
    @State private var isLoading = false
    @State private var isDownloading = false
    @State private var downloadStatus = ""
    @State private var downloadPercent = 0
    @State private var selectedModel: ModelInfo?
    @State private var errorMessage = ""
    @State private var hfToken = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            HStack {
                Label("Model Hub", systemImage: "square.grid.2x2")
                    .font(.headline)
                Spacer()
                Button(action: loadModels) {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .disabled(isLoading || isDownloading)
            }

            if isLoading {
                ProgressView("Loading model list...")
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            } else if models.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "square.grid.2x2.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)
                    Text("No models loaded")
                        .foregroundColor(.secondary)
                    Button("Load Available Models") {
                        loadModels()
                    }
                }
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(30)
            } else {
                // Model list
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(models) { model in
                            modelCard(model)
                        }
                    }
                }
            }

            // Download section
            if isDownloading {
                VStack(spacing: 8) {
                    ProgressView(value: Double(downloadPercent), total: 100) {
                        Text(downloadStatus)
                            .font(.caption)
                    }
                    Text("This may take several minutes for large models...")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(RoundedRectangle(cornerRadius: 8)
                    .fill(Color.blue.opacity(0.1)))
            }

            if !errorMessage.isEmpty {
                Label(errorMessage, systemImage: "exclamationmark.triangle")
                    .foregroundColor(.red)
                    .font(.caption)
            }
        }
        .padding()
        .onAppear { loadModels() }
    }

    @ViewBuilder
    private func modelCard(_ model: ModelInfo) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(model.displayName)
                            .font(.headline)
                        Text(model.paramsFormatted)
                            .font(.caption)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.blue.opacity(0.2)))
                        if model.hasGQA {
                            Text("GQA")
                                .font(.caption2)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1)
                                .background(Capsule().fill(Color.orange.opacity(0.2)))
                        }
                        if model.gated {
                            Image(systemName: "lock.fill")
                                .font(.caption2)
                                .foregroundColor(.orange)
                        }
                    }
                    Text(model.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                Button("Download") {
                    selectedModel = model
                    downloadModel(model)
                }
                .disabled(isDownloading)
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }

            // Architecture details
            HStack(spacing: 16) {
                detailPill("dim", "\(model.dim)")
                detailPill("hidden", "\(model.hiddenDim)")
                detailPill("layers", "\(model.nLayers)")
                detailPill("heads", "\(model.nHeads)")
                detailPill("vocab", "\(model.vocabSize)")
                detailPill("ctx", "\(model.seqLen)")
            }
            .font(.caption2)

            Text(model.repoId)
                .font(.caption2)
                .foregroundColor(.blue)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8)
            .fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.gray.opacity(0.2))
        )
    }

    private func detailPill(_ label: String, _ value: String) -> some View {
        HStack(spacing: 2) {
            Text(label)
                .foregroundColor(.secondary)
            Text(value)
                .fontWeight(.medium)
        }
    }

    private func loadModels() {
        isLoading = true
        models = []
        cliRunner.listModels(onModel: { json in
            let model = ModelInfo(from: json)
            models.append(model)
        }, onComplete: {
            Task { @MainActor in
                isLoading = false
            }
        })
    }

    private func downloadModel(_ model: ModelInfo) {
        isDownloading = true
        errorMessage = ""
        downloadStatus = "Starting download..."
        downloadPercent = 0

        let outputDir = project.checkpointPath.isEmpty
            ? NSHomeDirectory() + "/NeuralForge/models/\(model.name)"
            : (project.checkpointPath as NSString).deletingLastPathComponent + "/\(model.name)"

        cliRunner.downloadModel(
            modelName: model.name,
            outputPath: outputDir,
            hfToken: hfToken.isEmpty ? nil : hfToken,
            onProgress: { status, percent in
                downloadStatus = status.replacingOccurrences(of: "_", with: " ").capitalized
                downloadPercent = percent
            },
            onComplete: { success, modelPath, tokenizerPath in
                Task { @MainActor in
                    isDownloading = false
                    if success {
                        downloadStatus = "Download complete!"
                        downloadPercent = 100
                        if let mp = modelPath {
                            project.modelPath = mp
                        }
                    } else {
                        errorMessage = "Download failed. Check that Python dependencies are installed."
                    }
                }
            }
        )
    }
}
