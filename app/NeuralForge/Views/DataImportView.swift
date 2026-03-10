// DataImportView.swift — Document import + tokenization

import SwiftUI
import UniformTypeIdentifiers

struct DataImportView: View {
    @Binding var project: NFProject
    @EnvironmentObject var cliRunner: CLIRunner
    @State private var importedFiles: [URL] = []
    @State private var isImporting = false
    @State private var tokenizing = false
    @State private var tokenCount: Int?
    @State private var statusMessage: String?
    @State private var errorMessage: String?

    // Default tokenizer location
    private var tokenizerPath: String {
        // Look next to the CLI binary first, then vendor
        let cliDir = (cliRunner.cliBinaryPath as NSString).deletingLastPathComponent
        let candidates = [
            cliDir + "/tokenizer.bin",
            cliDir + "/../models/tokenizer.bin",
            cliDir + "/../vendor/ANE/assets/models/tokenizer.bin",
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) } ?? "tokenizer.bin"
    }

    var body: some View {
        VStack(spacing: 20) {
            GroupBox("Import Documents") {
                VStack(spacing: 12) {
                    if importedFiles.isEmpty {
                        dropZone
                    } else {
                        fileList
                    }

                    HStack {
                        Button("Add Files") { isImporting = true }
                            .buttonStyle(.bordered)
                        Spacer()
                        if !importedFiles.isEmpty {
                            Button("Clear All") {
                                importedFiles = []
                                statusMessage = nil
                                errorMessage = nil
                            }
                            .buttonStyle(.bordered)
                            .tint(.red)
                        }
                    }
                }
                .padding(8)
            }

            GroupBox("Token Data") {
                VStack(alignment: .leading, spacing: 10) {
                    if !project.dataPath.isEmpty {
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                            Text(URL(fileURLWithPath: project.dataPath).lastPathComponent)
                                .font(.body.monospaced())
                            Spacer()
                            if let count = tokenCount {
                                Text("\(count.formatted()) tokens")
                                    .foregroundStyle(.secondary)
                            }
                        }
                    } else {
                        Text("No token data loaded. Import documents and tokenize, or browse for a .bin file.")
                            .foregroundStyle(.secondary)
                    }

                    HStack {
                        Button("Browse .bin") {
                            let panel = NSOpenPanel()
                            panel.canChooseFiles = true
                            panel.allowedContentTypes = [UTType(filenameExtension: "bin")].compactMap { $0 }
                            if panel.runModal() == .OK, let url = panel.url {
                                project.dataPath = url.path
                                updateTokenCount()
                            }
                        }
                        .buttonStyle(.bordered)

                        if !importedFiles.isEmpty {
                            Button(action: tokenizeFiles) {
                                if tokenizing {
                                    ProgressView()
                                        .controlSize(.small)
                                    Text("Tokenizing...")
                                } else {
                                    Label("Tokenize", systemImage: "text.badge.checkmark")
                                }
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(tokenizing)
                        }
                    }
                }
                .padding(8)
            }

            // Status / error messages
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
        .fileImporter(isPresented: $isImporting,
                       allowedContentTypes: [.plainText, .pdf, .commaSeparatedText],
                       allowsMultipleSelection: true) { result in
            if case .success(let urls) = result {
                importedFiles.append(contentsOf: urls)
            }
        }
        .onAppear { updateTokenCount() }
    }

    var dropZone: some View {
        VStack(spacing: 8) {
            Image(systemName: "arrow.down.doc")
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text("Drop documents here")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("TXT, PDF, CSV, Markdown supported")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 150)
        .background(.bar, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [8, 4]))
                .foregroundStyle(.quaternary)
        )
    }

    var fileList: some View {
        List {
            ForEach(importedFiles, id: \.absoluteString) { url in
                HStack {
                    Image(systemName: iconForExtension(url.pathExtension))
                    Text(url.lastPathComponent)
                    Spacer()
                    Button(action: { importedFiles.removeAll { $0 == url } }) {
                        Image(systemName: "xmark.circle")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .frame(height: 150)
    }

    private func iconForExtension(_ ext: String) -> String {
        switch ext.lowercased() {
        case "pdf": "doc.richtext"
        case "csv": "tablecells"
        case "md": "doc.text"
        default: "doc.text"
        }
    }

    private func tokenizeFiles() {
        tokenizing = true
        statusMessage = nil
        errorMessage = nil

        // 1. Extract text from all imported files
        Task {
            do {
                let mergedText = try DocumentImporter.mergeFiles(importedFiles)
                let tempFile = try DocumentImporter.writeToTempFile(mergedText)

                // 2. Set output path in project directory
                let outputDir = project.directory.appendingPathComponent("data")
                try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
                let outputPath = outputDir.appendingPathComponent("tokens.bin").path

                // 3. Call CLI tokenizer
                cliRunner.tokenize(
                    inputPath: tempFile.path,
                    outputPath: outputPath,
                    tokenizerPath: tokenizerPath
                ) { success, count in
                    Task { @MainActor in
                        tokenizing = false
                        try? FileManager.default.removeItem(at: tempFile)

                        if success {
                            project.dataPath = outputPath
                            tokenCount = count
                            statusMessage = "Tokenized \(importedFiles.count) file(s) — \(count?.formatted() ?? "?") tokens"
                        } else {
                            errorMessage = "Tokenization failed. Check that tokenizer.bin exists at \(tokenizerPath)"
                        }
                    }
                }
            } catch {
                await MainActor.run {
                    tokenizing = false
                    errorMessage = "Text extraction failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func updateTokenCount() {
        guard !project.dataPath.isEmpty else { tokenCount = nil; return }
        if let attrs = try? FileManager.default.attributesOfItem(atPath: project.dataPath),
           let size = attrs[.size] as? Int {
            tokenCount = size / 2
        }
    }
}
