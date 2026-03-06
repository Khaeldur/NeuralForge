// ExportView.swift — Model export to various formats

import SwiftUI

struct ExportView: View {
    @Binding var project: NFProject
    @EnvironmentObject var cliRunner: CLIRunner
    @State private var selectedFormat: ExportFormat = .llama2c
    @State private var isExporting = false
    @State private var exportPath: String?
    @State private var exportError: String?

    enum ExportFormat: String, CaseIterable {
        case llama2c = "llama2c"
        case gguf = "GGUF"
        case coreml = "CoreML"

        var description: String {
            switch self {
            case .llama2c: "Native llama2.c format — lightweight, direct inference"
            case .gguf: "GGUF format — compatible with llama.cpp ecosystem"
            case .coreml: "CoreML — optimized for Apple device deployment"
            }
        }

        var icon: String {
            switch self {
            case .llama2c: "doc.zipper"
            case .gguf: "shippingbox"
            case .coreml: "apple.logo"
            }
        }

        var fileExtension: String {
            switch self {
            case .llama2c: "bin"
            case .gguf: "gguf"
            case .coreml: "mlpackage"
            }
        }

        var cliFormat: String {
            switch self {
            case .llama2c: "llama2c"
            case .gguf: "gguf"
            case .coreml: "llama2c"  // CoreML uses Python converter on llama2c output
            }
        }
    }

    var hasCheckpoint: Bool {
        !project.checkpointPath.isEmpty &&
        FileManager.default.fileExists(atPath: project.checkpointPath)
    }

    var body: some View {
        VStack(spacing: 20) {
            GroupBox("Export Format") {
                VStack(spacing: 0) {
                    ForEach(ExportFormat.allCases, id: \.self) { format in
                        Button(action: { selectedFormat = format }) {
                            HStack {
                                Image(systemName: format.icon)
                                    .frame(width: 24)
                                VStack(alignment: .leading) {
                                    Text(format.rawValue)
                                        .font(.headline)
                                    Text(format.description)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                if selectedFormat == format {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(.blue)
                                }
                            }
                            .padding(10)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        if format != ExportFormat.allCases.last {
                            Divider()
                        }
                    }
                }
            }

            GroupBox("Checkpoint") {
                HStack {
                    if hasCheckpoint {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        VStack(alignment: .leading) {
                            Text(URL(fileURLWithPath: project.checkpointPath).lastPathComponent)
                                .font(.body.monospaced())
                            if let attrs = try? FileManager.default.attributesOfItem(atPath: project.checkpointPath),
                               let size = attrs[.size] as? Int {
                                Text("\(ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    } else {
                        Image(systemName: "xmark.circle")
                            .foregroundStyle(.red)
                        Text("No checkpoint found. Train first.")
                            .foregroundStyle(.secondary)
                    }
                    Spacer()

                    Button("Browse") {
                        let panel = NSOpenPanel()
                        panel.canChooseFiles = true
                        panel.allowedContentTypes = [.data]
                        if panel.runModal() == .OK, let url = panel.url {
                            project.checkpointPath = url.path
                        }
                    }
                    .buttonStyle(.bordered)
                }
                .padding(8)
            }

            if let path = exportPath {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Text("Exported to: \(URL(fileURLWithPath: path).lastPathComponent)")
                        .font(.caption.monospaced())
                    Spacer()
                    Button("Reveal in Finder") {
                        NSWorkspace.shared.selectFile(path, inFileViewerRootedAtPath: "")
                    }
                    .buttonStyle(.bordered)
                }
                .padding(10)
                .background(.green.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            }

            if let error = exportError {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(error)
                        .foregroundStyle(.red)
                    Spacer()
                }
                .padding(10)
                .background(.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            }

            HStack {
                Spacer()
                Button(action: exportModel) {
                    if isExporting {
                        ProgressView()
                            .controlSize(.small)
                        Text("Exporting...")
                    } else {
                        Label("Export Model", systemImage: "square.and.arrow.up")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isExporting || !hasCheckpoint)
            }

            Spacer()
        }
    }

    private func exportModel() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "\(project.name).\(selectedFormat.fileExtension)"
        panel.allowedContentTypes = [.data]

        guard panel.runModal() == .OK, let url = panel.url else { return }

        isExporting = true
        exportPath = nil
        exportError = nil

        if selectedFormat == .coreml {
            cliRunner.exportCoreML(
                checkpointPath: project.checkpointPath,
                outputPath: url.path
            ) { success, error in
                Task { @MainActor in
                    isExporting = false
                    if success {
                        exportPath = url.path
                    } else {
                        exportError = error ?? "CoreML export failed (requires: pip install coremltools numpy)"
                    }
                }
            }
        } else {
            cliRunner.exportModel(
                checkpointPath: project.checkpointPath,
                format: selectedFormat.cliFormat,
                outputPath: url.path
            ) { success, error in
                Task { @MainActor in
                    isExporting = false
                    if success {
                        exportPath = url.path
                    } else {
                        exportError = error ?? "Export failed"
                    }
                }
            }
        }
    }
}
