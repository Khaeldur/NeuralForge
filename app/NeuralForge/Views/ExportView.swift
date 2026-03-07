// ExportView.swift — Model export to various formats

import SwiftUI

struct ExportView: View {
    @Binding var project: NFProject
    @EnvironmentObject var cliRunner: CLIRunner
    @ObservedObject private var quantService = QuantizationService.shared
    @State private var selectedFormat: ExportFormat = .llama2c
    @State private var selectedQuantType: QuantizationType = .q4_0
    @State private var isExporting = false
    @State private var exportPath: String?
    @State private var exportError: String?
    @State private var showQuantOptions = false

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

            // Quantization options (GGUF only)
            if selectedFormat == .gguf {
                GroupBox("Quantization") {
                    VStack(alignment: .leading, spacing: 8) {
                        Picker("Precision", selection: $selectedQuantType) {
                            ForEach(QuantizationType.allCases) { qt in
                                HStack {
                                    Text(qt.rawValue)
                                    Text("— Q\(qt.qualityRating)/10")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                .tag(qt)
                            }
                        }

                        HStack(spacing: 16) {
                            Label("\(String(format: "%.1f", selectedQuantType.bitsPerWeight)) bits/weight",
                                  systemImage: "memorychip")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            let estimatedSize = QuantizationType.estimateSize(
                                modelParams: 110_000_000,
                                quantType: selectedQuantType)
                            Label("~\(QuantizationType.formatSize(estimatedSize)) for 110M params",
                                  systemImage: "doc.zipper")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Text(selectedQuantType.description)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 4)
                }
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

            // Job history
            if !quantService.jobs.isEmpty {
                GroupBox("Export History") {
                    VStack(spacing: 4) {
                        ForEach(quantService.jobs) { job in
                            HStack {
                                Image(systemName: jobIcon(job.status))
                                    .foregroundColor(jobColor(job.status))
                                    .font(.caption)
                                Text(job.quantType.rawValue)
                                    .font(.caption.bold())
                                Text(job.progress)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Spacer()
                                if let size = job.outputSize {
                                    Text(QuantizationType.formatSize(size))
                                        .font(.caption2.monospaced())
                                        .foregroundColor(.secondary)
                                }
                                Text(job.timestamp, style: .relative)
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .padding(.vertical, 4)
                }
            }

            Spacer()
        }
    }

    private func jobIcon(_ status: QuantizationJob.JobStatus) -> String {
        switch status {
        case .pending: return "clock"
        case .running: return "arrow.clockwise"
        case .success: return "checkmark.circle.fill"
        case .failed: return "xmark.circle.fill"
        }
    }

    private func jobColor(_ status: QuantizationJob.JobStatus) -> Color {
        switch status {
        case .pending: return .secondary
        case .running: return .blue
        case .success: return .green
        case .failed: return .red
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
