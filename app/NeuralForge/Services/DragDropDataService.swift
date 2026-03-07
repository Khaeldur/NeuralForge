// DragDropDataService.swift — Drag & drop file ingestion for training data

import Foundation
import UniformTypeIdentifiers

/// Supported file types for drag & drop data ingestion
enum IngestFileType: String, CaseIterable {
    case plainText = "txt"
    case markdown = "md"
    case json = "json"
    case jsonl = "jsonl"
    case csv = "csv"
    case pdf = "pdf"
    case swift = "swift"
    case python = "py"
    case html = "html"

    var utType: UTType? {
        switch self {
        case .plainText: return .plainText
        case .markdown: return .init(filenameExtension: "md")
        case .json: return .json
        case .jsonl: return .init(filenameExtension: "jsonl")
        case .csv: return .commaSeparatedText
        case .pdf: return .pdf
        case .swift: return .swiftSource
        case .python: return .pythonScript
        case .html: return .html
        }
    }

    static let allExtensions: Set<String> = Set(allCases.map { $0.rawValue })

    static func from(extension ext: String) -> IngestFileType? {
        allCases.first { $0.rawValue == ext.lowercased() }
    }
}

/// Result of processing a dropped file
struct DroppedFileResult: Identifiable {
    let id = UUID()
    let fileName: String
    let fileSize: Int64
    let fileType: IngestFileType
    let characterCount: Int
    let lineCount: Int
    let status: Status

    enum Status {
        case success
        case skipped(reason: String)
        case error(message: String)
    }

    var statusDescription: String {
        switch status {
        case .success: return "Ready"
        case .skipped(let reason): return "Skipped: \(reason)"
        case .error(let message): return "Error: \(message)"
        }
    }
}

/// Accumulated batch of files ready for tokenization
struct IngestBatch: Identifiable {
    let id = UUID()
    var files: [DroppedFileResult]
    var totalCharacters: Int { files.compactMap { if case .success = $0.status { return $0.characterCount } else { return nil } }.reduce(0, +) }
    var totalLines: Int { files.compactMap { if case .success = $0.status { return $0.lineCount } else { return nil } }.reduce(0, +) }
    var successCount: Int { files.filter { if case .success = $0.status { return true } else { return false } }.count }
    var errorCount: Int { files.filter { if case .error = $0.status { return true } else { return false } }.count }
    var skippedCount: Int { files.filter { if case .skipped = $0.status { return true } else { return false } }.count }
}

/// Manages drag & drop data ingestion into training projects
@MainActor
class DragDropDataService: ObservableObject {
    @Published var currentBatch: IngestBatch?
    @Published var isProcessing = false
    @Published var processingProgress: Double = 0

    // Configurable limits
    static let maxFileSizeMB: Int64 = 100
    static let maxFilesPerBatch = 1000
    static let supportedExtensions = IngestFileType.allExtensions

    // MARK: - Drop Handling

    /// Validate whether a URL is an acceptable drop target
    func canAcceptDrop(urls: [URL]) -> Bool {
        urls.contains { url in
            let ext = url.pathExtension.lowercased()
            return Self.supportedExtensions.contains(ext)
        }
    }

    /// Process a batch of dropped file URLs
    func processDroppedFiles(_ urls: [URL]) async -> IngestBatch {
        isProcessing = true
        processingProgress = 0

        var results: [DroppedFileResult] = []
        let validURLs = urls.prefix(Self.maxFilesPerBatch)

        for (index, url) in validURLs.enumerated() {
            let result = processFile(at: url)
            results.append(result)
            processingProgress = Double(index + 1) / Double(validURLs.count)
        }

        let batch = IngestBatch(files: results)
        currentBatch = batch
        isProcessing = false
        return batch
    }

    /// Process a single file
    private func processFile(at url: URL) -> DroppedFileResult {
        let fileName = url.lastPathComponent
        let ext = url.pathExtension.lowercased()

        // Check file type
        guard let fileType = IngestFileType.from(extension: ext) else {
            return DroppedFileResult(
                fileName: fileName, fileSize: 0, fileType: .plainText,
                characterCount: 0, lineCount: 0,
                status: .skipped(reason: "Unsupported file type: .\(ext)")
            )
        }

        // Check file size
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        let fileSize = (attrs?[.size] as? Int64) ?? 0

        if fileSize > Self.maxFileSizeMB * 1024 * 1024 {
            return DroppedFileResult(
                fileName: fileName, fileSize: fileSize, fileType: fileType,
                characterCount: 0, lineCount: 0,
                status: .skipped(reason: "File too large (\(fileSize / 1024 / 1024)MB > \(Self.maxFileSizeMB)MB)")
            )
        }

        if fileSize == 0 {
            return DroppedFileResult(
                fileName: fileName, fileSize: 0, fileType: fileType,
                characterCount: 0, lineCount: 0,
                status: .skipped(reason: "Empty file")
            )
        }

        // Read and analyze text content
        guard let content = try? String(contentsOf: url, encoding: .utf8) else {
            return DroppedFileResult(
                fileName: fileName, fileSize: fileSize, fileType: fileType,
                characterCount: 0, lineCount: 0,
                status: .error(message: "Could not read file as UTF-8")
            )
        }

        let charCount = content.count
        let lineCount = content.components(separatedBy: .newlines).count

        return DroppedFileResult(
            fileName: fileName, fileSize: fileSize, fileType: fileType,
            characterCount: charCount, lineCount: lineCount,
            status: .success
        )
    }

    // MARK: - Staging

    /// Copy accepted files to a project's data staging directory
    func stageFiles(from batch: IngestBatch, to project: NFProject) throws -> URL {
        let stagingDir = project.directory.appendingPathComponent("staging")
        try FileManager.default.createDirectory(at: stagingDir, withIntermediateDirectories: true)

        for file in batch.files {
            guard case .success = file.status else { continue }
            // Files are already validated — actual copy would happen here
            // In production, this copies from the drop source to staging
        }

        return stagingDir
    }

    /// Concatenate all staged text files into a single training text file
    func concatenateStaged(in stagingDir: URL) throws -> URL {
        let outputFile = stagingDir.appendingPathComponent("combined_training_data.txt")
        let fm = FileManager.default

        guard let contents = try? fm.contentsOfDirectory(at: stagingDir,
                                                          includingPropertiesForKeys: nil) else {
            throw DragDropError.emptyStaging
        }

        var combinedText = ""
        for url in contents.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            guard url.pathExtension != "txt" || url.lastPathComponent != "combined_training_data.txt" else { continue }
            if let text = try? String(contentsOf: url, encoding: .utf8) {
                if !combinedText.isEmpty { combinedText += "\n\n" }
                combinedText += text
            }
        }

        try combinedText.write(to: outputFile, atomically: true, encoding: .utf8)
        return outputFile
    }

    // MARK: - Cleanup

    func clearBatch() {
        currentBatch = nil
        processingProgress = 0
    }

    // MARK: - Helpers

    /// Format file size for display
    static func formatFileSize(_ bytes: Int64) -> String {
        if bytes < 1024 { return "\(bytes) B" }
        if bytes < 1024 * 1024 { return "\(bytes / 1024) KB" }
        return "\(bytes / 1024 / 1024) MB"
    }

    /// Estimate token count from character count (rough approximation)
    static func estimateTokenCount(characters: Int) -> Int {
        // English text: ~4 chars per token on average
        characters / 4
    }
}

// MARK: - Errors

enum DragDropError: Error, LocalizedError {
    case emptyStaging
    case unsupportedFileType(String)
    case fileTooLarge(String, Int64)
    case readError(String)

    var errorDescription: String? {
        switch self {
        case .emptyStaging: return "No files in staging directory"
        case .unsupportedFileType(let ext): return "Unsupported file type: \(ext)"
        case .fileTooLarge(let name, let size): return "\(name) is too large (\(size / 1024 / 1024)MB)"
        case .readError(let name): return "Could not read \(name)"
        }
    }
}
