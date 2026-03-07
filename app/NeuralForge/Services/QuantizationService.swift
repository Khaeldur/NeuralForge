// QuantizationService.swift — GGUF quantization and CoreML conversion pipeline

import Foundation

// MARK: - Quantization Types

enum QuantizationType: String, CaseIterable, Identifiable {
    case f16 = "F16"
    case q8_0 = "Q8_0"
    case q5_1 = "Q5_1"
    case q5_0 = "Q5_0"
    case q4_1 = "Q4_1"
    case q4_0 = "Q4_0"
    case q3_K_M = "Q3_K_M"
    case q2_K = "Q2_K"

    var id: String { rawValue }

    var description: String {
        switch self {
        case .f16: return "Full 16-bit precision — no quality loss"
        case .q8_0: return "8-bit — near-lossless, ~50% size reduction"
        case .q5_1: return "5-bit (variant 1) — good quality, ~68% smaller"
        case .q5_0: return "5-bit (variant 0) — good quality, ~68% smaller"
        case .q4_1: return "4-bit (variant 1) — balanced quality/size"
        case .q4_0: return "4-bit (variant 0) — smallest practical 4-bit"
        case .q3_K_M: return "3-bit K-quant medium — aggressive compression"
        case .q2_K: return "2-bit K-quant — maximum compression, quality loss"
        }
    }

    var bitsPerWeight: Double {
        switch self {
        case .f16: return 16.0
        case .q8_0: return 8.5
        case .q5_1: return 5.5
        case .q5_0: return 5.5
        case .q4_1: return 5.0
        case .q4_0: return 4.5
        case .q3_K_M: return 3.9
        case .q2_K: return 3.35
        }
    }

    var qualityRating: Int {
        switch self {
        case .f16: return 10
        case .q8_0: return 9
        case .q5_1: return 8
        case .q5_0: return 7
        case .q4_1: return 7
        case .q4_0: return 6
        case .q3_K_M: return 5
        case .q2_K: return 3
        }
    }

    static func estimateSize(modelParams: Int, quantType: QuantizationType) -> Int64 {
        Int64(Double(modelParams) * quantType.bitsPerWeight / 8.0)
    }

    static func formatSize(_ bytes: Int64) -> String {
        if bytes >= 1_000_000_000 { return String(format: "%.1f GB", Double(bytes) / 1e9) }
        if bytes >= 1_000_000 { return String(format: "%.0f MB", Double(bytes) / 1e6) }
        return String(format: "%.0f KB", Double(bytes) / 1e3)
    }
}

// MARK: - Quantization Job

struct QuantizationJob: Identifiable {
    let id = UUID()
    let inputPath: String
    let outputPath: String
    let quantType: QuantizationType
    let timestamp: Date
    var status: JobStatus
    var progress: String
    var outputSize: Int64?

    enum JobStatus: Equatable {
        case pending
        case running(percent: Int)
        case success
        case failed(String)
    }
}

// MARK: - CoreML Conversion Config

struct CoreMLConfig {
    var computeUnits: ComputeUnit = .all
    var minimumDeploymentTarget: String = "macOS14"
    var precision: Precision = .float16

    enum ComputeUnit: String, CaseIterable {
        case all = "All (CPU + GPU + ANE)"
        case cpuAndGPU = "CPU + GPU"
        case cpuOnly = "CPU Only"

        var coremlValue: String {
            switch self {
            case .all: return "all"
            case .cpuAndGPU: return "cpu_and_gpu"
            case .cpuOnly: return "cpu_only"
            }
        }
    }

    enum Precision: String, CaseIterable {
        case float16 = "Float16"
        case float32 = "Float32"
    }
}

// MARK: - Service

@MainActor
class QuantizationService: ObservableObject {
    static let shared = QuantizationService()

    @Published var jobs: [QuantizationJob] = []
    @Published var isProcessing = false
    @Published var currentProgress = ""

    // MARK: - GGUF Quantization

    func quantizeGGUF(inputPath: String, outputPath: String, quantType: QuantizationType,
                       cliRunner: CLIRunner, onComplete: @escaping (Bool, String?) -> Void) {
        let job = QuantizationJob(
            inputPath: inputPath, outputPath: outputPath,
            quantType: quantType, timestamp: Date(),
            status: .running(percent: 0), progress: "Starting quantization..."
        )
        jobs.insert(job, at: 0)
        isProcessing = true
        currentProgress = "Quantizing to \(quantType.rawValue)..."

        cliRunner.exportModel(
            checkpointPath: inputPath,
            format: "gguf",
            outputPath: outputPath
        ) { [weak self] success, error in
            Task { @MainActor in
                guard let self = self else { return }
                self.isProcessing = false

                if let idx = self.jobs.firstIndex(where: { $0.id == job.id }) {
                    if success {
                        self.jobs[idx].status = .success
                        self.jobs[idx].progress = "Complete"
                        // Get output size
                        if let attrs = try? FileManager.default.attributesOfItem(atPath: outputPath),
                           let size = attrs[.size] as? Int64 {
                            self.jobs[idx].outputSize = size
                        }
                    } else {
                        self.jobs[idx].status = .failed(error ?? "Unknown error")
                        self.jobs[idx].progress = error ?? "Failed"
                    }
                }

                self.currentProgress = ""
                onComplete(success, error)
            }
        }
    }

    // MARK: - CoreML Conversion

    func convertCoreML(inputPath: String, outputPath: String, config: CoreMLConfig,
                        cliRunner: CLIRunner, onComplete: @escaping (Bool, String?) -> Void) {
        let job = QuantizationJob(
            inputPath: inputPath, outputPath: outputPath,
            quantType: .f16, timestamp: Date(),
            status: .running(percent: 0), progress: "Converting to CoreML..."
        )
        jobs.insert(job, at: 0)
        isProcessing = true
        currentProgress = "Converting to CoreML (\(config.computeUnits.rawValue))..."

        cliRunner.exportCoreML(
            checkpointPath: inputPath,
            outputPath: outputPath
        ) { [weak self] success, error in
            Task { @MainActor in
                guard let self = self else { return }
                self.isProcessing = false

                if let idx = self.jobs.firstIndex(where: { $0.id == job.id }) {
                    if success {
                        self.jobs[idx].status = .success
                        self.jobs[idx].progress = "Complete"
                    } else {
                        self.jobs[idx].status = .failed(error ?? "Conversion failed")
                        self.jobs[idx].progress = error ?? "Failed"
                    }
                }

                self.currentProgress = ""
                onComplete(success, error)
            }
        }
    }

    // MARK: - Helpers

    func clearJobs() {
        jobs.removeAll { job in
            if case .running = job.status { return false }
            return true
        }
    }

    var successCount: Int { jobs.filter { $0.status == .success }.count }
    var failedCount: Int {
        jobs.filter { job in
            if case .failed = job.status { return true }
            return false
        }.count
    }
}
