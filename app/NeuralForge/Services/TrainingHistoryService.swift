// TrainingHistoryService.swift — Persist and query completed training runs

import Foundation

// MARK: - Training Run Model

struct TrainingRun: Identifiable, Codable {
    let id: UUID
    let projectID: UUID
    let projectName: String

    // Model info
    let modelPath: String
    let modelParams: Int
    let modelLayers: Int
    let modelDim: Int

    // Config snapshot
    let config: TrainingRunConfig

    // Results
    let finalLoss: Double
    let bestLoss: Double
    let finalStep: Int
    let totalSteps: Int
    let totalTimeSeconds: Double
    let avgMsPerStep: Double
    let avgTFLOPS: Double

    // Loss curve (downsampled for storage)
    let lossHistory: [LossPoint]
    let valLossHistory: [LossPoint]

    // Timestamps
    let startedAt: Date
    let completedAt: Date

    // Optional notes
    var notes: String

    var durationFormatted: String {
        let s = totalTimeSeconds
        if s < 60 { return String(format: "%.0fs", s) }
        if s < 3600 { return String(format: "%.0fm %.0fs", s / 60, s.truncatingRemainder(dividingBy: 60)) }
        return String(format: "%.0fh %.0fm", s / 3600, (s / 60).truncatingRemainder(dividingBy: 60))
    }

    var completionRatio: Double {
        guard totalSteps > 0 else { return 0 }
        return Double(finalStep) / Double(totalSteps)
    }
}

struct LossPoint: Codable {
    let step: Int
    let loss: Double
}

struct TrainingRunConfig: Codable {
    let learningRate: Double
    let accumSteps: Int
    let checkpointEvery: Int
    let gradClipNorm: Double
    let seed: Int
    let warmupSteps: Int
    let lrSchedule: String
    let shuffle: Bool
    let loraRank: Int
    let loraAlpha: Double
    let dataPath: String
    let valDataPath: String

    init(from config: TrainingConfig, dataPath: String) {
        self.learningRate = config.learningRate
        self.accumSteps = config.accumSteps
        self.checkpointEvery = config.checkpointEvery
        self.gradClipNorm = config.gradClipNorm
        self.seed = config.seed
        self.warmupSteps = config.warmupSteps
        self.lrSchedule = config.lrSchedule
        self.shuffle = config.shuffle
        self.loraRank = config.loraRank
        self.loraAlpha = config.loraAlpha
        self.dataPath = dataPath
        self.valDataPath = config.valDataPath
    }

    // Direct init for tests / manual creation
    init(learningRate: Double = 3e-4, accumSteps: Int = 10,
         checkpointEvery: Int = 100, gradClipNorm: Double = 1.0,
         seed: Int = 42, warmupSteps: Int = 0, lrSchedule: String = "none",
         shuffle: Bool = false, loraRank: Int = 0, loraAlpha: Double = 16.0,
         dataPath: String = "", valDataPath: String = "") {
        self.learningRate = learningRate
        self.accumSteps = accumSteps
        self.checkpointEvery = checkpointEvery
        self.gradClipNorm = gradClipNorm
        self.seed = seed
        self.warmupSteps = warmupSteps
        self.lrSchedule = lrSchedule
        self.shuffle = shuffle
        self.loraRank = loraRank
        self.loraAlpha = loraAlpha
        self.dataPath = dataPath
        self.valDataPath = valDataPath
    }
}

// MARK: - Training History Service

@MainActor
class TrainingHistoryService: ObservableObject {
    static let shared = TrainingHistoryService()

    @Published var runs: [TrainingRun] = []

    private let storageURL: URL
    private let maxRuns: Int

    init(storageURL: URL? = nil, maxRuns: Int = 0) {
        let defaultURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("NeuralForge/training_history.json")

        self.storageURL = storageURL ?? defaultURL
        // 0 means use AppStorage value at runtime
        self.maxRuns = maxRuns
        loadRuns()
    }

    // MARK: - CRUD

    func addRun(_ run: TrainingRun) {
        runs.insert(run, at: 0)
        let limit = maxRuns > 0 ? maxRuns : UserDefaults.standard.integer(forKey: "maxHistoryEntries")
        let effectiveLimit = limit > 0 ? limit : 100
        if runs.count > effectiveLimit {
            runs = Array(runs.prefix(effectiveLimit))
        }
        saveRuns()
    }

    func deleteRun(id: UUID) {
        runs.removeAll { $0.id == id }
        saveRuns()
    }

    func deleteRuns(ids: Set<UUID>) {
        runs.removeAll { ids.contains($0.id) }
        saveRuns()
    }

    func updateNotes(id: UUID, notes: String) {
        if let idx = runs.firstIndex(where: { $0.id == id }) {
            runs[idx].notes = notes
            saveRuns()
        }
    }

    func clearHistory() {
        runs.removeAll()
        saveRuns()
    }

    // MARK: - Queries

    func runs(for projectID: UUID) -> [TrainingRun] {
        runs.filter { $0.projectID == projectID }
    }

    func bestRun(for projectID: UUID) -> TrainingRun? {
        runs(for: projectID).min(by: { $0.bestLoss < $1.bestLoss })
    }

    func recentRuns(limit: Int = 10) -> [TrainingRun] {
        Array(runs.prefix(limit))
    }

    func searchRuns(query: String) -> [TrainingRun] {
        let q = query.lowercased()
        return runs.filter {
            $0.projectName.lowercased().contains(q)
            || $0.notes.lowercased().contains(q)
            || $0.modelPath.lowercased().contains(q)
            || ($0.config.loraRank > 0 && "lora".contains(q))
        }
    }

    // MARK: - Auto-save from training completion

    func recordCompletedRun(project: NFProject, state: TrainingState) {
        let lossHistory = downsample(state.lossHistory, maxPoints: 500)
        let valLossHistory = downsample(state.valLossHistory, maxPoints: 200)

        let run = TrainingRun(
            id: UUID(),
            projectID: project.id,
            projectName: project.name,
            modelPath: project.modelPath,
            modelParams: state.modelParams,
            modelLayers: state.modelLayers,
            modelDim: state.modelDim,
            config: TrainingRunConfig(from: project.config, dataPath: project.dataPath),
            finalLoss: state.currentLoss,
            bestLoss: state.bestLoss,
            finalStep: state.currentStep,
            totalSteps: state.totalSteps,
            totalTimeSeconds: state.totalTimeSeconds,
            avgMsPerStep: state.msPerStep,
            avgTFLOPS: state.tflopsTotal,
            lossHistory: lossHistory.map { LossPoint(step: $0.step, loss: $0.loss) },
            valLossHistory: valLossHistory.map { LossPoint(step: $0.step, loss: $0.loss) },
            startedAt: Date().addingTimeInterval(-state.totalTimeSeconds),
            completedAt: Date(),
            notes: ""
        )

        addRun(run)
    }

    // MARK: - Persistence

    private func loadRuns() {
        guard FileManager.default.fileExists(atPath: storageURL.path) else { return }
        do {
            let data = try Data(contentsOf: storageURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            runs = try decoder.decode([TrainingRun].self, from: data)
        } catch {
            print("[TrainingHistory] Failed to load: \(error)")
        }
    }

    private func saveRuns() {
        do {
            let dir = storageURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(runs)
            try data.write(to: storageURL, options: .atomic)
        } catch {
            print("[TrainingHistory] Failed to save: \(error)")
        }
    }

    // MARK: - Helpers

    private func downsample(_ points: [(step: Int, loss: Double)], maxPoints: Int) -> [(step: Int, loss: Double)] {
        guard points.count > maxPoints else { return points }
        let stride = points.count / maxPoints
        var result: [(step: Int, loss: Double)] = []
        for i in Swift.stride(from: 0, to: points.count, by: stride) {
            result.append(points[i])
        }
        // Always include the last point
        if let last = points.last, result.last?.step != last.step {
            result.append(last)
        }
        return result
    }

    // MARK: - Export

    func exportCSV() -> String {
        var csv = "Run ID,Project,Model,Params,Final Loss,Best Loss,Steps,Duration (s),LR,LoRA Rank,Schedule,Started,Completed,Notes\n"
        for run in runs {
            let notes = run.notes.replacingOccurrences(of: "\"", with: "\"\"")
            csv += "\(run.id),\"\(run.projectName)\",\"\(run.modelPath)\",\(run.modelParams),"
            csv += String(format: "%.6f,%.6f,", run.finalLoss, run.bestLoss)
            csv += "\(run.finalStep),\(String(format: "%.1f", run.totalTimeSeconds)),"
            csv += "\(run.config.learningRate),\(run.config.loraRank),\(run.config.lrSchedule),"
            csv += "\(ISO8601DateFormatter().string(from: run.startedAt)),"
            csv += "\(ISO8601DateFormatter().string(from: run.completedAt)),"
            csv += "\"\(notes)\"\n"
        }
        return csv
    }
}
