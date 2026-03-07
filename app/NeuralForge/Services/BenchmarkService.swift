// BenchmarkService.swift — Model evaluation with perplexity scoring and checkpoint comparison

import Foundation

// MARK: - Benchmark Models

struct BenchmarkResult: Identifiable, Codable {
    let id: UUID
    let projectID: UUID
    let checkpointPath: String
    let checkpointStep: Int
    let evalDataPath: String

    // Perplexity metrics
    let perplexity: Double
    let avgLoss: Double
    let tokenCount: Int
    let evalTimeSeconds: Double

    // Model info
    let modelParams: Int
    let loraRank: Int

    // Timestamp
    let evaluatedAt: Date

    // Optional label
    var label: String

    var perplexityFormatted: String {
        String(format: "%.2f", perplexity)
    }

    var lossFormatted: String {
        String(format: "%.4f", avgLoss)
    }

    var tokensPerSecond: Double {
        guard evalTimeSeconds > 0 else { return 0 }
        return Double(tokenCount) / evalTimeSeconds
    }
}

struct RegressionAlert: Identifiable {
    let id = UUID()
    let current: BenchmarkResult
    let previous: BenchmarkResult
    let perplexityDelta: Double
    let severity: Severity

    enum Severity: String {
        case warning = "Warning"
        case critical = "Critical"

        var threshold: Double {
            switch self {
            case .warning: return 0.5
            case .critical: return 2.0
            }
        }
    }

    var message: String {
        let direction = perplexityDelta > 0 ? "increased" : "decreased"
        return "Perplexity \(direction) by \(String(format: "%.2f", abs(perplexityDelta))) " +
               "(step \(previous.checkpointStep) → \(current.checkpointStep))"
    }
}

// MARK: - Benchmark Service

@MainActor
class BenchmarkService: ObservableObject {
    static let shared = BenchmarkService()

    @Published var results: [BenchmarkResult] = []
    @Published var isEvaluating = false
    @Published var evaluationProgress = ""
    @Published var regressionAlerts: [RegressionAlert] = []

    private let storageURL: URL

    init(storageURL: URL? = nil) {
        let defaultURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("NeuralForge/benchmark_results.json")
        self.storageURL = storageURL ?? defaultURL
        loadResults()
    }

    // MARK: - Evaluate Perplexity

    func evaluatePerplexity(projectID: UUID, checkpointPath: String, checkpointStep: Int,
                            evalDataPath: String, modelParams: Int, loraRank: Int,
                            cliRunner: CLIRunner,
                            onComplete: @escaping (BenchmarkResult?) -> Void) {
        guard !isEvaluating else {
            onComplete(nil)
            return
        }

        isEvaluating = true
        evaluationProgress = "Starting evaluation..."

        let startTime = Date()
        var batchCount = 0

        cliRunner.evaluatePerplexity(
            checkpointPath: checkpointPath,
            evalDataPath: evalDataPath,
            onBatch: { [weak self] batch, loss, tokens in
                batchCount = batch
                Task { @MainActor [weak self] in
                    self?.evaluationProgress = "Evaluating... batch \(batch), loss: \(String(format: "%.4f", loss))"
                }
            },
            onComplete: { [weak self] success, avgLoss, perplexity, totalTokens in
                Task { @MainActor [weak self] in
                    guard let self = self else { return }
                    self.isEvaluating = false

                    guard success, let avgLoss = avgLoss, let totalTokens = totalTokens, totalTokens > 0 else {
                        self.evaluationProgress = "Evaluation failed — no tokens processed"
                        onComplete(nil)
                        return
                    }

                    let ppl = perplexity ?? exp(avgLoss)
                    let elapsed = Date().timeIntervalSince(startTime)

                    let result = BenchmarkResult(
                        id: UUID(),
                        projectID: projectID,
                        checkpointPath: checkpointPath,
                        checkpointStep: checkpointStep,
                        evalDataPath: evalDataPath,
                        perplexity: ppl,
                        avgLoss: avgLoss,
                        tokenCount: totalTokens,
                        evalTimeSeconds: elapsed,
                        modelParams: modelParams,
                        loraRank: loraRank,
                        evaluatedAt: Date(),
                        label: "Step \(checkpointStep)"
                    )

                    self.addResult(result)
                    self.checkForRegression(result)
                    self.evaluationProgress = "Evaluation complete — perplexity: \(result.perplexityFormatted)"
                    onComplete(result)
                }
            }
        )
    }

    // MARK: - Compute perplexity from loss (for manual / test use)

    static func perplexityFromLoss(_ avgLoss: Double) -> Double {
        return exp(avgLoss)
    }

    static func computeStats(results: [BenchmarkResult]) -> BenchmarkStats {
        guard !results.isEmpty else {
            return BenchmarkStats(count: 0, bestPerplexity: 0, worstPerplexity: 0,
                                 avgPerplexity: 0, trend: .stable)
        }

        let perplexities = results.map(\.perplexity)
        let sorted = results.sorted { $0.checkpointStep < $1.checkpointStep }

        let trend: BenchmarkTrend
        if sorted.count >= 2 {
            let first = sorted.first!.perplexity
            let last = sorted.last!.perplexity
            let delta = last - first
            if delta < -0.5 { trend = .improving }
            else if delta > 0.5 { trend = .degrading }
            else { trend = .stable }
        } else {
            trend = .stable
        }

        return BenchmarkStats(
            count: results.count,
            bestPerplexity: perplexities.min() ?? 0,
            worstPerplexity: perplexities.max() ?? 0,
            avgPerplexity: perplexities.reduce(0, +) / Double(perplexities.count),
            trend: trend
        )
    }

    // MARK: - CRUD

    func addResult(_ result: BenchmarkResult) {
        results.insert(result, at: 0)
        saveResults()
    }

    func deleteResult(id: UUID) {
        results.removeAll { $0.id == id }
        regressionAlerts.removeAll { $0.current.id == id || $0.previous.id == id }
        saveResults()
    }

    func updateLabel(id: UUID, label: String) {
        if let idx = results.firstIndex(where: { $0.id == id }) {
            results[idx].label = label
            saveResults()
        }
    }

    func clearResults(for projectID: UUID) {
        results.removeAll { $0.projectID == projectID }
        regressionAlerts.removeAll()
        saveResults()
    }

    // MARK: - Queries

    func results(for projectID: UUID) -> [BenchmarkResult] {
        results.filter { $0.projectID == projectID }
            .sorted { $0.checkpointStep < $1.checkpointStep }
    }

    func bestResult(for projectID: UUID) -> BenchmarkResult? {
        results(for: projectID).min(by: { $0.perplexity < $1.perplexity })
    }

    // MARK: - Regression Detection

    private func checkForRegression(_ newResult: BenchmarkResult) {
        let projectResults = results(for: newResult.projectID)
            .filter { $0.id != newResult.id }
            .sorted { $0.checkpointStep < $1.checkpointStep }

        guard let previous = projectResults.last else { return }

        let delta = newResult.perplexity - previous.perplexity
        if delta > RegressionAlert.Severity.critical.threshold {
            regressionAlerts.append(RegressionAlert(
                current: newResult, previous: previous,
                perplexityDelta: delta, severity: .critical
            ))
        } else if delta > RegressionAlert.Severity.warning.threshold {
            regressionAlerts.append(RegressionAlert(
                current: newResult, previous: previous,
                perplexityDelta: delta, severity: .warning
            ))
        }
    }

    func dismissAlert(id: UUID) {
        regressionAlerts.removeAll { $0.id == id }
    }

    // MARK: - Persistence

    private func loadResults() {
        guard FileManager.default.fileExists(atPath: storageURL.path) else { return }
        do {
            let data = try Data(contentsOf: storageURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            results = try decoder.decode([BenchmarkResult].self, from: data)
        } catch {
            print("[Benchmark] Failed to load: \(error)")
        }
    }

    private func saveResults() {
        do {
            let dir = storageURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(results)
            try data.write(to: storageURL, options: .atomic)
        } catch {
            print("[Benchmark] Failed to save: \(error)")
        }
    }

    // MARK: - Export

    func exportCSV(for projectID: UUID) -> String {
        let projectResults = results(for: projectID)
        var csv = "Label,Step,Perplexity,Avg Loss,Tokens,Eval Time (s),Tokens/s,Params,LoRA Rank,Evaluated At\n"
        for r in projectResults {
            let label = r.label.replacingOccurrences(of: "\"", with: "\"\"")
            csv += "\"\(label)\",\(r.checkpointStep),"
            csv += String(format: "%.4f,%.6f,", r.perplexity, r.avgLoss)
            csv += "\(r.tokenCount),\(String(format: "%.1f", r.evalTimeSeconds)),"
            csv += String(format: "%.0f,", r.tokensPerSecond)
            csv += "\(r.modelParams),\(r.loraRank),"
            csv += "\(ISO8601DateFormatter().string(from: r.evaluatedAt))\n"
        }
        return csv
    }
}

// MARK: - Supporting Types

struct BenchmarkStats {
    let count: Int
    let bestPerplexity: Double
    let worstPerplexity: Double
    let avgPerplexity: Double
    let trend: BenchmarkTrend
}

enum BenchmarkTrend: String {
    case improving = "Improving"
    case stable = "Stable"
    case degrading = "Degrading"

    var icon: String {
        switch self {
        case .improving: return "arrow.down.right"
        case .stable: return "arrow.right"
        case .degrading: return "arrow.up.right"
        }
    }

    var color: String {
        switch self {
        case .improving: return "green"
        case .stable: return "blue"
        case .degrading: return "red"
        }
    }
}
