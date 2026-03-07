// TrainingProgress.swift — JSON messages from CLI + observable state

import Foundation

// MARK: – CLI JSON Message Types

enum CLIMessage: Decodable {
    case init_(InitMsg)
    case step(StepMsg)
    case batch(BatchMsg)
    case checkpoint(CheckpointMsg)
    case restart(RestartMsg)
    case done(DoneMsg)
    case error(ErrorMsg)
    case info(InfoMsg)
    case val(ValMsg)
    case token(TokenMsg)
    case generateDone(GenerateDoneMsg)

    struct InitMsg: Decodable {
        let params, layers, dim, hidden, heads, seq, vocab: Int
        let timestamp: Int?
    }
    struct StepMsg: Decodable {
        let step, total: Int
        let loss: Double
        let lr: Double
        let ms: Double
        let tflops_ane: Double
        let tflops_total: Double
    }
    struct BatchMsg: Decodable {
        let batch, step: Int
        let avg_loss, best_loss: Double
        let compile_ms, train_ms: Double
        let compiles: Int
    }
    struct CheckpointMsg: Decodable {
        let path: String
        let step: Int
        let loss: Double
    }
    struct RestartMsg: Decodable {
        let step, compiles: Int
        let reason: String
    }
    struct DoneMsg: Decodable {
        let total_steps: Int
        let final_loss: Double
        let total_time_s: Double
        let tflops_ane: Double
        let tflops_total: Double
    }
    struct ErrorMsg: Decodable {
        let message: String
        let code: Int
    }
    struct InfoMsg: Decodable {
        let key: String
        let value: AnyCodable
    }
    struct ValMsg: Decodable {
        let step: Int
        let val_loss: Double
        let val_batches: Int
    }
    struct TokenMsg: Decodable {
        let token_id: Int
        let text: String
    }
    struct GenerateDoneMsg: Decodable {
        let tokens: Int
        let total_ms: Double
    }

    private enum CodingKeys: String, CodingKey { case type }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let type = try c.decode(String.self, forKey: .type)
        let sc = try decoder.singleValueContainer()
        switch type {
        case "init":       self = .init_(try sc.decode(InitMsg.self))
        case "step":       self = .step(try sc.decode(StepMsg.self))
        case "batch":      self = .batch(try sc.decode(BatchMsg.self))
        case "checkpoint": self = .checkpoint(try sc.decode(CheckpointMsg.self))
        case "restart":    self = .restart(try sc.decode(RestartMsg.self))
        case "done":       self = .done(try sc.decode(DoneMsg.self))
        case "error":      self = .error(try sc.decode(ErrorMsg.self))
        case "info":       self = .info(try sc.decode(InfoMsg.self))
        case "val":           self = .val(try sc.decode(ValMsg.self))
        case "token":         self = .token(try sc.decode(TokenMsg.self))
        case "generate_done": self = .generateDone(try sc.decode(GenerateDoneMsg.self))
        default: throw DecodingError.dataCorruptedError(forKey: .type, in: c, debugDescription: "Unknown type: \(type)")
        }
    }
}

// Simple Any wrapper for JSON
struct AnyCodable: Decodable {
    let value: Any

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let v = try? c.decode(String.self) { value = v }
        else if let v = try? c.decode(Int.self) { value = v }
        else if let v = try? c.decode(Double.self) { value = v }
        else if let v = try? c.decode(Bool.self) { value = v }
        else if let v = try? c.decode([String: AnyCodable].self) { value = v.mapValues(\.value) }
        else if let v = try? c.decode([AnyCodable].self) { value = v.map(\.value) }
        else { value = NSNull() }
    }
}

// MARK: – Observable Training State

@MainActor
class TrainingState: ObservableObject {
    enum Phase: String {
        case idle, compiling, training, paused, done, error
    }

    @Published var phase: Phase = .idle
    @Published var currentStep: Int = 0
    @Published var totalSteps: Int = 0
    @Published var currentLoss: Double = 0
    @Published var bestLoss: Double = .infinity
    @Published var learningRate: Double = 3e-4
    @Published var msPerStep: Double = 0
    @Published var tflopsANE: Double = 0
    @Published var tflopsTotal: Double = 0
    @Published var compileCount: Int = 0
    @Published var restartCount: Int = 0
    @Published var lossHistory: [(step: Int, loss: Double)] = []
    @Published var emaLossHistory: [(step: Int, loss: Double)] = []
    @Published var valLossHistory: [(step: Int, loss: Double)] = []
    @Published var tflopsHistory: [(step: Int, tflops: Double)] = []
    @Published var errorMessage: String?
    private var emaLoss: Double = 0
    @Published var lastCheckpoint: String?
    @Published var totalTimeSeconds: Double = 0
    @Published var compileStartTime: Date?

    var compileElapsed: TimeInterval {
        guard let start = compileStartTime else { return 0 }
        return Date().timeIntervalSince(start)
    }

    // Model info
    @Published var modelParams: Int = 0
    @Published var modelLayers: Int = 0
    @Published var modelDim: Int = 0
    @Published var modelVocab: Int = 0

    var progress: Double {
        guard totalSteps > 0 else { return 0 }
        return Double(currentStep) / Double(totalSteps)
    }

    var eta: String {
        guard msPerStep > 0, currentStep < totalSteps else { return "--" }
        let remaining = Double(totalSteps - currentStep) * msPerStep / 1000.0
        if remaining < 60 { return String(format: "%.0fs", remaining) }
        if remaining < 3600 { return String(format: "%.0fm", remaining / 60) }
        return String(format: "%.1fh", remaining / 3600)
    }

    func handle(_ msg: CLIMessage) {
        switch msg {
        case .init_(let m):
            modelParams = m.params
            modelLayers = m.layers
            modelDim = m.dim
            modelVocab = m.vocab
            phase = .compiling
            compileStartTime = Date()
        case .step(let m):
            phase = .training
            compileStartTime = nil
            currentStep = m.step
            totalSteps = m.total
            currentLoss = m.loss
            learningRate = m.lr
            msPerStep = m.ms
            tflopsANE = m.tflops_ane
            tflopsTotal = m.tflops_total
            if m.loss < bestLoss { bestLoss = m.loss }
            lossHistory.append((step: m.step, loss: m.loss))
            // EMA smoothing (alpha=0.98)
            if emaLoss == 0 { emaLoss = m.loss }
            else { emaLoss = 0.98 * emaLoss + 0.02 * m.loss }
            emaLossHistory.append((step: m.step, loss: emaLoss))
            // TFLOPS tracking
            tflopsHistory.append((step: m.step, tflops: m.tflops_ane))
            // Keep history manageable
            if lossHistory.count > 5000 {
                lossHistory = Array(lossHistory.suffix(3000))
                emaLossHistory = Array(emaLossHistory.suffix(3000))
                tflopsHistory = Array(tflopsHistory.suffix(3000))
            }
        case .batch(let m):
            compileCount = m.compiles
            if m.avg_loss < bestLoss { bestLoss = m.avg_loss }
        case .checkpoint(let m):
            lastCheckpoint = m.path
        case .restart(let m):
            restartCount += 1
            compileCount = m.compiles
            phase = .compiling
            compileStartTime = Date()
        case .done(let m):
            phase = .done
            currentLoss = m.final_loss
            totalTimeSeconds = m.total_time_s
            tflopsANE = m.tflops_ane
            tflopsTotal = m.tflops_total
        case .error(let m):
            phase = .error
            errorMessage = m.message
        case .val(let m):
            valLossHistory.append((step: m.step, loss: m.val_loss))
            if valLossHistory.count > 2000 {
                valLossHistory = Array(valLossHistory.suffix(1500))
            }
        case .info:
            break
        case .token, .generateDone:
            break  // Handled by GenerateView via CLIRunner streaming
        }
    }

    func reset() {
        phase = .idle
        currentStep = 0; totalSteps = 0
        currentLoss = 0; bestLoss = .infinity
        msPerStep = 0; tflopsANE = 0; tflopsTotal = 0
        compileCount = 0; restartCount = 0
        lossHistory = []; emaLossHistory = []; valLossHistory = []
        tflopsHistory = []; emaLoss = 0; errorMessage = nil
        lastCheckpoint = nil; totalTimeSeconds = 0; compileStartTime = nil
    }
}
