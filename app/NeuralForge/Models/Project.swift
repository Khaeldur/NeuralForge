// Project.swift — NeuralForge project model

import Foundation

struct NFProject: Identifiable, Codable {
    let id: UUID
    var name: String
    var modelPath: String
    var dataPath: String
    var checkpointPath: String
    var created: Date
    var lastTrained: Date?

    var config: TrainingConfig

    init(name: String) {
        self.id = UUID()
        self.name = name
        self.modelPath = ""
        self.dataPath = ""
        self.checkpointPath = ""
        self.created = Date()
        self.config = TrainingConfig()
    }

    var directory: URL {
        NFProject.projectsDir.appendingPathComponent(id.uuidString)
    }

    static var projectsDir: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("NeuralForge/Projects")
    }
}

struct TrainingConfig: Codable {
    var totalSteps: Int = 10000
    var learningRate: Double = 3e-4
    var accumSteps: Int = 10
    var checkpointEvery: Int = 100
    var gradClipNorm: Double = 1.0
    var beta1: Double = 0.9
    var beta2: Double = 0.999
    var eps: Double = 1e-8
    var useANEExtras: Bool = true
    var seed: Int = 42

    // LR Scheduler
    var warmupSteps: Int = 0
    var lrMin: Double = 1e-5
    var lrSchedule: String = "none"  // "none" or "cosine"

    // Data Pipeline
    var valDataPath: String = ""
    var valEvery: Int = 0       // 0 = disabled
    var valBatches: Int = 10
    var shuffle: Bool = false

    // LoRA
    var loraRank: Int = 0       // 0 = full fine-tune, 4-64 typical
    var loraAlpha: Double = 16.0
    var loraTargets: Int = 8    // bitmask: 8 = Wo only
}
