// TrainingProfileService.swift — Save/load full training config presets

import Foundation

/// A named training configuration preset that captures all hyperparameters
struct TrainingProfile: Identifiable, Codable {
    let id: UUID
    var name: String
    var description: String
    var config: TrainingConfig
    var created: Date
    var lastUsed: Date?
    var tags: [String]

    init(name: String, description: String = "", config: TrainingConfig, tags: [String] = []) {
        self.id = UUID()
        self.name = name
        self.description = description
        self.config = config
        self.created = Date()
        self.lastUsed = nil
        self.tags = tags
    }

    /// Built-in presets for common use cases
    enum BuiltIn: String, CaseIterable {
        case quickTest = "Quick Test"
        case standard = "Standard Training"
        case longRun = "Long Run"
        case loraFineTune = "LoRA Fine-Tune"
        case conservative = "Conservative"

        var profile: TrainingProfile {
            switch self {
            case .quickTest:
                var cfg = TrainingConfig()
                cfg.totalSteps = 100
                cfg.learningRate = 3e-4
                cfg.checkpointEvery = 50
                cfg.warmupSteps = 10
                cfg.lrSchedule = "cosine"
                return TrainingProfile(name: rawValue, description: "Fast iteration — 100 steps for testing", config: cfg, tags: ["quick", "test"])

            case .standard:
                var cfg = TrainingConfig()
                cfg.totalSteps = 5000
                cfg.learningRate = 3e-4
                cfg.warmupSteps = 100
                cfg.lrMin = 1e-5
                cfg.lrSchedule = "cosine"
                cfg.checkpointEvery = 500
                cfg.shuffle = true
                return TrainingProfile(name: rawValue, description: "Balanced training with cosine schedule", config: cfg, tags: ["standard", "cosine"])

            case .longRun:
                var cfg = TrainingConfig()
                cfg.totalSteps = 50000
                cfg.learningRate = 1e-4
                cfg.warmupSteps = 500
                cfg.lrMin = 1e-6
                cfg.lrSchedule = "cosine"
                cfg.checkpointEvery = 1000
                cfg.shuffle = true
                cfg.gradClipNorm = 0.5
                return TrainingProfile(name: rawValue, description: "Extended training for full convergence", config: cfg, tags: ["long", "production"])

            case .loraFineTune:
                var cfg = TrainingConfig()
                cfg.totalSteps = 2000
                cfg.learningRate = 1e-4
                cfg.warmupSteps = 50
                cfg.lrMin = 1e-6
                cfg.lrSchedule = "cosine"
                cfg.checkpointEvery = 200
                cfg.loraRank = 8
                cfg.loraAlpha = 16.0
                cfg.loraTargets = 15  // all QKVO
                cfg.shuffle = true
                return TrainingProfile(name: rawValue, description: "Memory-efficient LoRA r=8", config: cfg, tags: ["lora", "efficient"])

            case .conservative:
                var cfg = TrainingConfig()
                cfg.totalSteps = 10000
                cfg.learningRate = 1e-4
                cfg.warmupSteps = 200
                cfg.lrMin = 1e-6
                cfg.lrSchedule = "cosine"
                cfg.checkpointEvery = 100
                cfg.gradClipNorm = 0.5
                cfg.beta1 = 0.9
                cfg.beta2 = 0.95
                cfg.accumSteps = 20
                return TrainingProfile(name: rawValue, description: "Slow and steady — aggressive grad clip, high accumulation", config: cfg, tags: ["conservative", "stable"])
            }
        }
    }
}

/// Manages training profile persistence, search, and application
@MainActor
class TrainingProfileService: ObservableObject {
    @Published var profiles: [TrainingProfile] = []
    @Published var recentProfiles: [UUID] = []

    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    static let maxRecentProfiles = 5

    init() {
        encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        loadProfiles()
    }

    // MARK: - Directory

    static var profilesDir: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("NeuralForge/Profiles")
    }

    // MARK: - CRUD

    func save(_ profile: TrainingProfile) {
        let url = Self.profilesDir.appendingPathComponent("\(profile.id.uuidString).json")
        try? FileManager.default.createDirectory(at: Self.profilesDir, withIntermediateDirectories: true)
        if let data = try? encoder.encode(profile) {
            try? data.write(to: url, options: .atomic)
        }
        if let idx = profiles.firstIndex(where: { $0.id == profile.id }) {
            profiles[idx] = profile
        } else {
            profiles.append(profile)
        }
    }

    func delete(_ profile: TrainingProfile) {
        let url = Self.profilesDir.appendingPathComponent("\(profile.id.uuidString).json")
        try? FileManager.default.removeItem(at: url)
        profiles.removeAll { $0.id == profile.id }
        recentProfiles.removeAll { $0 == profile.id }
    }

    func duplicate(_ profile: TrainingProfile) -> TrainingProfile {
        let copy = TrainingProfile(
            name: "\(profile.name) Copy",
            description: profile.description,
            config: profile.config,
            tags: profile.tags
        )
        save(copy)
        return copy
    }

    /// Create a profile from an existing project's config
    func createFromProject(_ project: NFProject, name: String? = nil) -> TrainingProfile {
        let profileName = name ?? "From: \(project.name)"
        let profile = TrainingProfile(
            name: profileName,
            description: "Extracted from project \(project.name)",
            config: project.config,
            tags: ["project-derived"]
        )
        save(profile)
        return profile
    }

    /// Apply a profile's config to a project
    func apply(_ profile: TrainingProfile, to project: inout NFProject) {
        project.config = profile.config
        trackRecent(profile.id)

        // Update lastUsed timestamp
        var updated = profile
        updated.lastUsed = Date()
        save(updated)
    }

    // MARK: - Search & Filter

    func profiles(matching query: String) -> [TrainingProfile] {
        guard !query.isEmpty else { return profiles }
        let lower = query.lowercased()
        return profiles.filter {
            $0.name.lowercased().contains(lower)
                || $0.description.lowercased().contains(lower)
                || $0.tags.contains(where: { $0.lowercased().contains(lower) })
        }
    }

    func profiles(withTag tag: String) -> [TrainingProfile] {
        profiles.filter { $0.tags.contains(tag) }
    }

    var allTags: [String] {
        Array(Set(profiles.flatMap { $0.tags })).sorted()
    }

    // MARK: - Import/Export

    func exportProfile(_ profile: TrainingProfile) -> Data? {
        try? encoder.encode(profile)
    }

    func importProfile(from data: Data) -> TrainingProfile? {
        guard let profile = try? decoder.decode(TrainingProfile.self, from: data) else { return nil }
        save(profile)
        return profile
    }

    func exportAll() -> Data? {
        try? encoder.encode(profiles)
    }

    /// Compute a summary of config differences between two profiles
    static func diff(_ a: TrainingProfile, _ b: TrainingProfile) -> [String] {
        var diffs: [String] = []
        let ca = a.config, cb = b.config
        if ca.totalSteps != cb.totalSteps { diffs.append("Steps: \(ca.totalSteps) → \(cb.totalSteps)") }
        if ca.learningRate != cb.learningRate { diffs.append("LR: \(ca.learningRate) → \(cb.learningRate)") }
        if ca.accumSteps != cb.accumSteps { diffs.append("Accum: \(ca.accumSteps) → \(cb.accumSteps)") }
        if ca.warmupSteps != cb.warmupSteps { diffs.append("Warmup: \(ca.warmupSteps) → \(cb.warmupSteps)") }
        if ca.lrSchedule != cb.lrSchedule { diffs.append("Schedule: \(ca.lrSchedule) → \(cb.lrSchedule)") }
        if ca.loraRank != cb.loraRank { diffs.append("LoRA Rank: \(ca.loraRank) → \(cb.loraRank)") }
        if ca.gradClipNorm != cb.gradClipNorm { diffs.append("Grad Clip: \(ca.gradClipNorm) → \(cb.gradClipNorm)") }
        if ca.shuffle != cb.shuffle { diffs.append("Shuffle: \(ca.shuffle) → \(cb.shuffle)") }
        if ca.useANEExtras != cb.useANEExtras { diffs.append("ANE Extras: \(ca.useANEExtras) → \(cb.useANEExtras)") }
        return diffs
    }

    // MARK: - Built-in Profiles

    func installBuiltInProfiles() {
        for builtIn in TrainingProfile.BuiltIn.allCases {
            let existing = profiles.first { $0.name == builtIn.rawValue }
            if existing == nil {
                save(builtIn.profile)
            }
        }
    }

    // MARK: - Persistence

    private func loadProfiles() {
        let fm = FileManager.default
        try? fm.createDirectory(at: Self.profilesDir, withIntermediateDirectories: true)

        guard let contents = try? fm.contentsOfDirectory(at: Self.profilesDir,
                                                          includingPropertiesForKeys: nil) else { return }

        profiles = contents.compactMap { url -> TrainingProfile? in
            guard url.pathExtension == "json" else { return nil }
            guard let data = try? Data(contentsOf: url) else { return nil }
            return try? decoder.decode(TrainingProfile.self, from: data)
        }.sorted { $0.created > $1.created }
    }

    private func trackRecent(_ profileID: UUID) {
        recentProfiles.removeAll { $0 == profileID }
        recentProfiles.insert(profileID, at: 0)
        if recentProfiles.count > Self.maxRecentProfiles {
            recentProfiles = Array(recentProfiles.prefix(Self.maxRecentProfiles))
        }
    }
}
