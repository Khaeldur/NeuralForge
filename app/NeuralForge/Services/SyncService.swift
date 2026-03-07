// SyncService.swift — Checkpoint sync service with LaunchAgent support
// Monitors project checkpoints and syncs to a shared/backup directory.
// Can install a LaunchAgent plist for automatic periodic sync.

import Foundation

// MARK: - Sync Configuration

struct SyncConfig: Codable {
    var syncEnabled: Bool = false
    var syncDirectory: String = ""       // destination for checkpoint backups
    var intervalMinutes: Int = 30        // LaunchAgent interval
    var syncOnCheckpoint: Bool = true    // auto-sync when new checkpoint detected
    var lastSyncDate: Date?
    var syncedItems: [SyncedItem] = []

    static var configPath: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("NeuralForge/sync_config.json")
    }
}

struct SyncedItem: Codable, Identifiable {
    var id: String { sourcePath }
    let sourcePath: String
    let destPath: String
    let projectName: String
    let syncDate: Date
    let sizeBytes: Int64
    let checkpointStep: Int?
}

// MARK: - Sync Status

enum SyncStatus: Equatable {
    case idle
    case syncing(progress: String)
    case success(count: Int, date: Date)
    case error(String)

    var description: String {
        switch self {
        case .idle: return "Not synced"
        case .syncing(let msg): return msg
        case .success(let count, _): return "\(count) item(s) synced"
        case .error(let msg): return "Error: \(msg)"
        }
    }
}

// MARK: - Shared Registry Entry

struct SharedModelEntry: Identifiable {
    let id = UUID()
    let name: String
    let path: String
    let sizeBytes: Int64
    let modifiedDate: Date
    let projectName: String
    let isCheckpoint: Bool
    let isModel: Bool
}

// MARK: - Sync Service

@MainActor
class SyncService: ObservableObject {
    @Published var config = SyncConfig()
    @Published var status: SyncStatus = .idle
    @Published var sharedModels: [SharedModelEntry] = []
    @Published var pendingSync: [String] = []   // paths needing sync
    @Published var isAgentInstalled = false

    static let shared = SyncService()
    static let agentLabel = "com.neuralforge.checkpointsync"

    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = .prettyPrinted
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    private let fm = FileManager.default

    init() {
        loadConfig()
        checkAgentStatus()
        if config.syncEnabled && !config.syncDirectory.isEmpty {
            scanSharedDirectory()
        }
    }

    // MARK: - Configuration

    func loadConfig() {
        let path = SyncConfig.configPath
        guard fm.fileExists(atPath: path.path),
              let data = try? Data(contentsOf: path),
              let cfg = try? decoder.decode(SyncConfig.self, from: data) else {
            return
        }
        config = cfg
    }

    func saveConfig() {
        let dir = SyncConfig.configPath.deletingLastPathComponent()
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        if let data = try? encoder.encode(config) {
            try? data.write(to: SyncConfig.configPath, options: .atomic)
        }
    }

    func setSyncDirectory(_ path: String) {
        config.syncDirectory = path
        config.syncEnabled = !path.isEmpty
        saveConfig()
        if config.syncEnabled {
            // Create sync directory structure
            try? fm.createDirectory(atPath: path, withIntermediateDirectories: true)
            try? fm.createDirectory(
                atPath: (path as NSString).appendingPathComponent("checkpoints"),
                withIntermediateDirectories: true)
            try? fm.createDirectory(
                atPath: (path as NSString).appendingPathComponent("models"),
                withIntermediateDirectories: true)
            scanSharedDirectory()
        }
    }

    // MARK: - Sync Operations

    func syncAllProjects() {
        guard config.syncEnabled, !config.syncDirectory.isEmpty else {
            status = .error("Sync not configured")
            return
        }

        status = .syncing(progress: "Scanning projects...")

        // Find all project directories
        let projectsDir = NFProject.projectsDir
        guard let projectDirs = try? fm.contentsOfDirectory(
            at: projectsDir, includingPropertiesForKeys: [.isDirectoryKey]) else {
            status = .error("Cannot read projects directory")
            return
        }

        var synced = 0

        for dir in projectDirs {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else { continue }

            // Read project.json to get project name
            let projectFile = dir.appendingPathComponent("project.json")
            guard let data = try? Data(contentsOf: projectFile),
                  let project = try? decoder.decode(NFProject.self, from: data) else { continue }

            status = .syncing(progress: "Syncing \(project.name)...")

            // Sync checkpoint files
            let checkpointDir = dir.appendingPathComponent("checkpoints")
            if let files = try? fm.contentsOfDirectory(at: checkpointDir, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey]) {
                for file in files where file.pathExtension == "bin" {
                    if syncFile(file, projectName: project.name, category: "checkpoints") {
                        synced += 1
                    }
                }
            }

            // Sync model files in project dir
            if let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.fileSizeKey]) {
                for file in files where file.pathExtension == "bin" && file.lastPathComponent != "project.json" {
                    let name = file.lastPathComponent
                    if name.contains("stories") || name.contains("model") || name.contains("smollm") || name.contains("tinyllama") {
                        if syncFile(file, projectName: project.name, category: "models") {
                            synced += 1
                        }
                    }
                }
            }
        }

        config.lastSyncDate = Date()
        saveConfig()

        status = .success(count: synced, date: Date())
        scanSharedDirectory()
    }

    func syncCheckpoint(path: String, projectName: String) {
        guard config.syncEnabled, !config.syncDirectory.isEmpty else { return }

        let url = URL(fileURLWithPath: path)
        status = .syncing(progress: "Syncing checkpoint...")

        if syncFile(url, projectName: projectName, category: "checkpoints") {
            config.lastSyncDate = Date()
            saveConfig()
            status = .success(count: 1, date: Date())
        } else {
            status = .error("Failed to sync checkpoint")
        }

        scanSharedDirectory()
    }

    private func syncFile(_ source: URL, projectName: String, category: String) -> Bool {
        let destDir = (config.syncDirectory as NSString)
            .appendingPathComponent(category)
        let safeName = projectName.replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: " ", with: "_")

        // Create project subdirectory
        let projectDir = (destDir as NSString).appendingPathComponent(safeName)
        try? fm.createDirectory(atPath: projectDir, withIntermediateDirectories: true)

        let destPath = (projectDir as NSString).appendingPathComponent(source.lastPathComponent)
        let destURL = URL(fileURLWithPath: destPath)

        // Check if already synced and up-to-date
        if fm.fileExists(atPath: destPath) {
            let sourceAttrs = try? fm.attributesOfItem(atPath: source.path)
            let destAttrs = try? fm.attributesOfItem(atPath: destPath)
            let sourceDate = sourceAttrs?[.modificationDate] as? Date ?? Date.distantPast
            let destDate = destAttrs?[.modificationDate] as? Date ?? Date.distantPast
            if destDate >= sourceDate { return false }  // already up-to-date
        }

        // Copy file
        do {
            if fm.fileExists(atPath: destPath) {
                try fm.removeItem(at: destURL)
            }
            try fm.copyItem(at: source, to: destURL)

            // Record synced item
            let attrs = try? fm.attributesOfItem(atPath: source.path)
            let size = attrs?[.size] as? Int64 ?? 0

            // Try to extract step number from filename (e.g., checkpoint_step100.bin)
            var step: Int? = nil
            let name = source.lastPathComponent
            if let range = name.range(of: "step(\\d+)", options: .regularExpression) {
                let numStr = name[range].dropFirst(4)
                step = Int(numStr)
            }

            let item = SyncedItem(
                sourcePath: source.path,
                destPath: destPath,
                projectName: projectName,
                syncDate: Date(),
                sizeBytes: size,
                checkpointStep: step
            )
            config.syncedItems.append(item)

            return true
        } catch {
            return false
        }
    }

    // MARK: - Shared Directory Scanning

    func scanSharedDirectory() {
        guard !config.syncDirectory.isEmpty else {
            sharedModels = []
            return
        }

        var entries = [SharedModelEntry]()

        // Scan checkpoints
        scanCategory("checkpoints", isCheckpoint: true, isModel: false, into: &entries)
        // Scan models
        scanCategory("models", isCheckpoint: false, isModel: true, into: &entries)

        sharedModels = entries.sorted { $0.modifiedDate > $1.modifiedDate }
    }

    private func scanCategory(_ category: String, isCheckpoint: Bool, isModel: Bool,
                              into entries: inout [SharedModelEntry]) {
        let categoryDir = (config.syncDirectory as NSString).appendingPathComponent(category)
        guard let projectDirs = try? fm.contentsOfDirectory(atPath: categoryDir) else { return }

        for projectDir in projectDirs {
            let fullDir = (categoryDir as NSString).appendingPathComponent(projectDir)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: fullDir, isDirectory: &isDir), isDir.boolValue else { continue }

            guard let files = try? fm.contentsOfDirectory(atPath: fullDir) else { continue }
            for file in files {
                let filePath = (fullDir as NSString).appendingPathComponent(file)
                let attrs = try? fm.attributesOfItem(atPath: filePath)
                let size = attrs?[.size] as? Int64 ?? 0
                let date = attrs?[.modificationDate] as? Date ?? Date()

                entries.append(SharedModelEntry(
                    name: file,
                    path: filePath,
                    sizeBytes: size,
                    modifiedDate: date,
                    projectName: projectDir.replacingOccurrences(of: "_", with: " "),
                    isCheckpoint: isCheckpoint,
                    isModel: isModel
                ))
            }
        }
    }

    // MARK: - Pending Sync Detection

    func scanForPendingSync() {
        pendingSync = []
        let projectsDir = NFProject.projectsDir

        guard let projectDirs = try? fm.contentsOfDirectory(
            at: projectsDir, includingPropertiesForKeys: nil) else { return }

        for dir in projectDirs {
            let checkpointDir = dir.appendingPathComponent("checkpoints")
            guard let files = try? fm.contentsOfDirectory(
                at: checkpointDir, includingPropertiesForKeys: [.contentModificationDateKey]) else { continue }

            for file in files where file.pathExtension == "bin" {
                // Check if already in sync directory
                let synced = config.syncedItems.contains { $0.sourcePath == file.path }
                if !synced { pendingSync.append(file.path) }
            }
        }
    }

    // MARK: - LaunchAgent Management

    static var agentPlistPath: String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return "\(home)/Library/LaunchAgents/\(agentLabel).plist"
    }

    func checkAgentStatus() {
        isAgentInstalled = fm.fileExists(atPath: SyncService.agentPlistPath)
    }

    func generateAgentPlist() -> String {
        let syncDir = config.syncDirectory
        let interval = config.intervalMinutes * 60

        // The LaunchAgent runs a simple rsync-based sync script
        let logDir = fm.homeDirectoryForCurrentUser.path + "/Library/Logs/NeuralForge"
        let projectsDir = NFProject.projectsDir.path

        return """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>\(SyncService.agentLabel)</string>
            <key>ProgramArguments</key>
            <array>
                <string>/usr/bin/rsync</string>
                <string>-av</string>
                <string>--update</string>
                <string>--include=*.bin</string>
                <string>--include=*/</string>
                <string>--exclude=*</string>
                <string>\(projectsDir)/</string>
                <string>\(syncDir)/checkpoints/</string>
            </array>
            <key>StartInterval</key>
            <integer>\(interval)</integer>
            <key>RunAtLoad</key>
            <false/>
            <key>StandardOutPath</key>
            <string>\(logDir)/sync_stdout.log</string>
            <key>StandardErrorPath</key>
            <string>\(logDir)/sync_stderr.log</string>
        </dict>
        </plist>
        """
    }

    func installAgent() -> Bool {
        let plist = generateAgentPlist()
        let path = SyncService.agentPlistPath

        // Ensure LaunchAgents directory exists
        let agentDir = (path as NSString).deletingLastPathComponent
        try? fm.createDirectory(atPath: agentDir, withIntermediateDirectories: true)

        // Write plist
        guard (try? plist.write(toFile: path, atomically: true, encoding: .utf8)) != nil else {
            return false
        }

        // Load the agent
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        task.arguments = ["load", path]
        try? task.run()
        task.waitUntilExit()

        checkAgentStatus()
        return task.terminationStatus == 0
    }

    func uninstallAgent() -> Bool {
        let path = SyncService.agentPlistPath

        // Unload the agent
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        task.arguments = ["unload", path]
        try? task.run()
        task.waitUntilExit()

        // Remove plist
        try? fm.removeItem(atPath: path)

        checkAgentStatus()
        return true
    }

    // MARK: - Restore from Shared

    func restoreCheckpoint(_ entry: SharedModelEntry, to project: inout NFProject) -> Bool {
        let destDir = project.directory.appendingPathComponent("checkpoints")
        try? fm.createDirectory(at: destDir, withIntermediateDirectories: true)

        let dest = destDir.appendingPathComponent(entry.name)

        do {
            if fm.fileExists(atPath: dest.path) {
                try fm.removeItem(at: dest)
            }
            try fm.copyItem(atPath: entry.path, toPath: dest.path)
            project.checkpointPath = dest.path
            return true
        } catch {
            return false
        }
    }

    // MARK: - Helpers

    static func formatSize(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
