// AuditLogReader.swift — Read and verify NeuralForge audit logs
// Parses ~/Library/Logs/NeuralForge/audit.jsonl JSONL entries
// Verifies SHA-256 hash chain integrity for tamper detection

import Foundation
import CommonCrypto

// MARK: - Audit Entry Model

struct AuditEntry: Identifiable, Codable {
    let id: UUID
    let seq: Int
    let time: String
    let event: String
    let user: String
    let prevHash: String
    let hash: String

    // Optional detail fields (vary by event type)
    let model: String?
    let data: String?
    let steps: Int?
    let lr: Double?
    let accum: Int?
    let loraRank: Int?
    let loraAlpha: Double?
    let stepsDone: Int?
    let finalLoss: Double?
    let totalTimeS: Double?
    let reason: String?
    let path: String?
    let step: Int?
    let loss: Double?
    let input: String?
    let output: String?
    let format: String?
    let maxTokens: Int?
    let temperature: Double?
    let topP: Double?
    let tokens: Int?
    let totalMs: Double?

    var date: Date? {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.date(from: time)
    }

    var eventIcon: String {
        switch event {
        case "training_start": return "play.fill"
        case "training_stop": return "stop.fill"
        case "checkpoint_save": return "externaldrive.fill"
        case "export": return "square.and.arrow.up"
        case "generate_start": return "text.bubble"
        case "generate_done": return "checkmark.bubble"
        case "config_used": return "slider.horizontal.3"
        case "ingest_start": return "tray.and.arrow.down"
        case "ingest_done": return "tray.and.arrow.down.fill"
        default: return "doc.text"
        }
    }

    var eventColor: String {
        switch event {
        case "training_start": return "green"
        case "training_stop": return "red"
        case "checkpoint_save": return "blue"
        case "export": return "purple"
        case "generate_start", "generate_done": return "orange"
        default: return "gray"
        }
    }

    var summary: String {
        switch event {
        case "training_start":
            var parts = [String]()
            if let m = model { parts.append(URL(fileURLWithPath: m).lastPathComponent) }
            if let s = steps { parts.append("\(s) steps") }
            if let l = lr { parts.append("lr=\(String(format: "%.1e", l))") }
            if let r = loraRank, r > 0 { parts.append("LoRA r=\(r)") }
            return parts.joined(separator: ", ")
        case "training_stop":
            var parts = [String]()
            if let s = stepsDone { parts.append("\(s) steps") }
            if let l = finalLoss { parts.append("loss=\(String(format: "%.4f", l))") }
            if let t = totalTimeS { parts.append("\(String(format: "%.1f", t))s") }
            return parts.joined(separator: ", ")
        case "checkpoint_save":
            var parts = [String]()
            if let s = step { parts.append("step \(s)") }
            if let l = loss { parts.append("loss=\(String(format: "%.4f", l))") }
            return parts.joined(separator: ", ")
        case "export":
            var parts = [String]()
            if let f = format { parts.append(f) }
            if let o = output { parts.append(URL(fileURLWithPath: o).lastPathComponent) }
            return parts.joined(separator: " → ")
        case "generate_start":
            var parts = [String]()
            if let t = maxTokens { parts.append("\(t) tokens") }
            if let temp = temperature { parts.append("t=\(String(format: "%.1f", temp))") }
            return parts.joined(separator: ", ")
        case "generate_done":
            var parts = [String]()
            if let t = tokens { parts.append("\(t) tokens") }
            if let ms = totalMs { parts.append("\(String(format: "%.0f", ms))ms") }
            return parts.joined(separator: ", ")
        default:
            return ""
        }
    }

    init(from json: [String: Any]) {
        self.id = UUID()
        self.seq = json["seq"] as? Int ?? 0
        self.time = json["time"] as? String ?? ""
        self.event = json["event"] as? String ?? "unknown"
        self.user = json["user"] as? String ?? ""
        self.prevHash = json["prev_hash"] as? String ?? ""
        self.hash = json["hash"] as? String ?? ""
        self.model = json["model"] as? String
        self.data = json["data"] as? String
        self.steps = json["steps"] as? Int
        self.lr = json["lr"] as? Double
        self.accum = json["accum"] as? Int
        self.loraRank = json["lora_rank"] as? Int
        self.loraAlpha = json["lora_alpha"] as? Double
        self.stepsDone = json["steps_done"] as? Int
        self.finalLoss = json["final_loss"] as? Double
        self.totalTimeS = json["total_time_s"] as? Double
        self.reason = json["reason"] as? String
        self.path = json["path"] as? String
        self.step = json["step"] as? Int
        self.loss = json["loss"] as? Double
        self.input = json["input"] as? String
        self.output = json["output"] as? String
        self.format = json["format"] as? String
        self.maxTokens = json["max_tokens"] as? Int
        self.temperature = json["temperature"] as? Double
        self.topP = json["top_p"] as? Double
        self.tokens = json["tokens"] as? Int
        self.totalMs = json["total_ms"] as? Double
    }
}

// MARK: - Chain Verification Result

struct AuditVerification {
    let totalEntries: Int
    let validEntries: Int
    let isValid: Bool
    let firstBadSeq: Int?       // seq of first broken entry
    let verifiedAt: Date

    var statusText: String {
        if isValid { return "Chain intact — \(totalEntries) entries verified" }
        if let bad = firstBadSeq {
            return "TAMPER DETECTED at entry #\(bad) — \(validEntries)/\(totalEntries) valid"
        }
        return "Verification failed"
    }
}

// MARK: - Audit Statistics

struct AuditStats {
    let totalEntries: Int
    let trainingStarts: Int
    let trainingStops: Int
    let checkpoints: Int
    let exports: Int
    let generations: Int
    let users: Set<String>
    let dateRange: (first: Date?, last: Date?)
    let totalTrainingTime: Double     // seconds
    let bestLoss: Double?
}

// MARK: - Audit Log Reader

@MainActor
class AuditLogReader: ObservableObject {
    @Published var entries: [AuditEntry] = []
    @Published var isLoading = false
    @Published var verification: AuditVerification?
    @Published var stats: AuditStats?
    @Published var errorMessage: String?

    static var logPath: String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return "\(home)/Library/Logs/NeuralForge/audit.jsonl"
    }

    func loadEntries() {
        isLoading = true
        errorMessage = nil
        entries = []

        let path = AuditLogReader.logPath

        guard FileManager.default.fileExists(atPath: path) else {
            errorMessage = "No audit log found at \(path)"
            isLoading = false
            return
        }

        guard let content = try? String(contentsOfFile: path, encoding: .utf8) else {
            errorMessage = "Cannot read audit log"
            isLoading = false
            return
        }

        var parsed = [AuditEntry]()
        let lines = content.components(separatedBy: .newlines)

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }

            guard let data = trimmed.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                continue
            }

            parsed.append(AuditEntry(from: json))
        }

        entries = parsed.reversed()  // Most recent first
        computeStats()
        isLoading = false
    }

    func verifyChain() {
        let path = AuditLogReader.logPath

        guard FileManager.default.fileExists(atPath: path),
              let content = try? String(contentsOfFile: path, encoding: .utf8) else {
            verification = AuditVerification(
                totalEntries: 0, validEntries: 0, isValid: false,
                firstBadSeq: nil, verifiedAt: Date())
            return
        }

        let lines = content.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        let zeroHash = String(repeating: "0", count: 64)
        var prevHash = zeroHash
        var validCount = 0
        var firstBad: Int? = nil

        for line in lines {
            guard let data = line.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let seq = json["seq"] as? Int,
                  let timestamp = json["time"] as? String,
                  let event = json["event"] as? String,
                  let user = json["user"] as? String,
                  let entryPrevHash = json["prev_hash"] as? String,
                  let entryHash = json["hash"] as? String else {
                continue
            }

            // Check chain link
            if entryPrevHash != prevHash {
                firstBad = seq
                break
            }

            // Extract details (everything between user and prev_hash in the raw JSON)
            let details = extractDetails(from: line)

            // Reconstruct content string for hash verification
            let content: String
            if let d = details, !d.isEmpty {
                content = "\(entryPrevHash)|\(seq)|\(timestamp)|\(event)|\(user)|\(d)"
            } else {
                content = "\(entryPrevHash)|\(seq)|\(timestamp)|\(event)|\(user)"
            }

            let computedHash = sha256Hex(content)
            if computedHash != entryHash {
                firstBad = seq
                break
            }

            prevHash = entryHash
            validCount += 1
        }

        verification = AuditVerification(
            totalEntries: lines.count,
            validEntries: validCount,
            isValid: firstBad == nil && validCount == lines.count,
            firstBadSeq: firstBad,
            verifiedAt: Date()
        )
    }

    // MARK: - Filtering

    func filteredEntries(eventFilter: String?, userFilter: String?,
                         searchText: String) -> [AuditEntry] {
        var result = entries

        if let event = eventFilter, !event.isEmpty {
            result = result.filter { $0.event == event }
        }
        if let user = userFilter, !user.isEmpty {
            result = result.filter { $0.user == user }
        }
        if !searchText.isEmpty {
            let query = searchText.lowercased()
            result = result.filter {
                $0.event.lowercased().contains(query)
                || $0.user.lowercased().contains(query)
                || $0.summary.lowercased().contains(query)
                || ($0.model?.lowercased().contains(query) ?? false)
                || ($0.path?.lowercased().contains(query) ?? false)
            }
        }

        return result
    }

    var uniqueEvents: [String] {
        Array(Set(entries.map(\.event))).sorted()
    }

    var uniqueUsers: [String] {
        Array(Set(entries.map(\.user))).sorted()
    }

    // MARK: - CSV Export

    func exportCSV() -> String {
        var csv = "seq,time,event,user,summary,hash\n"
        for entry in entries.reversed() {  // chronological for export
            let summary = entry.summary.replacingOccurrences(of: ",", with: ";")
            csv += "\(entry.seq),\(entry.time),\(entry.event),\(entry.user),\"\(summary)\",\(entry.hash.prefix(16))...\n"
        }
        return csv
    }

    // MARK: - Private Helpers

    private func computeStats() {
        let chronological = entries.reversed()
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]

        var totalTrainingTime = 0.0
        var bestLoss: Double? = nil
        var users = Set<String>()
        var firstDate: Date? = nil
        var lastDate: Date? = nil

        for entry in chronological {
            users.insert(entry.user)
            if let d = formatter.date(from: entry.time) {
                if firstDate == nil { firstDate = d }
                lastDate = d
            }
            if let t = entry.totalTimeS { totalTrainingTime += t }
            if let l = entry.finalLoss {
                if bestLoss == nil || l < bestLoss! { bestLoss = l }
            }
            if let l = entry.loss {
                if bestLoss == nil || l < bestLoss! { bestLoss = l }
            }
        }

        stats = AuditStats(
            totalEntries: entries.count,
            trainingStarts: entries.filter { $0.event == "training_start" }.count,
            trainingStops: entries.filter { $0.event == "training_stop" }.count,
            checkpoints: entries.filter { $0.event == "checkpoint_save" }.count,
            exports: entries.filter { $0.event == "export" }.count,
            generations: entries.filter { $0.event.hasPrefix("generate") }.count,
            users: users,
            dateRange: (firstDate, lastDate),
            totalTrainingTime: totalTrainingTime,
            bestLoss: bestLoss
        )
    }

    private func extractDetails(from line: String) -> String? {
        // Find content between "user":"..." and ,"prev_hash"
        guard let userRange = line.range(of: "\"user\":\"") else { return nil }
        let afterUser = line[userRange.upperBound...]
        guard let userEnd = afterUser.firstIndex(of: "\"") else { return nil }
        let afterUserClose = line[line.index(after: userEnd)...]

        guard afterUserClose.hasPrefix(",") else { return nil }
        let afterComma = afterUserClose.dropFirst()

        guard let prevHashRange = afterComma.range(of: "\"prev_hash\":") else { return nil }
        let details = afterComma[afterComma.startIndex..<prevHashRange.lowerBound]

        var result = String(details).trimmingCharacters(in: .whitespacesAndNewlines)
        // Remove trailing comma
        while result.hasSuffix(",") { result = String(result.dropLast()) }
        return result.isEmpty ? nil : result
    }

    private func sha256Hex(_ string: String) -> String {
        guard let data = string.data(using: .utf8) else { return "" }
        var hash = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        data.withUnsafeBytes {
            _ = CC_SHA256($0.baseAddress, CC_LONG(data.count), &hash)
        }
        return hash.map { String(format: "%02x", $0) }.joined()
    }
}
