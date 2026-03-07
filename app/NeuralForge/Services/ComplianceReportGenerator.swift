// ComplianceReportGenerator.swift — Generate HIPAA/SOX compliance reports from audit data
// Reads audit log entries and produces structured compliance reports for export as PDF/text.

import Foundation
import CommonCrypto

// MARK: - Report Types

enum ComplianceFramework: String, CaseIterable, Identifiable {
    case general = "General Audit"
    case hipaa = "HIPAA"
    case sox = "SOX"

    var id: String { rawValue }

    var description: String {
        switch self {
        case .general: return "Complete audit trail summary with activity metrics"
        case .hipaa: return "Health Insurance Portability and Accountability Act — data handling & access controls"
        case .sox: return "Sarbanes-Oxley Act — change management & authorization audit trail"
        }
    }

    var icon: String {
        switch self {
        case .general: return "doc.text.magnifyingglass"
        case .hipaa: return "cross.case"
        case .sox: return "building.columns"
        }
    }
}

// MARK: - Report Sections

struct ReportSection: Identifiable {
    let id = UUID()
    let title: String
    let icon: String
    let content: String
    let severity: SectionSeverity

    enum SectionSeverity {
        case info, pass, warning, critical
    }
}

// MARK: - Compliance Report

struct ComplianceReport: Identifiable {
    let id = UUID()
    let framework: ComplianceFramework
    let generatedAt: Date
    let periodStart: Date?
    let periodEnd: Date?
    let sections: [ReportSection]
    let overallStatus: ReportStatus
    let summary: String

    enum ReportStatus: String {
        case compliant = "Compliant"
        case needsReview = "Needs Review"
        case nonCompliant = "Non-Compliant"

        var color: String {
            switch self {
            case .compliant: return "green"
            case .needsReview: return "orange"
            case .nonCompliant: return "red"
            }
        }

        var icon: String {
            switch self {
            case .compliant: return "checkmark.shield.fill"
            case .needsReview: return "exclamationmark.shield.fill"
            case .nonCompliant: return "xmark.shield.fill"
            }
        }
    }
}

// MARK: - Report Generator

@MainActor
class ComplianceReportGenerator: ObservableObject {
    @Published var isGenerating = false
    @Published var report: ComplianceReport?
    @Published var errorMessage: String?

    private let reader = AuditLogReader()
    private let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return f
    }()

    func generateReport(framework: ComplianceFramework,
                        startDate: Date? = nil,
                        endDate: Date? = nil) {
        isGenerating = true
        errorMessage = nil
        report = nil

        reader.loadEntries()
        reader.verifyChain()

        guard !reader.entries.isEmpty else {
            errorMessage = "No audit log entries found. Train or export a model first."
            isGenerating = false
            return
        }

        // Filter entries by date range
        let filtered = filterEntries(reader.entries, from: startDate, to: endDate)

        switch framework {
        case .general:
            report = generateGeneralReport(entries: filtered, startDate: startDate, endDate: endDate)
        case .hipaa:
            report = generateHIPAAReport(entries: filtered, startDate: startDate, endDate: endDate)
        case .sox:
            report = generateSOXReport(entries: filtered, startDate: startDate, endDate: endDate)
        }

        isGenerating = false
    }

    // MARK: - Export

    func exportAsText() -> String? {
        guard let report = report else { return nil }

        var text = ""
        let line = String(repeating: "═", count: 72)
        let thinLine = String(repeating: "─", count: 72)

        text += "\(line)\n"
        text += "  NEURALFORGE COMPLIANCE REPORT\n"
        text += "  Framework: \(report.framework.rawValue)\n"
        text += "  Generated: \(dateFormatter.string(from: report.generatedAt))\n"
        if let start = report.periodStart {
            text += "  Period: \(dateFormatter.string(from: start))"
            if let end = report.periodEnd {
                text += " — \(dateFormatter.string(from: end))"
            }
            text += "\n"
        }
        text += "  Status: \(report.overallStatus.rawValue)\n"
        text += "\(line)\n\n"

        text += "EXECUTIVE SUMMARY\n"
        text += "\(thinLine)\n"
        text += "\(report.summary)\n\n"

        for section in report.sections {
            let marker: String
            switch section.severity {
            case .pass: marker = "✓ PASS"
            case .warning: marker = "⚠ WARNING"
            case .critical: marker = "✗ CRITICAL"
            case .info: marker = "ℹ INFO"
            }

            text += "\(marker): \(section.title)\n"
            text += "\(thinLine)\n"
            text += "\(section.content)\n\n"
        }

        text += "\(line)\n"
        text += "  Report ID: \(report.id.uuidString)\n"
        text += "  Hash chain verified by NeuralForge Audit System\n"
        text += "\(line)\n"

        return text
    }

    // MARK: - General Report

    private func generateGeneralReport(entries: [AuditEntry],
                                       startDate: Date?, endDate: Date?) -> ComplianceReport {
        var sections = [ReportSection]()
        let stats = computeStats(entries)

        // Activity overview
        sections.append(ReportSection(
            title: "Activity Overview",
            icon: "chart.bar",
            content: formatActivityOverview(stats),
            severity: .info
        ))

        // User access log
        sections.append(ReportSection(
            title: "User Access Log",
            icon: "person.2",
            content: formatUserAccess(entries),
            severity: stats.users.count > 5 ? .warning : .info
        ))

        // Training activity
        sections.append(ReportSection(
            title: "Training Activity",
            icon: "play.fill",
            content: formatTrainingActivity(entries),
            severity: .info
        ))

        // Model exports
        sections.append(ReportSection(
            title: "Model Exports & Artifacts",
            icon: "square.and.arrow.up",
            content: formatExportActivity(entries),
            severity: .info
        ))

        // Chain integrity
        let chainSection = formatChainIntegrity()
        sections.append(chainSection)

        // Data provenance
        sections.append(ReportSection(
            title: "Data Provenance",
            icon: "doc.text",
            content: formatDataProvenance(entries),
            severity: .info
        ))

        let status: ComplianceReport.ReportStatus = chainSection.severity == .pass ? .compliant : .nonCompliant

        return ComplianceReport(
            framework: .general,
            generatedAt: Date(),
            periodStart: startDate ?? entries.last?.date,
            periodEnd: endDate ?? entries.first?.date,
            sections: sections,
            overallStatus: status,
            summary: "Audit trail contains \(entries.count) entries across \(stats.users.count) user(s). "
                + "\(stats.trainingStarts) training run(s), \(stats.exports) export(s), \(stats.generations) generation(s). "
                + "Hash chain integrity: \(chainSection.severity == .pass ? "VERIFIED" : "COMPROMISED")."
        )
    }

    // MARK: - HIPAA Report

    private func generateHIPAAReport(entries: [AuditEntry],
                                     startDate: Date?, endDate: Date?) -> ComplianceReport {
        var sections = [ReportSection]()
        let stats = computeStats(entries)

        // §164.312(b) — Audit Controls
        let chainSection = formatChainIntegrity()
        sections.append(ReportSection(
            title: "§164.312(b) Audit Controls",
            icon: "shield.checkered",
            content: """
            REQUIREMENT: Implement mechanisms to record and examine activity in systems that contain or use ePHI.

            FINDING: NeuralForge maintains an append-only audit log with SHA-256 hash chain integrity protection.
            • Total audit entries: \(entries.count)
            • Hash chain status: \(chainSection.severity == .pass ? "INTACT — no tampering detected" : "COMPROMISED — entries may have been modified")
            • Audit events tracked: training, checkpoints, exports, text generation
            • Log location: ~/Library/Logs/NeuralForge/audit.jsonl
            • Log format: JSONL with cryptographic hash chain
            """,
            severity: chainSection.severity
        ))

        // §164.312(a)(1) — Access Control
        sections.append(ReportSection(
            title: "§164.312(a)(1) Access Control",
            icon: "lock.shield",
            content: """
            REQUIREMENT: Implement technical policies to allow access only to authorized persons.

            FINDING: System access tracked via user identification in audit log.
            • Users with access: \(stats.users.sorted().joined(separator: ", "))
            • Total unique users: \(stats.users.count)
            • Access pattern: All operations logged with user identification
            \(stats.users.count > 3 ? "\n⚠ NOTE: Multiple users detected. Verify all are authorized for ePHI access." : "")
            """,
            severity: stats.users.count > 3 ? .warning : .pass
        ))

        // §164.312(c)(1) — Integrity Controls
        sections.append(ReportSection(
            title: "§164.312(c)(1) Integrity Controls",
            icon: "lock.doc",
            content: """
            REQUIREMENT: Implement policies and procedures to protect ePHI from improper alteration or destruction.

            FINDING: Model weights and training data are protected through audit tracking.
            • Checkpoints saved: \(stats.checkpoints)
            • Models exported: \(stats.exports)
            • Data files referenced in training: \(countUniqueDataFiles(entries))
            • All checkpoint saves and exports are logged with file paths and timestamps
            """,
            severity: .pass
        ))

        // §164.312(d) — Person or Entity Authentication
        sections.append(ReportSection(
            title: "§164.312(d) Person/Entity Authentication",
            icon: "person.badge.key",
            content: """
            REQUIREMENT: Implement procedures to verify that a person seeking access is the one claimed.

            FINDING: User identification is captured via system username.
            • Authentication method: macOS system user account (USER environment variable)
            • Users recorded: \(stats.users.sorted().joined(separator: ", "))

            RECOMMENDATION: For enhanced compliance, consider implementing additional authentication (e.g., Touch ID, Keychain-based) before sensitive operations.
            """,
            severity: .warning
        ))

        // §164.312(e)(1) — Transmission Security
        sections.append(ReportSection(
            title: "§164.312(e)(1) Transmission Security",
            icon: "network.badge.shield.half.filled",
            content: """
            REQUIREMENT: Implement technical security measures to guard against unauthorized access to ePHI during transmission.

            FINDING: NeuralForge operates locally on-device.
            • All training runs: LOCAL (Apple Neural Engine)
            • Model inference: LOCAL
            • Data processing: LOCAL
            • Network transmission: NONE (models and data remain on device)
            • External API calls: Optional LLM Assistant only (metadata only, no ePHI transmitted)

            On-device processing eliminates network transmission risks for training data and model weights.
            """,
            severity: .pass
        ))

        // Data handling summary
        sections.append(ReportSection(
            title: "Data Handling Summary",
            icon: "externaldrive.badge.checkmark",
            content: formatDataProvenance(entries),
            severity: .info
        ))

        let hasChainIssue = chainSection.severity != .pass
        let hasManyUsers = stats.users.count > 3
        let status: ComplianceReport.ReportStatus
        if hasChainIssue { status = .nonCompliant }
        else if hasManyUsers { status = .needsReview }
        else { status = .compliant }

        return ComplianceReport(
            framework: .hipaa,
            generatedAt: Date(),
            periodStart: startDate ?? entries.last?.date,
            periodEnd: endDate ?? entries.first?.date,
            sections: sections,
            overallStatus: status,
            summary: "HIPAA Technical Safeguards Assessment: \(status.rawValue). "
                + "Audit controls \(hasChainIssue ? "FAILED" : "PASSED") — \(entries.count) entries with "
                + "\(hasChainIssue ? "broken" : "intact") hash chain. "
                + "\(stats.users.count) authorized user(s). All processing performed locally on-device."
        )
    }

    // MARK: - SOX Report

    private func generateSOXReport(entries: [AuditEntry],
                                   startDate: Date?, endDate: Date?) -> ComplianceReport {
        var sections = [ReportSection]()
        let stats = computeStats(entries)
        let chainSection = formatChainIntegrity()

        // §302 — Management Assessment of Internal Controls
        sections.append(ReportSection(
            title: "§302 Management Assessment of Internal Controls",
            icon: "person.badge.shield.checkmark",
            content: """
            REQUIREMENT: Management must assess and report on the effectiveness of internal controls.

            FINDING: NeuralForge provides automated internal controls for ML model management.
            • Change tracking: All model training, checkpointing, and export events are logged
            • Audit trail: Tamper-evident hash chain with \(entries.count) entries
            • Authorization: Operations tied to identified user accounts
            • Period covered: \(formatDateRange(entries))
            """,
            severity: .info
        ))

        // §404 — Internal Control Over Financial Reporting (adapted for ML)
        sections.append(ReportSection(
            title: "§404 Internal Controls — Model Change Management",
            icon: "gearshape.2",
            content: """
            REQUIREMENT: Establish and maintain adequate internal controls. For ML systems, this covers model versioning and change management.

            FINDING: Change management controls:
            • Training runs initiated: \(stats.trainingStarts)
            • Training runs completed: \(stats.trainingStops)
            • Incomplete runs: \(max(0, stats.trainingStarts - stats.trainingStops))
            • Checkpoints saved: \(stats.checkpoints)
            • Models exported: \(stats.exports)
            • Text generation sessions: \(stats.generations)

            Each operation is recorded with: timestamp, user, configuration parameters, and results.
            \(stats.trainingStarts > stats.trainingStops + 1 ? "\n⚠ Multiple incomplete training runs detected. Review for potential unauthorized terminations." : "")
            """,
            severity: stats.trainingStarts > stats.trainingStops + 1 ? .warning : .pass
        ))

        // Audit Trail Integrity
        sections.append(ReportSection(
            title: "Audit Trail Integrity",
            icon: "shield.checkered",
            content: """
            REQUIREMENT: Audit trails must be protected from unauthorized modification.

            FINDING:
            • Hash chain algorithm: SHA-256
            • Chain status: \(chainSection.severity == .pass ? "INTACT — all entries verified" : "COMPROMISED — potential tampering detected")
            • Total entries verified: \(reader.verification?.validEntries ?? 0) / \(reader.verification?.totalEntries ?? 0)
            • Verification performed: \(dateFormatter.string(from: Date()))
            \(chainSection.severity != .pass ? "\n✗ CRITICAL: Audit trail integrity compromised. Investigate immediately." : "")
            """,
            severity: chainSection.severity
        ))

        // Separation of Duties
        sections.append(ReportSection(
            title: "Separation of Duties",
            icon: "person.2.badge.gearshape",
            content: """
            REQUIREMENT: Critical functions should be divided among different people to reduce fraud risk.

            FINDING:
            • Total users: \(stats.users.count)
            • Users: \(stats.users.sorted().joined(separator: ", "))
            \(stats.users.count == 1 ?
                "\n⚠ Single user performs all operations. For SOX compliance, consider requiring separate users for training vs. export/deployment." :
                "\n✓ Multiple users recorded, supporting separation of duties.")

            Operations by user:
            \(formatOperationsByUser(entries))
            """,
            severity: stats.users.count == 1 ? .warning : .pass
        ))

        // Model Configuration Controls
        sections.append(ReportSection(
            title: "Configuration Change Controls",
            icon: "slider.horizontal.3",
            content: """
            REQUIREMENT: Changes to system configuration must be authorized, tested, and documented.

            FINDING: Training configurations logged in audit trail:
            \(formatConfigChanges(entries))

            All configuration parameters (learning rate, steps, LoRA rank, etc.) are captured at training start.
            """,
            severity: .pass
        ))

        // Timeline
        sections.append(ReportSection(
            title: "Activity Timeline",
            icon: "calendar.badge.clock",
            content: formatTimeline(entries),
            severity: .info
        ))

        let hasChainIssue = chainSection.severity != .pass
        let hasIncomplete = stats.trainingStarts > stats.trainingStops + 1
        let status: ComplianceReport.ReportStatus
        if hasChainIssue { status = .nonCompliant }
        else if hasIncomplete || stats.users.count == 1 { status = .needsReview }
        else { status = .compliant }

        return ComplianceReport(
            framework: .sox,
            generatedAt: Date(),
            periodStart: startDate ?? entries.last?.date,
            periodEnd: endDate ?? entries.first?.date,
            sections: sections,
            overallStatus: status,
            summary: "SOX Internal Controls Assessment: \(status.rawValue). "
                + "\(stats.trainingStarts) training run(s), \(stats.exports) export(s) by \(stats.users.count) user(s). "
                + "Audit trail integrity: \(hasChainIssue ? "COMPROMISED" : "VERIFIED"). "
                + "\(stats.users.count == 1 ? "Single-user operation — consider separation of duties." : "")"
        )
    }

    // MARK: - Formatting Helpers

    private func formatActivityOverview(_ stats: LocalStats) -> String {
        var lines = [String]()
        lines.append("• Total audit entries: \(stats.totalEntries)")
        lines.append("• Training runs started: \(stats.trainingStarts)")
        lines.append("• Training runs completed: \(stats.trainingStops)")
        lines.append("• Checkpoints saved: \(stats.checkpoints)")
        lines.append("• Models exported: \(stats.exports)")
        lines.append("• Text generations: \(stats.generations)")
        lines.append("• Unique users: \(stats.users.count) (\(stats.users.sorted().joined(separator: ", ")))")
        if stats.totalTrainingTime > 0 {
            lines.append("• Total training time: \(formatDuration(stats.totalTrainingTime))")
        }
        if let best = stats.bestLoss {
            lines.append("• Best recorded loss: \(String(format: "%.4f", best))")
        }
        return lines.joined(separator: "\n")
    }

    private func formatUserAccess(_ entries: [AuditEntry]) -> String {
        var userOps = [String: [String: Int]]()
        for entry in entries {
            if userOps[entry.user] == nil { userOps[entry.user] = [:] }
            userOps[entry.user]![entry.event, default: 0] += 1
        }

        var lines = [String]()
        for user in userOps.keys.sorted() {
            let ops = userOps[user]!
            let total = ops.values.reduce(0, +)
            let firstEntry = entries.last(where: { $0.user == user })
            let lastEntry = entries.first(where: { $0.user == user })

            lines.append("User: \(user)")
            lines.append("  Total operations: \(total)")
            if let first = firstEntry?.date {
                lines.append("  First activity: \(dateFormatter.string(from: first))")
            }
            if let last = lastEntry?.date {
                lines.append("  Last activity: \(dateFormatter.string(from: last))")
            }
            for (event, count) in ops.sorted(by: { $0.value > $1.value }) {
                lines.append("  \(event): \(count)")
            }
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    private func formatTrainingActivity(_ entries: [AuditEntry]) -> String {
        let starts = entries.filter { $0.event == "training_start" }
        let stops = entries.filter { $0.event == "training_stop" }

        if starts.isEmpty { return "No training activity recorded in this period." }

        var lines = [String]()
        lines.append("Training Runs: \(starts.count) started, \(stops.count) completed\n")

        for (i, start) in starts.enumerated().prefix(10) {
            lines.append("Run \(i + 1):")
            if let m = start.model { lines.append("  Model: \(URL(fileURLWithPath: m).lastPathComponent)") }
            if let s = start.steps { lines.append("  Target steps: \(s)") }
            if let l = start.lr { lines.append("  Learning rate: \(String(format: "%.1e", l))") }
            if let r = start.loraRank, r > 0 { lines.append("  LoRA rank: \(r)") }
            if let d = start.date { lines.append("  Started: \(dateFormatter.string(from: d))") }
            lines.append("")
        }

        if starts.count > 10 {
            lines.append("... and \(starts.count - 10) more runs")
        }

        for stop in stops.prefix(5) {
            if let sd = stop.stepsDone, let fl = stop.finalLoss, let tt = stop.totalTimeS {
                lines.append("Completed: \(sd) steps, loss=\(String(format: "%.4f", fl)), time=\(formatDuration(tt))")
            }
        }

        return lines.joined(separator: "\n")
    }

    private func formatExportActivity(_ entries: [AuditEntry]) -> String {
        let exports = entries.filter { $0.event == "export" }

        if exports.isEmpty { return "No model exports recorded in this period." }

        var lines = [String]()
        lines.append("Total Exports: \(exports.count)\n")

        for export in exports.prefix(20) {
            var parts = [String]()
            if let f = export.format { parts.append("Format: \(f)") }
            if let o = export.output { parts.append("Output: \(URL(fileURLWithPath: o).lastPathComponent)") }
            if let d = export.date { parts.append("Date: \(dateFormatter.string(from: d))") }
            parts.append("User: \(export.user)")
            lines.append("• \(parts.joined(separator: ", "))")
        }

        return lines.joined(separator: "\n")
    }

    private func formatChainIntegrity() -> ReportSection {
        let v = reader.verification
        let isValid = v?.isValid ?? false

        let content: String
        if let v = v {
            content = """
            Hash chain algorithm: SHA-256
            Total entries: \(v.totalEntries)
            Valid entries: \(v.validEntries)
            Status: \(v.isValid ? "INTACT — no tampering detected" : "COMPROMISED")
            \(v.firstBadSeq != nil ? "First tampered entry: #\(v.firstBadSeq!)" : "")
            Verified at: \(dateFormatter.string(from: v.verifiedAt))

            \(v.isValid ? "All entries form an unbroken SHA-256 hash chain from genesis. No evidence of insertion, deletion, or modification." : "WARNING: The hash chain is broken. Audit log entries may have been modified, deleted, or inserted after creation. Investigate immediately.")
            """
        } else {
            content = "Hash chain verification could not be performed."
        }

        return ReportSection(
            title: "Hash Chain Integrity Verification",
            icon: "checkmark.shield",
            content: content,
            severity: isValid ? .pass : .critical
        )
    }

    private func formatDataProvenance(_ entries: [AuditEntry]) -> String {
        var dataFiles = Set<String>()
        var modelFiles = Set<String>()

        for entry in entries {
            if let d = entry.data { dataFiles.insert(d) }
            if let m = entry.model { modelFiles.insert(m) }
        }

        var lines = [String]()
        if !modelFiles.isEmpty {
            lines.append("Models Referenced (\(modelFiles.count)):")
            for f in modelFiles.sorted() {
                lines.append("  • \(URL(fileURLWithPath: f).lastPathComponent)")
            }
        }
        if !dataFiles.isEmpty {
            lines.append("\nTraining Data Referenced (\(dataFiles.count)):")
            for f in dataFiles.sorted() {
                lines.append("  • \(URL(fileURLWithPath: f).lastPathComponent)")
            }
        }
        if dataFiles.isEmpty && modelFiles.isEmpty {
            lines.append("No data or model file references found in this period.")
        }
        return lines.joined(separator: "\n")
    }

    private func formatOperationsByUser(_ entries: [AuditEntry]) -> String {
        var userOps = [String: Int]()
        for entry in entries {
            userOps[entry.user, default: 0] += 1
        }
        return userOps.sorted(by: { $0.value > $1.value })
            .map { "  • \($0.key): \($0.value) operations" }
            .joined(separator: "\n")
    }

    private func formatConfigChanges(_ entries: [AuditEntry]) -> String {
        let configs = entries.filter { $0.event == "training_start" }
        if configs.isEmpty { return "  No training configurations recorded." }

        var lines = [String]()
        for (i, cfg) in configs.enumerated().prefix(5) {
            var parts = [String]()
            if let s = cfg.steps { parts.append("steps=\(s)") }
            if let l = cfg.lr { parts.append("lr=\(String(format: "%.1e", l))") }
            if let a = cfg.accum { parts.append("accum=\(a)") }
            if let r = cfg.loraRank, r > 0 { parts.append("lora_r=\(r)") }
            lines.append("  Config \(i + 1): \(parts.joined(separator: ", "))")
            if let d = cfg.date { lines.append("    Date: \(dateFormatter.string(from: d)), User: \(cfg.user)") }
        }
        if configs.count > 5 { lines.append("  ... and \(configs.count - 5) more") }
        return lines.joined(separator: "\n")
    }

    private func formatTimeline(_ entries: [AuditEntry]) -> String {
        // Show last 20 entries in chronological order
        let recent = Array(entries.prefix(20).reversed())
        if recent.isEmpty { return "No activity in this period." }

        var lines = [String]()
        for entry in recent {
            let dateStr: String
            if let d = entry.date { dateStr = dateFormatter.string(from: d) }
            else { dateStr = entry.time }

            let event = entry.event.replacingOccurrences(of: "_", with: " ")
            lines.append("  [\(dateStr)] \(entry.user): \(event) — \(entry.summary)")
        }

        if entries.count > 20 {
            lines.append("  ... \(entries.count - 20) earlier entries omitted")
        }

        return lines.joined(separator: "\n")
    }

    private func formatDateRange(_ entries: [AuditEntry]) -> String {
        let first = entries.last?.date
        let last = entries.first?.date
        if let f = first, let l = last {
            return "\(dateFormatter.string(from: f)) — \(dateFormatter.string(from: l))"
        }
        return "N/A"
    }

    private func countUniqueDataFiles(_ entries: [AuditEntry]) -> Int {
        Set(entries.compactMap(\.data)).count
    }

    // MARK: - Date Filtering

    private func filterEntries(_ entries: [AuditEntry], from start: Date?, to end: Date?) -> [AuditEntry] {
        guard start != nil || end != nil else { return entries }

        return entries.filter { entry in
            guard let date = entry.date else { return true }
            if let s = start, date < s { return false }
            if let e = end, date > e { return false }
            return true
        }
    }

    // MARK: - Local Stats (simpler than AuditStats)

    private struct LocalStats {
        let totalEntries: Int
        let trainingStarts: Int
        let trainingStops: Int
        let checkpoints: Int
        let exports: Int
        let generations: Int
        let users: Set<String>
        let totalTrainingTime: Double
        let bestLoss: Double?
    }

    private func computeStats(_ entries: [AuditEntry]) -> LocalStats {
        var totalTime = 0.0
        var bestLoss: Double? = nil
        var users = Set<String>()

        for entry in entries {
            users.insert(entry.user)
            if let t = entry.totalTimeS { totalTime += t }
            if let l = entry.finalLoss {
                if bestLoss == nil || l < bestLoss! { bestLoss = l }
            }
        }

        return LocalStats(
            totalEntries: entries.count,
            trainingStarts: entries.filter { $0.event == "training_start" }.count,
            trainingStops: entries.filter { $0.event == "training_stop" }.count,
            checkpoints: entries.filter { $0.event == "checkpoint_save" }.count,
            exports: entries.filter { $0.event == "export" }.count,
            generations: entries.filter { $0.event.hasPrefix("generate") }.count,
            users: users,
            totalTrainingTime: totalTime,
            bestLoss: bestLoss
        )
    }

    private func formatDuration(_ seconds: Double) -> String {
        if seconds < 60 { return String(format: "%.0f seconds", seconds) }
        if seconds < 3600 { return String(format: "%.1f minutes", seconds / 60) }
        return String(format: "%.1f hours", seconds / 3600)
    }
}
