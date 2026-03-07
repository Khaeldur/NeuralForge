// AuditDashboardView.swift — Compliance audit log viewer with hash chain verification
import SwiftUI

struct AuditDashboardView: View {
    @Binding var project: NFProject
    @StateObject private var reader = AuditLogReader()
    @State private var eventFilter: String?
    @State private var searchText = ""
    @State private var showVerification = false
    @State private var showExport = false
    @State private var selectedEntry: AuditEntry?

    var body: some View {
        VStack(spacing: 0) {
            // Header with stats
            headerSection

            Divider()

            if reader.isLoading {
                Spacer()
                ProgressView("Loading audit log...")
                Spacer()
            } else if reader.entries.isEmpty {
                emptyState
            } else {
                // Filters
                filterBar

                Divider()

                // Entry list
                entryList
            }

            if let error = reader.errorMessage {
                Label(error, systemImage: "exclamationmark.triangle")
                    .foregroundColor(.red)
                    .font(.caption)
                    .padding()
            }
        }
        .onAppear { reader.loadEntries() }
        .sheet(isPresented: $showVerification) { verificationSheet }
        .sheet(item: $selectedEntry) { entry in entryDetailSheet(entry) }
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(spacing: 12) {
            HStack {
                Label("Audit Dashboard", systemImage: "shield.checkered")
                    .font(.headline)
                Spacer()
                Button(action: { reader.loadEntries() }) {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .controlSize(.small)
                Button(action: {
                    reader.verifyChain()
                    showVerification = true
                }) {
                    Label("Verify Chain", systemImage: "checkmark.shield")
                }
                .controlSize(.small)
                .buttonStyle(.borderedProminent)
                Menu {
                    Button("Export CSV...") { exportCSV() }
                } label: {
                    Label("Export", systemImage: "square.and.arrow.up")
                }
                .controlSize(.small)
            }

            if let stats = reader.stats {
                statsBar(stats)
            }
        }
        .padding()
    }

    private func statsBar(_ stats: AuditStats) -> some View {
        HStack(spacing: 16) {
            statPill("Entries", "\(stats.totalEntries)", "doc.text")
            statPill("Trainings", "\(stats.trainingStarts)", "play.fill")
            statPill("Checkpoints", "\(stats.checkpoints)", "externaldrive.fill")
            statPill("Exports", "\(stats.exports)", "square.and.arrow.up")
            statPill("Generations", "\(stats.generations)", "text.bubble")

            if stats.totalTrainingTime > 0 {
                statPill("Train Time", formatDuration(stats.totalTrainingTime), "clock")
            }
            if let best = stats.bestLoss {
                statPill("Best Loss", String(format: "%.4f", best), "chart.line.downtrend.xyaxis")
            }

            Spacer()

            if stats.users.count > 0 {
                HStack(spacing: 4) {
                    Image(systemName: "person.2")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(stats.users.joined(separator: ", "))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
    }

    private func statPill(_ label: String, _ value: String, _ icon: String) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption2)
                .foregroundColor(.blue)
            VStack(alignment: .leading, spacing: 0) {
                Text(value)
                    .font(.caption.bold().monospacedDigit())
                Text(label)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }

    // MARK: - Filter Bar

    private var filterBar: some View {
        HStack(spacing: 12) {
            Image(systemName: "line.3.horizontal.decrease")
                .foregroundColor(.secondary)

            Picker("Event", selection: Binding(
                get: { eventFilter ?? "" },
                set: { eventFilter = $0.isEmpty ? nil : $0 }
            )) {
                Text("All Events").tag("")
                ForEach(reader.uniqueEvents, id: \.self) { event in
                    Text(event.replacingOccurrences(of: "_", with: " ").capitalized)
                        .tag(event)
                }
            }
            .frame(width: 180)

            TextField("Search...", text: $searchText)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 200)

            Spacer()

            Text("\(filteredEntries.count) entries")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    private var filteredEntries: [AuditEntry] {
        reader.filteredEntries(eventFilter: eventFilter, userFilter: nil, searchText: searchText)
    }

    // MARK: - Entry List

    private var entryList: some View {
        ScrollView {
            LazyVStack(spacing: 1) {
                ForEach(filteredEntries) { entry in
                    entryRow(entry)
                        .onTapGesture { selectedEntry = entry }
                }
            }
        }
    }

    private func entryRow(_ entry: AuditEntry) -> some View {
        HStack(spacing: 10) {
            // Event icon
            Image(systemName: entry.eventIcon)
                .frame(width: 20)
                .foregroundColor(colorFor(entry.eventColor))

            // Seq number
            Text("#\(entry.seq)")
                .font(.caption.monospaced())
                .foregroundColor(.secondary)
                .frame(width: 40, alignment: .trailing)

            // Event type
            Text(entry.event.replacingOccurrences(of: "_", with: " "))
                .font(.caption.bold())
                .frame(width: 120, alignment: .leading)

            // Summary
            Text(entry.summary)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(1)

            Spacer()

            // User
            Text(entry.user)
                .font(.caption2)
                .foregroundColor(.secondary)

            // Timestamp
            if let date = entry.date {
                Text(date, style: .date)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text(date, style: .time)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            // Hash (truncated)
            Text(String(entry.hash.prefix(8)))
                .font(.caption2.monospaced())
                .foregroundColor(.secondary.opacity(0.6))
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(Color(nsColor: .controlBackgroundColor))
        .contentShape(Rectangle())
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer()
            Image(systemName: "shield.checkered")
                .font(.system(size: 40))
                .foregroundColor(.secondary)
            Text("No audit entries found")
                .foregroundColor(.secondary)
            Text("Train, generate, or export a model to see audit entries here.")
                .font(.caption)
                .foregroundColor(.secondary)
            Button("Reload") { reader.loadEntries() }
                .buttonStyle(.bordered)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Verification Sheet

    private var verificationSheet: some View {
        VStack(spacing: 20) {
            if let v = reader.verification {
                Image(systemName: v.isValid ? "checkmark.shield.fill" : "xmark.shield.fill")
                    .font(.system(size: 48))
                    .foregroundColor(v.isValid ? .green : .red)

                Text(v.isValid ? "Hash Chain Verified" : "Chain Integrity Compromised")
                    .font(.title2.bold())

                Text(v.statusText)
                    .font(.body)
                    .multilineTextAlignment(.center)

                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Total Entries:")
                        Spacer()
                        Text("\(v.totalEntries)")
                            .monospacedDigit()
                    }
                    HStack {
                        Text("Valid Entries:")
                        Spacer()
                        Text("\(v.validEntries)")
                            .monospacedDigit()
                            .foregroundColor(v.isValid ? .green : .red)
                    }
                    if let bad = v.firstBadSeq {
                        HStack {
                            Text("First Tampered:")
                            Spacer()
                            Text("Entry #\(bad)")
                                .foregroundColor(.red)
                                .bold()
                        }
                    }
                    HStack {
                        Text("Verified At:")
                        Spacer()
                        Text(v.verifiedAt, style: .date)
                        Text(v.verifiedAt, style: .time)
                    }
                }
                .padding()
                .background(RoundedRectangle(cornerRadius: 8)
                    .fill(Color(nsColor: .controlBackgroundColor)))

                if v.isValid {
                    Text("All entries form an unbroken SHA-256 hash chain. No tampering detected.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                } else {
                    Text("The hash chain is broken. This may indicate that log entries were modified, deleted, or inserted after creation.")
                        .font(.caption)
                        .foregroundColor(.red)
                        .multilineTextAlignment(.center)
                }
            } else {
                ProgressView("Verifying hash chain...")
            }

            Button("Done") { showVerification = false }
                .buttonStyle(.borderedProminent)
        }
        .padding(30)
        .frame(width: 400, height: 400)
    }

    // MARK: - Entry Detail Sheet

    private func entryDetailSheet(_ entry: AuditEntry) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: entry.eventIcon)
                    .foregroundColor(colorFor(entry.eventColor))
                Text(entry.event.replacingOccurrences(of: "_", with: " ").capitalized)
                    .font(.headline)
                Spacer()
                Text("#\(entry.seq)")
                    .font(.caption.monospaced())
                    .foregroundColor(.secondary)
            }

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    detailRow("Time", entry.time)
                    detailRow("User", entry.user)

                    if let m = entry.model { detailRow("Model", m) }
                    if let d = entry.data { detailRow("Data", d) }
                    if let s = entry.steps { detailRow("Steps", "\(s)") }
                    if let l = entry.lr { detailRow("Learning Rate", String(format: "%.6e", l)) }
                    if let a = entry.accum { detailRow("Accumulation", "\(a)") }
                    if let r = entry.loraRank, r > 0 { detailRow("LoRA Rank", "\(r)") }
                    if let sd = entry.stepsDone { detailRow("Steps Done", "\(sd)") }
                    if let fl = entry.finalLoss { detailRow("Final Loss", String(format: "%.6f", fl)) }
                    if let tt = entry.totalTimeS { detailRow("Total Time", "\(String(format: "%.2f", tt))s") }
                    if let r = entry.reason { detailRow("Reason", r) }
                    if let p = entry.path { detailRow("Path", p) }
                    if let s = entry.step { detailRow("Step", "\(s)") }
                    if let l = entry.loss { detailRow("Loss", String(format: "%.6f", l)) }
                    if let i = entry.input { detailRow("Input", i) }
                    if let o = entry.output { detailRow("Output", o) }
                    if let f = entry.format { detailRow("Format", f) }
                    if let t = entry.maxTokens { detailRow("Max Tokens", "\(t)") }
                    if let t = entry.temperature { detailRow("Temperature", String(format: "%.2f", t)) }
                    if let t = entry.tokens { detailRow("Tokens", "\(t)") }
                    if let ms = entry.totalMs { detailRow("Time (ms)", String(format: "%.1f", ms)) }

                    Divider()

                    Text("Hash Chain")
                        .font(.subheadline.bold())
                    detailRow("Prev Hash", String(entry.prevHash.prefix(32)) + "...")
                    detailRow("Entry Hash", String(entry.hash.prefix(32)) + "...")
                }
            }

            HStack {
                Spacer()
                Button("Done") { selectedEntry = nil }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding()
        .frame(width: 500, height: 450)
    }

    private func detailRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .top) {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(width: 100, alignment: .trailing)
            Text(value)
                .font(.caption.monospaced())
                .textSelection(.enabled)
        }
    }

    // MARK: - Helpers

    private func colorFor(_ name: String) -> Color {
        switch name {
        case "green": return .green
        case "red": return .red
        case "blue": return .blue
        case "purple": return .purple
        case "orange": return .orange
        default: return .gray
        }
    }

    private func formatDuration(_ seconds: Double) -> String {
        if seconds < 60 { return String(format: "%.0fs", seconds) }
        if seconds < 3600 { return String(format: "%.0fm", seconds / 60) }
        return String(format: "%.1fh", seconds / 3600)
    }

    private func exportCSV() {
        let csv = reader.exportCSV()
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.commaSeparatedText]
        panel.nameFieldStringValue = "neuralforge_audit.csv"
        if panel.runModal() == .OK, let url = panel.url {
            try? csv.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
