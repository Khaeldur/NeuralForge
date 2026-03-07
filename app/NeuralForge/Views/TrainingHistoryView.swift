// TrainingHistoryView.swift — Browse, search, and compare past training runs

import SwiftUI
import Charts

struct TrainingHistoryView: View {
    @Binding var project: NFProject
    @ObservedObject private var history = TrainingHistoryService.shared
    @State private var searchText = ""
    @State private var selectedRuns: Set<UUID> = []
    @State private var showComparison = false
    @State private var showDetail: TrainingRun?
    @State private var sortOrder: SortOrder = .newest

    enum SortOrder: String, CaseIterable {
        case newest = "Newest First"
        case oldest = "Oldest First"
        case bestLoss = "Best Loss"
        case longest = "Longest"
    }

    private var projectRuns: [TrainingRun] {
        var runs = history.runs(for: project.id)
        if !searchText.isEmpty {
            let q = searchText.lowercased()
            runs = runs.filter {
                $0.notes.lowercased().contains(q)
                || $0.modelPath.lowercased().contains(q)
                || String(format: "%.4f", $0.bestLoss).contains(q)
            }
        }
        switch sortOrder {
        case .newest: runs.sort { $0.completedAt > $1.completedAt }
        case .oldest: runs.sort { $0.completedAt < $1.completedAt }
        case .bestLoss: runs.sort { $0.bestLoss < $1.bestLoss }
        case .longest: runs.sort { $0.totalTimeSeconds > $1.totalTimeSeconds }
        }
        return runs
    }

    var body: some View {
        VStack(spacing: 0) {
            headerBar
            Divider()

            if projectRuns.isEmpty && searchText.isEmpty {
                emptyState
            } else if projectRuns.isEmpty {
                noResultsState
            } else {
                runsList
            }
        }
        .sheet(item: $showDetail) { run in
            RunDetailSheet(run: run)
        }
        .sheet(isPresented: $showComparison) {
            let compared = history.runs.filter { selectedRuns.contains($0.id) }
            ComparisonSheet(runs: compared)
        }
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack {
            Image(systemName: "clock.arrow.circlepath")
                .foregroundColor(.blue)
            Text("Training History")
                .font(.headline)
            Text("(\(projectRuns.count) runs)")
                .font(.caption)
                .foregroundColor(.secondary)

            Spacer()

            Picker("Sort", selection: $sortOrder) {
                ForEach(SortOrder.allCases, id: \.self) { order in
                    Text(order.rawValue).tag(order)
                }
            }
            .frame(width: 160)

            if selectedRuns.count >= 2 {
                Button("Compare (\(selectedRuns.count))") { showComparison = true }
                    .buttonStyle(.borderedProminent)
            }

            Button(action: exportCSV) {
                Image(systemName: "square.and.arrow.up")
            }
            .help("Export as CSV")
            .disabled(projectRuns.isEmpty)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    // MARK: - Runs List

    private var runsList: some View {
        VStack(spacing: 0) {
            // Search
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                TextField("Search runs...", text: $searchText)
                    .textFieldStyle(.plain)
                if !searchText.isEmpty {
                    Button(action: { searchText = "" }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.borderless)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 6)
            .background(Color(nsColor: .controlBackgroundColor))

            Divider()

            // Table header
            HStack(spacing: 0) {
                Text("").frame(width: 30)
                Text("Date").frame(width: 100, alignment: .leading)
                Text("Steps").frame(width: 70, alignment: .trailing)
                Text("Best Loss").frame(width: 90, alignment: .trailing)
                Text("Final Loss").frame(width: 90, alignment: .trailing)
                Text("Duration").frame(width: 80, alignment: .trailing)
                Text("LR").frame(width: 80, alignment: .trailing)
                Text("LoRA").frame(width: 50, alignment: .center)
                Text("TFLOPS").frame(width: 70, alignment: .trailing)
                Spacer()
            }
            .font(.caption.bold())
            .foregroundColor(.secondary)
            .padding(.horizontal)
            .padding(.vertical, 4)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))

            Divider()

            // Rows
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(projectRuns) { run in
                        RunRow(run: run, isSelected: selectedRuns.contains(run.id),
                               isBest: run.id == history.bestRun(for: project.id)?.id,
                               onToggleSelect: {
                                   if selectedRuns.contains(run.id) {
                                       selectedRuns.remove(run.id)
                                   } else {
                                       selectedRuns.insert(run.id)
                                   }
                               })
                            .onTapGesture { showDetail = run }
                            .contextMenu {
                                Button("View Details") { showDetail = run }
                                Divider()
                                Button("Delete Run", role: .destructive) {
                                    history.deleteRun(id: run.id)
                                    selectedRuns.remove(run.id)
                                }
                            }
                        Divider()
                    }
                }
            }
        }
    }

    // MARK: - Run Row

    private struct RunRow: View {
        let run: TrainingRun
        let isSelected: Bool
        let isBest: Bool
        var onToggleSelect: () -> Void = {}

        var body: some View {
            HStack(spacing: 0) {
                // Selection checkbox + best indicator
                ZStack {
                    Button(action: onToggleSelect) {
                        Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                            .foregroundColor(isSelected ? .accentColor : .secondary.opacity(0.5))
                            .font(.caption)
                    }
                    .buttonStyle(.borderless)

                    if isBest {
                        Image(systemName: "trophy.fill")
                            .foregroundColor(.yellow)
                            .font(.system(size: 6))
                            .offset(x: 8, y: -6)
                    }
                }
                .frame(width: 30)

                // Date
                Text(run.completedAt, format: .dateTime.month(.abbreviated).day().hour().minute())
                    .font(.caption.monospaced())
                    .frame(width: 100, alignment: .leading)

                // Steps
                Text("\(run.finalStep)/\(run.totalSteps)")
                    .font(.caption.monospaced())
                    .frame(width: 70, alignment: .trailing)
                    .foregroundColor(run.finalStep >= run.totalSteps ? .primary : .orange)

                // Best Loss
                Text(String(format: "%.4f", run.bestLoss))
                    .font(.caption.monospaced())
                    .frame(width: 90, alignment: .trailing)
                    .foregroundColor(isBest ? .green : .primary)

                // Final Loss
                Text(String(format: "%.4f", run.finalLoss))
                    .font(.caption.monospaced())
                    .frame(width: 90, alignment: .trailing)

                // Duration
                Text(run.durationFormatted)
                    .font(.caption.monospaced())
                    .frame(width: 80, alignment: .trailing)

                // LR
                Text(String(format: "%.1e", run.config.learningRate))
                    .font(.caption.monospaced())
                    .frame(width: 80, alignment: .trailing)

                // LoRA
                if run.config.loraRank > 0 {
                    Text("r\(run.config.loraRank)")
                        .font(.caption2.bold())
                        .foregroundColor(.purple)
                        .frame(width: 50, alignment: .center)
                } else {
                    Text("Full")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .frame(width: 50, alignment: .center)
                }

                // TFLOPS
                Text(String(format: "%.1f", run.avgTFLOPS))
                    .font(.caption.monospaced())
                    .frame(width: 70, alignment: .trailing)

                Spacer()
            }
            .padding(.horizontal)
            .padding(.vertical, 6)
            .background(isSelected ? Color.accentColor.opacity(0.1) : Color.clear)
        }
    }

    // MARK: - Empty States

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "clock.arrow.circlepath")
                .font(.system(size: 40))
                .foregroundColor(.secondary)
            Text("No Training History")
                .font(.title3.bold())
            Text("Completed training runs will appear here automatically.\nYou can compare runs, track improvements, and export results.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private var noResultsState: some View {
        VStack(spacing: 12) {
            Spacer()
            Image(systemName: "magnifyingglass")
                .font(.system(size: 30))
                .foregroundColor(.secondary)
            Text("No matching runs")
                .font(.subheadline)
                .foregroundColor(.secondary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Export

    private func exportCSV() {
        let projectRunsToExport = history.runs(for: project.id)
        var csv = "Run ID,Project,Model,Params,Final Loss,Best Loss,Steps,Duration (s),LR,LoRA Rank,Schedule,Started,Completed,Notes\n"
        for run in projectRunsToExport {
            let notes = run.notes.replacingOccurrences(of: "\"", with: "\"\"")
            csv += "\(run.id),\"\(run.projectName)\",\"\(run.modelPath)\",\(run.modelParams),"
            csv += String(format: "%.6f,%.6f,", run.finalLoss, run.bestLoss)
            csv += "\(run.finalStep),\(String(format: "%.1f", run.totalTimeSeconds)),"
            csv += "\(run.config.learningRate),\(run.config.loraRank),\(run.config.lrSchedule),"
            csv += "\(ISO8601DateFormatter().string(from: run.startedAt)),"
            csv += "\(ISO8601DateFormatter().string(from: run.completedAt)),"
            csv += "\"\(notes)\"\n"
        }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.commaSeparatedText]
        panel.nameFieldStringValue = "\(project.name)_training_history.csv"
        if panel.runModal() == .OK, let url = panel.url {
            try? csv.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}

// MARK: - Run Detail Sheet

struct RunDetailSheet: View {
    let run: TrainingRun
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading) {
                    Text("Training Run Details")
                        .font(.title3.bold())
                    Text(run.completedAt, format: .dateTime.year().month().day().hour().minute())
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                Button("Done") { dismiss() }
            }
            .padding()

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Results
                    infoSection("Results") {
                        infoRow("Final Loss", String(format: "%.6f", run.finalLoss))
                        infoRow("Best Loss", String(format: "%.6f", run.bestLoss))
                        infoRow("Steps", "\(run.finalStep) / \(run.totalSteps)")
                        infoRow("Duration", run.durationFormatted)
                        infoRow("Avg ms/step", String(format: "%.1f", run.avgMsPerStep))
                        infoRow("Avg TFLOPS", String(format: "%.1f", run.avgTFLOPS))
                    }

                    // Loss chart
                    if !run.lossHistory.isEmpty {
                        infoSection("Loss Curve") {
                            Chart {
                                ForEach(run.lossHistory, id: \.step) { pt in
                                    LineMark(x: .value("Step", pt.step), y: .value("Loss", pt.loss))
                                        .foregroundStyle(.blue.opacity(0.5))
                                }
                                ForEach(run.valLossHistory, id: \.step) { pt in
                                    LineMark(x: .value("Step", pt.step), y: .value("Val Loss", pt.loss))
                                        .foregroundStyle(.orange)
                                        .lineStyle(StrokeStyle(dash: [4, 3]))
                                }
                            }
                            .frame(height: 150)
                            .chartYScale(domain: .automatic(includesZero: false))
                        }
                    }

                    // Config
                    infoSection("Configuration") {
                        infoRow("Learning Rate", String(format: "%.1e", run.config.learningRate))
                        infoRow("Accumulation Steps", "\(run.config.accumSteps)")
                        infoRow("Checkpoint Every", "\(run.config.checkpointEvery)")
                        infoRow("Grad Clip Norm", String(format: "%.1f", run.config.gradClipNorm))
                        infoRow("Seed", "\(run.config.seed)")
                        if run.config.lrSchedule != "none" {
                            infoRow("LR Schedule", run.config.lrSchedule)
                            infoRow("Warmup Steps", "\(run.config.warmupSteps)")
                        }
                        if run.config.loraRank > 0 {
                            infoRow("LoRA Rank", "\(run.config.loraRank)")
                            infoRow("LoRA Alpha", String(format: "%.1f", run.config.loraAlpha))
                        }
                    }

                    // Model
                    infoSection("Model") {
                        infoRow("Path", run.modelPath)
                        infoRow("Parameters", formatParams(run.modelParams))
                        infoRow("Layers", "\(run.modelLayers)")
                        infoRow("Dimension", "\(run.modelDim)")
                    }

                    // Notes
                    infoSection("Notes") {
                        Text(run.notes.isEmpty ? "No notes" : run.notes)
                            .font(.caption)
                            .foregroundColor(run.notes.isEmpty ? .secondary : .primary)
                    }
                }
                .padding()
            }
        }
        .frame(width: 500, height: 600)
    }

    private func infoSection<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.subheadline.bold())
            content()
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
    }

    private func infoRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.caption.monospaced())
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    private func formatParams(_ p: Int) -> String {
        if p >= 1_000_000_000 { return String(format: "%.1fB", Double(p) / 1e9) }
        if p >= 1_000_000 { return String(format: "%.0fM", Double(p) / 1e6) }
        if p >= 1_000 { return String(format: "%.0fK", Double(p) / 1e3) }
        return "\(p)"
    }
}

// MARK: - Comparison Sheet

struct ComparisonSheet: View {
    let runs: [TrainingRun]
    @Environment(\.dismiss) private var dismiss

    private let colors: [Color] = [.blue, .orange, .green, .purple, .red, .cyan]

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Run Comparison (\(runs.count) runs)")
                    .font(.title3.bold())
                Spacer()
                Button("Done") { dismiss() }
            }
            .padding()

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Loss overlay chart
                    if runs.contains(where: { !$0.lossHistory.isEmpty }) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Loss Curves").font(.subheadline.bold())
                            Chart {
                                ForEach(Array(runs.enumerated()), id: \.element.id) { idx, run in
                                    let color = colors[idx % colors.count]
                                    ForEach(run.lossHistory, id: \.step) { pt in
                                        LineMark(
                                            x: .value("Step", pt.step),
                                            y: .value("Loss", pt.loss),
                                            series: .value("Run", run.projectName + " " + run.completedAt.formatted(.dateTime.month(.abbreviated).day()))
                                        )
                                        .foregroundStyle(color)
                                    }
                                }
                            }
                            .frame(height: 200)
                            .chartYScale(domain: .automatic(includesZero: false))
                        }
                        .padding()
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
                    }

                    // Comparison table
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Side-by-Side").font(.subheadline.bold())

                        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                            // Header
                            GridRow {
                                Text("Metric").font(.caption.bold()).foregroundColor(.secondary)
                                ForEach(Array(runs.enumerated()), id: \.element.id) { idx, run in
                                    VStack {
                                        Circle().fill(colors[idx % colors.count]).frame(width: 8, height: 8)
                                        Text(run.completedAt, format: .dateTime.month(.abbreviated).day())
                                            .font(.caption2)
                                    }
                                }
                            }
                            Divider()

                            comparisonRow("Best Loss", runs.map { String(format: "%.4f", $0.bestLoss) })
                            comparisonRow("Final Loss", runs.map { String(format: "%.4f", $0.finalLoss) })
                            comparisonRow("Steps", runs.map { "\($0.finalStep)/\($0.totalSteps)" })
                            comparisonRow("Duration", runs.map { $0.durationFormatted })
                            comparisonRow("LR", runs.map { String(format: "%.1e", $0.config.learningRate) })
                            comparisonRow("LoRA Rank", runs.map { $0.config.loraRank > 0 ? "r\($0.config.loraRank)" : "Full" })
                            comparisonRow("Schedule", runs.map { $0.config.lrSchedule })
                            comparisonRow("Shuffle", runs.map { $0.config.shuffle ? "Yes" : "No" })
                            comparisonRow("TFLOPS", runs.map { String(format: "%.1f", $0.avgTFLOPS) })
                            comparisonRow("ms/step", runs.map { String(format: "%.1f", $0.avgMsPerStep) })
                        }
                    }
                    .padding()
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
                }
                .padding()
            }
        }
        .frame(width: 700, height: 550)
    }

    private func comparisonRow(_ label: String, _ values: [String]) -> some View {
        GridRow {
            Text(label).font(.caption).foregroundColor(.secondary)
            ForEach(Array(values.enumerated()), id: \.offset) { _, val in
                Text(val).font(.caption.monospaced())
            }
        }
    }
}
