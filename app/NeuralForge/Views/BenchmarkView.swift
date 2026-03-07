// BenchmarkView.swift — Model evaluation benchmarks with perplexity scoring

import SwiftUI
import Charts

struct BenchmarkView: View {
    @Binding var project: NFProject
    @EnvironmentObject var cliRunner: CLIRunner
    @ObservedObject private var benchmarks = BenchmarkService.shared
    @State private var evalDataPath = ""
    @State private var showDetail: BenchmarkResult?

    private var projectResults: [BenchmarkResult] {
        benchmarks.results(for: project.id)
    }

    private var stats: BenchmarkStats {
        BenchmarkService.computeStats(results: projectResults)
    }

    var body: some View {
        VStack(spacing: 0) {
            headerSection
            Divider()

            if projectResults.isEmpty && !benchmarks.isEvaluating {
                emptyState
            } else {
                ScrollView {
                    VStack(spacing: 16) {
                        // Regression alerts
                        if !benchmarks.regressionAlerts.isEmpty {
                            alertsSection
                        }

                        // Stats bar
                        if !projectResults.isEmpty {
                            statsSection
                        }

                        // Perplexity chart
                        if projectResults.count >= 2 {
                            chartSection
                        }

                        // Evaluation controls
                        evalSection

                        // Results table
                        if !projectResults.isEmpty {
                            resultsSection
                        }
                    }
                    .padding()
                }
            }
        }
        .sheet(item: $showDetail) { result in
            BenchmarkDetailSheet(result: result)
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        HStack {
            Image(systemName: "gauge.with.dots.needle.33percent")
                .foregroundColor(.blue)
            Text("Model Benchmarks")
                .font(.headline)
            if !projectResults.isEmpty {
                Text("(\(projectResults.count) evaluations)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            if benchmarks.isEvaluating {
                ProgressView()
                    .controlSize(.small)
                    .padding(.trailing, 4)
                Text(benchmarks.evaluationProgress)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Button(action: exportCSV) {
                Image(systemName: "square.and.arrow.up")
            }
            .help("Export as CSV")
            .disabled(projectResults.isEmpty)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    // MARK: - Alerts

    private var alertsSection: some View {
        VStack(spacing: 8) {
            ForEach(benchmarks.regressionAlerts) { alert in
                HStack {
                    Image(systemName: alert.severity == .critical ? "exclamationmark.triangle.fill" : "exclamationmark.circle.fill")
                        .foregroundColor(alert.severity == .critical ? .red : .orange)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Quality \(alert.severity.rawValue)")
                            .font(.caption.bold())
                        Text(alert.message)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    Button(action: { benchmarks.dismissAlert(id: alert.id) }) {
                        Image(systemName: "xmark")
                            .font(.caption)
                    }
                    .buttonStyle(.borderless)
                }
                .padding(10)
                .background(RoundedRectangle(cornerRadius: 8)
                    .fill(alert.severity == .critical ? Color.red.opacity(0.1) : Color.orange.opacity(0.1)))
            }
        }
    }

    // MARK: - Stats

    private var statsSection: some View {
        HStack(spacing: 20) {
            statBadge("Best PPL", String(format: "%.2f", stats.bestPerplexity), color: .green)
            statBadge("Avg PPL", String(format: "%.2f", stats.avgPerplexity), color: .blue)
            statBadge("Worst PPL", String(format: "%.2f", stats.worstPerplexity), color: .orange)
            statBadge("Evaluations", "\(stats.count)", color: .secondary)

            HStack(spacing: 4) {
                Image(systemName: stats.trend.icon)
                Text(stats.trend.rawValue)
            }
            .font(.caption.bold())
            .foregroundColor(trendColor(stats.trend))
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(RoundedRectangle(cornerRadius: 6)
                .fill(trendColor(stats.trend).opacity(0.1)))
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(RoundedRectangle(cornerRadius: 10)
            .fill(Color(nsColor: .controlBackgroundColor)))
    }

    private func statBadge(_ label: String, _ value: String, color: Color) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.title3.bold().monospaced())
                .foregroundColor(color)
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }

    private func trendColor(_ trend: BenchmarkTrend) -> Color {
        switch trend {
        case .improving: return .green
        case .stable: return .blue
        case .degrading: return .red
        }
    }

    // MARK: - Chart

    private var chartSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Perplexity Over Checkpoints")
                .font(.subheadline.bold())

            Chart {
                ForEach(projectResults, id: \.id) { result in
                    LineMark(
                        x: .value("Step", result.checkpointStep),
                        y: .value("Perplexity", result.perplexity)
                    )
                    .foregroundStyle(.blue)
                    .symbol(.circle)

                    PointMark(
                        x: .value("Step", result.checkpointStep),
                        y: .value("Perplexity", result.perplexity)
                    )
                    .foregroundStyle(.blue)
                }
            }
            .frame(height: 180)
            .chartYScale(domain: .automatic(includesZero: false))
            .chartYAxisLabel("Perplexity")
            .chartXAxisLabel("Checkpoint Step")
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10)
            .fill(Color(nsColor: .controlBackgroundColor)))
    }

    // MARK: - Evaluation Controls

    private var evalSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Run Evaluation")
                .font(.subheadline.bold())

            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Evaluation Data")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    HStack {
                        TextField("Path to evaluation data...", text: $evalDataPath)
                            .textFieldStyle(.roundedBorder)
                        Button("Browse") {
                            let panel = NSOpenPanel()
                            panel.canChooseFiles = true
                            panel.canChooseDirectories = false
                            panel.allowedContentTypes = [.data]
                            if panel.runModal() == .OK, let url = panel.url {
                                evalDataPath = url.path
                            }
                        }
                    }
                }
            }

            HStack {
                Button(action: runEvaluation) {
                    HStack {
                        Image(systemName: "play.fill")
                        Text("Evaluate Checkpoint")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(benchmarks.isEvaluating || project.checkpointPath.isEmpty)

                if project.checkpointPath.isEmpty {
                    Text("No checkpoint available — train first")
                        .font(.caption)
                        .foregroundColor(.orange)
                }

                Spacer()

                if !benchmarks.evaluationProgress.isEmpty {
                    Text(benchmarks.evaluationProgress)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10)
            .fill(Color(nsColor: .controlBackgroundColor)))
    }

    // MARK: - Results Table

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Evaluation Results")
                .font(.subheadline.bold())

            // Table header
            HStack(spacing: 0) {
                Text("Label").frame(width: 100, alignment: .leading)
                Text("Step").frame(width: 70, alignment: .trailing)
                Text("Perplexity").frame(width: 90, alignment: .trailing)
                Text("Avg Loss").frame(width: 90, alignment: .trailing)
                Text("Tokens").frame(width: 80, alignment: .trailing)
                Text("Tokens/s").frame(width: 80, alignment: .trailing)
                Text("Eval Time").frame(width: 80, alignment: .trailing)
                Text("Date").frame(width: 100, alignment: .trailing)
                Spacer()
            }
            .font(.caption.bold())
            .foregroundColor(.secondary)
            .padding(.horizontal, 8)

            Divider()

            ForEach(projectResults) { result in
                let isBest = result.id == benchmarks.bestResult(for: project.id)?.id
                HStack(spacing: 0) {
                    HStack(spacing: 4) {
                        if isBest {
                            Image(systemName: "star.fill")
                                .foregroundColor(.yellow)
                                .font(.caption2)
                        }
                        Text(result.label)
                            .lineLimit(1)
                    }
                    .frame(width: 100, alignment: .leading)

                    Text("\(result.checkpointStep)")
                        .frame(width: 70, alignment: .trailing)

                    Text(result.perplexityFormatted)
                        .foregroundColor(isBest ? .green : .primary)
                        .frame(width: 90, alignment: .trailing)

                    Text(result.lossFormatted)
                        .frame(width: 90, alignment: .trailing)

                    Text("\(result.tokenCount)")
                        .frame(width: 80, alignment: .trailing)

                    Text(String(format: "%.0f", result.tokensPerSecond))
                        .frame(width: 80, alignment: .trailing)

                    Text(formatTime(result.evalTimeSeconds))
                        .frame(width: 80, alignment: .trailing)

                    Text(result.evaluatedAt, format: .dateTime.month(.abbreviated).day())
                        .frame(width: 100, alignment: .trailing)

                    Spacer()
                }
                .font(.caption.monospaced())
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .onTapGesture { showDetail = result }
                .contextMenu {
                    Button("View Details") { showDetail = result }
                    Divider()
                    Button("Delete", role: .destructive) {
                        benchmarks.deleteResult(id: result.id)
                    }
                }

                Divider()
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10)
            .fill(Color(nsColor: .controlBackgroundColor)))
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "gauge.with.dots.needle.33percent")
                .font(.system(size: 40))
                .foregroundColor(.secondary)
            Text("No Benchmarks Yet")
                .font(.title3.bold())
            Text("Evaluate checkpoints to measure perplexity,\ncompare quality across training steps,\nand detect regressions automatically.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Actions

    private func runEvaluation() {
        let dataPath = evalDataPath.isEmpty ? project.dataPath : evalDataPath
        guard !dataPath.isEmpty else { return }

        let ckptDir = project.directory.appendingPathComponent("checkpoints")
        let ckptPath = ckptDir.appendingPathComponent("checkpoint.bin").path

        benchmarks.evaluatePerplexity(
            projectID: project.id,
            checkpointPath: ckptPath,
            checkpointStep: cliRunner.state.currentStep,
            evalDataPath: dataPath,
            modelParams: cliRunner.state.modelParams,
            loraRank: project.config.loraRank,
            cliRunner: cliRunner
        ) { _ in }
    }

    private func exportCSV() {
        let csv = benchmarks.exportCSV(for: project.id)
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.commaSeparatedText]
        panel.nameFieldStringValue = "\(project.name)_benchmarks.csv"
        if panel.runModal() == .OK, let url = panel.url {
            try? csv.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    private func formatTime(_ seconds: Double) -> String {
        if seconds < 60 { return String(format: "%.0fs", seconds) }
        return String(format: "%.0fm", seconds / 60)
    }
}

// MARK: - Detail Sheet

struct BenchmarkDetailSheet: View {
    let result: BenchmarkResult
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading) {
                    Text("Benchmark Detail")
                        .font(.title3.bold())
                    Text(result.label)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                Button("Done") { dismiss() }
            }
            .padding()

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    detailSection("Perplexity Metrics") {
                        detailRow("Perplexity", result.perplexityFormatted)
                        detailRow("Average Loss", result.lossFormatted)
                        detailRow("Tokens Evaluated", "\(result.tokenCount)")
                        detailRow("Tokens/Second", String(format: "%.0f", result.tokensPerSecond))
                        detailRow("Evaluation Time", formatTime(result.evalTimeSeconds))
                    }

                    detailSection("Checkpoint Info") {
                        detailRow("Checkpoint Step", "\(result.checkpointStep)")
                        detailRow("Checkpoint Path", result.checkpointPath)
                        detailRow("Eval Data", result.evalDataPath)
                        detailRow("Model Parameters", formatParams(result.modelParams))
                        if result.loraRank > 0 {
                            detailRow("LoRA Rank", "\(result.loraRank)")
                        }
                    }

                    detailSection("Metadata") {
                        detailRow("Evaluated At",
                                  result.evaluatedAt.formatted(.dateTime.year().month().day().hour().minute()))
                        detailRow("ID", result.id.uuidString)
                    }
                }
                .padding()
            }
        }
        .frame(width: 450, height: 450)
    }

    private func detailSection<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.subheadline.bold())
            content()
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
    }

    private func detailRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).font(.caption).foregroundColor(.secondary)
            Spacer()
            Text(value).font(.caption.monospaced()).lineLimit(1).truncationMode(.middle)
        }
    }

    private func formatParams(_ p: Int) -> String {
        if p >= 1_000_000_000 { return String(format: "%.1fB", Double(p) / 1e9) }
        if p >= 1_000_000 { return String(format: "%.0fM", Double(p) / 1e6) }
        if p >= 1_000 { return String(format: "%.0fK", Double(p) / 1e3) }
        return "\(p)"
    }

    private func formatTime(_ seconds: Double) -> String {
        if seconds < 60 { return String(format: "%.1fs", seconds) }
        return String(format: "%.0fm %.0fs", seconds / 60, seconds.truncatingRemainder(dividingBy: 60))
    }
}
