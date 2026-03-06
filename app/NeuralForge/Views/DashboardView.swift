// DashboardView.swift — Live training dashboard with loss curve

import SwiftUI
import Charts

struct DashboardView: View {
    @Binding var project: NFProject
    @EnvironmentObject var cliRunner: CLIRunner

    var state: TrainingState { cliRunner.state }

    var body: some View {
        VStack(spacing: 16) {
            // Controls bar
            HStack {
                statusBadge
                Spacer()
                controlButtons
            }

            // Metrics cards
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: 4), spacing: 12) {
                MetricCard(title: "Step", value: "\(state.currentStep)/\(state.totalSteps)",
                           icon: "number", color: .blue)
                MetricCard(title: "Loss", value: String(format: "%.4f", state.currentLoss),
                           icon: "chart.line.downtrend.xyaxis", color: .orange)
                MetricCard(title: "Best Loss", value: state.bestLoss < .infinity ? String(format: "%.4f", state.bestLoss) : "--",
                           icon: "star", color: .yellow)
                MetricCard(title: "ANE TFLOPS", value: String(format: "%.2f", state.tflopsANE),
                           icon: "cpu", color: .green)
            }

            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: 4), spacing: 12) {
                MetricCard(title: "ms/step", value: String(format: "%.1f", state.msPerStep),
                           icon: "timer", color: .purple)
                MetricCard(title: "ETA", value: state.eta,
                           icon: "clock", color: .cyan)
                MetricCard(title: "Restarts", value: "\(state.restartCount)",
                           icon: "arrow.counterclockwise", color: .indigo)
                MetricCard(title: "Compiles", value: "\(state.compileCount)",
                           icon: "hammer", color: .mint)
            }

            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: 4), spacing: 12) {
                MetricCard(title: "Learning Rate", value: String(format: "%.1e", state.learningRate),
                           icon: "dial.low", color: .teal)
                MetricCard(title: "Total TFLOPS", value: String(format: "%.2f", state.tflopsTotal),
                           icon: "bolt", color: .pink)
                MetricCard(title: "Wall Time", value: formatTime(state.totalTimeSeconds),
                           icon: "stopwatch", color: .brown)
                MetricCard(title: "Checkpoint", value: checkpointLabel,
                           icon: "externaldrive", color: .gray)
            }

            // Model info bar
            if state.modelParams > 0 {
                HStack(spacing: 16) {
                    Label("\(String(format: "%.0fM", Double(state.modelParams) / 1e6)) params", systemImage: "brain")
                    Label("\(state.modelLayers) layers", systemImage: "square.stack.3d.up")
                    Label("dim \(state.modelDim)", systemImage: "ruler")
                    Label("\(state.modelVocab / 1000)K vocab", systemImage: "textformat.abc")
                    Spacer()
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 4)
            }

            // Progress bar
            if state.phase == .training || state.phase == .compiling {
                ProgressView(value: state.progress) {
                    Text("\(Int(state.progress * 100))%")
                        .font(.caption.monospacedDigit())
                }
                .tint(.blue)
            }

            // Loss curve
            GroupBox("Loss Curve") {
                if state.lossHistory.isEmpty {
                    ContentUnavailableView("No training data yet", systemImage: "chart.line.uptrend.xyaxis")
                        .frame(height: 250)
                } else {
                    Chart {
                        ForEach(Array(state.lossHistory.enumerated()), id: \.offset) { _, point in
                            LineMark(
                                x: .value("Step", point.step),
                                y: .value("Loss", point.loss)
                            )
                            .foregroundStyle(.blue.gradient)
                        }
                    }
                    .chartYScale(domain: .automatic(includesZero: false))
                    .chartXAxisLabel("Step")
                    .chartYAxisLabel("Loss")
                    .frame(height: 250)
                }
            }

            // Error display
            if let error = state.errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(error)
                        .foregroundStyle(.red)
                    Spacer()
                }
                .padding(10)
                .background(.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            }

            Spacer()
        }
    }

    @ViewBuilder
    var statusBadge: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
            Text(state.phase.rawValue.capitalized)
                .font(.subheadline.bold())
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(.bar, in: Capsule())
    }

    var checkpointLabel: String {
        guard let path = state.lastCheckpoint else { return "--" }
        return URL(fileURLWithPath: path).lastPathComponent
    }

    func formatTime(_ seconds: Double) -> String {
        guard seconds > 0 else { return "--" }
        if seconds < 60 { return String(format: "%.0fs", seconds) }
        if seconds < 3600 { return String(format: "%dm %02ds", Int(seconds) / 60, Int(seconds) % 60) }
        return String(format: "%dh %02dm", Int(seconds) / 3600, (Int(seconds) % 3600) / 60)
    }

    var statusColor: Color {
        switch state.phase {
        case .idle: .gray
        case .compiling: .orange
        case .training: .green
        case .paused: .yellow
        case .done: .blue
        case .error: .red
        }
    }

    @ViewBuilder
    var controlButtons: some View {
        HStack(spacing: 8) {
            if cliRunner.isRunning {
                Button(action: { cliRunner.stopTraining() }) {
                    Label("Stop", systemImage: "stop.fill")
                }
                .tint(.red)
            } else {
                Button(action: { cliRunner.startTraining(project: project) }) {
                    Label("Start Training", systemImage: "play.fill")
                }
                .tint(.green)
                .disabled(project.modelPath.isEmpty || project.dataPath.isEmpty)
            }
        }
        .buttonStyle(.borderedProminent)
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: icon)
                    .foregroundStyle(color)
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Text(value)
                .font(.title3.monospacedDigit().bold())
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(.bar, in: RoundedRectangle(cornerRadius: 10))
    }
}
