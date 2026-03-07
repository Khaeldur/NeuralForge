// MenuBarManager.swift — Menu bar extra showing training status at a glance

import SwiftUI

@MainActor
class MenuBarManager: ObservableObject {
    static let shared = MenuBarManager()

    @Published var isTraining = false
    @Published var currentStep = 0
    @Published var totalSteps = 0
    @Published var currentLoss: Double = 0
    @Published var bestLoss: Double = .infinity
    @Published var tflops: Double = 0
    @Published var msPerStep: Double = 0
    @Published var projectName = ""
    @Published var elapsedSeconds: Double = 0

    private var timer: Timer?
    private var startTime: Date?

    var progress: Double {
        guard totalSteps > 0 else { return 0 }
        return Double(currentStep) / Double(totalSteps)
    }

    var progressPercent: Int {
        Int(progress * 100)
    }

    var statusIcon: String {
        if !isTraining { return "cpu" }
        // Pulse between two states while training
        return "cpu.fill"
    }

    var menuBarTitle: String {
        guard isTraining else { return "" }
        return "\(progressPercent)%"
    }

    var etaFormatted: String {
        guard isTraining, currentStep > 0, msPerStep > 0 else { return "—" }
        let remainingSteps = totalSteps - currentStep
        let remainingSec = Double(remainingSteps) * msPerStep / 1000.0
        if remainingSec < 60 { return String(format: "%.0fs", remainingSec) }
        if remainingSec < 3600 { return String(format: "%.0fm", remainingSec / 60) }
        return String(format: "%.1fh", remainingSec / 3600)
    }

    var elapsedFormatted: String {
        let s = elapsedSeconds
        if s < 60 { return String(format: "%.0fs", s) }
        if s < 3600 { return String(format: "%.0fm %.0fs", s / 60, s.truncatingRemainder(dividingBy: 60)) }
        return String(format: "%.0fh %.0fm", s / 3600, (s / 60).truncatingRemainder(dividingBy: 60))
    }

    func startTraining(project: String, total: Int) {
        isTraining = true
        projectName = project
        totalSteps = total
        currentStep = 0
        currentLoss = 0
        bestLoss = .infinity
        startTime = Date()
        elapsedSeconds = 0

        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self = self, let start = self.startTime else { return }
                self.elapsedSeconds = Date().timeIntervalSince(start)
            }
        }
    }

    func updateStep(step: Int, loss: Double, tflops: Double, msPerStep: Double) {
        self.currentStep = step
        self.currentLoss = loss
        self.tflops = tflops
        self.msPerStep = msPerStep
        if loss < bestLoss { bestLoss = loss }
    }

    func stopTraining() {
        isTraining = false
        timer?.invalidate()
        timer = nil
    }
}

// MARK: - Menu Bar View

struct MenuBarView: View {
    @ObservedObject var manager = MenuBarManager.shared
    @ObservedObject var cliRunner: CLIRunner

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if manager.isTraining {
                trainingSection
            } else {
                idleSection
            }

            Divider()

            // Quick actions
            HStack {
                Button("Open NeuralForge") {
                    NSApp.activate(ignoringOtherApps: true)
                    if let window = NSApp.windows.first {
                        window.makeKeyAndOrderFront(nil)
                    }
                }
                Spacer()
                Button("Quit") {
                    NSApp.terminate(nil)
                }
                .foregroundColor(.red)
            }
            .font(.caption)
        }
        .padding()
        .frame(width: 280)
    }

    private var trainingSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "cpu.fill")
                    .foregroundColor(.green)
                Text("Training: \(manager.projectName)")
                    .font(.subheadline.bold())
                    .lineLimit(1)
            }

            ProgressView(value: manager.progress) {
                HStack {
                    Text("Step \(manager.currentStep)/\(manager.totalSteps)")
                        .font(.caption.monospaced())
                    Spacer()
                    Text("\(manager.progressPercent)%")
                        .font(.caption.bold())
                }
            }

            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
                GridRow {
                    metricLabel("Loss")
                    metricValue(String(format: "%.4f", manager.currentLoss))
                    metricLabel("Best")
                    metricValue(String(format: "%.4f", manager.bestLoss))
                }
                GridRow {
                    metricLabel("TFLOPS")
                    metricValue(String(format: "%.1f", manager.tflops))
                    metricLabel("ms/step")
                    metricValue(String(format: "%.0f", manager.msPerStep))
                }
                GridRow {
                    metricLabel("Elapsed")
                    metricValue(manager.elapsedFormatted)
                    metricLabel("ETA")
                    metricValue(manager.etaFormatted)
                }
            }
        }
    }

    private var idleSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: "cpu")
                    .foregroundColor(.secondary)
                Text("NeuralForge")
                    .font(.subheadline.bold())
            }
            Text("No active training session")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    private func metricLabel(_ text: String) -> some View {
        Text(text)
            .font(.caption2)
            .foregroundColor(.secondary)
    }

    private func metricValue(_ text: String) -> some View {
        Text(text)
            .font(.caption.monospaced())
    }
}
