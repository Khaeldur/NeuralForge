// TrainingConfigView.swift — Training hyperparameter configuration

import SwiftUI

struct TrainingConfigView: View {
    @Binding var project: NFProject

    var body: some View {
        Form {
            Section("Model & Data") {
                FilePickerRow(label: "Model Weights", path: $project.modelPath,
                              allowedTypes: ["bin", "pt", "safetensors"])
                FilePickerRow(label: "Token Data", path: $project.dataPath,
                              allowedTypes: ["bin"])
            }

            Section("Training") {
                HStack {
                    Text("Total Steps")
                    Spacer()
                    TextField("Steps", value: $project.config.totalSteps, format: .number)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 100)
                }

                VStack(alignment: .leading) {
                    HStack {
                        Text("Learning Rate")
                        Spacer()
                        Text(String(format: "%.1e", project.config.learningRate))
                            .monospacedDigit()
                    }
                    Slider(value: $project.config.learningRate, in: 1e-5...1e-2)
                }

                HStack {
                    Text("Gradient Accumulation")
                    Spacer()
                    TextField("Accum", value: $project.config.accumSteps, format: .number)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 80)
                }

                HStack {
                    Text("Checkpoint Every")
                    Spacer()
                    TextField("Steps", value: $project.config.checkpointEvery, format: .number)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 80)
                    Text("steps")
                        .foregroundStyle(.secondary)
                }

                HStack {
                    Text("Random Seed")
                    Spacer()
                    TextField("Seed", value: $project.config.seed, format: .number)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 80)
                }
            }

            Section("Optimizer (Adam)") {
                HStack {
                    Text("Beta1")
                    Spacer()
                    TextField("", value: $project.config.beta1, format: .number)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 80)
                }
                HStack {
                    Text("Beta2")
                    Spacer()
                    TextField("", value: $project.config.beta2, format: .number)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 80)
                }
                HStack {
                    Text("Epsilon")
                    Spacer()
                    Text(String(format: "%.0e", project.config.eps))
                        .monospacedDigit()
                }
                HStack {
                    Text("Grad Clip Norm")
                    Spacer()
                    TextField("", value: $project.config.gradClipNorm, format: .number)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 80)
                }
            }

            Section("Hardware") {
                Toggle("ANE Extras (classifier, softmax, rmsnorm_bwd on ANE)", isOn: $project.config.useANEExtras)
            }
        }
        .formStyle(.grouped)
    }
}

struct FilePickerRow: View {
    let label: String
    @Binding var path: String
    let allowedTypes: [String]

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            if path.isEmpty {
                Text("Not set")
                    .foregroundStyle(.secondary)
            } else {
                Text(URL(fileURLWithPath: path).lastPathComponent)
                    .font(.caption.monospaced())
                    .lineLimit(1)
            }
            Button("Browse") {
                let panel = NSOpenPanel()
                panel.canChooseFiles = true
                panel.canChooseDirectories = false
                panel.allowsMultipleSelection = false
                panel.allowedContentTypes = allowedTypes.compactMap {
                    .init(filenameExtension: $0)
                }
                if panel.runModal() == .OK, let url = panel.url {
                    path = url.path
                }
            }
            .buttonStyle(.bordered)
        }
    }
}
