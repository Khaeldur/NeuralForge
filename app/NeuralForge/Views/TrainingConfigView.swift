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

            Section("LR Schedule") {
                Picker("Schedule", selection: $project.config.lrSchedule) {
                    Text("Constant").tag("none")
                    Text("Cosine Annealing").tag("cosine")
                }
                .pickerStyle(.segmented)

                HStack {
                    Text("Warmup Steps")
                    Spacer()
                    TextField("", value: $project.config.warmupSteps, format: .number)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 80)
                }

                if project.config.lrSchedule == "cosine" {
                    VStack(alignment: .leading) {
                        HStack {
                            Text("Min Learning Rate")
                            Spacer()
                            Text(String(format: "%.1e", project.config.lrMin))
                                .monospacedDigit()
                        }
                        Slider(value: $project.config.lrMin, in: 0...1e-3)
                    }
                }
            }

            Section("Data Pipeline") {
                FilePickerRow(label: "Validation Data", path: $project.config.valDataPath,
                              allowedTypes: ["bin"])

                HStack {
                    Text("Validate Every")
                    Spacer()
                    TextField("Steps", value: $project.config.valEvery, format: .number)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 80)
                    Text("batches")
                        .foregroundStyle(.secondary)
                }

                if project.config.valEvery > 0 {
                    HStack {
                        Text("Val Batch Count")
                        Spacer()
                        TextField("", value: $project.config.valBatches, format: .number)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 80)
                    }
                }

                Toggle("Shuffle Data", isOn: $project.config.shuffle)
            }

            Section("LoRA Fine-Tuning") {
                Picker("Mode", selection: $project.config.loraRank) {
                    Text("Full Fine-Tune").tag(0)
                    Text("LoRA r=4").tag(4)
                    Text("LoRA r=8").tag(8)
                    Text("LoRA r=16").tag(16)
                    Text("LoRA r=32").tag(32)
                    Text("LoRA r=64").tag(64)
                }

                if project.config.loraRank > 0 {
                    HStack {
                        Text("Alpha")
                        Spacer()
                        TextField("Alpha", value: $project.config.loraAlpha, format: .number)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 80)
                    }

                    let loraParams = 2 * 768 * project.config.loraRank * 12
                    HStack {
                        Text("LoRA Parameters")
                        Spacer()
                        Text("\(loraParams.formatted()) (\(String(format: "%.2f", Double(loraParams) / 110e6 * 100))%)")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }

                    HStack {
                        Text("Target")
                        Spacer()
                        Text("Wo (attention output)")
                            .foregroundStyle(.secondary)
                    }
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
