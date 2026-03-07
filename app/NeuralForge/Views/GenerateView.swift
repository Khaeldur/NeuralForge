// GenerateView.swift — Text generation interface with streaming output

import SwiftUI
import AppKit

struct GenerateView: View {
    @Binding var project: NFProject
    @EnvironmentObject var cliRunner: CLIRunner

    @State private var prompt = "Once upon a time"
    @State private var generatedText = ""
    @State private var temperature = 0.8
    @State private var topP = 0.9
    @State private var maxTokens = 256
    @State private var isGenerating = false
    @State private var tokensGenerated: Int?
    @State private var generationTimeMs: Double?
    @State private var tokenizerPath = ""

    var body: some View {
        HSplitView {
            // Left: Controls
            VStack(alignment: .leading, spacing: 16) {
                GroupBox("Prompt") {
                    TextEditor(text: $prompt)
                        .font(.system(.body, design: .monospaced))
                        .frame(minHeight: 80)
                }

                GroupBox("Parameters") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Temperature")
                            Spacer()
                            Text(String(format: "%.2f", temperature))
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $temperature, in: 0.1...2.0, step: 0.05)

                        Divider()

                        HStack {
                            Text("Top-p")
                            Spacer()
                            Text(String(format: "%.2f", topP))
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $topP, in: 0.1...1.0, step: 0.05)

                        Divider()

                        HStack {
                            Text("Max Tokens")
                            Spacer()
                            Text("\(maxTokens)")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: Binding(
                            get: { Double(maxTokens) },
                            set: { maxTokens = Int($0) }
                        ), in: 1...512, step: 1)
                    }
                    .padding(.vertical, 4)
                }

                GroupBox("Tokenizer") {
                    HStack {
                        TextField("tokenizer.bin path", text: $tokenizerPath)
                            .textFieldStyle(.roundedBorder)
                        Button("Browse") {
                            let panel = NSOpenPanel()
                            panel.allowedContentTypes = [.data]
                            panel.nameFieldStringValue = "tokenizer.bin"
                            if panel.runModal() == .OK, let url = panel.url {
                                tokenizerPath = url.path
                            }
                        }
                    }
                }

                HStack {
                    Button(action: startGeneration) {
                        Label(isGenerating ? "Generating..." : "Generate",
                              systemImage: "text.bubble")
                    }
                    .disabled(isGenerating || project.modelPath.isEmpty || tokenizerPath.isEmpty)
                    .controlSize(.large)

                    if isGenerating {
                        ProgressView()
                            .scaleEffect(0.7)
                    }

                    Spacer()

                    if let tokens = tokensGenerated, let ms = generationTimeMs {
                        let tps = ms > 0 ? Double(tokens) / (ms / 1000.0) : 0
                        Text("\(tokens) tok \u{2022} \(String(format: "%.0f", ms))ms \u{2022} \(String(format: "%.1f", tps)) tok/s")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                Spacer()
            }
            .frame(minWidth: 280, maxWidth: 320)
            .padding()

            // Right: Output
            VStack(alignment: .leading) {
                HStack {
                    Text("Output")
                        .font(.headline)
                    Spacer()
                    if !generatedText.isEmpty {
                        Button("Copy") {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(generatedText, forType: .string)
                        }
                        .buttonStyle(.borderless)

                        Button("Clear") {
                            generatedText = ""
                            tokensGenerated = nil
                            generationTimeMs = nil
                        }
                        .buttonStyle(.borderless)
                    }
                }

                ScrollView {
                    Text(generatedText.isEmpty ? "Generated text will appear here..." : generatedText)
                        .font(.system(.body, design: .monospaced))
                        .foregroundColor(generatedText.isEmpty ? .secondary : .primary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(8)
                }
                .background(Color(nsColor: .textBackgroundColor))
                .cornerRadius(6)
            }
            .padding()
        }
        .onAppear { autoDetectTokenizer() }
    }

    private func autoDetectTokenizer() {
        guard tokenizerPath.isEmpty, !project.modelPath.isEmpty else { return }
        let modelDir = (project.modelPath as NSString).deletingLastPathComponent
        let candidates = ["tokenizer.bin", "tokenizer.model"]
        for name in candidates {
            let path = (modelDir as NSString).appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: path) {
                tokenizerPath = path
                return
            }
        }
    }

    private func startGeneration() {
        guard !isGenerating else { return }
        isGenerating = true
        generatedText = ""
        tokensGenerated = nil
        generationTimeMs = nil

        cliRunner.generate(
            modelPath: project.modelPath,
            tokenizerPath: tokenizerPath,
            prompt: prompt,
            temperature: temperature,
            topP: topP,
            maxTokens: maxTokens,
            seed: project.config.seed,
            onToken: { text in
                generatedText += text
            },
            onComplete: { success, tokens, ms in
                isGenerating = false
                tokensGenerated = tokens
                generationTimeMs = ms
            }
        )
    }
}
