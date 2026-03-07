// AssistantView.swift — AI training assistant chat + hyperparameter suggestions + text evaluation
import SwiftUI

struct AssistantView: View {
    @Binding var project: NFProject
    @ObservedObject private var intelligence = NFIntelligence.shared
    @EnvironmentObject var cliRunner: CLIRunner
    @State private var inputText = ""
    @State private var showSettings = false
    @State private var apiKeyInput = ""
    @State private var showSuggestions = false
    @State private var suggestion: NFHyperparamSuggestion?
    @State private var showEvaluation = false
    @State private var evalReport: NFQualityReport?
    @State private var evalSamples: [String] = []

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Label("Training Assistant", systemImage: "brain")
                    .font(.headline)
                Spacer()
                if intelligence.hasAPIKey {
                    Menu {
                        Button("Suggest Hyperparameters") { requestSuggestions() }
                        Button("Evaluate Generated Text") { requestEvaluation() }
                        Divider()
                        Button("Clear Chat") { intelligence.clearChat() }
                        Button("API Key Settings...") { showSettings = true }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                } else {
                    Button("Configure API Key") { showSettings = true }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                }
            }
            .padding()

            Divider()

            if !intelligence.hasAPIKey {
                apiKeyPromptView
            } else {
                chatView
            }
        }
        .sheet(isPresented: $showSettings) { settingsSheet }
        .sheet(isPresented: $showSuggestions) { suggestionsSheet }
        .sheet(isPresented: $showEvaluation) { evaluationSheet }
    }

    // MARK: - API Key Prompt

    private var apiKeyPromptView: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "brain.head.profile")
                .font(.system(size: 48))
                .foregroundColor(.secondary)

            Text("AI Training Assistant")
                .font(.title2)

            Text("Get intelligent advice about your training runs, hyperparameter suggestions, and generated text evaluation.")
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
                .padding(.horizontal, 40)

            VStack(spacing: 8) {
                Text("Enter your Claude API key to get started")
                    .font(.caption)
                    .foregroundColor(.secondary)

                HStack {
                    SecureField("sk-ant-...", text: $apiKeyInput)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 300)

                    Button("Save") {
                        intelligence.setAPIKey(apiKeyInput)
                        apiKeyInput = ""
                    }
                    .disabled(apiKeyInput.isEmpty)
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding()
            .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))

            Text("Your API key is stored securely in the macOS Keychain. Only training metadata is sent — never model weights or training data.")
                .font(.caption2)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            Spacer()
        }
        .padding()
    }

    // MARK: - Chat View

    private var chatView: some View {
        VStack(spacing: 0) {
            if intelligence.chatHistory.isEmpty {
                emptyStateView
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(intelligence.chatHistory) { msg in
                                chatBubble(msg)
                                    .id(msg.id)
                            }
                            if intelligence.isProcessing {
                                HStack(spacing: 8) {
                                    ProgressView()
                                        .controlSize(.small)
                                    Text("Thinking...")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                .padding(.leading, 12)
                                .id("thinking")
                            }
                        }
                        .padding()
                    }
                    .onChange(of: intelligence.chatHistory.count) {
                        if let last = intelligence.chatHistory.last {
                            withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                        }
                    }
                }
            }

            Divider()

            // Input bar
            HStack(spacing: 8) {
                TextField("Ask about your training...", text: $inputText)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { sendMessage() }
                    .disabled(intelligence.isProcessing)

                Button(action: sendMessage) {
                    Image(systemName: "paperplane.fill")
                }
                .disabled(inputText.isEmpty || intelligence.isProcessing)
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }
            .padding()

            if let error = intelligence.lastError {
                Label(error, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundColor(.red)
                    .padding(.horizontal)
                    .padding(.bottom, 8)
            }
        }
    }

    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "text.bubble")
                .font(.system(size: 36))
                .foregroundColor(.secondary)

            Text("Ask me anything about your training")
                .font(.headline)
                .foregroundColor(.secondary)

            VStack(alignment: .leading, spacing: 8) {
                suggestionChip("My loss plateaued — what should I change?")
                suggestionChip("Should I use LoRA or full fine-tuning?")
                suggestionChip("Is my learning rate too high?")
                suggestionChip("How many steps should I train for?")
            }
            Spacer()
        }
    }

    private func suggestionChip(_ text: String) -> some View {
        Button(action: {
            inputText = text
            sendMessage()
        }) {
            HStack {
                Image(systemName: "sparkles")
                    .font(.caption)
                Text(text)
                    .font(.caption)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(RoundedRectangle(cornerRadius: 16)
                .fill(Color.blue.opacity(0.1)))
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private func chatBubble(_ msg: NFChatMessage) -> some View {
        HStack {
            if msg.role == "user" { Spacer(minLength: 60) }

            VStack(alignment: msg.role == "user" ? .trailing : .leading, spacing: 4) {
                Text(msg.content)
                    .padding(10)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(msg.role == "user"
                                  ? Color.blue.opacity(0.15)
                                  : Color(nsColor: .controlBackgroundColor))
                    )
                    .textSelection(.enabled)

                Text(msg.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            if msg.role == "assistant" { Spacer(minLength: 60) }
        }
    }

    // MARK: - Settings Sheet

    private var settingsSheet: some View {
        VStack(spacing: 20) {
            Text("API Key Settings")
                .font(.headline)

            VStack(alignment: .leading, spacing: 8) {
                Text("Claude API Key")
                    .font(.subheadline)
                HStack {
                    SecureField("sk-ant-...", text: $apiKeyInput)
                        .textFieldStyle(.roundedBorder)
                    Button("Save") {
                        intelligence.setAPIKey(apiKeyInput)
                        apiKeyInput = ""
                        showSettings = false
                    }
                    .disabled(apiKeyInput.isEmpty)
                }

                if intelligence.hasAPIKey {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                        Text("API key is configured")
                            .font(.caption)
                        Spacer()
                        Button("Remove Key") {
                            intelligence.removeAPIKey()
                        }
                        .foregroundColor(.red)
                        .font(.caption)
                    }
                }
            }

            Text("Your API key is stored in the macOS Keychain and never leaves your device.")
                .font(.caption)
                .foregroundColor(.secondary)

            Spacer()

            Button("Done") { showSettings = false }
                .buttonStyle(.borderedProminent)
        }
        .padding()
        .frame(width: 400, height: 250)
    }

    // MARK: - Suggestions Sheet

    private var suggestionsSheet: some View {
        VStack(spacing: 16) {
            HStack {
                Label("Hyperparameter Suggestions", systemImage: "wand.and.stars")
                    .font(.headline)
                Spacer()
                Button("Done") { showSuggestions = false }
            }

            if intelligence.isProcessing {
                Spacer()
                ProgressView("Analyzing your setup...")
                Spacer()
            } else if let s = suggestion {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        if let lr = s.learningRate { suggestionRow("Learning Rate", lr) }
                        if let ws = s.warmupSteps { suggestionRow("Warmup Steps", ws) }
                        if let ts = s.totalSteps { suggestionRow("Total Steps", ts) }
                        if let lora = s.loraRank { suggestionRow("LoRA Rank", lora) }
                        if let sched = s.lrSchedule { suggestionRow("LR Schedule", sched) }
                        if let acc = s.accumSteps { suggestionRow("Accumulation", acc) }

                        if !s.warnings.isEmpty {
                            VStack(alignment: .leading, spacing: 4) {
                                Label("Warnings", systemImage: "exclamationmark.triangle")
                                    .font(.subheadline.bold())
                                    .foregroundColor(.orange)
                                ForEach(s.warnings, id: \.self) { w in
                                    Text("• \(w)")
                                        .font(.caption)
                                }
                            }
                            .padding()
                            .background(RoundedRectangle(cornerRadius: 8)
                                .fill(Color.orange.opacity(0.1)))
                        }

                        Text(s.summary)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.top, 4)
                    }
                }

                Button("Apply Suggestions") { applySuggestions(s) }
                    .buttonStyle(.borderedProminent)
            } else {
                Spacer()
                Text("No suggestions available")
                    .foregroundColor(.secondary)
                Spacer()
            }
        }
        .padding()
        .frame(width: 500, height: 450)
    }

    private func suggestionRow(_ label: String, _ item: NFHyperparamSuggestion.SuggestionItem) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .font(.subheadline.bold())
                Spacer()
                Text(item.value)
                    .font(.subheadline.monospaced())
                    .foregroundColor(.blue)
            }
            Text(item.reason)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 6)
            .fill(Color(nsColor: .controlBackgroundColor)))
    }

    // MARK: - Evaluation Sheet

    private var evaluationSheet: some View {
        VStack(spacing: 16) {
            HStack {
                Label("Text Quality Evaluation", systemImage: "text.magnifyingglass")
                    .font(.headline)
                Spacer()
                Button("Done") { showEvaluation = false }
            }

            if intelligence.isProcessing {
                Spacer()
                ProgressView("Evaluating generated text...")
                Spacer()
            } else if let r = evalReport {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        // Score bars
                        HStack(spacing: 20) {
                            scoreGauge("Fluency", r.fluencyScore)
                            scoreGauge("Coherence", r.coherenceScore)
                            scoreGauge("Creativity", r.creativityScore)
                            scoreGauge("Overall", r.overallScore)
                        }
                        .padding()
                        .background(RoundedRectangle(cornerRadius: 8)
                            .fill(Color(nsColor: .controlBackgroundColor)))

                        if !r.grammarIssues.isEmpty {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Grammar Issues")
                                    .font(.subheadline.bold())
                                ForEach(r.grammarIssues, id: \.self) { issue in
                                    Text("• \(issue)")
                                        .font(.caption)
                                }
                            }
                        }

                        Text(r.summary)
                            .font(.caption)
                            .padding()
                            .background(RoundedRectangle(cornerRadius: 8)
                                .fill(Color.blue.opacity(0.05)))

                        if !evalSamples.isEmpty {
                            DisclosureGroup("Generated Samples") {
                                ForEach(evalSamples.indices, id: \.self) { i in
                                    Text(evalSamples[i])
                                        .font(.caption.monospaced())
                                        .padding(8)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .background(RoundedRectangle(cornerRadius: 4)
                                            .fill(Color(nsColor: .textBackgroundColor)))
                                }
                            }
                        }
                    }
                }
            } else {
                Spacer()
                Text("No evaluation available")
                    .foregroundColor(.secondary)
                Spacer()
            }
        }
        .padding()
        .frame(width: 550, height: 500)
    }

    private func scoreGauge(_ label: String, _ score: Int) -> some View {
        VStack(spacing: 4) {
            ZStack {
                Circle()
                    .stroke(Color.gray.opacity(0.2), lineWidth: 4)
                Circle()
                    .trim(from: 0, to: Double(score) / 10.0)
                    .stroke(scoreColor(score), style: StrokeStyle(lineWidth: 4, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                Text("\(score)")
                    .font(.title3.bold())
            }
            .frame(width: 50, height: 50)

            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }

    private func scoreColor(_ score: Int) -> Color {
        if score >= 7 { return .green }
        if score >= 4 { return .orange }
        return .red
    }

    // MARK: - Actions

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        inputText = ""

        let configSnap = TrainingConfigSnapshot(
            learningRate: project.config.learningRate,
            totalSteps: project.config.totalSteps,
            warmupSteps: project.config.warmupSteps,
            lrSchedule: project.config.lrSchedule,
            loraRank: project.config.loraRank,
            loraAlpha: project.config.loraAlpha,
            accumSteps: project.config.accumSteps
        )

        let stateSnap = TrainingStateSnapshot()

        let context = TrainingContext(project: configSnap, state: stateSnap)

        Task {
            await intelligence.sendMessage(text, trainingContext: context)
        }
    }

    private func requestSuggestions() {
        let configSnap = TrainingConfigSnapshot(
            learningRate: project.config.learningRate,
            totalSteps: project.config.totalSteps,
            warmupSteps: project.config.warmupSteps,
            lrSchedule: project.config.lrSchedule,
            loraRank: project.config.loraRank,
            loraAlpha: project.config.loraAlpha,
            accumSteps: project.config.accumSteps
        )
        showSuggestions = true
        Task {
            suggestion = await intelligence.suggestHyperparams(
                config: configSnap,
                dataTokens: 100_000,
                modelParams: 110_000_000,
                modelDim: 768
            )
        }
    }

    private func requestEvaluation() {
        guard !project.modelPath.isEmpty else {
            evalSamples = ["No model loaded. Train or download a model first, then use this feature to evaluate its output quality."]
            evalReport = nil
            showEvaluation = true
            return
        }

        // Auto-detect tokenizer
        let modelDir = (project.modelPath as NSString).deletingLastPathComponent
        var tokPath = ""
        for name in ["tokenizer.bin", "tokenizer.model"] {
            let p = (modelDir as NSString).appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: p) { tokPath = p; break }
        }

        guard !tokPath.isEmpty else {
            evalSamples = ["No tokenizer found alongside the model. Place tokenizer.bin in the same directory as your model."]
            evalReport = nil
            showEvaluation = true
            return
        }

        showEvaluation = true
        evalReport = nil
        evalSamples = []

        // Generate 3 text samples, then evaluate them via Claude
        let prompts = [
            "Once upon a time",
            "The most important thing about",
            "In this paper, we propose"
        ]
        var collectedSamples: [String] = []
        let group = DispatchGroup()

        for prompt in prompts {
            group.enter()
            var sampleText = ""
            cliRunner.generate(
                modelPath: project.modelPath,
                tokenizerPath: tokPath,
                prompt: prompt,
                temperature: 0.8, topP: 0.9,
                maxTokens: 100, seed: Int.random(in: 1...9999),
                onToken: { token in sampleText += token },
                onComplete: { success, _, _ in
                    if success && !sampleText.isEmpty {
                        collectedSamples.append(prompt + sampleText)
                    }
                    group.leave()
                }
            )
        }

        group.notify(queue: .main) { [self] in
            if collectedSamples.isEmpty {
                self.evalSamples = ["Generation failed. Check that the model and tokenizer are valid."]
                return
            }

            self.evalSamples = collectedSamples

            // Now evaluate via Claude API
            let modelName = URL(fileURLWithPath: project.modelPath).lastPathComponent
            Task {
                let report = await intelligence.evaluateGeneratedText(
                    samples: collectedSamples, modelName: modelName)
                await MainActor.run {
                    self.evalReport = report
                }
            }
        }
    }

    private func applySuggestions(_ s: NFHyperparamSuggestion) {
        if let lr = s.learningRate, let val = Double(lr.value) {
            project.config.learningRate = val
        }
        if let ws = s.warmupSteps, let val = Int(ws.value) {
            project.config.warmupSteps = val
        }
        if let ts = s.totalSteps, let val = Int(ts.value) {
            project.config.totalSteps = val
        }
        if let lr = s.loraRank, let val = Int(lr.value) {
            project.config.loraRank = val
        }
        if let sched = s.lrSchedule {
            project.config.lrSchedule = sched.value
        }
        if let acc = s.accumSteps, let val = Int(acc.value) {
            project.config.accumSteps = val
        }
        showSuggestions = false
    }
}
