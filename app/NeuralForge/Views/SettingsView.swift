// SettingsView.swift — App settings with CLI path, defaults, API keys, and appearance

import SwiftUI
import Security

struct SettingsView: View {
    @EnvironmentObject var cliRunner: CLIRunner
    @State private var selectedTab = "general"

    var body: some View {
        TabView(selection: $selectedTab) {
            GeneralSettingsTab()
                .environmentObject(cliRunner)
                .tabItem { Label("General", systemImage: "gearshape") }
                .tag("general")

            TrainingDefaultsTab()
                .tabItem { Label("Training", systemImage: "slider.horizontal.3") }
                .tag("training")

            APIKeysTab()
                .tabItem { Label("API Keys", systemImage: "key") }
                .tag("api")

            AboutTab()
                .environmentObject(cliRunner)
                .tabItem { Label("About", systemImage: "info.circle") }
                .tag("about")
        }
        .padding(20)
        .frame(width: 580, height: 460)
    }
}

// MARK: - General Settings

struct GeneralSettingsTab: View {
    @EnvironmentObject var cliRunner: CLIRunner
    @State private var cliPath = ""
    @AppStorage("autoSaveInterval") private var autoSaveInterval = 30
    @AppStorage("showCompileTimer") private var showCompileTimer = true
    @AppStorage("defaultExportFormat") private var defaultExportFormat = "gguf"
    @AppStorage("maxHistoryEntries") private var maxHistoryEntries = 100

    var body: some View {
        Form {
            Section("CLI Binary") {
                HStack {
                    if cliRunner.cliFound {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text("Found")
                            .foregroundStyle(.green)
                    } else {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.red)
                        Text("Not Found")
                            .foregroundStyle(.red)
                    }
                    Spacer()
                }

                TextField("Path to neuralforge binary", text: $cliPath)
                    .onAppear { cliPath = cliRunner.cliBinaryPath }
                    .onSubmit { cliRunner.setCLIPath(cliPath) }

                HStack {
                    Button("Browse...") {
                        let panel = NSOpenPanel()
                        panel.canChooseFiles = true
                        panel.canChooseDirectories = false
                        panel.allowsMultipleSelection = false
                        if panel.runModal() == .OK, let url = panel.url {
                            cliPath = url.path
                            cliRunner.setCLIPath(cliPath)
                        }
                    }

                    Button("Auto-Detect") {
                        cliPath = cliRunner.cliBinaryPath
                    }

                    Spacer()
                    Text(cliRunner.cliBinaryPath)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }

            Section("Preferences") {
                Toggle("Show compile timer during ANE compilation", isOn: $showCompileTimer)

                Picker("Default export format", selection: $defaultExportFormat) {
                    Text("GGUF (llama.cpp)").tag("gguf")
                    Text("llama2c (.bin)").tag("llama2c")
                    Text("CoreML (.mlpackage)").tag("coreml")
                }

                Stepper("Auto-save interval: \(autoSaveInterval)s", value: $autoSaveInterval, in: 10...120, step: 10)

                Stepper("Max training history entries: \(maxHistoryEntries)", value: $maxHistoryEntries, in: 10...1000, step: 10)
            }

            Section("Build Instructions") {
                VStack(alignment: .leading, spacing: 4) {
                    Text("To build the CLI from source:")
                        .font(.subheadline.bold())
                    Text("cd NeuralForge/cli && make")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
        }
    }
}

// MARK: - Training Defaults

struct TrainingDefaultsTab: View {
    @AppStorage("default_totalSteps") private var totalSteps = 10000
    @AppStorage("default_learningRate") private var learningRate = 3e-4
    @AppStorage("default_accumSteps") private var accumSteps = 10
    @AppStorage("default_checkpointEvery") private var checkpointEvery = 100
    @AppStorage("default_gradClipNorm") private var gradClipNorm = 1.0
    @AppStorage("default_seed") private var seed = 42
    @AppStorage("default_warmupSteps") private var warmupSteps = 0
    @AppStorage("default_lrSchedule") private var lrSchedule = "none"
    @AppStorage("default_shuffle") private var shuffle = false
    @AppStorage("default_loraRank") private var loraRank = 0

    var body: some View {
        Form {
            Section("Default Hyperparameters") {
                HStack {
                    Text("Total steps")
                    Spacer()
                    TextField("", value: $totalSteps, format: .number)
                        .frame(width: 80)
                        .textFieldStyle(.roundedBorder)
                }

                HStack {
                    Text("Learning rate")
                    Spacer()
                    TextField("", value: $learningRate, format: .number)
                        .frame(width: 100)
                        .textFieldStyle(.roundedBorder)
                }

                HStack {
                    Text("Accumulation steps")
                    Spacer()
                    TextField("", value: $accumSteps, format: .number)
                        .frame(width: 80)
                        .textFieldStyle(.roundedBorder)
                }

                HStack {
                    Text("Checkpoint every")
                    Spacer()
                    TextField("", value: $checkpointEvery, format: .number)
                        .frame(width: 80)
                        .textFieldStyle(.roundedBorder)
                }

                HStack {
                    Text("Gradient clip norm")
                    Spacer()
                    TextField("", value: $gradClipNorm, format: .number)
                        .frame(width: 80)
                        .textFieldStyle(.roundedBorder)
                }

                HStack {
                    Text("Seed")
                    Spacer()
                    TextField("", value: $seed, format: .number)
                        .frame(width: 80)
                        .textFieldStyle(.roundedBorder)
                }
            }

            Section("Default Scheduler") {
                Picker("LR schedule", selection: $lrSchedule) {
                    Text("None").tag("none")
                    Text("Cosine Annealing").tag("cosine")
                }

                HStack {
                    Text("Warmup steps")
                    Spacer()
                    TextField("", value: $warmupSteps, format: .number)
                        .frame(width: 80)
                        .textFieldStyle(.roundedBorder)
                }

                Toggle("Shuffle data", isOn: $shuffle)
            }

            Section("Default LoRA") {
                Picker("LoRA rank", selection: $loraRank) {
                    Text("Off (full fine-tune)").tag(0)
                    Text("4").tag(4)
                    Text("8").tag(8)
                    Text("16").tag(16)
                    Text("32").tag(32)
                    Text("64").tag(64)
                }
            }

            Section {
                Button("Reset to Defaults") {
                    totalSteps = 10000; learningRate = 3e-4; accumSteps = 10
                    checkpointEvery = 100; gradClipNorm = 1.0; seed = 42
                    warmupSteps = 0; lrSchedule = "none"; shuffle = false
                    loraRank = 0
                }
                .foregroundColor(.red)
            }
        }
    }
}

// MARK: - API Keys

struct APIKeysTab: View {
    @State private var claudeKey = ""
    @State private var hasClaudeKey = false
    @State private var showClaudeKey = false
    @State private var hfToken = ""
    @State private var hasHFToken = false
    @State private var showHFToken = false
    @State private var statusMessage = ""

    var body: some View {
        Form {
            Section("Claude API Key (Anthropic)") {
                HStack {
                    if hasClaudeKey {
                        Image(systemName: "checkmark.circle.fill").foregroundColor(.green)
                        Text("Saved in Keychain").foregroundColor(.green)
                    } else {
                        Image(systemName: "circle.dotted").foregroundColor(.secondary)
                        Text("Not configured").foregroundColor(.secondary)
                    }
                    Spacer()
                }

                HStack {
                    if showClaudeKey {
                        TextField("sk-ant-...", text: $claudeKey)
                            .textFieldStyle(.roundedBorder)
                    } else {
                        SecureField("sk-ant-...", text: $claudeKey)
                            .textFieldStyle(.roundedBorder)
                    }
                    Button(action: { showClaudeKey.toggle() }) {
                        Image(systemName: showClaudeKey ? "eye.slash" : "eye")
                    }
                    .buttonStyle(.borderless)
                }

                HStack {
                    Button("Save") { saveClaudeKey() }
                        .disabled(claudeKey.isEmpty)
                    if hasClaudeKey {
                        Button("Remove", role: .destructive) { removeClaudeKey() }
                    }
                    Spacer()
                    Text("Used by: Assistant tab")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }

            Section("HuggingFace Token") {
                HStack {
                    if hasHFToken {
                        Image(systemName: "checkmark.circle.fill").foregroundColor(.green)
                        Text("Saved in Keychain").foregroundColor(.green)
                    } else {
                        Image(systemName: "circle.dotted").foregroundColor(.secondary)
                        Text("Not configured (optional)").foregroundColor(.secondary)
                    }
                    Spacer()
                }

                HStack {
                    if showHFToken {
                        TextField("hf_...", text: $hfToken)
                            .textFieldStyle(.roundedBorder)
                    } else {
                        SecureField("hf_...", text: $hfToken)
                            .textFieldStyle(.roundedBorder)
                    }
                    Button(action: { showHFToken.toggle() }) {
                        Image(systemName: showHFToken ? "eye.slash" : "eye")
                    }
                    .buttonStyle(.borderless)
                }

                HStack {
                    Button("Save") { saveHFToken() }
                        .disabled(hfToken.isEmpty)
                    if hasHFToken {
                        Button("Remove", role: .destructive) { removeHFToken() }
                    }
                    Spacer()
                    Text("Used by: Model downloads (for gated models)")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }

            if !statusMessage.isEmpty {
                Section {
                    Text(statusMessage)
                        .foregroundColor(.blue)
                        .font(.caption)
                }
            }
        }
        .onAppear {
            hasClaudeKey = keychainLoad(service: "com.neuralforge.claude-api") != nil
            hasHFToken = keychainLoad(service: "com.neuralforge.hf-token") != nil
        }
    }

    // MARK: - Keychain Helpers

    private func saveClaudeKey() {
        if keychainSave(service: "com.neuralforge.claude-api", data: claudeKey) {
            hasClaudeKey = true
            claudeKey = ""
            statusMessage = "Claude API key saved to Keychain"
        }
    }

    private func removeClaudeKey() {
        keychainDelete(service: "com.neuralforge.claude-api")
        hasClaudeKey = false
        statusMessage = "Claude API key removed"
    }

    private func saveHFToken() {
        if keychainSave(service: "com.neuralforge.hf-token", data: hfToken) {
            hasHFToken = true
            hfToken = ""
            statusMessage = "HuggingFace token saved to Keychain"
        }
    }

    private func removeHFToken() {
        keychainDelete(service: "com.neuralforge.hf-token")
        hasHFToken = false
        statusMessage = "HuggingFace token removed"
    }

    private func keychainSave(service: String, data: String) -> Bool {
        guard let d = data.data(using: .utf8) else { return false }
        keychainDelete(service: service)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: "default",
            kSecValueData as String: d,
        ]
        return SecItemAdd(query as CFDictionary, nil) == errSecSuccess
    }

    private func keychainLoad(service: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: "default",
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private func keychainDelete(service: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: "default",
        ]
        SecItemDelete(query as CFDictionary)
    }
}

// MARK: - About

struct AboutTab: View {
    @EnvironmentObject var cliRunner: CLIRunner
    let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev"
    let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"

    var body: some View {
        VStack(spacing: 20) {
            Spacer()

            Image(systemName: "cpu")
                .font(.system(size: 48))
                .foregroundColor(.blue)

            Text("NeuralForge")
                .font(.title.bold())
            Text("On-device LLM fine-tuning using Apple Neural Engine")
                .font(.subheadline)
                .foregroundColor(.secondary)

            VStack(spacing: 4) {
                Text("Version \(version) (build \(build))")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("CLI: \(cliRunner.cliFound ? "Available" : "Not Found")")
                    .font(.caption)
                    .foregroundColor(cliRunner.cliFound ? .green : .red)
            }

            VStack(spacing: 8) {
                featureItem("Core Training", "ANE forward/backward, Adam optimizer, checkpoint resume")
                featureItem("Fine-Tuning", "LoRA (rank 4-64), cosine scheduler, multi-shard data")
                featureItem("Generation", "Autoregressive inference, top-p sampling, streaming output")
                featureItem("Enterprise", "Audit log, HIPAA/SOX compliance, team sync")
                featureItem("Cluster", "Bonjour discovery, TFLOPS aggregation, shard distribution")
            }
            .padding()
            .background(RoundedRectangle(cornerRadius: 12)
                .fill(Color(nsColor: .controlBackgroundColor)))

            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private func featureItem(_ title: String, _ desc: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)
                .font(.caption)
                .padding(.top, 2)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.caption.bold())
                Text(desc).font(.caption2).foregroundColor(.secondary)
            }
            Spacer()
        }
    }
}
