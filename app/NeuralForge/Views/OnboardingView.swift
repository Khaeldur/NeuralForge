// OnboardingView.swift — First-run wizard for new users

import SwiftUI

struct OnboardingView: View {
    @EnvironmentObject var projectManager: ProjectManager
    @EnvironmentObject var cliRunner: CLIRunner
    @Binding var isComplete: Bool
    @State private var currentPage = 0
    @State private var cliPath = ""
    @State private var cliValid = false
    @State private var hfToken = ""
    @State private var projectName = "My First Project"
    @State private var selectedGoal: TrainingGoal = .experiment

    enum TrainingGoal: String, CaseIterable {
        case experiment = "Experiment & Learn"
        case finetune = "Fine-tune a Model"
        case production = "Production Deployment"

        var icon: String {
            switch self {
            case .experiment: return "flask"
            case .finetune: return "cpu"
            case .production: return "shippingbox"
            }
        }

        var description: String {
            switch self {
            case .experiment: return "Great for exploring LLM fine-tuning on your Mac"
            case .finetune: return "Train a model for a specific task or domain"
            case .production: return "Build production-ready models with full audit trails"
            }
        }
    }

    private let totalPages = 4

    var body: some View {
        VStack(spacing: 0) {
            // Progress dots
            HStack(spacing: 8) {
                ForEach(0..<totalPages, id: \.self) { i in
                    Circle()
                        .fill(i <= currentPage ? Color.accentColor : Color.secondary.opacity(0.3))
                        .frame(width: 8, height: 8)
                }
            }
            .padding(.top, 24)

            // Page content
            TabView(selection: $currentPage) {
                welcomePage.tag(0)
                setupPage.tag(1)
                goalPage.tag(2)
                readyPage.tag(3)
            }
            .tabViewStyle(.automatic)

            // Navigation
            HStack {
                if currentPage > 0 {
                    Button("Back") { withAnimation { currentPage -= 1 } }
                        .buttonStyle(.bordered)
                }
                Spacer()
                if currentPage < totalPages - 1 {
                    Button("Next") { withAnimation { currentPage += 1 } }
                        .buttonStyle(.borderedProminent)
                        .disabled(currentPage == 1 && !cliValid)
                } else {
                    Button("Get Started") { finishOnboarding() }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                }
            }
            .padding(24)
        }
        .frame(width: 620, height: 520)
        .onAppear { detectCLI() }
    }

    // MARK: - Page 1: Welcome

    private var welcomePage: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "cpu.fill")
                .font(.system(size: 64))
                .foregroundStyle(.linearGradient(
                    colors: [.blue, .purple],
                    startPoint: .topLeading, endPoint: .bottomTrailing))

            Text("Welcome to NeuralForge")
                .font(.largeTitle.bold())

            Text("Fine-tune language models on your Mac\nusing Apple's Neural Engine")
                .font(.title3)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            HStack(spacing: 32) {
                featurePill("bolt.fill", "Neural Engine", "Hardware-accelerated training")
                featurePill("lock.shield", "100% Local", "Your data never leaves your Mac")
                featurePill("chart.line.uptrend.xyaxis", "Real-time", "Live loss curves & metrics")
            }
            .padding(.top, 8)

            Spacer()
        }
        .padding()
    }

    private func featurePill(_ icon: String, _ title: String, _ subtitle: String) -> some View {
        VStack(spacing: 6) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.blue)
            Text(title)
                .font(.caption.bold())
            Text(subtitle)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(width: 140)
    }

    // MARK: - Page 2: Setup

    private var setupPage: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "wrench.and.screwdriver")
                .font(.system(size: 48))
                .foregroundColor(.orange)

            Text("Quick Setup")
                .font(.title.bold())

            Text("NeuralForge needs its CLI binary to run training and inference.")
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: cliValid ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .foregroundColor(cliValid ? .green : .red)
                    Text("CLI Binary")
                        .font(.subheadline.bold())
                }

                HStack {
                    TextField("Path to neuralforge binary", text: $cliPath)
                        .textFieldStyle(.roundedBorder)
                    Button("Browse") {
                        let panel = NSOpenPanel()
                        panel.canChooseFiles = true
                        panel.allowedContentTypes = [.unixExecutable]
                        if panel.runModal() == .OK, let url = panel.url {
                            cliPath = url.path
                            validateCLI()
                        }
                    }
                    Button("Auto-detect") { detectCLI() }
                }
                .onChange(of: cliPath) { validateCLI() }

                if cliValid {
                    Label("CLI binary found and ready", systemImage: "checkmark")
                        .font(.caption)
                        .foregroundColor(.green)
                } else if !cliPath.isEmpty {
                    Label("Binary not found at this path", systemImage: "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundColor(.orange)
                }

                Divider()

                HStack {
                    Image(systemName: "key")
                        .foregroundColor(.secondary)
                    Text("HuggingFace Token")
                        .font(.subheadline.bold())
                    Text("(optional)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                SecureField("hf_...", text: $hfToken)
                    .textFieldStyle(.roundedBorder)

                Text("Required only for downloading gated models (e.g., Llama). Stored in Keychain.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(RoundedRectangle(cornerRadius: 12)
                .fill(Color(nsColor: .controlBackgroundColor)))
            .padding(.horizontal, 30)

            Spacer()
        }
        .padding()
    }

    // MARK: - Page 3: Goal

    private var goalPage: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "target")
                .font(.system(size: 48))
                .foregroundColor(.green)

            Text("What's Your Goal?")
                .font(.title.bold())

            Text("This helps us configure sensible defaults for your first project.")
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            VStack(spacing: 12) {
                ForEach(TrainingGoal.allCases, id: \.self) { goal in
                    Button(action: { selectedGoal = goal }) {
                        HStack(spacing: 12) {
                            Image(systemName: goal.icon)
                                .font(.title3)
                                .foregroundColor(.blue)
                                .frame(width: 32)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(goal.rawValue)
                                    .font(.subheadline.bold())
                                    .foregroundColor(.primary)
                                Text(goal.description)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                            if selectedGoal == goal {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.blue)
                            }
                        }
                        .padding(12)
                        .background(RoundedRectangle(cornerRadius: 10)
                            .fill(selectedGoal == goal
                                  ? Color.accentColor.opacity(0.08)
                                  : Color(nsColor: .controlBackgroundColor)))
                        .overlay(RoundedRectangle(cornerRadius: 10)
                            .stroke(selectedGoal == goal ? Color.accentColor : Color.clear, lineWidth: 1.5))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 60)

            Spacer()
        }
        .padding()
    }

    // MARK: - Page 4: Ready

    private var readyPage: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 56))
                .foregroundStyle(.linearGradient(
                    colors: [.green, .blue],
                    startPoint: .topLeading, endPoint: .bottomTrailing))

            Text("You're All Set!")
                .font(.title.bold())

            VStack(alignment: .leading, spacing: 12) {
                Text("Your first project:")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                TextField("Project name", text: $projectName)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 300)
            }

            VStack(alignment: .leading, spacing: 8) {
                readyRow("square.grid.2x2", "Download a model from the Model Hub")
                readyRow("doc.text", "Import or create training data")
                readyRow("play.fill", "Hit Train and watch the loss curve drop")
                readyRow("text.bubble", "Generate text with your fine-tuned model")
            }
            .padding()
            .background(RoundedRectangle(cornerRadius: 12)
                .fill(Color(nsColor: .controlBackgroundColor)))

            Text("Press **Get Started** to create your project and begin!")
                .font(.caption)
                .foregroundColor(.secondary)

            Spacer()
        }
        .padding()
    }

    private func readyRow(_ icon: String, _ text: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 20)
            Text(text)
                .font(.caption)
        }
    }

    // MARK: - Actions

    private func detectCLI() {
        let candidates = [
            "/usr/local/bin/neuralforge",
            NSHomeDirectory() + "/NeuralForge/cli/neuralforge",
            NSHomeDirectory() + "/Desktop/sl/NeuralForge/cli/neuralforge",
            Bundle.main.bundlePath + "/Contents/Resources/neuralforge"
        ]
        // Also check @AppStorage
        let storedPath = UserDefaults.standard.string(forKey: "cliPath") ?? ""
        let allPaths = [storedPath] + candidates

        for path in allPaths where !path.isEmpty {
            if FileManager.default.isExecutableFile(atPath: path) {
                cliPath = path
                cliValid = true
                return
            }
        }
    }

    private func validateCLI() {
        cliValid = FileManager.default.isExecutableFile(atPath: cliPath)
    }

    private func finishOnboarding() {
        // Save CLI path
        if cliValid {
            UserDefaults.standard.set(cliPath, forKey: "cliPath")
        }

        // Save HF token
        if !hfToken.isEmpty {
            _ = NFKeychain.save(service: "com.neuralforge.app", account: "hf-token", value: hfToken)
        }

        // Apply goal-based defaults
        switch selectedGoal {
        case .experiment:
            UserDefaults.standard.set(1000, forKey: "defaultTotalSteps")
            UserDefaults.standard.set(3e-4, forKey: "defaultLearningRate")
        case .finetune:
            UserDefaults.standard.set(5000, forKey: "defaultTotalSteps")
            UserDefaults.standard.set(2e-4, forKey: "defaultLearningRate")
        case .production:
            UserDefaults.standard.set(10000, forKey: "defaultTotalSteps")
            UserDefaults.standard.set(1e-4, forKey: "defaultLearningRate")
        }

        // Create first project
        let trimmed = projectName.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty {
            let _ = projectManager.create(name: trimmed)
        }

        // Mark onboarding complete
        UserDefaults.standard.set(true, forKey: "onboardingComplete")
        isComplete = true
    }
}

// Extension for NFKeychain to support arbitrary keys
extension NFKeychain {
    static func save(service: String, account: String, value: String) -> Bool {
        guard let data = value.data(using: .utf8) else { return false }
        let delQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        SecItemDelete(delQuery as CFDictionary)
        let addQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String: data
        ]
        return SecItemAdd(addQuery as CFDictionary, nil) == errSecSuccess
    }

    static func load(service: String, account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data,
              let val = String(data: data, encoding: .utf8) else { return nil }
        return val
    }
}
