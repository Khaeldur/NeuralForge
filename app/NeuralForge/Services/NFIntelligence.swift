// NFIntelligence.swift — Claude API client for training assistance
// 100% optional — NeuralForge works fully offline without this.
// Only metadata (loss curves, config, generated text) is sent — never weights or training data.

import Foundation
import Security

// MARK: - Keychain Helper

struct NFKeychain {
    private static let service = "com.neuralforge.app"
    private static let account = "claude-api-key"

    static func save(apiKey: String) -> Bool {
        guard let data = apiKey.data(using: .utf8) else { return false }
        // Delete existing
        let delQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        SecItemDelete(delQuery as CFDictionary)
        // Add new
        let addQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String: data
        ]
        return SecItemAdd(addQuery as CFDictionary, nil) == errSecSuccess
    }

    static func load() -> String? {
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
              let key = String(data: data, encoding: .utf8) else { return nil }
        return key
    }

    static func delete() -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        return SecItemDelete(query as CFDictionary) == errSecSuccess
    }
}

// MARK: - API Response Types

struct NFChatMessage: Identifiable, Codable {
    let id: UUID
    let role: String          // "user" or "assistant"
    let content: String
    let timestamp: Date

    init(role: String, content: String) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.timestamp = Date()
    }
}

struct NFHyperparamSuggestion: Codable {
    let learningRate: SuggestionItem?
    let warmupSteps: SuggestionItem?
    let totalSteps: SuggestionItem?
    let loraRank: SuggestionItem?
    let lrSchedule: SuggestionItem?
    let accumSteps: SuggestionItem?
    let warnings: [String]
    let summary: String

    struct SuggestionItem: Codable {
        let value: String
        let reason: String
    }
}

struct NFQualityReport: Codable {
    let fluencyScore: Int
    let coherenceScore: Int
    let creativityScore: Int
    let grammarIssues: [String]
    let summary: String
    let overallScore: Int
}

// MARK: - Claude API Client

@MainActor
class NFIntelligence: ObservableObject {
    static let shared = NFIntelligence()

    @Published var hasAPIKey: Bool = false
    @Published var isProcessing: Bool = false
    @Published var chatHistory: [NFChatMessage] = []
    @Published var lastError: String?

    private let apiURL = URL(string: "https://api.anthropic.com/v1/messages")!
    private let model = "claude-sonnet-4-20250514"
    private var rateLimitTokens: Int = 5    // requests per minute
    private var lastRequestTime: Date = .distantPast

    init() {
        hasAPIKey = NFKeychain.load() != nil
    }

    var apiKey: String? { NFKeychain.load() }

    func setAPIKey(_ key: String) {
        if NFKeychain.save(apiKey: key) {
            hasAPIKey = true
            lastError = nil
        } else {
            lastError = "Failed to save API key to Keychain"
        }
    }

    func removeAPIKey() {
        _ = NFKeychain.delete()
        hasAPIKey = false
    }

    // MARK: - Training Assistant Chat

    func sendMessage(_ text: String, trainingContext: TrainingContext) async {
        guard let key = apiKey else {
            lastError = "No API key configured"
            return
        }

        let userMsg = NFChatMessage(role: "user", content: text)
        chatHistory.append(userMsg)
        isProcessing = true
        lastError = nil

        let systemPrompt = buildSystemPrompt(context: trainingContext)

        // Build messages array (last 10 messages for context window)
        let recentMessages = chatHistory.suffix(10).map { msg -> [String: String] in
            ["role": msg.role, "content": msg.content]
        }

        do {
            let response = try await callClaude(
                system: systemPrompt,
                messages: recentMessages,
                apiKey: key,
                maxTokens: 1024
            )
            let assistantMsg = NFChatMessage(role: "assistant", content: response)
            chatHistory.append(assistantMsg)
        } catch {
            lastError = error.localizedDescription
            let errorMsg = NFChatMessage(role: "assistant",
                content: "Sorry, I couldn't process that request. \(error.localizedDescription)")
            chatHistory.append(errorMsg)
        }

        isProcessing = false
    }

    // MARK: - Hyperparameter Suggestions

    func suggestHyperparams(config: TrainingConfigSnapshot, dataTokens: Int,
                            modelParams: Int, modelDim: Int) async -> NFHyperparamSuggestion? {
        guard let key = apiKey else {
            lastError = "No API key configured"
            return nil
        }

        isProcessing = true
        lastError = nil

        let prompt = """
        You are an ML training expert. Analyze this fine-tuning setup and suggest optimal hyperparameters.

        MODEL: \(modelParams / 1_000_000)M parameters, dim=\(modelDim)
        DATASET: \(dataTokens) tokens (~\(dataTokens / 1000)K tokens)
        Data/Param ratio: \(String(format: "%.2f", Double(dataTokens) / Double(modelParams)))

        CURRENT CONFIG:
        - Learning rate: \(config.learningRate)
        - Total steps: \(config.totalSteps)
        - Warmup steps: \(config.warmupSteps)
        - LR schedule: \(config.lrSchedule)
        - Grad accumulation: \(config.accumSteps)
        - LoRA rank: \(config.loraRank) (0 = full fine-tune)
        - Sequence length: 256

        Respond with ONLY a JSON object (no markdown, no code fences) in this exact format:
        {
          "learning_rate": {"value": "2e-4", "reason": "explanation"},
          "warmup_steps": {"value": "50", "reason": "explanation"},
          "total_steps": {"value": "1000", "reason": "explanation"},
          "lora_rank": {"value": "8", "reason": "explanation"},
          "lr_schedule": {"value": "cosine", "reason": "explanation"},
          "accum_steps": {"value": "10", "reason": "explanation"},
          "warnings": ["warning1", "warning2"],
          "summary": "Brief overall recommendation"
        }

        Consider: overfitting risk (data/param ratio), learning rate stability, \
        LoRA vs full fine-tune tradeoffs, warmup proportion, total epoch count.
        """

        do {
            let response = try await callClaude(
                system: "You are an ML hyperparameter optimization expert. Respond only with valid JSON.",
                messages: [["role": "user", "content": prompt]],
                apiKey: key,
                maxTokens: 512
            )
            let suggestion = parseHyperparamResponse(response)
            isProcessing = false
            return suggestion
        } catch {
            lastError = error.localizedDescription
            isProcessing = false
            return nil
        }
    }

    // MARK: - Generated Text Evaluation

    func evaluateGeneratedText(samples: [String], modelName: String) async -> NFQualityReport? {
        guard let key = apiKey else {
            lastError = "No API key configured"
            return nil
        }
        guard !samples.isEmpty else {
            lastError = "No samples to evaluate"
            return nil
        }

        isProcessing = true
        lastError = nil

        let samplesText = samples.enumerated().map { i, s in
            "--- Sample \(i + 1) ---\n\(s)"
        }.joined(separator: "\n\n")

        let prompt = """
        Evaluate the quality of these text samples generated by a fine-tuned language model (\(modelName)).

        \(samplesText)

        Respond with ONLY a JSON object (no markdown, no code fences) in this exact format:
        {
          "fluency_score": 7,
          "coherence_score": 6,
          "creativity_score": 5,
          "grammar_issues": ["issue1", "issue2"],
          "summary": "Overall assessment of generation quality",
          "overall_score": 6
        }

        Scores are 1-10. Be specific about issues found. Assess whether the model produces \
        meaningful, grammatically correct, and contextually appropriate text.
        """

        do {
            let response = try await callClaude(
                system: "You are a text quality evaluator. Respond only with valid JSON.",
                messages: [["role": "user", "content": prompt]],
                apiKey: key,
                maxTokens: 512
            )
            let report = parseQualityResponse(response)
            isProcessing = false
            return report
        } catch {
            lastError = error.localizedDescription
            isProcessing = false
            return nil
        }
    }

    // MARK: - API Call

    private func callClaude(system: String, messages: [[String: String]],
                            apiKey: String, maxTokens: Int) async throws -> String {
        // Rate limiting: ensure at least 12s between requests (5/min)
        let elapsed = Date().timeIntervalSince(lastRequestTime)
        if elapsed < 12 {
            try await Task.sleep(nanoseconds: UInt64((12 - elapsed) * 1_000_000_000))
        }

        var request = URLRequest(url: apiURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        request.timeoutInterval = 60

        let body: [String: Any] = [
            "model": model,
            "max_tokens": maxTokens,
            "system": system,
            "messages": messages
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        lastRequestTime = Date()

        let (data, httpResponse) = try await URLSession.shared.data(for: request)

        guard let response = httpResponse as? HTTPURLResponse else {
            throw NFIntelligenceError.invalidResponse
        }

        if response.statusCode == 401 {
            throw NFIntelligenceError.invalidAPIKey
        }
        if response.statusCode == 429 {
            throw NFIntelligenceError.rateLimited
        }
        if response.statusCode != 200 {
            let body = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw NFIntelligenceError.apiError(response.statusCode, body)
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let content = json["content"] as? [[String: Any]],
              let first = content.first,
              let text = first["text"] as? String else {
            throw NFIntelligenceError.parseError
        }

        return text
    }

    // MARK: - Response Parsers

    private func parseHyperparamResponse(_ text: String) -> NFHyperparamSuggestion {
        // Try to extract JSON from response (handle markdown fences)
        let cleaned = extractJSON(from: text)

        guard let data = cleaned.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return NFHyperparamSuggestion(
                learningRate: nil, warmupSteps: nil, totalSteps: nil,
                loraRank: nil, lrSchedule: nil, accumSteps: nil,
                warnings: ["Could not parse suggestion response"],
                summary: text
            )
        }

        func parseItem(_ key: String) -> NFHyperparamSuggestion.SuggestionItem? {
            guard let dict = json[key] as? [String: String],
                  let value = dict["value"],
                  let reason = dict["reason"] else { return nil }
            return .init(value: value, reason: reason)
        }

        return NFHyperparamSuggestion(
            learningRate: parseItem("learning_rate"),
            warmupSteps: parseItem("warmup_steps"),
            totalSteps: parseItem("total_steps"),
            loraRank: parseItem("lora_rank"),
            lrSchedule: parseItem("lr_schedule"),
            accumSteps: parseItem("accum_steps"),
            warnings: json["warnings"] as? [String] ?? [],
            summary: json["summary"] as? String ?? "See suggestions above."
        )
    }

    private func parseQualityResponse(_ text: String) -> NFQualityReport {
        let cleaned = extractJSON(from: text)

        guard let data = cleaned.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return NFQualityReport(fluencyScore: 0, coherenceScore: 0, creativityScore: 0,
                                   grammarIssues: [], summary: text, overallScore: 0)
        }

        return NFQualityReport(
            fluencyScore: json["fluency_score"] as? Int ?? 0,
            coherenceScore: json["coherence_score"] as? Int ?? 0,
            creativityScore: json["creativity_score"] as? Int ?? 0,
            grammarIssues: json["grammar_issues"] as? [String] ?? [],
            summary: json["summary"] as? String ?? "",
            overallScore: json["overall_score"] as? Int ?? 0
        )
    }

    private func extractJSON(from text: String) -> String {
        // Strip markdown code fences if present
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("```json") { s = String(s.dropFirst(7)) }
        else if s.hasPrefix("```") { s = String(s.dropFirst(3)) }
        if s.hasSuffix("```") { s = String(s.dropLast(3)) }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - System Prompt Builder

    private func buildSystemPrompt(context: TrainingContext) -> String {
        var parts = [String]()
        parts.append("You are an expert ML training assistant embedded in NeuralForge, a macOS app for fine-tuning language models on Apple's Neural Engine.")
        parts.append("Be concise and actionable. Focus on practical advice for the user's specific situation.")
        parts.append("")
        parts.append("CURRENT TRAINING CONTEXT:")

        if context.modelParams > 0 {
            parts.append("- Model: \(context.modelParams / 1_000_000)M params, dim=\(context.modelDim), \(context.modelLayers) layers, vocab=\(context.modelVocab)")
        }
        parts.append("- Config: lr=\(String(format: "%.1e", context.learningRate)), steps=\(context.totalSteps), warmup=\(context.warmupSteps), schedule=\(context.lrSchedule)")
        if context.loraRank > 0 {
            parts.append("- LoRA: rank=\(context.loraRank), alpha=\(context.loraAlpha)")
        } else {
            parts.append("- Mode: Full fine-tuning")
        }
        parts.append("- Accumulation steps: \(context.accumSteps)")

        if context.currentStep > 0 {
            parts.append("")
            parts.append("TRAINING PROGRESS:")
            parts.append("- Step: \(context.currentStep)/\(context.totalSteps)")
            parts.append("- Current loss: \(String(format: "%.4f", context.currentLoss))")
            parts.append("- Best loss: \(String(format: "%.4f", context.bestLoss))")
            parts.append("- TFLOPS (ANE): \(String(format: "%.2f", context.tflopsANE))")
            parts.append("- ms/step: \(String(format: "%.1f", context.msPerStep))")

            if !context.lossHistory.isEmpty {
                let recent = context.lossHistory.suffix(10)
                let lossStr = recent.map { String(format: "%.3f", $0.loss) }.joined(separator: ", ")
                parts.append("- Recent loss trend: [\(lossStr)]")
            }
            if let valLoss = context.valLossHistory.last {
                parts.append("- Last val loss: \(String(format: "%.4f", valLoss.loss)) (step \(valLoss.step))")
            }
        }

        return parts.joined(separator: "\n")
    }

    func clearChat() {
        chatHistory = []
        lastError = nil
    }
}

// MARK: - Training Context Snapshot

struct TrainingContext {
    let modelParams: Int
    let modelDim: Int
    let modelLayers: Int
    let modelVocab: Int
    let learningRate: Double
    let totalSteps: Int
    let warmupSteps: Int
    let lrSchedule: String
    let loraRank: Int
    let loraAlpha: Double
    let accumSteps: Int
    let currentStep: Int
    let currentLoss: Double
    let bestLoss: Double
    let tflopsANE: Double
    let msPerStep: Double
    let lossHistory: [(step: Int, loss: Double)]
    let valLossHistory: [(step: Int, loss: Double)]

    init(project: TrainingConfigSnapshot, state: TrainingStateSnapshot) {
        self.learningRate = project.learningRate
        self.totalSteps = project.totalSteps
        self.warmupSteps = project.warmupSteps
        self.lrSchedule = project.lrSchedule
        self.loraRank = project.loraRank
        self.loraAlpha = project.loraAlpha
        self.accumSteps = project.accumSteps

        self.modelParams = state.modelParams
        self.modelDim = state.modelDim
        self.modelLayers = state.modelLayers
        self.modelVocab = state.modelVocab
        self.currentStep = state.currentStep
        self.currentLoss = state.currentLoss
        self.bestLoss = state.bestLoss
        self.tflopsANE = state.tflopsANE
        self.msPerStep = state.msPerStep
        self.lossHistory = state.lossHistory
        self.valLossHistory = state.valLossHistory
    }

    static let empty = TrainingContext(
        project: TrainingConfigSnapshot(), state: TrainingStateSnapshot()
    )
}

struct TrainingConfigSnapshot {
    var learningRate: Double = 3e-4
    var totalSteps: Int = 10000
    var warmupSteps: Int = 0
    var lrSchedule: String = "none"
    var loraRank: Int = 0
    var loraAlpha: Double = 16.0
    var accumSteps: Int = 10
}

struct TrainingStateSnapshot {
    var modelParams: Int = 0
    var modelDim: Int = 0
    var modelLayers: Int = 0
    var modelVocab: Int = 0
    var currentStep: Int = 0
    var currentLoss: Double = 0
    var bestLoss: Double = .infinity
    var tflopsANE: Double = 0
    var msPerStep: Double = 0
    var lossHistory: [(step: Int, loss: Double)] = []
    var valLossHistory: [(step: Int, loss: Double)] = []
}

// MARK: - Error Types

enum NFIntelligenceError: LocalizedError {
    case invalidAPIKey
    case rateLimited
    case invalidResponse
    case parseError
    case apiError(Int, String)
    case noAPIKey

    var errorDescription: String? {
        switch self {
        case .invalidAPIKey: return "Invalid API key. Check your Claude API key in settings."
        case .rateLimited: return "Rate limited. Please wait a moment and try again."
        case .invalidResponse: return "Invalid response from API."
        case .parseError: return "Failed to parse API response."
        case .apiError(let code, let msg): return "API error (\(code)): \(msg)"
        case .noAPIKey: return "No API key configured. Add your Claude API key in settings."
        }
    }
}
