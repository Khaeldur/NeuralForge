// WebhookNotificationService.swift — Slack/Discord/generic webhook alerts

import Foundation

/// Webhook target type
enum WebhookProvider: String, Codable, CaseIterable, Identifiable {
    case slack = "Slack"
    case discord = "Discord"
    case generic = "Generic"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .slack: return "bubble.left.fill"
        case .discord: return "gamecontroller.fill"
        case .generic: return "globe"
        }
    }

    var placeholderURL: String {
        switch self {
        case .slack: return "https://hooks.slack.com/services/T.../B.../xxx"
        case .discord: return "https://discord.com/api/webhooks/1234/abcdef"
        case .generic: return "https://example.com/webhook"
        }
    }
}

/// Events that can trigger webhook notifications
enum WebhookEvent: String, Codable, CaseIterable, Identifiable {
    case trainingStarted = "training_started"
    case trainingCompleted = "training_completed"
    case trainingFailed = "training_failed"
    case checkpointSaved = "checkpoint_saved"
    case validationImproved = "validation_improved"
    case lossTarget = "loss_target_reached"
    case exportCompleted = "export_completed"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .trainingStarted: return "Training Started"
        case .trainingCompleted: return "Training Completed"
        case .trainingFailed: return "Training Failed"
        case .checkpointSaved: return "Checkpoint Saved"
        case .validationImproved: return "Validation Improved"
        case .lossTarget: return "Loss Target Reached"
        case .exportCompleted: return "Export Completed"
        }
    }
}

/// A configured webhook endpoint
struct WebhookConfig: Identifiable, Codable {
    let id: UUID
    var name: String
    var provider: WebhookProvider
    var url: String
    var enabledEvents: Set<String>  // WebhookEvent rawValues
    var isEnabled: Bool
    var includeMetrics: Bool  // Include loss, step count, etc. in message
    var customMessage: String  // Optional extra text
    var created: Date

    init(name: String, provider: WebhookProvider, url: String) {
        self.id = UUID()
        self.name = name
        self.provider = provider
        self.url = url
        self.enabledEvents = Set(WebhookEvent.allCases.map { $0.rawValue })
        self.isEnabled = true
        self.includeMetrics = true
        self.customMessage = ""
        self.created = Date()
    }
}

/// Delivery attempt record
struct WebhookDelivery: Identifiable {
    let id = UUID()
    let webhookID: UUID
    let event: WebhookEvent
    let timestamp: Date
    let statusCode: Int
    let success: Bool
    let responseBody: String?
    let errorMessage: String?
}

/// Manages webhook configurations and sends notifications
@MainActor
class WebhookNotificationService: ObservableObject {
    @Published var webhooks: [WebhookConfig] = []
    @Published var recentDeliveries: [WebhookDelivery] = []
    @Published var isSending = false

    private let encoder: JSONEncoder
    private let decoder: JSONDecoder
    private let session: URLSession

    static let maxDeliveryHistory = 100

    init() {
        encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 30
        session = URLSession(configuration: config)

        loadWebhooks()
    }

    // MARK: - Directory

    static var webhooksDir: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("NeuralForge/Webhooks")
    }

    // MARK: - CRUD

    func save(_ webhook: WebhookConfig) {
        let url = Self.webhooksDir.appendingPathComponent("\(webhook.id.uuidString).json")
        try? FileManager.default.createDirectory(at: Self.webhooksDir, withIntermediateDirectories: true)
        if let data = try? encoder.encode(webhook) {
            try? data.write(to: url, options: .atomic)
        }
        if let idx = webhooks.firstIndex(where: { $0.id == webhook.id }) {
            webhooks[idx] = webhook
        } else {
            webhooks.append(webhook)
        }
    }

    func delete(_ webhook: WebhookConfig) {
        let url = Self.webhooksDir.appendingPathComponent("\(webhook.id.uuidString).json")
        try? FileManager.default.removeItem(at: url)
        webhooks.removeAll { $0.id == webhook.id }
    }

    // MARK: - Sending

    /// Send a notification to all enabled webhooks for a given event
    func notify(event: WebhookEvent, project: String, details: [String: Any] = [:]) async {
        let enabled = webhooks.filter { $0.isEnabled && $0.enabledEvents.contains(event.rawValue) }
        guard !enabled.isEmpty else { return }

        isSending = true
        for webhook in enabled {
            let delivery = await sendWebhook(webhook, event: event, project: project, details: details)
            recentDeliveries.insert(delivery, at: 0)
            if recentDeliveries.count > Self.maxDeliveryHistory {
                recentDeliveries = Array(recentDeliveries.prefix(Self.maxDeliveryHistory))
            }
        }
        isSending = false
    }

    /// Send a test notification to verify webhook configuration
    func sendTest(_ webhook: WebhookConfig) async -> WebhookDelivery {
        let delivery = await sendWebhook(webhook, event: .trainingCompleted, project: "Test Project", details: [
            "step": 1000,
            "loss": 2.45,
            "duration": "15m 30s",
            "test": true
        ])
        recentDeliveries.insert(delivery, at: 0)
        return delivery
    }

    // MARK: - Payload Building

    /// Build the webhook payload based on provider type
    nonisolated static func buildPayload(provider: WebhookProvider, event: WebhookEvent, project: String,
                              details: [String: Any], includeMetrics: Bool, customMessage: String) -> [String: Any] {
        let message = buildMessage(event: event, project: project, details: details,
                                    includeMetrics: includeMetrics, customMessage: customMessage)

        switch provider {
        case .slack:
            return buildSlackPayload(message: message, event: event)
        case .discord:
            return buildDiscordPayload(message: message, event: event, project: project)
        case .generic:
            return buildGenericPayload(event: event, project: project, message: message, details: details)
        }
    }

    nonisolated private static func buildMessage(event: WebhookEvent, project: String, details: [String: Any],
                                      includeMetrics: Bool, customMessage: String) -> String {
        var lines: [String] = []

        let emoji: String
        switch event {
        case .trainingStarted: emoji = "🚀"
        case .trainingCompleted: emoji = "✅"
        case .trainingFailed: emoji = "❌"
        case .checkpointSaved: emoji = "💾"
        case .validationImproved: emoji = "📈"
        case .lossTarget: emoji = "🎯"
        case .exportCompleted: emoji = "📦"
        }

        lines.append("\(emoji) \(event.displayName)")
        lines.append("Project: \(project)")

        if includeMetrics {
            if let step = details["step"] { lines.append("Step: \(step)") }
            if let loss = details["loss"] { lines.append("Loss: \(loss)") }
            if let valLoss = details["val_loss"] { lines.append("Val Loss: \(valLoss)") }
            if let duration = details["duration"] { lines.append("Duration: \(duration)") }
        }

        if !customMessage.isEmpty {
            lines.append(customMessage)
        }

        return lines.joined(separator: "\n")
    }

    nonisolated private static func buildSlackPayload(message: String, event: WebhookEvent) -> [String: Any] {
        let color: String
        switch event {
        case .trainingCompleted, .validationImproved, .lossTarget, .exportCompleted: color = "#36a64f"
        case .trainingFailed: color = "#ff0000"
        default: color = "#439FE0"
        }

        return [
            "attachments": [
                [
                    "color": color,
                    "text": message,
                    "footer": "NeuralForge",
                    "ts": Int(Date().timeIntervalSince1970)
                ]
            ]
        ]
    }

    nonisolated private static func buildDiscordPayload(message: String, event: WebhookEvent, project: String) -> [String: Any] {
        let color: Int
        switch event {
        case .trainingCompleted, .validationImproved, .lossTarget, .exportCompleted: color = 0x36A64F
        case .trainingFailed: color = 0xFF0000
        default: color = 0x439FE0
        }

        return [
            "embeds": [
                [
                    "title": "NeuralForge — \(event.displayName)",
                    "description": message,
                    "color": color,
                    "footer": ["text": "NeuralForge"]
                ]
            ]
        ]
    }

    nonisolated private static func buildGenericPayload(event: WebhookEvent, project: String,
                                              message: String, details: [String: Any]) -> [String: Any] {
        var payload: [String: Any] = [
            "event": event.rawValue,
            "project": project,
            "message": message,
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "source": "NeuralForge"
        ]
        // Add serializable details
        for (key, value) in details {
            if value is String || value is Int || value is Double || value is Bool {
                payload[key] = value
            }
        }
        return payload
    }

    // MARK: - Network

    private nonisolated func sendWebhook(_ webhook: WebhookConfig, event: WebhookEvent,
                                          project: String, details: [String: Any]) async -> WebhookDelivery {
        guard let url = URL(string: webhook.url) else {
            return WebhookDelivery(
                webhookID: webhook.id, event: event, timestamp: Date(),
                statusCode: 0, success: false, responseBody: nil,
                errorMessage: "Invalid webhook URL"
            )
        }

        let payload = WebhookNotificationService.buildPayload(
            provider: webhook.provider, event: event, project: project,
            details: details, includeMetrics: webhook.includeMetrics,
            customMessage: webhook.customMessage
        )

        guard let body = try? JSONSerialization.data(withJSONObject: payload) else {
            return WebhookDelivery(
                webhookID: webhook.id, event: event, timestamp: Date(),
                statusCode: 0, success: false, responseBody: nil,
                errorMessage: "Failed to serialize payload"
            )
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("NeuralForge/1.0", forHTTPHeaderField: "User-Agent")
        request.httpBody = body

        do {
            let (data, response) = try await session.data(for: request)
            let httpResponse = response as? HTTPURLResponse
            let statusCode = httpResponse?.statusCode ?? 0
            let responseBody = String(data: data, encoding: .utf8)

            return WebhookDelivery(
                webhookID: webhook.id, event: event, timestamp: Date(),
                statusCode: statusCode, success: (200..<300).contains(statusCode),
                responseBody: responseBody, errorMessage: nil
            )
        } catch {
            return WebhookDelivery(
                webhookID: webhook.id, event: event, timestamp: Date(),
                statusCode: 0, success: false, responseBody: nil,
                errorMessage: error.localizedDescription
            )
        }
    }

    // MARK: - Persistence

    private func loadWebhooks() {
        let fm = FileManager.default
        try? fm.createDirectory(at: Self.webhooksDir, withIntermediateDirectories: true)

        guard let contents = try? fm.contentsOfDirectory(at: Self.webhooksDir,
                                                          includingPropertiesForKeys: nil) else { return }

        webhooks = contents.compactMap { url -> WebhookConfig? in
            guard url.pathExtension == "json" else { return nil }
            guard let data = try? Data(contentsOf: url) else { return nil }
            return try? decoder.decode(WebhookConfig.self, from: data)
        }.sorted { $0.created > $1.created }
    }

    // MARK: - Stats

    var successRate: Double {
        guard !recentDeliveries.isEmpty else { return 1.0 }
        let successes = recentDeliveries.filter { $0.success }.count
        return Double(successes) / Double(recentDeliveries.count)
    }

    func deliveries(for webhookID: UUID) -> [WebhookDelivery] {
        recentDeliveries.filter { $0.webhookID == webhookID }
    }
}
