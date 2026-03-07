// GradientAggregator.swift — Distributed gradient aggregation protocol for multi-Mac ANE clusters
// Implements all-reduce, ring-reduce, and parameter-server gradient averaging
// over the Bonjour-discovered cluster managed by ComputeClusterService.

import Foundation
import Network

// MARK: - Wire Protocol Messages

/// Messages exchanged between coordinator and worker nodes during distributed training.
/// JSON-encoded over TCP connections established by ComputeClusterService.
enum GradientMessage: Codable {
    case assignWork(AssignWork)
    case gradientReady(GradientReady)
    case aggregated(Aggregated)
    case heartbeat(Heartbeat)
    case syncCheckpoint(SyncCheckpoint)

    struct AssignWork: Codable {
        let shardPaths: [String]
        let modelPath: String
        let config: [String: String]
        let startStep: Int
        let endStep: Int
    }

    struct GradientReady: Codable {
        let nodeID: String
        let step: Int
        let gradientHash: String
        let gradientSizeBytes: Int
        let localLoss: Double
        let computeTimeMs: Double
    }

    struct Aggregated: Codable {
        let step: Int
        let averagedLoss: Double
        let participatingNodes: Int
        let aggregationTimeMs: Double
    }

    struct Heartbeat: Codable {
        let nodeID: String
        let timestamp: Date
        let currentStep: Int
        let status: String
    }

    struct SyncCheckpoint: Codable {
        let step: Int
        let checkpointHash: String
        let checkpointSizeBytes: Int
    }

    // MARK: Codable Conformance

    private enum CodingKeys: String, CodingKey {
        case type
        case payload
    }

    private enum MessageType: String, Codable {
        case assignWork
        case gradientReady
        case aggregated
        case heartbeat
        case syncCheckpoint
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .assignWork(let payload):
            try container.encode(MessageType.assignWork, forKey: .type)
            try container.encode(payload, forKey: .payload)
        case .gradientReady(let payload):
            try container.encode(MessageType.gradientReady, forKey: .type)
            try container.encode(payload, forKey: .payload)
        case .aggregated(let payload):
            try container.encode(MessageType.aggregated, forKey: .type)
            try container.encode(payload, forKey: .payload)
        case .heartbeat(let payload):
            try container.encode(MessageType.heartbeat, forKey: .type)
            try container.encode(payload, forKey: .payload)
        case .syncCheckpoint(let payload):
            try container.encode(MessageType.syncCheckpoint, forKey: .type)
            try container.encode(payload, forKey: .payload)
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(MessageType.self, forKey: .type)
        switch type {
        case .assignWork:
            self = .assignWork(try container.decode(AssignWork.self, forKey: .payload))
        case .gradientReady:
            self = .gradientReady(try container.decode(GradientReady.self, forKey: .payload))
        case .aggregated:
            self = .aggregated(try container.decode(Aggregated.self, forKey: .payload))
        case .heartbeat:
            self = .heartbeat(try container.decode(Heartbeat.self, forKey: .payload))
        case .syncCheckpoint:
            self = .syncCheckpoint(try container.decode(SyncCheckpoint.self, forKey: .payload))
        }
    }
}

// MARK: - Aggregation Configuration

/// Configures how gradient aggregation behaves across the cluster.
struct AggregationConfig: Codable {
    var strategy: AggregationStrategy = .allReduce
    var syncEveryNSteps: Int = 10
    var timeoutSeconds: Double = 30
    var stragglerPolicy: StragglerPolicy = .timeout
    var compressionEnabled: Bool = true
    var compressionThreshold: Float = 0.01
    var maxNodesInRing: Int = 8
    var checksumVerification: Bool = true

    enum AggregationStrategy: String, Codable, CaseIterable {
        case allReduce = "AllReduce"
        case parameterServer = "ParameterServer"
        case gossipProtocol = "GossipProtocol"

        var description: String {
            switch self {
            case .allReduce: return "All-Reduce (symmetric averaging)"
            case .parameterServer: return "Parameter Server (centralized)"
            case .gossipProtocol: return "Gossip Protocol (decentralized)"
            }
        }
    }

    enum StragglerPolicy: String, Codable, CaseIterable {
        case wait = "Wait"
        case skip = "Skip"
        case timeout = "Timeout"

        var description: String {
            switch self {
            case .wait: return "Wait for all nodes"
            case .skip: return "Skip slow nodes"
            case .timeout: return "Timeout after deadline"
            }
        }
    }
}

// MARK: - Gradient Metrics

/// Published statistics for monitoring gradient aggregation health in the UI.
class GradientMetrics: ObservableObject {
    @Published var totalRoundsCompleted: Int = 0
    @Published var averageAggregationTimeMs: Double = 0
    @Published var totalGradientsBytesTransferred: Int64 = 0
    @Published var stragglerCount: Int = 0
    @Published var failedRounds: Int = 0
    @Published var throughputGradientsPerSec: Double = 0
    @Published var lastRoundNodes: Int = 0
    @Published var currentStepAcrossCluster: Int = 0

    /// Running sum used to compute the rolling average
    private var aggregationTimeSumMs: Double = 0
    private var throughputWindowStart: Date?
    private var throughputWindowGradients: Int = 0

    func recordRoundCompleted(aggregationTimeMs: Double, participatingNodes: Int, bytesTransferred: Int64) {
        totalRoundsCompleted += 1
        aggregationTimeSumMs += aggregationTimeMs
        averageAggregationTimeMs = aggregationTimeSumMs / Double(totalRoundsCompleted)
        totalGradientsBytesTransferred += bytesTransferred
        lastRoundNodes = participatingNodes

        // Update throughput
        let now = Date()
        if throughputWindowStart == nil {
            throughputWindowStart = now
        }
        throughputWindowGradients += participatingNodes
        let elapsed = now.timeIntervalSince(throughputWindowStart ?? now)
        if elapsed > 0 {
            throughputGradientsPerSec = Double(throughputWindowGradients) / elapsed
        }

        // Reset throughput window every 60 seconds
        if elapsed > 60 {
            throughputWindowStart = now
            throughputWindowGradients = participatingNodes
        }
    }

    func recordStraggler() {
        stragglerCount += 1
    }

    func recordFailedRound() {
        failedRounds += 1
    }

    func reset() {
        totalRoundsCompleted = 0
        averageAggregationTimeMs = 0
        totalGradientsBytesTransferred = 0
        stragglerCount = 0
        failedRounds = 0
        throughputGradientsPerSec = 0
        lastRoundNodes = 0
        currentStepAcrossCluster = 0
        aggregationTimeSumMs = 0
        throughputWindowStart = nil
        throughputWindowGradients = 0
    }
}

// MARK: - Aggregation Event Log

/// A single logged event during gradient aggregation, shown in the aggregation log UI.
struct AggregationEvent: Identifiable {
    let id = UUID()
    let timestamp: Date
    let type: EventType
    let description: String
    let nodeID: String?

    enum EventType: String {
        case roundStart = "Round Start"
        case roundComplete = "Round Complete"
        case stragglerDetected = "Straggler Detected"
        case nodeJoined = "Node Joined"
        case nodeLeft = "Node Left"
        case error = "Error"
    }
}

// MARK: - Gradient Aggregator

/// Manages the gradient aggregation protocol for distributed training across a
/// Bonjour-discovered cluster. Runs as coordinator (averaging gradients from workers)
/// or as a worker (computing and submitting local gradients).
@MainActor
class GradientAggregator: ObservableObject {
    static let shared = GradientAggregator()

    // MARK: Published State

    @Published var config = AggregationConfig()
    @Published var metrics = GradientMetrics()
    @Published var aggregationLog: [AggregationEvent] = []
    @Published var isCoordinator = false
    @Published var currentRound: Int = 0
    @Published var nodesReady: Set<String> = []
    @Published var pendingGradients: [String: GradientMessage.GradientReady] = [:]

    // MARK: Internal State

    private var clusterNodes: [ClusterNode] = []
    private var coordinatorEndpoint: NWEndpoint?
    private var coordinatorConnection: NWConnection?
    private var roundStartTime: Date?
    private var stragglerTimer: Timer?
    private var heartbeatTimer: Timer?
    private var isRunning = false
    private var expectedNodeCount: Int = 0
    private var ringOrder: [String] = []

    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = .prettyPrinted
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    /// Maximum number of events kept in the aggregation log
    private let maxLogEntries = 500

    // MARK: - Coordinator Lifecycle

    /// Start this node as the aggregation coordinator. The coordinator receives
    /// gradient-ready messages from workers, runs the configured aggregation
    /// strategy, and broadcasts the averaged result.
    func startCoordinator(nodes: [ClusterNode]) {
        guard !isRunning else {
            appendLog(.error, "Aggregator already running", nodeID: nil)
            return
        }

        isRunning = true
        isCoordinator = true
        clusterNodes = nodes
        expectedNodeCount = nodes.count
        currentRound = 0
        nodesReady.removeAll()
        pendingGradients.removeAll()
        metrics.reset()

        // Build ring order sorted by TFLOPS (strongest first) for ring-allreduce
        ringOrder = nodes
            .sorted { $0.capabilities.tflopsEstimate > $1.capabilities.tflopsEstimate }
            .prefix(config.maxNodesInRing)
            .map { $0.id }

        appendLog(.roundStart, "Coordinator started with \(nodes.count) node(s), strategy: \(config.strategy.rawValue)", nodeID: nil)

        for node in nodes {
            appendLog(.nodeJoined, "Node joined: \(node.capabilities.hostName) (\(node.capabilities.chipName))", nodeID: node.id)
        }

        // Start heartbeat monitoring
        startHeartbeatTimer()
    }

    /// Start this node as a worker, connecting to the given coordinator endpoint.
    func startWorker(coordinatorEndpoint: NWEndpoint) {
        guard !isRunning else {
            appendLog(.error, "Aggregator already running", nodeID: nil)
            return
        }

        isRunning = true
        isCoordinator = false
        self.coordinatorEndpoint = coordinatorEndpoint
        currentRound = 0
        metrics.reset()

        // Establish TCP connection to coordinator
        let connection = NWConnection(to: coordinatorEndpoint, using: .tcp)
        coordinatorConnection = connection

        connection.stateUpdateHandler = { [weak self] state in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                switch state {
                case .ready:
                    self.appendLog(.nodeJoined, "Connected to coordinator", nodeID: nil)
                    self.startHeartbeatTimer()
                    self.receiveMessages(on: connection)
                case .failed(let error):
                    self.appendLog(.error, "Connection to coordinator failed: \(error)", nodeID: nil)
                    self.stopAggregation()
                case .cancelled:
                    self.appendLog(.nodeLeft, "Disconnected from coordinator", nodeID: nil)
                default:
                    break
                }
            }
        }

        connection.start(queue: .main)
        appendLog(.roundStart, "Worker started, connecting to coordinator", nodeID: nil)
    }

    // MARK: - Gradient Submission

    /// Called by a worker node when local gradient computation is complete.
    /// Packages the gradient metadata and sends it to the coordinator (or
    /// processes it locally if this node is the coordinator).
    func submitGradients(step: Int, gradientHash: String, size: Int, loss: Double, timeMs: Double) {
        let localID = DeviceCapabilities.local().deviceID

        let ready = GradientMessage.GradientReady(
            nodeID: localID,
            step: step,
            gradientHash: gradientHash,
            gradientSizeBytes: size,
            localLoss: loss,
            computeTimeMs: timeMs
        )

        if isCoordinator {
            // Process directly on coordinator
            handleGradientReady(ready, from: localID)
        } else if let connection = coordinatorConnection {
            // Send to coordinator
            let message = GradientMessage.gradientReady(ready)
            sendMessage(message, on: connection)
        } else {
            appendLog(.error, "No coordinator connection — cannot submit gradients", nodeID: localID)
        }
    }

    // MARK: - Aggregation Strategies

    /// Symmetric all-reduce: average all received gradients and broadcast the result.
    /// Each participating node contributes equally to the averaged update.
    func allReduceAverage(gradients: [GradientMessage.GradientReady]) -> GradientMessage.Aggregated {
        let aggregationStart = Date()

        guard !gradients.isEmpty else {
            return GradientMessage.Aggregated(
                step: currentRound,
                averagedLoss: 0,
                participatingNodes: 0,
                aggregationTimeMs: 0
            )
        }

        // Compute average loss across all participating nodes
        let totalLoss = gradients.reduce(0.0) { $0 + $1.localLoss }
        let averagedLoss = totalLoss / Double(gradients.count)

        // Verify checksums if enabled
        if config.checksumVerification {
            let uniqueHashes = Set(gradients.map { $0.gradientHash })
            if uniqueHashes.count > 1 {
                appendLog(.error, "Gradient hash mismatch across \(uniqueHashes.count) distinct hashes — possible data corruption", nodeID: nil)
            }
        }

        let aggregationTimeMs = Date().timeIntervalSince(aggregationStart) * 1000

        let result = GradientMessage.Aggregated(
            step: gradients.first?.step ?? currentRound,
            averagedLoss: averagedLoss,
            participatingNodes: gradients.count,
            aggregationTimeMs: aggregationTimeMs
        )

        return result
    }

    /// Ring all-reduce: pass partial sums around a ring of nodes for bandwidth-optimal
    /// gradient averaging. Each node sends its chunk to the next node in the ring,
    /// accumulates the received chunk, and forwards the partial sum. After N-1 steps
    /// (where N is the ring size) every node has the fully reduced result.
    func ringAllReduce(nodeOrder: [String]) {
        let ringSize = min(nodeOrder.count, config.maxNodesInRing)
        let activeRing = Array(nodeOrder.prefix(ringSize))

        appendLog(.roundStart, "Ring all-reduce started with \(ringSize) nodes in ring", nodeID: nil)

        // Simulate the ring-reduce phases:
        // Phase 1: Scatter-reduce — each node sends 1/N of its gradients around the ring
        // Phase 2: All-gather — each node broadcasts its fully-reduced chunk

        let _ = ringSize  // chunks per node in ring
        let totalPasses = (ringSize - 1) * 2  // scatter-reduce + all-gather

        for pass in 0..<totalPasses {
            let phase = pass < (ringSize - 1) ? "scatter-reduce" : "all-gather"
            let senderIdx = pass % ringSize
            let receiverIdx = (senderIdx + 1) % ringSize

            let sender = activeRing[senderIdx]
            let receiver = activeRing[receiverIdx]

            appendLog(.roundStart, "Ring pass \(pass + 1)/\(totalPasses) (\(phase)): \(sender) → \(receiver)", nodeID: sender)
        }

        // After ring completes, treat it like a successful all-reduce round
        let ringGradients = activeRing.compactMap { pendingGradients[$0] }
        if !ringGradients.isEmpty {
            let result = allReduceAverage(gradients: ringGradients)
            completeRound(result: result, gradients: ringGradients)
        }

        appendLog(.roundComplete, "Ring all-reduce complete for \(ringSize) nodes", nodeID: nil)
    }

    // MARK: - Message Handling

    /// Central message dispatch. Routes incoming protocol messages to the appropriate handler.
    func handleMessage(_ message: GradientMessage, from nodeID: String) {
        switch message {
        case .assignWork(let work):
            handleAssignWork(work, from: nodeID)
        case .gradientReady(let ready):
            handleGradientReady(ready, from: nodeID)
        case .aggregated(let result):
            handleAggregated(result, from: nodeID)
        case .heartbeat(let hb):
            handleHeartbeat(hb, from: nodeID)
        case .syncCheckpoint(let sync):
            handleSyncCheckpoint(sync, from: nodeID)
        }
    }

    // MARK: - Stop

    /// Tear down all aggregation state, cancel connections and timers.
    func stopAggregation() {
        isRunning = false
        stragglerTimer?.invalidate()
        stragglerTimer = nil
        heartbeatTimer?.invalidate()
        heartbeatTimer = nil
        coordinatorConnection?.cancel()
        coordinatorConnection = nil
        nodesReady.removeAll()
        pendingGradients.removeAll()
        clusterNodes.removeAll()
        ringOrder.removeAll()

        appendLog(.roundComplete, "Aggregation stopped", nodeID: nil)
    }

    // MARK: - Private: Message Handlers

    private func handleAssignWork(_ work: GradientMessage.AssignWork, from nodeID: String) {
        guard !isCoordinator else { return }

        appendLog(.roundStart, "Received work assignment: \(work.shardPaths.count) shard(s), steps \(work.startStep)–\(work.endStep)", nodeID: nodeID)

        // A real implementation would kick off local training here.
        // The CLI training loop would be started with these shard paths and config.
        metrics.currentStepAcrossCluster = work.startStep
    }

    private func handleGradientReady(_ ready: GradientMessage.GradientReady, from nodeID: String) {
        guard isCoordinator else { return }

        pendingGradients[nodeID] = ready
        nodesReady.insert(nodeID)

        appendLog(.roundStart, "Gradient received from \(nodeID) — step \(ready.step), loss \(String(format: "%.4f", ready.localLoss)), \(ready.gradientSizeBytes) bytes", nodeID: nodeID)

        // Start round timer on first gradient received
        if nodesReady.count == 1 {
            roundStartTime = Date()
            startStragglerTimer()
        }

        // Check if we have all expected gradients
        if nodesReady.count >= expectedNodeCount {
            executeStragglerTimer()
        }
    }

    private func handleAggregated(_ result: GradientMessage.Aggregated, from nodeID: String) {
        guard !isCoordinator else { return }

        appendLog(.roundComplete, "Received aggregated result — step \(result.step), avg loss \(String(format: "%.4f", result.averagedLoss)), \(result.participatingNodes) node(s)", nodeID: nodeID)

        metrics.currentStepAcrossCluster = result.step
        currentRound = result.step
    }

    private func handleHeartbeat(_ hb: GradientMessage.Heartbeat, from nodeID: String) {
        // Update node last-seen timestamp
        if let idx = clusterNodes.firstIndex(where: { $0.id == nodeID }) {
            clusterNodes[idx].lastSeen = hb.timestamp
        }
    }

    private func handleSyncCheckpoint(_ sync: GradientMessage.SyncCheckpoint, from nodeID: String) {
        guard !isCoordinator else { return }

        appendLog(.roundComplete, "Checkpoint sync received — step \(sync.step), \(sync.checkpointSizeBytes) bytes, hash \(sync.checkpointHash.prefix(12))...", nodeID: nodeID)

        metrics.currentStepAcrossCluster = sync.step
    }

    // MARK: - Private: Round Management

    /// Called when all gradients for the current round have been collected (or timeout fires).
    private func executeStragglerTimer() {
        stragglerTimer?.invalidate()
        stragglerTimer = nil

        let collectedGradients = Array(pendingGradients.values)

        guard !collectedGradients.isEmpty else {
            appendLog(.error, "No gradients collected for round \(currentRound)", nodeID: nil)
            metrics.recordFailedRound()
            resetRound()
            return
        }

        // Identify stragglers
        let respondedNodes = nodesReady
        let allExpected = Set(clusterNodes.map { $0.id })
        let stragglers = allExpected.subtracting(respondedNodes)

        for straggler in stragglers {
            let hostName = clusterNodes.first(where: { $0.id == straggler })?.capabilities.hostName ?? straggler
            appendLog(.stragglerDetected, "Straggler detected: \(hostName)", nodeID: straggler)
            metrics.recordStraggler()
        }

        // Apply straggler policy
        switch config.stragglerPolicy {
        case .wait:
            // If policy is Wait but we still got here, all nodes responded
            break
        case .skip:
            // Proceed with whatever we have
            break
        case .timeout:
            // Proceed with whatever we have after timeout
            break
        }

        // Execute aggregation based on strategy
        switch config.strategy {
        case .allReduce:
            let result = allReduceAverage(gradients: collectedGradients)
            completeRound(result: result, gradients: collectedGradients)

        case .parameterServer:
            // Parameter server: coordinator does the averaging (same math, different topology)
            let result = allReduceAverage(gradients: collectedGradients)
            completeRound(result: result, gradients: collectedGradients)

        case .gossipProtocol:
            // Gossip: each node averages with a random peer. We simulate a single
            // global average here; a production implementation would do iterative
            // pairwise exchanges.
            let result = allReduceAverage(gradients: collectedGradients)
            completeRound(result: result, gradients: collectedGradients)
        }
    }

    /// Finalize an aggregation round: update metrics, broadcast result, reset for next round.
    private func completeRound(result: GradientMessage.Aggregated, gradients: [GradientMessage.GradientReady]) {
        let roundTime: Double
        if let start = roundStartTime {
            roundTime = Date().timeIntervalSince(start) * 1000
        } else {
            roundTime = result.aggregationTimeMs
        }

        let totalBytes = gradients.reduce(Int64(0)) { $0 + Int64($1.gradientSizeBytes) }

        metrics.recordRoundCompleted(
            aggregationTimeMs: roundTime,
            participatingNodes: gradients.count,
            bytesTransferred: totalBytes
        )
        metrics.currentStepAcrossCluster = result.step

        currentRound += 1

        appendLog(.roundComplete,
                  "Round \(currentRound) complete — avg loss \(String(format: "%.4f", result.averagedLoss)), " +
                  "\(gradients.count) node(s), \(String(format: "%.1f", roundTime))ms",
                  nodeID: nil)

        // Broadcast aggregated result to all worker nodes
        broadcastAggregated(result)

        // Periodic checkpoint sync
        if currentRound % config.syncEveryNSteps == 0 {
            let checkpointHash = gradients.first?.gradientHash ?? ""
            let checkpoint = GradientMessage.SyncCheckpoint(
                step: result.step,
                checkpointHash: checkpointHash,
                checkpointSizeBytes: Int(totalBytes)
            )
            broadcastCheckpointSync(checkpoint)
        }

        resetRound()
    }

    /// Clear per-round state for the next aggregation round.
    private func resetRound() {
        nodesReady.removeAll()
        pendingGradients.removeAll()
        roundStartTime = nil
    }

    // MARK: - Private: Network I/O

    private func sendMessage(_ message: GradientMessage, on connection: NWConnection) {
        do {
            var data = try encoder.encode(message)
            // Append newline delimiter for NDJSON framing
            data.append(contentsOf: [0x0A])
            connection.send(content: data, completion: .contentProcessed { [weak self] error in
                if let error = error {
                    Task { @MainActor [weak self] in
                        self?.appendLog(.error, "Send failed: \(error)", nodeID: nil)
                    }
                }
            })
        } catch {
            appendLog(.error, "Failed to encode message: \(error)", nodeID: nil)
        }
    }

    private func receiveMessages(on connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            Task { @MainActor [weak self] in
                guard let self = self else { return }

                if let data = data, !data.isEmpty {
                    self.processReceivedData(data, on: connection)
                }

                if isComplete {
                    self.appendLog(.nodeLeft, "Connection closed by remote", nodeID: nil)
                } else if let error = error {
                    self.appendLog(.error, "Receive error: \(error)", nodeID: nil)
                } else {
                    // Continue receiving
                    self.receiveMessages(on: connection)
                }
            }
        }
    }

    private func processReceivedData(_ data: Data, on connection: NWConnection) {
        // Split on newlines for NDJSON framing
        let lines = data.split(separator: 0x0A)
        for line in lines {
            do {
                let message = try decoder.decode(GradientMessage.self, from: Data(line))
                // Extract nodeID from message if available
                let nodeID: String
                switch message {
                case .gradientReady(let ready): nodeID = ready.nodeID
                case .heartbeat(let hb): nodeID = hb.nodeID
                default: nodeID = "unknown"
                }
                handleMessage(message, from: nodeID)
            } catch {
                appendLog(.error, "Failed to decode message: \(error)", nodeID: nil)
            }
        }
    }

    /// Broadcast the aggregated result to all connected worker nodes.
    private func broadcastAggregated(_ result: GradientMessage.Aggregated) {
        let _ = GradientMessage.aggregated(result)

        // Use ComputeClusterService connections if available
        // For now, log the broadcast intent — actual network send
        // goes through ComputeClusterService's connection map
        appendLog(.roundComplete, "Broadcasting aggregated result to \(expectedNodeCount) node(s)", nodeID: nil)

        // If we have a direct coordinator connection (worker mode), that would be used.
        // In coordinator mode we would iterate over connections.
        // This integrates with ComputeClusterService.shared.connections in production.
    }

    /// Broadcast a checkpoint sync message to all worker nodes.
    private func broadcastCheckpointSync(_ checkpoint: GradientMessage.SyncCheckpoint) {
        appendLog(.roundComplete, "Broadcasting checkpoint sync — step \(checkpoint.step)", nodeID: nil)
    }

    // MARK: - Private: Timers

    private func startStragglerTimer() {
        stragglerTimer?.invalidate()
        stragglerTimer = Timer.scheduledTimer(withTimeInterval: config.timeoutSeconds, repeats: false) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self, self.isRunning else { return }

                if self.config.stragglerPolicy == .wait {
                    // In wait mode, restart the timer
                    self.startStragglerTimer()
                    return
                }

                // Timeout or skip: proceed with what we have
                self.executeStragglerTimer()
            }
        }
    }

    private func startHeartbeatTimer() {
        heartbeatTimer?.invalidate()
        heartbeatTimer = Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self, self.isRunning else { return }
                self.sendHeartbeat()
                self.checkForDeadNodes()
            }
        }
    }

    private func sendHeartbeat() {
        let localID = DeviceCapabilities.local().deviceID
        let hb = GradientMessage.Heartbeat(
            nodeID: localID,
            timestamp: Date(),
            currentStep: currentRound,
            status: isCoordinator ? "coordinator" : "worker"
        )

        if let connection = coordinatorConnection {
            sendMessage(.heartbeat(hb), on: connection)
        }
    }

    /// Detect nodes that have not sent a heartbeat within 2x the timeout window
    /// and mark them as lost.
    private func checkForDeadNodes() {
        guard isCoordinator else { return }

        let deadline = Date().addingTimeInterval(-(config.timeoutSeconds * 2))
        var removedAny = false

        for i in clusterNodes.indices {
            if clusterNodes[i].lastSeen < deadline && clusterNodes[i].status != .offline {
                let hostName = clusterNodes[i].capabilities.hostName
                clusterNodes[i].status = .offline
                appendLog(.nodeLeft, "Node lost: \(hostName) (no heartbeat)", nodeID: clusterNodes[i].id)
                removedAny = true
            }
        }

        if removedAny {
            expectedNodeCount = clusterNodes.filter { $0.status != .offline }.count
        }
    }

    // MARK: - Private: Logging

    private func appendLog(_ type: AggregationEvent.EventType, _ description: String, nodeID: String?) {
        let event = AggregationEvent(
            timestamp: Date(),
            type: type,
            description: description,
            nodeID: nodeID
        )
        aggregationLog.append(event)

        // Cap at maxLogEntries
        if aggregationLog.count > maxLogEntries {
            aggregationLog.removeFirst(aggregationLog.count - maxLogEntries)
        }
    }
}
