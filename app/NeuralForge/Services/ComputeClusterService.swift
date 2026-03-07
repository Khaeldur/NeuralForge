// ComputeClusterService.swift — Bonjour-based multi-Mac ANE compute cluster
// Discovers NeuralForge instances on the local network via Bonjour,
// advertises local device capabilities, and coordinates distributed training.

import Foundation
import Network

// MARK: - Device Capabilities

struct DeviceCapabilities: Codable, Identifiable {
    var id: String { deviceID }
    let deviceID: String          // unique per machine
    let hostName: String
    let chipName: String          // e.g. "Apple M4 Pro"
    let totalMemoryGB: Double
    let availableMemoryGB: Double
    let aneCoreClusters: Int      // Neural Engine core clusters
    let cpuCores: Int
    let gpuCores: Int
    let tflopsEstimate: Double    // estimated ANE TFLOPS
    let osVersion: String
    let neuralForgeVersion: String
    let timestamp: Date

    static func local() -> DeviceCapabilities {
        let processInfo = ProcessInfo.processInfo
        let physicalMem = Double(processInfo.physicalMemory) / (1024 * 1024 * 1024)

        // Get chip name via sysctl
        var chipName = "Unknown Apple Silicon"
        var size: size_t = 0
        if sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0 {
            var buffer = [CChar](repeating: 0, count: size)
            if sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0 {
                chipName = String(cString: buffer)
            }
        }

        // CPU core count
        let cpuCores = processInfo.processorCount

        // GPU cores estimate based on chip
        let gpuCores = Self.estimateGPUCores(chip: chipName)

        // ANE core cluster estimate
        let aneClusters = Self.estimateANEClusters(chip: chipName)

        // TFLOPS estimate based on chip family
        let tflops = Self.estimateTFLOPS(chip: chipName)

        // Available memory (approximation)
        var vmStats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &vmStats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        let availableMem: Double
        if result == KERN_SUCCESS {
            let pageSize = Double(vm_kernel_page_size)
            let freePages = Double(vmStats.free_count) + Double(vmStats.inactive_count)
            availableMem = (freePages * pageSize) / (1024 * 1024 * 1024)
        } else {
            availableMem = physicalMem * 0.5  // fallback estimate
        }

        let osVersion = processInfo.operatingSystemVersionString

        return DeviceCapabilities(
            deviceID: Self.machineID(),
            hostName: Host.current().localizedName ?? processInfo.hostName,
            chipName: chipName,
            totalMemoryGB: physicalMem,
            availableMemoryGB: availableMem,
            aneCoreClusters: aneClusters,
            cpuCores: cpuCores,
            gpuCores: gpuCores,
            tflopsEstimate: tflops,
            osVersion: osVersion,
            neuralForgeVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev",
            timestamp: Date()
        )
    }

    private static func machineID() -> String {
        // Use IOPlatformUUID as stable machine identifier
        let platformExpert = IOServiceGetMatchingService(
            kIOMainPortDefault,
            IOServiceMatching("IOPlatformExpertDevice")
        )
        guard platformExpert != 0 else {
            return UUID().uuidString  // fallback
        }
        defer { IOObjectRelease(platformExpert) }

        if let uuid = IORegistryEntryCreateCFProperty(
            platformExpert,
            "IOPlatformUUID" as CFString,
            kCFAllocatorDefault, 0
        )?.takeRetainedValue() as? String {
            return uuid
        }
        return UUID().uuidString
    }

    private static func estimateGPUCores(chip: String) -> Int {
        let lower = chip.lowercased()
        if lower.contains("m4 ultra") { return 80 }
        if lower.contains("m4 max") { return 40 }
        if lower.contains("m4 pro") { return 20 }
        if lower.contains("m4") { return 10 }
        if lower.contains("m3 ultra") { return 76 }
        if lower.contains("m3 max") { return 40 }
        if lower.contains("m3 pro") { return 18 }
        if lower.contains("m3") { return 10 }
        if lower.contains("m2 ultra") { return 76 }
        if lower.contains("m2 max") { return 38 }
        if lower.contains("m2 pro") { return 19 }
        if lower.contains("m2") { return 10 }
        if lower.contains("m1 ultra") { return 64 }
        if lower.contains("m1 max") { return 32 }
        if lower.contains("m1 pro") { return 16 }
        if lower.contains("m1") { return 8 }
        return 10
    }

    private static func estimateANEClusters(chip: String) -> Int {
        let lower = chip.lowercased()
        if lower.contains("ultra") { return 2 * 16 }
        if lower.contains("max") { return 16 }
        if lower.contains("pro") { return 16 }
        return 16  // all Apple Silicon has 16 ANE cores
    }

    private static func estimateTFLOPS(chip: String) -> Double {
        let lower = chip.lowercased()
        if lower.contains("m4 ultra") { return 72.0 }
        if lower.contains("m4 max") { return 38.0 }
        if lower.contains("m4 pro") { return 38.0 }
        if lower.contains("m4") { return 38.0 }
        if lower.contains("m3 ultra") { return 31.6 }
        if lower.contains("m3 max") { return 15.8 }
        if lower.contains("m3 pro") { return 15.8 }
        if lower.contains("m3") { return 15.8 }
        if lower.contains("m2 ultra") { return 31.6 }
        if lower.contains("m2 max") { return 15.8 }
        if lower.contains("m2 pro") { return 15.8 }
        if lower.contains("m2") { return 15.8 }
        if lower.contains("m1 ultra") { return 22.0 }
        if lower.contains("m1 max") { return 11.0 }
        if lower.contains("m1 pro") { return 11.0 }
        if lower.contains("m1") { return 11.0 }
        return 10.0
    }
}

// MARK: - Cluster Node

struct ClusterNode: Identifiable {
    let id: String               // deviceID
    var capabilities: DeviceCapabilities
    var status: NodeStatus
    var endpoint: NWEndpoint?
    var lastSeen: Date
    var assignedShards: [String]  // shard filenames assigned to this node

    enum NodeStatus: String {
        case discovered  // found on network, not yet handshaked
        case available   // ready to accept work
        case training    // actively training assigned shards
        case syncing     // syncing checkpoint/gradients
        case error       // communication error
        case offline     // lost contact
    }
}

// MARK: - Cluster Service

@MainActor
class ComputeClusterService: ObservableObject {
    @Published var localCapabilities: DeviceCapabilities
    @Published var nodes: [ClusterNode] = []
    @Published var isAdvertising = false
    @Published var isBrowsing = false
    @Published var clusterStatus: ClusterStatus = .idle
    @Published var totalClusterTFLOPS: Double = 0
    @Published var totalClusterMemoryGB: Double = 0
    @Published var shardDistribution: [String: [String]] = [:]  // nodeID → [shard names]

    static let shared = ComputeClusterService()
    static let serviceType = "_neuralforge._tcp"
    static let serviceDomain = "local."

    private var browser: NWBrowser?
    private var listener: NWListener?
    private var connections: [String: NWConnection] = [:]  // nodeID → connection

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

    enum ClusterStatus: Equatable {
        case idle
        case discovering
        case ready(nodeCount: Int)
        case distributing
        case training(progress: String)
        case aggregating
        case error(String)

        var description: String {
            switch self {
            case .idle: return "Cluster inactive"
            case .discovering: return "Discovering nodes..."
            case .ready(let n): return "\(n) node(s) ready"
            case .distributing: return "Distributing shards..."
            case .training(let p): return p
            case .aggregating: return "Aggregating gradients..."
            case .error(let msg): return "Error: \(msg)"
            }
        }
    }

    init() {
        localCapabilities = DeviceCapabilities.local()
        updateClusterMetrics()
    }

    // MARK: - Service Advertisement (Bonjour Publisher)

    func startAdvertising() {
        guard !isAdvertising else { return }

        do {
            let params = NWParameters.tcp
            params.includePeerToPeer = true

            listener = try NWListener(using: params)
            listener?.service = NWListener.Service(
                name: localCapabilities.hostName,
                type: ComputeClusterService.serviceType,
                domain: ComputeClusterService.serviceDomain
            )

            // Advertise capabilities as TXT record metadata
            let txtData = buildTXTRecord()
            listener?.service = NWListener.Service(
                name: localCapabilities.hostName,
                type: ComputeClusterService.serviceType,
                domain: ComputeClusterService.serviceDomain,
                txtRecord: txtData
            )

            listener?.stateUpdateHandler = { [weak self] state in
                Task { @MainActor [weak self] in
                    switch state {
                    case .ready:
                        self?.isAdvertising = true
                    case .failed(let error):
                        self?.isAdvertising = false
                        self?.clusterStatus = .error("Listener failed: \(error)")
                    case .cancelled:
                        self?.isAdvertising = false
                    default:
                        break
                    }
                }
            }

            listener?.newConnectionHandler = { [weak self] connection in
                Task { @MainActor [weak self] in
                    self?.handleIncomingConnection(connection)
                }
            }

            listener?.start(queue: .main)
            isAdvertising = true
        } catch {
            clusterStatus = .error("Failed to start listener: \(error)")
        }
    }

    func stopAdvertising() {
        listener?.cancel()
        listener = nil
        isAdvertising = false
    }

    // MARK: - Service Discovery (Bonjour Browser)

    func startBrowsing() {
        guard !isBrowsing else { return }

        let params = NWParameters()
        params.includePeerToPeer = true

        browser = NWBrowser(for: .bonjour(
            type: ComputeClusterService.serviceType,
            domain: ComputeClusterService.serviceDomain
        ), using: params)

        browser?.stateUpdateHandler = { [weak self] state in
            Task { @MainActor [weak self] in
                switch state {
                case .ready:
                    self?.isBrowsing = true
                    self?.clusterStatus = .discovering
                case .failed(let error):
                    self?.isBrowsing = false
                    self?.clusterStatus = .error("Browser failed: \(error)")
                case .cancelled:
                    self?.isBrowsing = false
                default:
                    break
                }
            }
        }

        browser?.browseResultsChangedHandler = { [weak self] results, changes in
            Task { @MainActor [weak self] in
                self?.handleBrowseResults(results, changes: changes)
            }
        }

        browser?.start(queue: .main)
        clusterStatus = .discovering
    }

    func stopBrowsing() {
        browser?.cancel()
        browser = nil
        isBrowsing = false
        if clusterStatus == .discovering {
            clusterStatus = .idle
        }
    }

    // MARK: - Cluster Management

    func startCluster() {
        startAdvertising()
        startBrowsing()
    }

    func stopCluster() {
        stopBrowsing()
        stopAdvertising()
        for (_, conn) in connections {
            conn.cancel()
        }
        connections.removeAll()
        nodes.removeAll()
        clusterStatus = .idle
        updateClusterMetrics()
    }

    func refreshCapabilities() {
        localCapabilities = DeviceCapabilities.local()
        updateClusterMetrics()
    }

    // MARK: - Shard Distribution

    func distributeShards(_ shardPaths: [String]) {
        guard !nodes.isEmpty else {
            clusterStatus = .error("No nodes available for distribution")
            return
        }

        clusterStatus = .distributing

        let availableNodes = nodes.filter { $0.status == .available }
        guard !availableNodes.isEmpty else {
            clusterStatus = .error("No nodes in available state")
            return
        }

        // Round-robin distribution weighted by TFLOPS
        var distribution: [String: [String]] = [:]
        let totalTFLOPS = availableNodes.reduce(0.0) { $0 + $1.capabilities.tflopsEstimate }

        var shardIdx = 0
        for node in availableNodes {
            let weight = node.capabilities.tflopsEstimate / totalTFLOPS
            let shardCount = max(1, Int(Double(shardPaths.count) * weight))
            var nodeShards: [String] = []

            for _ in 0..<shardCount where shardIdx < shardPaths.count {
                nodeShards.append(shardPaths[shardIdx])
                shardIdx += 1
            }

            distribution[node.id] = nodeShards
        }

        // Assign any remaining shards to the strongest node
        if shardIdx < shardPaths.count, let strongest = availableNodes.max(by: { $0.capabilities.tflopsEstimate < $1.capabilities.tflopsEstimate }) {
            var remaining = distribution[strongest.id] ?? []
            while shardIdx < shardPaths.count {
                remaining.append(shardPaths[shardIdx])
                shardIdx += 1
            }
            distribution[strongest.id] = remaining
        }

        shardDistribution = distribution

        // Update node assignments
        for i in nodes.indices {
            nodes[i].assignedShards = distribution[nodes[i].id] ?? []
        }

        let nodeCount = availableNodes.count
        clusterStatus = .ready(nodeCount: nodeCount)
    }

    // MARK: - Private Helpers

    private func buildTXTRecord() -> NWTXTRecord {
        let dict: [String: String] = [
            "deviceID": localCapabilities.deviceID,
            "chip": localCapabilities.chipName,
            "memGB": String(format: "%.0f", localCapabilities.totalMemoryGB),
            "tflops": String(format: "%.1f", localCapabilities.tflopsEstimate),
            "cpuCores": "\(localCapabilities.cpuCores)",
            "gpuCores": "\(localCapabilities.gpuCores)",
            "version": localCapabilities.neuralForgeVersion,
        ]
        // Build TXT record from dictionary entries
        var data = Data()
        for (key, value) in dict {
            let entry = "\(key)=\(value)"
            let entryBytes = Array(entry.utf8)
            data.append(UInt8(entryBytes.count))
            data.append(contentsOf: entryBytes)
        }
        return NWTXTRecord(data)
    }

    private func handleBrowseResults(_ results: Set<NWBrowser.Result>, changes: Set<NWBrowser.Result.Change>) {
        for change in changes {
            switch change {
            case .added(let result):
                handleDiscoveredService(result)
            case .removed(let result):
                handleRemovedService(result)
            case .changed(old: _, new: let result, flags: _):
                handleDiscoveredService(result)
            case .identical:
                break
            @unknown default:
                break
            }
        }
        updateClusterMetrics()
    }

    private func handleDiscoveredService(_ result: NWBrowser.Result) {
        // Parse TXT record for capabilities
        guard case .bonjour(let txt) = result.metadata else { return }

        let deviceID = txt.dictionary["deviceID"] ?? UUID().uuidString
        // Skip self
        if deviceID == localCapabilities.deviceID { return }

        let chip = txt.dictionary["chip"] ?? "Unknown"
        let memGB = Double(txt.dictionary["memGB"] ?? "0") ?? 0
        let tflops = Double(txt.dictionary["tflops"] ?? "0") ?? 0
        let cpuCores = Int(txt.dictionary["cpuCores"] ?? "0") ?? 0
        let gpuCores = Int(txt.dictionary["gpuCores"] ?? "0") ?? 0
        let version = txt.dictionary["version"] ?? "unknown"

        var hostName = "Unknown"
        if case .service(let name, _, _, _) = result.endpoint {
            hostName = name
        }

        let caps = DeviceCapabilities(
            deviceID: deviceID,
            hostName: hostName,
            chipName: chip,
            totalMemoryGB: memGB,
            availableMemoryGB: memGB * 0.5,  // estimate
            aneCoreClusters: 16,
            cpuCores: cpuCores,
            gpuCores: gpuCores,
            tflopsEstimate: tflops,
            osVersion: "",
            neuralForgeVersion: version,
            timestamp: Date()
        )

        if let idx = nodes.firstIndex(where: { $0.id == deviceID }) {
            nodes[idx].capabilities = caps
            nodes[idx].lastSeen = Date()
            nodes[idx].endpoint = result.endpoint
            if nodes[idx].status == .offline {
                nodes[idx].status = .discovered
            }
        } else {
            let node = ClusterNode(
                id: deviceID,
                capabilities: caps,
                status: .discovered,
                endpoint: result.endpoint,
                lastSeen: Date(),
                assignedShards: []
            )
            nodes.append(node)
        }
    }

    private func handleRemovedService(_ result: NWBrowser.Result) {
        guard case .bonjour(let txt) = result.metadata else { return }
        let deviceID = txt.dictionary["deviceID"] ?? ""
        if let idx = nodes.firstIndex(where: { $0.id == deviceID }) {
            nodes[idx].status = .offline
        }
    }

    private func handleIncomingConnection(_ connection: NWConnection) {
        connection.stateUpdateHandler = { [weak self] state in
            Task { @MainActor [weak self] in
                switch state {
                case .ready:
                    // Connection established — read capabilities handshake
                    self?.readHandshake(connection)
                case .failed:
                    connection.cancel()
                default:
                    break
                }
            }
        }
        connection.start(queue: .main)
    }

    private func readHandshake(_ connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, _, error in
            Task { @MainActor [weak self] in
                guard let self = self, let data = data, error == nil else { return }
                if let caps = try? self.decoder.decode(DeviceCapabilities.self, from: data) {
                    self.connections[caps.deviceID] = connection
                    if let idx = self.nodes.firstIndex(where: { $0.id == caps.deviceID }) {
                        self.nodes[idx].capabilities = caps
                        self.nodes[idx].status = .available
                    }
                }
            }
        }
    }

    private func updateClusterMetrics() {
        let activeNodes = nodes.filter { $0.status != .offline && $0.status != .error }
        totalClusterTFLOPS = localCapabilities.tflopsEstimate
            + activeNodes.reduce(0.0) { $0 + $1.capabilities.tflopsEstimate }
        totalClusterMemoryGB = localCapabilities.totalMemoryGB
            + activeNodes.reduce(0.0) { $0 + $1.capabilities.totalMemoryGB }

        let readyCount = activeNodes.count
        if readyCount > 0 && clusterStatus == .discovering {
            clusterStatus = .ready(nodeCount: readyCount)
        }
    }

    // MARK: - Helpers

    static func formatTFLOPS(_ tflops: Double) -> String {
        if tflops >= 100 { return String(format: "%.0f TFLOPS", tflops) }
        return String(format: "%.1f TFLOPS", tflops)
    }

    static func formatMemory(_ gb: Double) -> String {
        if gb >= 100 { return String(format: "%.0f GB", gb) }
        return String(format: "%.1f GB", gb)
    }
}

// MARK: - NWTXTRecord Helpers

extension NWTXTRecord {
    /// Extract known keys from the TXT record as a dictionary
    var dictionary: [String: String] {
        var dict: [String: String] = [:]
        let knownKeys = ["deviceID", "chip", "memGB", "tflops", "cpuCores", "gpuCores", "version"]
        for key in knownKeys {
            if let value = getString(for: key) {
                dict[key] = value
            }
        }
        return dict
    }

    /// Get a string value for a TXT record key
    func getString(for key: String) -> String? {
        guard let entry = self.getEntry(for: key) else { return nil }
        if case .string(let value) = entry {
            return value
        }
        return nil
    }
}
