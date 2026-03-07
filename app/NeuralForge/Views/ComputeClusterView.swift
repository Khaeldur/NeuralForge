// ComputeClusterView.swift — Distributed ANE compute cluster dashboard
// Displays local device capabilities, discovered nodes, shard distribution,
// and cluster-wide metrics for multi-Mac training coordination.

import SwiftUI

struct ComputeClusterView: View {
    @Binding var project: NFProject
    @ObservedObject private var cluster = ComputeClusterService.shared

    var body: some View {
        VStack(spacing: 0) {
            headerSection
            Divider()

            ScrollView {
                VStack(spacing: 16) {
                    localDeviceSection
                    clusterMetricsSection

                    if cluster.isBrowsing || !cluster.nodes.isEmpty {
                        discoveredNodesSection
                    }

                    if !cluster.shardDistribution.isEmpty {
                        shardDistributionSection
                    }

                    if !cluster.isBrowsing && cluster.nodes.isEmpty {
                        setupPrompt
                    }
                }
                .padding()
            }
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(spacing: 12) {
            HStack {
                Label("Compute Cluster", systemImage: "server.rack")
                    .font(.headline)
                Spacer()

                if cluster.isBrowsing || cluster.isAdvertising {
                    Button(action: { cluster.stopCluster() }) {
                        Label("Stop Cluster", systemImage: "stop.circle")
                    }
                    .buttonStyle(.bordered)
                    .foregroundColor(.red)
                } else {
                    Button(action: { cluster.startCluster() }) {
                        Label("Start Cluster", systemImage: "play.circle")
                    }
                    .buttonStyle(.borderedProminent)
                }
            }

            HStack(spacing: 16) {
                statusBadge
                Spacer()

                if cluster.isBrowsing {
                    HStack(spacing: 4) {
                        ProgressView()
                            .controlSize(.small)
                        Text("Scanning network...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .padding()
    }

    private var statusBadge: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
            Text(cluster.clusterStatus.description)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    private var statusColor: Color {
        switch cluster.clusterStatus {
        case .idle: return .secondary
        case .discovering: return .orange
        case .ready: return .green
        case .distributing: return .blue
        case .training: return .purple
        case .aggregating: return .blue
        case .error: return .red
        }
    }

    // MARK: - Local Device

    private var localDeviceSection: some View {
        GroupBox {
            HStack(spacing: 16) {
                Image(systemName: "desktopcomputer")
                    .font(.system(size: 32))
                    .foregroundColor(.blue)

                VStack(alignment: .leading, spacing: 4) {
                    Text(cluster.localCapabilities.hostName)
                        .font(.subheadline.bold())
                    Text(cluster.localCapabilities.chipName)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(cluster.localCapabilities.osVersion)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 4) {
                    capBadge("ANE", ComputeClusterService.formatTFLOPS(cluster.localCapabilities.tflopsEstimate), "bolt.fill", .orange)
                    capBadge("RAM", ComputeClusterService.formatMemory(cluster.localCapabilities.totalMemoryGB), "memorychip", .blue)
                    capBadge("CPU", "\(cluster.localCapabilities.cpuCores) cores", "cpu", .green)
                }
            }
            .padding(4)
        } label: {
            HStack {
                Label("This Mac", systemImage: "desktopcomputer")
                Spacer()
                Button(action: { cluster.refreshCapabilities() }) {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .controlSize(.small)
            }
        }
    }

    private func capBadge(_ label: String, _ value: String, _ icon: String, _ color: Color) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption2)
                .foregroundColor(color)
            VStack(alignment: .trailing, spacing: 0) {
                Text(value)
                    .font(.caption.bold().monospacedDigit())
                Text(label)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }

    // MARK: - Cluster Metrics

    private var clusterMetricsSection: some View {
        GroupBox {
            HStack(spacing: 24) {
                metricCard("Total TFLOPS",
                          ComputeClusterService.formatTFLOPS(cluster.totalClusterTFLOPS),
                          "bolt.fill", .orange)

                Divider().frame(height: 40)

                metricCard("Total Memory",
                          ComputeClusterService.formatMemory(cluster.totalClusterMemoryGB),
                          "memorychip", .blue)

                Divider().frame(height: 40)

                metricCard("Nodes",
                          "\(cluster.nodes.filter { $0.status != .offline }.count + 1)",
                          "server.rack", .green)

                Divider().frame(height: 40)

                metricCard("Available",
                          "\(cluster.nodes.filter { $0.status == .available || $0.status == .discovered }.count + 1)",
                          "checkmark.circle", .green)
            }
            .padding(8)
        } label: {
            Label("Cluster Resources", systemImage: "chart.bar")
        }
    }

    private func metricCard(_ label: String, _ value: String, _ icon: String, _ color: Color) -> some View {
        VStack(spacing: 6) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(color)
            Text(value)
                .font(.subheadline.bold().monospacedDigit())
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Discovered Nodes

    private var discoveredNodesSection: some View {
        GroupBox {
            if cluster.nodes.isEmpty {
                HStack {
                    Spacer()
                    VStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.regular)
                        Text("Searching for NeuralForge instances...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("Other Macs running NeuralForge on this network will appear here.")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    Spacer()
                }
            } else {
                LazyVStack(spacing: 1) {
                    ForEach(cluster.nodes) { node in
                        nodeRow(node)
                    }
                }
            }
        } label: {
            HStack {
                Label("Discovered Nodes (\(cluster.nodes.count))", systemImage: "network")
                Spacer()
                if !cluster.nodes.isEmpty {
                    Button(action: { cluster.startBrowsing() }) {
                        Image(systemName: "arrow.clockwise")
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                }
            }
        }
    }

    private func nodeRow(_ node: ClusterNode) -> some View {
        HStack(spacing: 10) {
            nodeStatusIcon(node.status)
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 2) {
                Text(node.capabilities.hostName)
                    .font(.caption.bold())
                Text(node.capabilities.chipName)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Text(ComputeClusterService.formatTFLOPS(node.capabilities.tflopsEstimate))
                .font(.caption2.monospacedDigit())
                .foregroundColor(.orange)

            Text(ComputeClusterService.formatMemory(node.capabilities.totalMemoryGB))
                .font(.caption2.monospacedDigit())
                .foregroundColor(.blue)

            Text("\(node.capabilities.cpuCores) CPU")
                .font(.caption2.monospacedDigit())
                .foregroundColor(.secondary)

            nodeStatusLabel(node.status)

            if !node.assignedShards.isEmpty {
                Text("\(node.assignedShards.count) shard(s)")
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color.blue.opacity(0.2)))
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    private func nodeStatusIcon(_ status: ClusterNode.NodeStatus) -> some View {
        Group {
            switch status {
            case .discovered:
                Image(systemName: "circle.dotted")
                    .foregroundColor(.orange)
            case .available:
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            case .training:
                ProgressView()
                    .controlSize(.mini)
            case .syncing:
                Image(systemName: "arrow.triangle.2.circlepath")
                    .foregroundColor(.blue)
            case .error:
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.red)
            case .offline:
                Image(systemName: "circle.slash")
                    .foregroundColor(.secondary)
            }
        }
        .font(.caption)
    }

    private func nodeStatusLabel(_ status: ClusterNode.NodeStatus) -> some View {
        Text(status.rawValue.capitalized)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                Capsule().fill(nodeStatusColor(status).opacity(0.15))
            )
            .foregroundColor(nodeStatusColor(status))
    }

    private func nodeStatusColor(_ status: ClusterNode.NodeStatus) -> Color {
        switch status {
        case .discovered: return .orange
        case .available: return .green
        case .training: return .purple
        case .syncing: return .blue
        case .error: return .red
        case .offline: return .secondary
        }
    }

    // MARK: - Shard Distribution

    private var shardDistributionSection: some View {
        GroupBox {
            LazyVStack(spacing: 8) {
                ForEach(Array(cluster.shardDistribution.keys.sorted()), id: \.self) { nodeID in
                    if let shards = cluster.shardDistribution[nodeID] {
                        let nodeName = nodeDisplayName(nodeID)
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Image(systemName: "desktopcomputer")
                                    .font(.caption)
                                    .foregroundColor(.blue)
                                Text(nodeName)
                                    .font(.caption.bold())
                                Spacer()
                                Text("\(shards.count) shard(s)")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }

                            // Shard bar
                            GeometryReader { geo in
                                let total = cluster.shardDistribution.values.reduce(0) { $0 + $1.count }
                                let fraction = total > 0 ? CGFloat(shards.count) / CGFloat(total) : 0
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(Color.blue.opacity(0.3))
                                    .frame(width: geo.size.width * fraction)
                            }
                            .frame(height: 8)
                            .background(RoundedRectangle(cornerRadius: 4)
                                .fill(Color(nsColor: .separatorColor).opacity(0.3)))
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                    }
                }
            }
        } label: {
            Label("Shard Distribution", systemImage: "rectangle.split.3x3")
        }
    }

    private func nodeDisplayName(_ nodeID: String) -> String {
        if nodeID == cluster.localCapabilities.deviceID {
            return "\(cluster.localCapabilities.hostName) (this Mac)"
        }
        return cluster.nodes.first { $0.id == nodeID }?.capabilities.hostName ?? nodeID
    }

    // MARK: - Setup Prompt

    private var setupPrompt: some View {
        VStack(spacing: 20) {
            Spacer()
            Image(systemName: "server.rack")
                .font(.system(size: 48))
                .foregroundColor(.secondary)

            Text("Distributed ANE Compute")
                .font(.title2.bold())
            Text("Combine multiple Macs for parallel training.\nEach Mac contributes its Neural Engine for faster throughput.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            VStack(alignment: .leading, spacing: 12) {
                featureRow("network", "Automatic device discovery via Bonjour")
                featureRow("bolt.fill", "Neural Engine TFLOPS aggregation")
                featureRow("rectangle.split.3x3", "Automatic data shard distribution")
                featureRow("arrow.triangle.2.circlepath", "Gradient aggregation across nodes")
            }
            .padding()
            .background(RoundedRectangle(cornerRadius: 12)
                .fill(Color(nsColor: .controlBackgroundColor)))

            Button(action: { cluster.startCluster() }) {
                Label("Start Cluster", systemImage: "play.circle")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)

            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding()
    }

    private func featureRow(_ icon: String, _ text: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 24)
            Text(text)
                .font(.caption)
        }
    }
}
