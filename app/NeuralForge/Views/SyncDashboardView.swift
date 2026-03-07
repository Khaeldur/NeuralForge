// SyncDashboardView.swift — Checkpoint sync configuration, status, and shared model browser
import SwiftUI

struct SyncDashboardView: View {
    @Binding var project: NFProject
    @ObservedObject private var sync = SyncService.shared
    @State private var showAgentSheet = false
    @State private var showRestoreConfirm = false
    @State private var selectedRestore: SharedModelEntry?

    var body: some View {
        VStack(spacing: 0) {
            headerSection
            Divider()

            if sync.config.syncEnabled && !sync.config.syncDirectory.isEmpty {
                ScrollView {
                    VStack(spacing: 16) {
                        statusSection
                        sharedBrowserSection
                        syncHistorySection
                    }
                    .padding()
                }
            } else {
                setupPrompt
            }
        }
        .sheet(isPresented: $showAgentSheet) { agentSheet }
        .alert("Restore Checkpoint?", isPresented: $showRestoreConfirm) {
            Button("Restore", role: .destructive) {
                if let entry = selectedRestore {
                    _ = sync.restoreCheckpoint(entry, to: &project)
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            if let entry = selectedRestore {
                Text("Copy \(entry.name) from \(entry.projectName) to this project? This will replace the current checkpoint.")
            }
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(spacing: 12) {
            HStack {
                Label("Team Sync", systemImage: "arrow.triangle.2.circlepath")
                    .font(.headline)
                Spacer()

                if sync.config.syncEnabled {
                    Button(action: { sync.syncAllProjects() }) {
                        Label("Sync Now", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.bordered)
                    .disabled(isSyncing)

                    Button(action: { showAgentSheet = true }) {
                        Label("Agent", systemImage: sync.isAgentInstalled ? "clock.badge.checkmark" : "clock")
                    }
                    .buttonStyle(.bordered)
                }
            }

            HStack(spacing: 16) {
                // Sync directory
                HStack(spacing: 4) {
                    Image(systemName: "folder")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    if sync.config.syncDirectory.isEmpty {
                        Text("No sync directory configured")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    } else {
                        Text(abbreviatePath(sync.config.syncDirectory))
                            .font(.caption.monospaced())
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                }

                Spacer()

                Button(action: chooseSyncDirectory) {
                    Label(sync.config.syncDirectory.isEmpty ? "Set Sync Folder" : "Change",
                          systemImage: "folder.badge.gearshape")
                }
                .controlSize(.small)
            }
        }
        .padding()
    }

    // MARK: - Status Section

    private var statusSection: some View {
        GroupBox {
            HStack(spacing: 16) {
                statusIcon
                    .font(.system(size: 28))

                VStack(alignment: .leading, spacing: 4) {
                    Text(statusTitle)
                        .font(.subheadline.bold())
                    Text(sync.status.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    if let lastSync = sync.config.lastSyncDate {
                        Text("Last sync: \(lastSync, style: .relative) ago")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 4) {
                    HStack(spacing: 12) {
                        statBadge("Synced", "\(sync.config.syncedItems.count)", "externaldrive.fill")
                        statBadge("Shared", "\(sync.sharedModels.count)", "square.grid.2x2")
                        statBadge("Pending", "\(sync.pendingSync.count)", "clock")
                    }
                }
            }
            .padding(4)
        } label: {
            Label("Sync Status", systemImage: "arrow.triangle.2.circlepath")
        }
        .onAppear { sync.scanForPendingSync() }
    }

    private var statusIcon: some View {
        Group {
            switch sync.status {
            case .idle:
                Image(systemName: "arrow.triangle.2.circlepath")
                    .foregroundColor(.secondary)
            case .syncing:
                ProgressView()
                    .controlSize(.regular)
            case .success:
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            case .error:
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.red)
            }
        }
    }

    private var statusTitle: String {
        switch sync.status {
        case .idle: return "Ready to Sync"
        case .syncing(let msg): return msg
        case .success(let count, _): return "\(count) item(s) synced successfully"
        case .error(let msg): return "Sync Error: \(msg)"
        }
    }

    private func statBadge(_ label: String, _ value: String, _ icon: String) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption2)
                .foregroundColor(.blue)
            VStack(alignment: .leading, spacing: 0) {
                Text(value)
                    .font(.caption.bold().monospacedDigit())
                Text(label)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }

    // MARK: - Shared Browser

    private var sharedBrowserSection: some View {
        GroupBox {
            if sync.sharedModels.isEmpty {
                HStack {
                    Spacer()
                    VStack(spacing: 8) {
                        Image(systemName: "tray")
                            .font(.title2)
                            .foregroundColor(.secondary)
                        Text("No shared models yet")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("Sync checkpoints to populate the shared registry.")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    Spacer()
                }
            } else {
                LazyVStack(spacing: 1) {
                    ForEach(sync.sharedModels) { entry in
                        sharedModelRow(entry)
                    }
                }
            }
        } label: {
            HStack {
                Label("Shared Model Registry", systemImage: "square.grid.2x2")
                Spacer()
                Button(action: { sync.scanSharedDirectory() }) {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .controlSize(.small)
            }
        }
    }

    private func sharedModelRow(_ entry: SharedModelEntry) -> some View {
        HStack(spacing: 10) {
            Image(systemName: entry.isCheckpoint ? "externaldrive.fill" : "cube")
                .foregroundColor(entry.isCheckpoint ? .blue : .purple)
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 2) {
                Text(entry.name)
                    .font(.caption.bold())
                Text(entry.projectName)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Text(SyncService.formatSize(entry.sizeBytes))
                .font(.caption2.monospacedDigit())
                .foregroundColor(.secondary)

            Text(entry.modifiedDate, style: .date)
                .font(.caption2)
                .foregroundColor(.secondary)

            if entry.isCheckpoint {
                Button("Restore") {
                    selectedRestore = entry
                    showRestoreConfirm = true
                }
                .controlSize(.mini)
                .buttonStyle(.bordered)
            }

            Button(action: {
                NSWorkspace.shared.selectFile(entry.path, inFileViewerRootedAtPath: "")
            }) {
                Image(systemName: "folder")
            }
            .controlSize(.mini)
            .buttonStyle(.borderless)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: - Sync History

    private var syncHistorySection: some View {
        GroupBox {
            if sync.config.syncedItems.isEmpty {
                HStack {
                    Spacer()
                    Text("No sync history")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding()
                    Spacer()
                }
            } else {
                LazyVStack(spacing: 1) {
                    ForEach(sync.config.syncedItems.suffix(20).reversed(), id: \.sourcePath) { item in
                        HStack(spacing: 8) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                                .font(.caption)

                            Text(URL(fileURLWithPath: item.sourcePath).lastPathComponent)
                                .font(.caption.monospaced())

                            Text(item.projectName)
                                .font(.caption2)
                                .foregroundColor(.secondary)

                            if let step = item.checkpointStep {
                                Text("step \(step)")
                                    .font(.caption2)
                                    .foregroundColor(.blue)
                            }

                            Spacer()

                            Text(SyncService.formatSize(item.sizeBytes))
                                .font(.caption2.monospacedDigit())
                                .foregroundColor(.secondary)

                            Text(item.syncDate, style: .relative)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                    }
                }
            }
        } label: {
            Label("Sync History", systemImage: "clock.arrow.circlepath")
        }
    }

    // MARK: - Setup Prompt

    private var setupPrompt: some View {
        VStack(spacing: 20) {
            Spacer()
            Image(systemName: "arrow.triangle.2.circlepath")
                .font(.system(size: 48))
                .foregroundColor(.secondary)

            Text("Team Sync")
                .font(.title2.bold())
            Text("Sync checkpoints and models to a shared directory\nfor team collaboration and backup.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            VStack(alignment: .leading, spacing: 12) {
                featureRow("externaldrive.fill", "Automatic checkpoint backup to shared folder")
                featureRow("clock.badge.checkmark", "LaunchAgent for scheduled background sync")
                featureRow("square.grid.2x2", "Shared model registry across team members")
                featureRow("arrow.down.circle", "Restore checkpoints from shared directory")
            }
            .padding()
            .background(RoundedRectangle(cornerRadius: 12)
                .fill(Color(nsColor: .controlBackgroundColor)))

            Button(action: chooseSyncDirectory) {
                Label("Choose Sync Folder", systemImage: "folder.badge.gearshape")
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

    // MARK: - LaunchAgent Sheet

    private var agentSheet: some View {
        VStack(spacing: 20) {
            Image(systemName: sync.isAgentInstalled ? "clock.badge.checkmark.fill" : "clock")
                .font(.system(size: 48))
                .foregroundColor(sync.isAgentInstalled ? .green : .secondary)

            Text(sync.isAgentInstalled ? "LaunchAgent Installed" : "Install LaunchAgent")
                .font(.title2.bold())

            Text(sync.isAgentInstalled
                 ? "Checkpoints sync automatically every \(sync.config.intervalMinutes) minutes."
                 : "Install a LaunchAgent to sync checkpoints in the background.")
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)

            if !sync.isAgentInstalled {
                Stepper("Interval: \(sync.config.intervalMinutes) min",
                        value: $sync.config.intervalMinutes, in: 5...120, step: 5)
                    .frame(width: 250)
            }

            HStack(spacing: 12) {
                if sync.isAgentInstalled {
                    Button("Uninstall Agent") {
                        _ = sync.uninstallAgent()
                    }
                    .buttonStyle(.bordered)
                    .foregroundColor(.red)
                } else {
                    Button("Install Agent") {
                        sync.saveConfig()
                        _ = sync.installAgent()
                    }
                    .buttonStyle(.borderedProminent)
                }

                Button("Done") {
                    sync.saveConfig()
                    showAgentSheet = false
                }
                    .buttonStyle(.bordered)
            }

            if sync.isAgentInstalled {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Agent Label: \(SyncService.agentLabel)")
                        .font(.caption.monospaced())
                    Text("Plist: \(SyncService.agentPlistPath)")
                        .font(.caption2.monospaced())
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(RoundedRectangle(cornerRadius: 8)
                    .fill(Color(nsColor: .controlBackgroundColor)))
            }
        }
        .padding(30)
        .frame(width: 450, height: 380)
    }

    // MARK: - Helpers

    private var isSyncing: Bool {
        if case .syncing = sync.status { return true }
        return false
    }

    private func chooseSyncDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Choose Sync Folder"
        panel.message = "Select a directory for checkpoint sync (can be a network mount or shared folder)"

        if panel.runModal() == .OK, let url = panel.url {
            sync.setSyncDirectory(url.path)
        }
    }

    private func abbreviatePath(_ path: String) -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        if path.hasPrefix(home) {
            return "~" + path.dropFirst(home.count)
        }
        return path
    }
}
