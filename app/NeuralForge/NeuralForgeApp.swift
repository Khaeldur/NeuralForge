// NeuralForgeApp.swift — App entry point

import SwiftUI

@main
struct NeuralForgeApp: App {
    @StateObject private var projectManager = ProjectManager()
    @StateObject private var cliRunner = CLIRunner()

    var body: some Scene {
        WindowGroup {
            MainView()
                .environmentObject(projectManager)
                .environmentObject(cliRunner)
        }
        .defaultSize(width: 1200, height: 800)

        Settings {
            SettingsView()
                .environmentObject(cliRunner)
        }
    }
}

struct SettingsView: View {
    @EnvironmentObject var cliRunner: CLIRunner
    @State private var cliPath = ""

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
                    .onSubmit {
                        cliRunner.setCLIPath(cliPath)
                    }

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
                    Spacer()
                    Text(cliRunner.cliBinaryPath)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
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
        .padding()
        .frame(width: 500)
    }
}
