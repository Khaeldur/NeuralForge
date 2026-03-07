// NeuralForgeApp.swift — App entry point

import SwiftUI

@main
struct NeuralForgeApp: App {
    @StateObject private var projectManager = ProjectManager()
    @StateObject private var cliRunner = CLIRunner()
    @AppStorage("onboardingComplete") private var onboardingComplete = false

    var body: some Scene {
        WindowGroup {
            if onboardingComplete {
                MainView()
                    .environmentObject(projectManager)
                    .environmentObject(cliRunner)
            } else {
                OnboardingView(isComplete: $onboardingComplete)
                    .environmentObject(projectManager)
                    .environmentObject(cliRunner)
            }
        }
        .defaultSize(width: onboardingComplete ? 1200 : 620, height: onboardingComplete ? 800 : 520)

        Settings {
            SettingsView()
                .environmentObject(cliRunner)
        }

        MenuBarExtra {
            MenuBarView(cliRunner: cliRunner)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: MenuBarManager.shared.statusIcon)
                if !MenuBarManager.shared.menuBarTitle.isEmpty {
                    Text(MenuBarManager.shared.menuBarTitle)
                }
            }
        }
        .menuBarExtraStyle(.window)
    }
}
