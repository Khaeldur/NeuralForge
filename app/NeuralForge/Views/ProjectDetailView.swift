// ProjectDetailView.swift — Tab view for config, dashboard, export

import SwiftUI

struct ProjectDetailView: View {
    @Binding var project: NFProject
    @EnvironmentObject var cliRunner: CLIRunner
    @State private var selectedTab = "dashboard"

    var body: some View {
        TabView(selection: $selectedTab) {
            DashboardView(project: $project)
                .tabItem { Label("Dashboard", systemImage: "chart.line.uptrend.xyaxis") }
                .tag("dashboard")

            TrainingConfigView(project: $project)
                .tabItem { Label("Config", systemImage: "slider.horizontal.3") }
                .tag("config")

            DataImportView(project: $project)
                .tabItem { Label("Data", systemImage: "doc.text") }
                .tag("data")

            ExportView(project: $project)
                .tabItem { Label("Export", systemImage: "square.and.arrow.up") }
                .tag("export")
        }
        .padding()
        .navigationTitle(project.name)
    }
}
