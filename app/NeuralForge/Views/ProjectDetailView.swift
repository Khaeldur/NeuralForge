// ProjectDetailView.swift — Tab view for config, dashboard, export

import SwiftUI

struct ProjectDetailView: View {
    @Binding var project: NFProject
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

            ModelCardView(project: $project)
                .tabItem { Label("Models", systemImage: "square.grid.2x2") }
                .tag("models")

            IngestView(project: $project)
                .tabItem { Label("Ingest", systemImage: "tray.and.arrow.down") }
                .tag("ingest")

            GenerateView(project: $project)
                .tabItem { Label("Generate", systemImage: "text.bubble") }
                .tag("generate")

            AssistantView(project: $project)
                .tabItem { Label("Assistant", systemImage: "brain") }
                .tag("assistant")

            AuditDashboardView(project: $project)
                .tabItem { Label("Audit", systemImage: "shield.checkered") }
                .tag("audit")

            ComplianceReportView(project: $project)
                .tabItem { Label("Reports", systemImage: "doc.text.magnifyingglass") }
                .tag("reports")

            SyncDashboardView(project: $project)
                .tabItem { Label("Sync", systemImage: "arrow.triangle.2.circlepath") }
                .tag("sync")

            ComputeClusterView(project: $project)
                .tabItem { Label("Cluster", systemImage: "server.rack") }
                .tag("cluster")

            TrainingHistoryView(project: $project)
                .tabItem { Label("History", systemImage: "clock.arrow.circlepath") }
                .tag("history")

            BenchmarkView(project: $project)
                .tabItem { Label("Benchmarks", systemImage: "gauge.with.dots.needle.33percent") }
                .tag("benchmarks")

            ExportView(project: $project)
                .tabItem { Label("Export", systemImage: "square.and.arrow.up") }
                .tag("export")
        }
        .padding()
        .navigationTitle(project.name)
    }
}
