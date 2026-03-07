// ProjectDetailView.swift — Sidebar + detail navigation for project features

import SwiftUI

struct ProjectDetailView: View {
    @Binding var project: NFProject
    @State private var selectedTab = "dashboard"

    var body: some View {
        HSplitView {
            // Left sidebar with grouped navigation
            List(selection: $selectedTab) {
                Section("Training") {
                    sidebarItem("Dashboard", icon: "chart.line.uptrend.xyaxis", tag: "dashboard")
                    sidebarItem("Config", icon: "slider.horizontal.3", tag: "config")
                    sidebarItem("Data", icon: "doc.text", tag: "data")
                    sidebarItem("Generate", icon: "text.bubble", tag: "generate")
                    sidebarItem("Export", icon: "square.and.arrow.up", tag: "export")
                }

                Section("Data & Models") {
                    sidebarItem("Models", icon: "square.grid.2x2", tag: "models")
                    sidebarItem("Ingest", icon: "tray.and.arrow.down", tag: "ingest")
                    sidebarItem("History", icon: "clock.arrow.circlepath", tag: "history")
                    sidebarItem("Benchmarks", icon: "gauge.with.dots.needle.33percent", tag: "benchmarks")
                }

                Section("Tools") {
                    sidebarItem("Assistant", icon: "brain", tag: "assistant")
                    sidebarItem("Sync", icon: "arrow.triangle.2.circlepath", tag: "sync")
                    sidebarItem("Cluster", icon: "server.rack", tag: "cluster")
                }

                Section("Compliance") {
                    sidebarItem("Audit", icon: "shield.checkered", tag: "audit")
                    sidebarItem("Reports", icon: "doc.text.magnifyingglass", tag: "reports")
                }
            }
            .listStyle(.sidebar)
            .frame(minWidth: 160, idealWidth: 180, maxWidth: 220)

            // Right content area
            detailContent
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .navigationTitle(project.name)
    }

    private func sidebarItem(_ title: String, icon: String, tag: String) -> some View {
        Label(title, systemImage: icon)
            .tag(tag)
    }

    @ViewBuilder
    private var detailContent: some View {
        switch selectedTab {
        case "dashboard":   DashboardView(project: $project)
        case "config":      TrainingConfigView(project: $project)
        case "data":        DataImportView(project: $project)
        case "models":      ModelCardView(project: $project)
        case "ingest":      IngestView(project: $project)
        case "generate":    GenerateView(project: $project)
        case "assistant":   AssistantView(project: $project)
        case "audit":       AuditDashboardView(project: $project)
        case "reports":     ComplianceReportView(project: $project)
        case "sync":        SyncDashboardView(project: $project)
        case "cluster":     ComputeClusterView(project: $project)
        case "history":     TrainingHistoryView(project: $project)
        case "benchmarks":  BenchmarkView(project: $project)
        case "export":      ExportView(project: $project)
        default:            DashboardView(project: $project)
        }
    }
}
