// ProjectListView.swift — Sidebar project list

import SwiftUI

struct ProjectListView: View {
    @EnvironmentObject var projectManager: ProjectManager
    @Binding var selectedProject: NFProject?
    @State private var showNewProject = false

    var body: some View {
        List(selection: $selectedProject) {
            ForEach(projectManager.projects) { project in
                ProjectRow(project: project)
                    .tag(project)
                    .contextMenu {
                        Button("Delete", role: .destructive) {
                            if selectedProject?.id == project.id {
                                selectedProject = nil
                            }
                            projectManager.delete(project)
                        }
                    }
            }
        }
        .navigationTitle("Projects")
        .toolbar {
            ToolbarItem {
                Button(action: { showNewProject = true }) {
                    Image(systemName: "plus")
                }
            }
        }
        .sheet(isPresented: $showNewProject) {
            NewProjectSheet { name in
                let project = projectManager.create(name: name)
                selectedProject = project
            }
        }
    }
}

struct ProjectRow: View {
    let project: NFProject

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(project.name)
                .font(.headline)
            HStack(spacing: 8) {
                if !project.modelPath.isEmpty {
                    Label(URL(fileURLWithPath: project.modelPath).lastPathComponent, systemImage: "brain")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let trained = project.lastTrained {
                    Text(trained, style: .relative)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .padding(.vertical, 3)
    }
}

extension NFProject: Hashable {
    static func == (lhs: NFProject, rhs: NFProject) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}
