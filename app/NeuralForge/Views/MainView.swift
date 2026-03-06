// MainView.swift — Sidebar + detail navigation

import SwiftUI

struct MainView: View {
    @EnvironmentObject var projectManager: ProjectManager
    @State private var selectedProject: NFProject?
    @State private var showNewProject = false

    var body: some View {
        NavigationSplitView {
            ProjectListView(selectedProject: $selectedProject)
                .navigationSplitViewColumnWidth(min: 200, ideal: 250)
        } detail: {
            if let project = selectedProject {
                ProjectDetailView(project: binding(for: project))
            } else {
                ContentUnavailableView {
                    Label("No Project Selected", systemImage: "cpu")
                } description: {
                    Text("Select or create a project to begin training.")
                } actions: {
                    Button("New Project") { showNewProject = true }
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

    private func binding(for project: NFProject) -> Binding<NFProject> {
        Binding(
            get: {
                projectManager.projects.first { $0.id == project.id } ?? project
            },
            set: { updated in
                projectManager.update(updated)
                selectedProject = updated
            }
        )
    }
}

struct NewProjectSheet: View {
    @Environment(\.dismiss) var dismiss
    @State private var name = ""
    let onCreate: (String) -> Void

    var body: some View {
        VStack(spacing: 20) {
            Text("New Project")
                .font(.title2.bold())
            TextField("Project name", text: $name)
                .textFieldStyle(.roundedBorder)
                .frame(width: 300)
                .onSubmit { create() }
            HStack {
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Create") { create() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(30)
    }

    private func create() {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        onCreate(trimmed)
        dismiss()
    }
}
