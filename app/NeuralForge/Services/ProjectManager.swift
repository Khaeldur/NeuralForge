// ProjectManager.swift — CRUD for NeuralForge projects

import Foundation

@MainActor
class ProjectManager: ObservableObject {
    @Published var projects: [NFProject] = []

    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init() {
        encoder.outputFormatting = .prettyPrinted
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601
        loadAll()
    }

    var projectsDir: URL { NFProject.projectsDir }

    // MARK: – CRUD

    func create(name: String) -> NFProject {
        var project = NFProject(name: name)
        let dir = project.directory
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        project.checkpointPath = dir.appendingPathComponent("checkpoints/checkpoint.bin").path
        save(project)
        projects.append(project)
        return project
    }

    func save(_ project: NFProject) {
        let url = project.directory.appendingPathComponent("project.json")
        try? FileManager.default.createDirectory(at: project.directory, withIntermediateDirectories: true)
        if let data = try? encoder.encode(project) {
            try? data.write(to: url, options: .atomic)
        }
    }

    func delete(_ project: NFProject) {
        projects.removeAll { $0.id == project.id }
        try? FileManager.default.removeItem(at: project.directory)
    }

    func update(_ project: NFProject) {
        if let idx = projects.firstIndex(where: { $0.id == project.id }) {
            projects[idx] = project
            save(project)
        }
    }

    // MARK: – Loading

    private func loadAll() {
        let fm = FileManager.default
        try? fm.createDirectory(at: projectsDir, withIntermediateDirectories: true)

        guard let contents = try? fm.contentsOfDirectory(at: projectsDir,
                                                          includingPropertiesForKeys: nil) else { return }

        projects = contents.compactMap { dir -> NFProject? in
            let file = dir.appendingPathComponent("project.json")
            guard let data = try? Data(contentsOf: file) else { return nil }
            return try? decoder.decode(NFProject.self, from: data)
        }.sorted { $0.created > $1.created }
    }
}
