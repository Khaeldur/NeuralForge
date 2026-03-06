// NeuralForgeTests.swift — Unit tests for NeuralForge Swift code
//
// Build: swiftc -o test_swift -framework Foundation NeuralForgeTests.swift
// Run:   ./test_swift

import Foundation

// ============================================================
// Inline copies of the types we're testing (avoids Xcode target deps)
// ============================================================

// MARK: - CLIMessage (from TrainingProgress.swift)

enum CLIMessage: Decodable {
    case init_(InitMsg)
    case step(StepMsg)
    case batch(BatchMsg)
    case checkpoint(CheckpointMsg)
    case restart(RestartMsg)
    case done(DoneMsg)
    case error(ErrorMsg)
    case info(InfoMsg)

    struct InitMsg: Decodable {
        let params, layers, dim, hidden, heads, seq, vocab: Int
        let timestamp: Int?
    }
    struct StepMsg: Decodable {
        let step, total: Int
        let loss: Double
        let lr: Double
        let ms: Double
        let tflops_ane: Double
        let tflops_total: Double
    }
    struct BatchMsg: Decodable {
        let batch, step: Int
        let avg_loss, best_loss: Double
        let compile_ms, train_ms: Double
        let compiles: Int
    }
    struct CheckpointMsg: Decodable {
        let path: String
        let step: Int
        let loss: Double
    }
    struct RestartMsg: Decodable {
        let step, compiles: Int
        let reason: String
    }
    struct DoneMsg: Decodable {
        let total_steps: Int
        let final_loss: Double
        let total_time_s: Double
        let tflops_ane: Double
        let tflops_total: Double
    }
    struct ErrorMsg: Decodable {
        let message: String
        let code: Int
    }
    struct InfoMsg: Decodable {
        let key: String
        let value: AnyCodable
    }

    private enum CodingKeys: String, CodingKey { case type }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let type = try c.decode(String.self, forKey: .type)
        let sc = try decoder.singleValueContainer()
        switch type {
        case "init":       self = .init_(try sc.decode(InitMsg.self))
        case "step":       self = .step(try sc.decode(StepMsg.self))
        case "batch":      self = .batch(try sc.decode(BatchMsg.self))
        case "checkpoint": self = .checkpoint(try sc.decode(CheckpointMsg.self))
        case "restart":    self = .restart(try sc.decode(RestartMsg.self))
        case "done":       self = .done(try sc.decode(DoneMsg.self))
        case "error":      self = .error(try sc.decode(ErrorMsg.self))
        case "info":       self = .info(try sc.decode(InfoMsg.self))
        default: throw DecodingError.dataCorruptedError(forKey: .type, in: c, debugDescription: "Unknown type: \(type)")
        }
    }
}

struct AnyCodable: Decodable {
    let value: Any
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let v = try? c.decode(String.self) { value = v }
        else if let v = try? c.decode(Int.self) { value = v }
        else if let v = try? c.decode(Double.self) { value = v }
        else if let v = try? c.decode(Bool.self) { value = v }
        else { value = NSNull() }
    }
}

// MARK: - NFProject (from Project.swift)

struct NFProject: Identifiable, Codable {
    let id: UUID
    var name: String
    var modelPath: String
    var dataPath: String
    var checkpointPath: String
    var created: Date
    var lastTrained: Date?
    var config: TrainingConfig

    init(name: String) {
        self.id = UUID()
        self.name = name
        self.modelPath = ""
        self.dataPath = ""
        self.checkpointPath = ""
        self.created = Date()
        self.config = TrainingConfig()
    }
}

struct TrainingConfig: Codable {
    var totalSteps: Int = 10000
    var learningRate: Double = 3e-4
    var accumSteps: Int = 10
    var checkpointEvery: Int = 100
    var gradClipNorm: Double = 1.0
    var beta1: Double = 0.9
    var beta2: Double = 0.999
    var eps: Double = 1e-8
    var useANEExtras: Bool = true
    var seed: Int = 42
}

// ============================================================
// Test Framework
// ============================================================

var testsRun = 0
var testsPassed = 0

func test(_ name: String, _ body: () throws -> Bool) {
    testsRun += 1
    let label = "  TEST: \(name)..."
    do {
        if try body() {
            testsPassed += 1
            print("\(label) PASS")
        } else {
            print("\(label) FAIL")
        }
    } catch {
        print("\(label) FAIL (\(error))")
    }
}

// ============================================================
// Tests
// ============================================================

// MARK: - JSON Parsing Tests

func parseMessage(_ json: String) throws -> CLIMessage {
    let data = json.data(using: .utf8)!
    return try JSONDecoder().decode(CLIMessage.self, from: data)
}

print("\n=== NeuralForge Swift Tests ===\n")

print("[CLI Message Parsing]")

test("parse_step_message") {
    let msg = try parseMessage("""
    {"type":"step","step":5,"total":100,"loss":3.14,"lr":0.0003,"ms":42.0,"tflops_ane":1.5,"tflops_total":2.0}
    """)
    guard case .step(let s) = msg else { return false }
    return s.step == 5 && s.total == 100 && abs(s.loss - 3.14) < 0.01
}

test("parse_init_message") {
    let msg = try parseMessage("""
    {"type":"init","params":110000000,"layers":12,"dim":768,"hidden":2048,"heads":12,"seq":256,"vocab":32000,"timestamp":1234}
    """)
    guard case .init_(let m) = msg else { return false }
    return m.layers == 12 && m.dim == 768 && m.vocab == 32000
}

test("parse_batch_message") {
    let msg = try parseMessage("""
    {"type":"batch","batch":3,"step":30,"avg_loss":2.5,"best_loss":2.1,"compile_ms":100,"train_ms":500,"compiles":5}
    """)
    guard case .batch(let b) = msg else { return false }
    return b.batch == 3 && b.compiles == 5 && abs(b.avg_loss - 2.5) < 0.01
}

test("parse_checkpoint_message") {
    let msg = try parseMessage("""
    {"type":"checkpoint","path":"/tmp/ckpt.bin","step":50,"loss":2.5}
    """)
    guard case .checkpoint(let c) = msg else { return false }
    return c.step == 50 && c.path == "/tmp/ckpt.bin"
}

test("parse_restart_message") {
    let msg = try parseMessage("""
    {"type":"restart","step":50,"compiles":99,"reason":"compile_budget"}
    """)
    guard case .restart(let r) = msg else { return false }
    return r.step == 50 && r.reason == "compile_budget"
}

test("parse_done_message") {
    let msg = try parseMessage("""
    {"type":"done","total_steps":1000,"final_loss":1.5,"total_time_s":60.0,"tflops_ane":3.0,"tflops_total":4.0}
    """)
    guard case .done(let d) = msg else { return false }
    return d.total_steps == 1000 && abs(d.final_loss - 1.5) < 0.01
}

test("parse_error_message") {
    let msg = try parseMessage("""
    {"type":"error","message":"test error","code":42}
    """)
    guard case .error(let e) = msg else { return false }
    return e.message == "test error" && e.code == 42
}

test("parse_unknown_type_fails") {
    do {
        _ = try parseMessage("""
        {"type":"unknown_type","data":"hello"}
        """)
        return false // Should have thrown
    } catch {
        return true
    }
}

test("parse_invalid_json_fails") {
    do {
        _ = try parseMessage("not json at all")
        return false
    } catch {
        return true
    }
}

// MARK: - Project Tests

print("\n[Project Model]")

test("project_creation") {
    let p = NFProject(name: "Test Project")
    return p.name == "Test Project" && p.modelPath.isEmpty && p.config.totalSteps == 10000
}

test("project_config_defaults") {
    let cfg = TrainingConfig()
    return cfg.totalSteps == 10000
        && cfg.accumSteps == 10
        && cfg.seed == 42
        && abs(cfg.learningRate - 3e-4) < 1e-6
        && cfg.useANEExtras == true
}

test("project_codable_roundtrip") {
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601

    var p = NFProject(name: "Roundtrip Test")
    p.modelPath = "/path/to/model.bin"
    p.dataPath = "/path/to/data.bin"
    p.config.totalSteps = 500
    p.config.learningRate = 1e-4

    let data = try encoder.encode(p)
    let p2 = try decoder.decode(NFProject.self, from: data)

    return p2.name == "Roundtrip Test"
        && p2.modelPath == "/path/to/model.bin"
        && p2.config.totalSteps == 500
        && abs(p2.config.learningRate - 1e-4) < 1e-8
        && p2.id == p.id
}

test("project_json_structure") {
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    encoder.outputFormatting = .sortedKeys

    let p = NFProject(name: "JSON Test")
    let data = try encoder.encode(p)
    let json = String(data: data, encoding: .utf8)!

    // Should contain expected keys
    return json.contains("\"name\"")
        && json.contains("\"config\"")
        && json.contains("\"totalSteps\"")
        && json.contains("\"id\"")
}

// MARK: - ProjectManager File Tests

print("\n[ProjectManager I/O]")

test("project_save_load_roundtrip") {
    let tmpDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("nf_test_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmpDir) }

    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601

    // Save
    var p = NFProject(name: "FileTest")
    p.config.totalSteps = 999
    let file = tmpDir.appendingPathComponent("project.json")
    let data = try encoder.encode(p)
    try data.write(to: file, options: .atomic)

    // Load
    let loaded = try Data(contentsOf: file)
    let p2 = try decoder.decode(NFProject.self, from: loaded)

    return p2.name == "FileTest" && p2.config.totalSteps == 999 && p2.id == p.id
}

test("project_delete_removes_file") {
    let tmpDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("nf_test_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)

    let file = tmpDir.appendingPathComponent("project.json")
    try "{}".write(to: file, atomically: true, encoding: .utf8)

    // Delete directory
    try FileManager.default.removeItem(at: tmpDir)

    return !FileManager.default.fileExists(atPath: tmpDir.path)
}

// MARK: - TrainingConfig Optimizer Params

print("\n[Optimizer Config]")

test("config_optimizer_defaults") {
    let cfg = TrainingConfig()
    return abs(cfg.beta1 - 0.9) < 1e-6
        && abs(cfg.beta2 - 0.999) < 1e-6
        && abs(cfg.eps - 1e-8) < 1e-12
        && abs(cfg.gradClipNorm - 1.0) < 1e-6
}

test("config_optimizer_roundtrip") {
    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    var cfg = TrainingConfig()
    cfg.beta1 = 0.85
    cfg.beta2 = 0.995
    cfg.eps = 1e-7
    cfg.gradClipNorm = 0.5

    let data = try encoder.encode(cfg)
    let cfg2 = try decoder.decode(TrainingConfig.self, from: data)

    return abs(cfg2.beta1 - 0.85) < 1e-6
        && abs(cfg2.beta2 - 0.995) < 1e-6
        && abs(cfg2.eps - 1e-7) < 1e-12
        && abs(cfg2.gradClipNorm - 0.5) < 1e-6
}

test("config_all_fields_present") {
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    let cfg = TrainingConfig()
    let data = try encoder.encode(cfg)
    let json = String(data: data, encoding: .utf8)!

    return json.contains("\"beta1\"")
        && json.contains("\"beta2\"")
        && json.contains("\"eps\"")
        && json.contains("\"gradClipNorm\"")
        && json.contains("\"totalSteps\"")
        && json.contains("\"learningRate\"")
        && json.contains("\"accumSteps\"")
        && json.contains("\"checkpointEvery\"")
        && json.contains("\"useANEExtras\"")
        && json.contains("\"seed\"")
}

// MARK: - NDJSON Multi-line Parsing Test

print("\n[NDJSON Parsing]")

test("ndjson_multiline") {
    let ndjson = """
    {"type":"step","step":1,"total":10,"loss":5.0,"lr":0.0003,"ms":100,"tflops_ane":1.0,"tflops_total":1.5}
    {"type":"step","step":2,"total":10,"loss":4.5,"lr":0.0003,"ms":95,"tflops_ane":1.1,"tflops_total":1.6}
    {"type":"done","total_steps":2,"final_loss":4.5,"total_time_s":0.2,"tflops_ane":1.1,"tflops_total":1.6}
    """

    let decoder = JSONDecoder()
    var messages: [CLIMessage] = []

    for line in ndjson.components(separatedBy: "\n") where !line.trimmingCharacters(in: .whitespaces).isEmpty {
        if let data = line.data(using: .utf8) {
            if let msg = try? decoder.decode(CLIMessage.self, from: data) {
                messages.append(msg)
            }
        }
    }

    guard messages.count == 3 else { return false }
    guard case .step(let s1) = messages[0], s1.step == 1 else { return false }
    guard case .step(let s2) = messages[1], s2.step == 2 else { return false }
    guard case .done(let d) = messages[2], d.total_steps == 2 else { return false }
    return true
}

// MARK: - Security Tests — Malicious JSON Input

print("\n[Security — Malicious JSON]")

test("json_injection_in_path") {
    // Checkpoint path with embedded JSON injection attempt
    let msg = try parseMessage("""
    {"type":"checkpoint","path":"/tmp/test\\\"injected\\\"value","step":1,"loss":1.0}
    """)
    guard case .checkpoint(let c) = msg else { return false }
    return c.path.contains("injected")  // Escaped quotes in path should be parsed correctly
}

test("json_huge_numbers") {
    // Very large numbers should not crash the parser
    let msg = try parseMessage("""
    {"type":"step","step":2147483647,"total":2147483647,"loss":999999.999,"lr":1e-30,"ms":0.001,"tflops_ane":0.0,"tflops_total":0.0}
    """)
    guard case .step(let s) = msg else { return false }
    return s.step == 2147483647
}

test("json_negative_values") {
    // Negative values in step/loss should parse without crashing
    let msg = try parseMessage("""
    {"type":"step","step":-1,"total":100,"loss":-99.0,"lr":0.0003,"ms":0.0,"tflops_ane":0.0,"tflops_total":0.0}
    """)
    guard case .step(let s) = msg else { return false }
    return s.step == -1 && s.loss < 0
}

test("json_unicode_in_error") {
    // Unicode characters in error message
    let msg = try parseMessage("""
    {"type":"error","message":"error: файл не найден 文件未找到","code":1}
    """)
    guard case .error(let e) = msg else { return false }
    return e.message.contains("файл")
}

test("json_empty_strings") {
    let msg = try parseMessage("""
    {"type":"checkpoint","path":"","step":0,"loss":0.0}
    """)
    guard case .checkpoint(let c) = msg else { return false }
    return c.path.isEmpty && c.step == 0
}

test("json_special_chars_in_path") {
    let msg = try parseMessage("""
    {"type":"checkpoint","path":"/tmp/model (copy)/ckpt [v2].bin","step":1,"loss":1.0}
    """)
    guard case .checkpoint(let c) = msg else { return false }
    return c.path.contains("(copy)") && c.path.contains("[v2]")
}

// MARK: - Security Tests — Project Config Bounds

print("\n[Security — Config Bounds]")

test("config_extreme_values") {
    var cfg = TrainingConfig()
    cfg.totalSteps = -1
    cfg.learningRate = -999
    cfg.accumSteps = 0
    cfg.beta1 = 999
    // Should survive encode/decode without crash
    let encoder = JSONEncoder()
    let decoder = JSONDecoder()
    let data = try encoder.encode(cfg)
    let cfg2 = try decoder.decode(TrainingConfig.self, from: data)
    return cfg2.totalSteps == -1 && cfg2.learningRate == -999
}

test("project_very_long_name") {
    let longName = String(repeating: "A", count: 10000)
    let p = NFProject(name: longName)
    return p.name.count == 10000
}

test("project_special_chars_name") {
    let p = NFProject(name: "test\n\t\"special\"<script>alert(1)</script>")
    let encoder = JSONEncoder()
    let decoder = JSONDecoder()
    encoder.dateEncodingStrategy = .iso8601
    decoder.dateDecodingStrategy = .iso8601
    let data = try encoder.encode(p)
    let p2 = try decoder.decode(NFProject.self, from: data)
    return p2.name.contains("<script>") && p2.name.contains("\"special\"")
}

// MARK: - Stability Tests

print("\n[Stability]")

test("parse_100_messages_rapidly") {
    let decoder = JSONDecoder()
    var count = 0
    for i in 0..<100 {
        let json = "{\"type\":\"step\",\"step\":\(i),\"total\":100,\"loss\":5.0,\"lr\":0.0003,\"ms\":42.0,\"tflops_ane\":1.5,\"tflops_total\":2.0}"
        if let data = json.data(using: .utf8),
           let _ = try? decoder.decode(CLIMessage.self, from: data) {
            count += 1
        }
    }
    return count == 100
}

test("parse_message_missing_fields_fails") {
    // Missing required fields should throw, not crash
    do {
        _ = try parseMessage("{\"type\":\"step\",\"step\":1}")
        return false  // Should have thrown
    } catch {
        return true
    }
}

test("project_codable_empty_paths") {
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601

    var p = NFProject(name: "")
    p.modelPath = ""
    p.dataPath = ""
    p.checkpointPath = ""

    let data = try encoder.encode(p)
    let p2 = try decoder.decode(NFProject.self, from: data)
    return p2.name.isEmpty && p2.modelPath.isEmpty
}

test("ndjson_malformed_lines_skipped") {
    let ndjson = """
    {"type":"step","step":1,"total":10,"loss":5.0,"lr":0.0003,"ms":100,"tflops_ane":1.0,"tflops_total":1.5}
    NOT VALID JSON AT ALL
    {"type":"done","total_steps":1,"final_loss":5.0,"total_time_s":0.1,"tflops_ane":1.0,"tflops_total":1.5}
    """

    let decoder = JSONDecoder()
    var messages: [CLIMessage] = []
    for line in ndjson.components(separatedBy: "\n") where !line.trimmingCharacters(in: .whitespaces).isEmpty {
        if let data = line.data(using: .utf8),
           let msg = try? decoder.decode(CLIMessage.self, from: data) {
            messages.append(msg)
        }
    }
    // Should parse 2 valid messages, skip the garbage line
    return messages.count == 2
}

// Results
print("\n=== Results: \(testsPassed)/\(testsRun) passed ===\n")
exit(testsPassed == testsRun ? 0 : 1)
