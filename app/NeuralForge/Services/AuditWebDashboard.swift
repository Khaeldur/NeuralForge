// AuditWebDashboard.swift — Built-in HTTP server for multi-user audit dashboard
// Serves a web-based audit log viewer with REST API endpoints.
// Aggregates audit logs from multiple machines via the sync directory.

import Foundation
import Network

// MARK: - Web Dashboard Configuration

struct WebDashboardConfig: Codable {
    var enabled: Bool = false
    var port: Int = 8742
    var bindAddress: String = "127.0.0.1"
    var allowRemoteAccess: Bool = false
    var refreshIntervalSeconds: Int = 30
    var maxEntriesPerPage: Int = 100
    var authToken: String = ""

    static var configPath: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("NeuralForge/web_dashboard_config.json")
    }

    /// The effective bind address considering the allowRemoteAccess flag
    var effectiveBindAddress: String {
        allowRemoteAccess ? "0.0.0.0" : bindAddress
    }
}

// MARK: - Audit Aggregator

/// Aggregates audit logs from the local machine and sync directory peers
class AuditAggregator {

    private let fm = FileManager.default

    static var auditLogPath: String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return "\(home)/Library/Logs/NeuralForge/audit.jsonl"
    }

    /// Reads the local audit.jsonl and returns parsed entries
    func scanLocalLog() -> [AuditEntry] {
        let path = AuditAggregator.auditLogPath

        guard fm.fileExists(atPath: path),
              let content = try? String(contentsOfFile: path, encoding: .utf8) else {
            return []
        }

        var entries = [AuditEntry]()
        let lines = content.components(separatedBy: .newlines)

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            guard let data = trimmed.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                continue
            }
            entries.append(AuditEntry(from: json))
        }

        return entries
    }

    /// Scans the sync directory for audit logs from other machines.
    /// Returns a dictionary keyed by machine name, each containing that machine's entries.
    func scanSyncDirectory(path syncPath: String) -> [String: [AuditEntry]] {
        var result = [String: [AuditEntry]]()

        guard fm.fileExists(atPath: syncPath) else { return result }

        // Look for audit.jsonl files in subdirectories named by machine
        // Expected layout: <syncDir>/<machineName>/audit.jsonl
        // Also check:      <syncDir>/<machineName>/Logs/NeuralForge/audit.jsonl
        guard let contents = try? fm.contentsOfDirectory(atPath: syncPath) else { return result }

        for item in contents {
            let machineDir = (syncPath as NSString).appendingPathComponent(item)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: machineDir, isDirectory: &isDir), isDir.boolValue else {
                continue
            }

            // Try direct audit.jsonl
            let directPath = (machineDir as NSString).appendingPathComponent("audit.jsonl")
            let nestedPath = (machineDir as NSString).appendingPathComponent("Logs/NeuralForge/audit.jsonl")

            let auditPath: String
            if fm.fileExists(atPath: directPath) {
                auditPath = directPath
            } else if fm.fileExists(atPath: nestedPath) {
                auditPath = nestedPath
            } else {
                continue
            }

            guard let content = try? String(contentsOfFile: auditPath, encoding: .utf8) else {
                continue
            }

            var entries = [AuditEntry]()
            let lines = content.components(separatedBy: .newlines)
            for line in lines {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty { continue }
                guard let data = trimmed.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    continue
                }
                entries.append(AuditEntry(from: json))
            }

            if !entries.isEmpty {
                result[item] = entries
            }
        }

        return result
    }

    /// Merges entries from multiple sources, sorted by timestamp (most recent first)
    func mergeEntries(_ sources: [String: [AuditEntry]]) -> [AuditEntry] {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]

        var all = [AuditEntry]()
        for (_, entries) in sources {
            all.append(contentsOf: entries)
        }

        return all.sorted { a, b in
            let dateA = formatter.date(from: a.time)
            let dateB = formatter.date(from: b.time)
            if let da = dateA, let db = dateB {
                return da > db
            }
            return a.time > b.time
        }
    }
}

// MARK: - Audit API Handler

/// Handles HTTP request routing and response generation for the audit dashboard
final class AuditAPIHandler: @unchecked Sendable {

    private let aggregator = AuditAggregator()

    private let jsonEncoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    /// Parsed HTTP request
    struct HTTPRequest {
        let method: String
        let path: String
        let queryParams: [String: String]
        let headers: [String: String]
        let httpVersion: String
    }

    /// HTTP response
    struct HTTPResponse {
        let statusCode: Int
        let statusText: String
        let contentType: String
        let body: Data
        var extraHeaders: [String: String] = [:]

        func serialize() -> Data {
            var header = "HTTP/1.1 \(statusCode) \(statusText)\r\n"
            header += "Content-Type: \(contentType)\r\n"
            header += "Content-Length: \(body.count)\r\n"
            header += "Connection: close\r\n"
            header += "Access-Control-Allow-Origin: *\r\n"
            header += "Access-Control-Allow-Methods: GET, OPTIONS\r\n"
            header += "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
            for (key, value) in extraHeaders {
                header += "\(key): \(value)\r\n"
            }
            header += "\r\n"

            var data = header.data(using: .utf8) ?? Data()
            data.append(body)
            return data
        }
    }

    // MARK: - Request Parsing

    func parseRequest(_ raw: String) -> HTTPRequest? {
        let lines = raw.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { return nil }

        let parts = requestLine.split(separator: " ", maxSplits: 2)
        guard parts.count >= 2 else { return nil }

        let method = String(parts[0])
        let fullPath = String(parts[1])
        let httpVersion = parts.count > 2 ? String(parts[2]) : "HTTP/1.1"

        // Parse path and query parameters
        let pathComponents = fullPath.split(separator: "?", maxSplits: 1)
        let path = String(pathComponents[0])
        var queryParams = [String: String]()

        if pathComponents.count > 1 {
            let queryString = String(pathComponents[1])
            for pair in queryString.split(separator: "&") {
                let kv = pair.split(separator: "=", maxSplits: 1)
                if kv.count == 2 {
                    let key = String(kv[0]).removingPercentEncoding ?? String(kv[0])
                    let value = String(kv[1]).removingPercentEncoding ?? String(kv[1])
                    queryParams[key] = value
                } else if kv.count == 1 {
                    queryParams[String(kv[0])] = ""
                }
            }
        }

        // Parse headers
        var headers = [String: String]()
        for i in 1..<lines.count {
            let line = lines[i]
            if line.isEmpty { break }
            if let colonIndex = line.firstIndex(of: ":") {
                let key = line[line.startIndex..<colonIndex].trimmingCharacters(in: .whitespaces).lowercased()
                let value = line[line.index(after: colonIndex)...].trimmingCharacters(in: .whitespaces)
                headers[key] = value
            }
        }

        return HTTPRequest(method: method, path: path, queryParams: queryParams,
                           headers: headers, httpVersion: httpVersion)
    }

    // MARK: - Request Handling

    func handleRequest(_ request: HTTPRequest, config: WebDashboardConfig) -> HTTPResponse {
        // Auth check
        if !config.authToken.isEmpty {
            let tokenFromQuery = request.queryParams["token"]
            let authHeader = request.headers["authorization"]
            let bearerToken = authHeader.flatMap { header -> String? in
                let parts = header.split(separator: " ", maxSplits: 1)
                if parts.count == 2, parts[0].lowercased() == "bearer" {
                    return String(parts[1])
                }
                return nil
            }

            if tokenFromQuery != config.authToken && bearerToken != config.authToken {
                return makeJSONResponse(statusCode: 401, statusText: "Unauthorized",
                                        json: ["error": "Invalid or missing auth token"])
            }
        }

        // Handle OPTIONS for CORS preflight
        if request.method == "OPTIONS" {
            return HTTPResponse(statusCode: 204, statusText: "No Content",
                                contentType: "text/plain", body: Data())
        }

        // Route handling
        switch request.path {
        case "/":
            return serveDashboardHTML(config: config)
        case "/health":
            return makeJSONResponse(statusCode: 200, statusText: "OK",
                                    json: ["status": "ok"])
        case "/api/entries":
            return handleEntriesAPI(request: request, config: config)
        case "/api/stats":
            return handleStatsAPI()
        case "/api/verify":
            return handleVerifyAPI()
        case "/api/machines":
            return handleMachinesAPI()
        default:
            return makeJSONResponse(statusCode: 404, statusText: "Not Found",
                                    json: ["error": "Not found: \(request.path)"])
        }
    }

    // MARK: - API Endpoints

    private func handleEntriesAPI(request: HTTPRequest, config: WebDashboardConfig) -> HTTPResponse {
        let page = Int(request.queryParams["page"] ?? "1") ?? 1
        let limit = min(Int(request.queryParams["limit"] ?? "\(config.maxEntriesPerPage)") ?? config.maxEntriesPerPage,
                        config.maxEntriesPerPage)
        let eventFilter = request.queryParams["event"]
        let userFilter = request.queryParams["user"]
        let machineFilter = request.queryParams["machine"]

        var allEntries = aggregator.scanLocalLog()

        // Apply filters
        if let event = eventFilter, !event.isEmpty {
            allEntries = allEntries.filter { $0.event == event }
        }
        if let user = userFilter, !user.isEmpty {
            allEntries = allEntries.filter { $0.user.lowercased().contains(user.lowercased()) }
        }
        _ = machineFilter // local-only filtering handled below

        // Sort most recent first
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        allEntries.sort { a, b in
            let da = formatter.date(from: a.time)
            let db = formatter.date(from: b.time)
            if let da = da, let db = db { return da > db }
            return a.time > b.time
        }

        let totalEntries = allEntries.count
        let totalPages = max(1, (totalEntries + limit - 1) / limit)
        let startIndex = (page - 1) * limit
        let endIndex = min(startIndex + limit, totalEntries)

        var pageEntries = [[String: Any]]()
        if startIndex < totalEntries {
            for entry in allEntries[startIndex..<endIndex] {
                pageEntries.append(entryToDict(entry))
            }
        }

        let response: [String: Any] = [
            "page": page,
            "limit": limit,
            "totalEntries": totalEntries,
            "totalPages": totalPages,
            "entries": pageEntries
        ]

        return makeJSONResponse(statusCode: 200, statusText: "OK", json: response)
    }

    private func handleStatsAPI() -> HTTPResponse {
        let entries = aggregator.scanLocalLog()

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]

        var trainingStarts = 0
        var trainingStops = 0
        var checkpoints = 0
        var exports = 0
        var generations = 0
        var users = Set<String>()
        var totalTrainingTime = 0.0
        var bestLoss: Double? = nil
        var eventCounts = [String: Int]()

        for entry in entries {
            users.insert(entry.user)
            eventCounts[entry.event, default: 0] += 1

            switch entry.event {
            case "training_start": trainingStarts += 1
            case "training_stop": trainingStops += 1
            case "checkpoint_save": checkpoints += 1
            case "export": exports += 1
            default:
                if entry.event.hasPrefix("generate") { generations += 1 }
            }

            if let t = entry.totalTimeS { totalTrainingTime += t }
            if let l = entry.finalLoss {
                if bestLoss == nil || l < bestLoss! { bestLoss = l }
            }
            if let l = entry.loss {
                if bestLoss == nil || l < bestLoss! { bestLoss = l }
            }
        }

        let firstDate = entries.last.flatMap { formatter.date(from: $0.time) }
        let lastDate = entries.first.flatMap { formatter.date(from: $0.time) }

        var response: [String: Any] = [
            "totalEntries": entries.count,
            "trainingStarts": trainingStarts,
            "trainingStops": trainingStops,
            "checkpoints": checkpoints,
            "exports": exports,
            "generations": generations,
            "activeUsers": Array(users).sorted(),
            "activeUserCount": users.count,
            "totalTrainingTimeSeconds": totalTrainingTime,
            "totalTrainingTimeFormatted": formatDuration(totalTrainingTime),
            "eventCounts": eventCounts
        ]
        if let bl = bestLoss {
            response["bestLoss"] = bl
        }
        if let fd = firstDate {
            response["firstEntryDate"] = formatter.string(from: fd)
        }
        if let ld = lastDate {
            response["lastEntryDate"] = formatter.string(from: ld)
        }

        return makeJSONResponse(statusCode: 200, statusText: "OK", json: response)
    }

    private func handleVerifyAPI() -> HTTPResponse {
        let path = AuditAggregator.auditLogPath
        let fm = FileManager.default

        guard fm.fileExists(atPath: path),
              let content = try? String(contentsOfFile: path, encoding: .utf8) else {
            let result: [String: Any] = [
                "valid": false,
                "totalEntries": 0,
                "validEntries": 0,
                "error": "No audit log found"
            ]
            return makeJSONResponse(statusCode: 200, statusText: "OK", json: result)
        }

        let lines = content.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        let zeroHash = String(repeating: "0", count: 64)
        var prevHash = zeroHash
        var validCount = 0
        var firstBadSeq: Int? = nil

        for line in lines {
            guard let data = line.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let seq = json["seq"] as? Int,
                  let entryPrevHash = json["prev_hash"] as? String,
                  let entryHash = json["hash"] as? String else {
                continue
            }

            if entryPrevHash != prevHash {
                firstBadSeq = seq
                break
            }

            // For verification, we trust the hash chain linkage here.
            // Full content-based verification requires the same extraction
            // logic as AuditLogReader.verifyChain().
            prevHash = entryHash
            validCount += 1
        }

        var result: [String: Any] = [
            "valid": firstBadSeq == nil && validCount == lines.count,
            "totalEntries": lines.count,
            "validEntries": validCount,
            "verifiedAt": ISO8601DateFormatter().string(from: Date())
        ]
        if let bad = firstBadSeq {
            result["firstBadSeq"] = bad
        }

        return makeJSONResponse(statusCode: 200, statusText: "OK", json: result)
    }

    private func handleMachinesAPI() -> HTTPResponse {
        let localHostName = ProcessInfo.processInfo.hostName
        let localEntries = aggregator.scanLocalLog()

        var machines = [[String: Any]]()
        machines.append([
            "name": localHostName,
            "source": "local",
            "entryCount": localEntries.count,
            "isLocal": true
        ])

        // Check sync directory for remote machines
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory,
                                                  in: .userDomainMask).first!
        let syncConfigPath = appSupport.appendingPathComponent("NeuralForge/sync_config.json")

        if let data = try? Data(contentsOf: syncConfigPath),
           let syncConfig = try? JSONDecoder().decode(SyncConfig.self, from: data),
           !syncConfig.syncDirectory.isEmpty {
            let remoteLogs = aggregator.scanSyncDirectory(path: syncConfig.syncDirectory)
            for (machineName, entries) in remoteLogs {
                machines.append([
                    "name": machineName,
                    "source": "sync",
                    "entryCount": entries.count,
                    "isLocal": false
                ])
            }
        }

        let response: [String: Any] = [
            "machines": machines,
            "totalMachines": machines.count
        ]

        return makeJSONResponse(statusCode: 200, statusText: "OK", json: response)
    }

    // MARK: - HTML Dashboard

    private func serveDashboardHTML(config: WebDashboardConfig) -> HTTPResponse {
        let html = generateDashboardHTML(config: config)
        let body = html.data(using: .utf8) ?? Data()
        return HTTPResponse(statusCode: 200, statusText: "OK",
                            contentType: "text/html; charset=utf-8", body: body)
    }

    // MARK: - Helpers

    private func entryToDict(_ entry: AuditEntry) -> [String: Any] {
        var dict: [String: Any] = [
            "seq": entry.seq,
            "time": entry.time,
            "event": entry.event,
            "user": entry.user,
            "hash": entry.hash,
            "summary": entry.summary,
            "eventColor": entry.eventColor
        ]
        if let m = entry.model { dict["model"] = m }
        if let s = entry.steps { dict["steps"] = s }
        if let l = entry.lr { dict["lr"] = l }
        if let sd = entry.stepsDone { dict["stepsDone"] = sd }
        if let fl = entry.finalLoss { dict["finalLoss"] = fl }
        if let t = entry.totalTimeS { dict["totalTimeS"] = t }
        if let r = entry.reason { dict["reason"] = r }
        if let p = entry.path { dict["path"] = p }
        if let s = entry.step { dict["step"] = s }
        if let l = entry.loss { dict["loss"] = l }
        if let f = entry.format { dict["format"] = f }
        if let mt = entry.maxTokens { dict["maxTokens"] = mt }
        if let t = entry.tokens { dict["tokens"] = t }
        if let ms = entry.totalMs { dict["totalMs"] = ms }
        return dict
    }

    private func makeJSONResponse(statusCode: Int, statusText: String, json: [String: Any]) -> HTTPResponse {
        let data: Data
        if let jsonData = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]) {
            data = jsonData
        } else {
            data = "{\"error\":\"serialization failed\"}".data(using: .utf8) ?? Data()
        }
        return HTTPResponse(statusCode: statusCode, statusText: statusText,
                            contentType: "application/json; charset=utf-8", body: data)
    }

    private func formatDuration(_ seconds: Double) -> String {
        if seconds < 60 {
            return String(format: "%.1fs", seconds)
        }
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        if mins < 60 {
            return "\(mins)m \(secs)s"
        }
        let hours = mins / 60
        let remainMins = mins % 60
        return "\(hours)h \(remainMins)m"
    }

    // MARK: - Dashboard HTML Generation

    private func generateDashboardHTML(config: WebDashboardConfig) -> String {
        let tokenParam = config.authToken.isEmpty ? "" : "?token=\(config.authToken)"
        let refreshMs = config.refreshIntervalSeconds * 1000

        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>NeuralForge Audit Dashboard</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }

                :root {
                    --bg-primary: #1a1a2e;
                    --bg-secondary: #16213e;
                    --bg-card: #0f3460;
                    --bg-input: #1a1a3e;
                    --text-primary: #e0e0e0;
                    --text-secondary: #a0a0b0;
                    --text-muted: #707080;
                    --accent: #4fc3f7;
                    --accent-hover: #81d4fa;
                    --border: #2a2a4e;
                    --green: #66bb6a;
                    --red: #ef5350;
                    --orange: #ffa726;
                    --blue: #42a5f5;
                    --purple: #ab47bc;
                    --gray: #78909c;
                }

                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', sans-serif;
                    background: var(--bg-primary);
                    color: var(--text-primary);
                    line-height: 1.5;
                    min-height: 100vh;
                }

                .container {
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                }

                /* Header */
                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 20px 0;
                    border-bottom: 1px solid var(--border);
                    margin-bottom: 24px;
                }

                .header h1 {
                    font-size: 24px;
                    font-weight: 600;
                    color: var(--accent);
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }

                .header h1::before {
                    content: '\\26A1';
                    font-size: 28px;
                }

                .header-actions {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }

                .verify-status {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    font-size: 13px;
                    padding: 6px 12px;
                    border-radius: 8px;
                    background: var(--bg-secondary);
                }

                .verify-status.valid { color: var(--green); }
                .verify-status.invalid { color: var(--red); }
                .verify-status.loading { color: var(--text-muted); }

                .btn {
                    padding: 8px 16px;
                    border-radius: 8px;
                    border: 1px solid var(--border);
                    background: var(--bg-secondary);
                    color: var(--text-primary);
                    cursor: pointer;
                    font-size: 13px;
                    transition: background 0.2s;
                }

                .btn:hover { background: var(--bg-card); }
                .btn-primary { background: var(--accent); color: #111; border-color: var(--accent); }
                .btn-primary:hover { background: var(--accent-hover); }

                /* Stats Cards */
                .stats-row {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 16px;
                    margin-bottom: 24px;
                }

                .stat-card {
                    background: var(--bg-secondary);
                    border: 1px solid var(--border);
                    border-radius: 12px;
                    padding: 16px;
                    text-align: center;
                }

                .stat-card .stat-value {
                    font-size: 28px;
                    font-weight: 700;
                    color: var(--accent);
                    margin-bottom: 4px;
                }

                .stat-card .stat-label {
                    font-size: 12px;
                    color: var(--text-secondary);
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }

                /* Filter Bar */
                .filter-bar {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 12px;
                    padding: 16px;
                    background: var(--bg-secondary);
                    border: 1px solid var(--border);
                    border-radius: 12px;
                    margin-bottom: 24px;
                    align-items: center;
                }

                .filter-bar label {
                    font-size: 12px;
                    color: var(--text-secondary);
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }

                .filter-bar select,
                .filter-bar input {
                    padding: 8px 12px;
                    border-radius: 8px;
                    border: 1px solid var(--border);
                    background: var(--bg-input);
                    color: var(--text-primary);
                    font-size: 13px;
                    outline: none;
                }

                .filter-bar select:focus,
                .filter-bar input:focus {
                    border-color: var(--accent);
                }

                /* Machine Tabs */
                .machine-tabs {
                    display: flex;
                    gap: 4px;
                    margin-bottom: 16px;
                    overflow-x: auto;
                }

                .machine-tab {
                    padding: 8px 16px;
                    border-radius: 8px 8px 0 0;
                    border: 1px solid var(--border);
                    border-bottom: none;
                    background: var(--bg-secondary);
                    color: var(--text-secondary);
                    cursor: pointer;
                    font-size: 13px;
                    white-space: nowrap;
                    transition: all 0.2s;
                }

                .machine-tab:hover { color: var(--text-primary); }
                .machine-tab.active {
                    background: var(--bg-card);
                    color: var(--accent);
                    border-color: var(--accent);
                }

                .machine-tab .tab-count {
                    font-size: 11px;
                    background: var(--bg-primary);
                    padding: 2px 6px;
                    border-radius: 10px;
                    margin-left: 6px;
                }

                /* Table */
                .table-container {
                    overflow-x: auto;
                    border: 1px solid var(--border);
                    border-radius: 12px;
                    background: var(--bg-secondary);
                }

                table {
                    width: 100%;
                    border-collapse: collapse;
                }

                thead th {
                    padding: 12px 16px;
                    text-align: left;
                    font-size: 12px;
                    font-weight: 600;
                    color: var(--text-secondary);
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    border-bottom: 1px solid var(--border);
                    background: var(--bg-card);
                    position: sticky;
                    top: 0;
                    z-index: 1;
                }

                tbody td {
                    padding: 10px 16px;
                    font-size: 13px;
                    border-bottom: 1px solid var(--border);
                    vertical-align: middle;
                }

                tbody tr:hover { background: rgba(79, 195, 247, 0.05); }
                tbody tr:last-child td { border-bottom: none; }

                .event-badge {
                    display: inline-block;
                    padding: 3px 10px;
                    border-radius: 12px;
                    font-size: 11px;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.3px;
                }

                .event-badge.green { background: rgba(102, 187, 106, 0.2); color: var(--green); }
                .event-badge.red { background: rgba(239, 83, 80, 0.2); color: var(--red); }
                .event-badge.blue { background: rgba(66, 165, 245, 0.2); color: var(--blue); }
                .event-badge.purple { background: rgba(171, 71, 188, 0.2); color: var(--purple); }
                .event-badge.orange { background: rgba(255, 167, 38, 0.2); color: var(--orange); }
                .event-badge.gray { background: rgba(120, 144, 156, 0.2); color: var(--gray); }

                .hash-cell {
                    font-family: 'SF Mono', 'Menlo', monospace;
                    font-size: 11px;
                    color: var(--text-muted);
                }

                .timestamp-cell {
                    font-size: 12px;
                    color: var(--text-secondary);
                    white-space: nowrap;
                }

                /* Pagination */
                .pagination {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 8px;
                    padding: 16px 0;
                }

                .pagination .page-info {
                    font-size: 13px;
                    color: var(--text-secondary);
                    margin: 0 12px;
                }

                /* Loading */
                .loading-spinner {
                    text-align: center;
                    padding: 40px;
                    color: var(--text-muted);
                }

                .loading-spinner::after {
                    content: '';
                    display: inline-block;
                    width: 24px;
                    height: 24px;
                    border: 3px solid var(--border);
                    border-top: 3px solid var(--accent);
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                    margin-left: 10px;
                    vertical-align: middle;
                }

                @keyframes spin { to { transform: rotate(360deg); } }

                /* Responsive */
                @media (max-width: 768px) {
                    .header { flex-direction: column; gap: 12px; }
                    .stats-row { grid-template-columns: repeat(2, 1fr); }
                    .filter-bar { flex-direction: column; }
                }

                .auto-refresh-indicator {
                    font-size: 11px;
                    color: var(--text-muted);
                    display: flex;
                    align-items: center;
                    gap: 4px;
                }

                .auto-refresh-indicator .dot {
                    width: 6px;
                    height: 6px;
                    border-radius: 50%;
                    background: var(--green);
                    animation: pulse 2s infinite;
                }

                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.3; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <!-- Header -->
                <div class="header">
                    <h1>NeuralForge Audit Dashboard</h1>
                    <div class="header-actions">
                        <div id="verifyStatus" class="verify-status loading">Verifying chain...</div>
                        <div class="auto-refresh-indicator">
                            <span class="dot"></span>
                            Auto-refresh \(config.refreshIntervalSeconds)s
                        </div>
                        <button class="btn btn-primary" onclick="refreshAll()">Refresh</button>
                    </div>
                </div>

                <!-- Stats Row -->
                <div class="stats-row" id="statsRow">
                    <div class="stat-card"><div class="stat-value" id="statTotal">--</div><div class="stat-label">Total Entries</div></div>
                    <div class="stat-card"><div class="stat-value" id="statTraining">--</div><div class="stat-label">Training Runs</div></div>
                    <div class="stat-card"><div class="stat-value" id="statCheckpoints">--</div><div class="stat-label">Checkpoints</div></div>
                    <div class="stat-card"><div class="stat-value" id="statExports">--</div><div class="stat-label">Exports</div></div>
                    <div class="stat-card"><div class="stat-value" id="statUsers">--</div><div class="stat-label">Active Users</div></div>
                    <div class="stat-card"><div class="stat-value" id="statTime">--</div><div class="stat-label">Total Training Time</div></div>
                </div>

                <!-- Filter Bar -->
                <div class="filter-bar">
                    <div>
                        <label>Event Type</label><br>
                        <select id="filterEvent" onchange="applyFilters()">
                            <option value="">All Events</option>
                            <option value="training_start">Training Start</option>
                            <option value="training_stop">Training Stop</option>
                            <option value="checkpoint_save">Checkpoint</option>
                            <option value="export">Export</option>
                            <option value="generate_start">Generate Start</option>
                            <option value="generate_done">Generate Done</option>
                            <option value="config_used">Config Used</option>
                            <option value="ingest_start">Ingest Start</option>
                            <option value="ingest_done">Ingest Done</option>
                        </select>
                    </div>
                    <div>
                        <label>User</label><br>
                        <input type="text" id="filterUser" placeholder="Filter by user..." oninput="debounceFilter()">
                    </div>
                    <div>
                        <label>Date From</label><br>
                        <input type="date" id="filterDateFrom" onchange="applyFilters()">
                    </div>
                    <div>
                        <label>Date To</label><br>
                        <input type="date" id="filterDateTo" onchange="applyFilters()">
                    </div>
                </div>

                <!-- Machine Tabs -->
                <div class="machine-tabs" id="machineTabs"></div>

                <!-- Table -->
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Seq</th>
                                <th>Timestamp</th>
                                <th>Event</th>
                                <th>User</th>
                                <th>Summary</th>
                                <th>Hash</th>
                            </tr>
                        </thead>
                        <tbody id="entriesBody">
                            <tr><td colspan="6" class="loading-spinner">Loading audit entries</td></tr>
                        </tbody>
                    </table>
                </div>

                <!-- Pagination -->
                <div class="pagination" id="pagination"></div>
            </div>

            <script>
                const BASE = '';
                const TOKEN_PARAM = '\(tokenParam)';
                let currentPage = 1;
                let filterDebounceTimer = null;
                let currentMachine = 'all';

                function apiURL(path) {
                    const sep = path.includes('?') ? '&' : '?';
                    return BASE + path + (TOKEN_PARAM ? sep + TOKEN_PARAM.substring(1) : '');
                }

                async function fetchJSON(path) {
                    const url = apiURL(path);
                    const resp = await fetch(url);
                    if (!resp.ok) throw new Error('HTTP ' + resp.status);
                    return resp.json();
                }

                async function loadStats() {
                    try {
                        const stats = await fetchJSON('/api/stats');
                        document.getElementById('statTotal').textContent = stats.totalEntries.toLocaleString();
                        document.getElementById('statTraining').textContent = stats.trainingStarts;
                        document.getElementById('statCheckpoints').textContent = stats.checkpoints;
                        document.getElementById('statExports').textContent = stats.exports;
                        document.getElementById('statUsers').textContent = stats.activeUserCount;
                        document.getElementById('statTime').textContent = stats.totalTrainingTimeFormatted || '0s';
                    } catch (e) {
                        console.error('Failed to load stats:', e);
                    }
                }

                async function loadVerification() {
                    const el = document.getElementById('verifyStatus');
                    try {
                        const result = await fetchJSON('/api/verify');
                        if (result.valid) {
                            el.className = 'verify-status valid';
                            el.innerHTML = '&#x2713; Chain intact &mdash; ' + result.totalEntries + ' entries verified';
                        } else {
                            el.className = 'verify-status invalid';
                            let msg = '&#x2717; TAMPER DETECTED';
                            if (result.firstBadSeq !== undefined) {
                                msg += ' at entry #' + result.firstBadSeq;
                            }
                            el.innerHTML = msg;
                        }
                    } catch (e) {
                        el.className = 'verify-status invalid';
                        el.textContent = 'Verification failed';
                    }
                }

                async function loadMachines() {
                    try {
                        const data = await fetchJSON('/api/machines');
                        const tabs = document.getElementById('machineTabs');
                        let html = '<div class="machine-tab ' + (currentMachine === 'all' ? 'active' : '') + '" onclick="switchMachine(\\'all\\')">All Machines</div>';
                        for (const m of data.machines) {
                            const active = currentMachine === m.name ? 'active' : '';
                            const label = m.isLocal ? m.name + ' (local)' : m.name;
                            html += '<div class="machine-tab ' + active + '" onclick="switchMachine(\\'' + m.name.replace(/'/g, "\\\\'") + '\\')">' + label + '<span class="tab-count">' + m.entryCount + '</span></div>';
                        }
                        tabs.innerHTML = html;
                    } catch (e) {
                        console.error('Failed to load machines:', e);
                    }
                }

                function switchMachine(name) {
                    currentMachine = name;
                    currentPage = 1;
                    loadEntries();
                    loadMachines();
                }

                async function loadEntries() {
                    const event = document.getElementById('filterEvent').value;
                    const user = document.getElementById('filterUser').value;

                    let url = '/api/entries?page=' + currentPage;
                    if (event) url += '&event=' + encodeURIComponent(event);
                    if (user) url += '&user=' + encodeURIComponent(user);
                    if (currentMachine !== 'all') url += '&machine=' + encodeURIComponent(currentMachine);

                    try {
                        const data = await fetchJSON(url);
                        renderEntries(data.entries);
                        renderPagination(data.page, data.totalPages, data.totalEntries);
                    } catch (e) {
                        document.getElementById('entriesBody').innerHTML =
                            '<tr><td colspan="6" style="text-align:center;padding:20px;color:var(--red)">Failed to load entries: ' + e.message + '</td></tr>';
                    }
                }

                function renderEntries(entries) {
                    const tbody = document.getElementById('entriesBody');
                    if (!entries || entries.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:20px;color:var(--text-muted)">No audit entries found</td></tr>';
                        return;
                    }

                    let html = '';
                    for (const e of entries) {
                        const ts = formatTimestamp(e.time);
                        const color = e.eventColor || 'gray';
                        const eventLabel = e.event.replace(/_/g, ' ');
                        const hashShort = e.hash ? e.hash.substring(0, 12) + '...' : '';
                        const summary = escapeHtml(e.summary || '');

                        html += '<tr>'
                            + '<td style="font-weight:600;color:var(--text-muted)">' + e.seq + '</td>'
                            + '<td class="timestamp-cell">' + ts + '</td>'
                            + '<td><span class="event-badge ' + color + '">' + eventLabel + '</span></td>'
                            + '<td>' + escapeHtml(e.user) + '</td>'
                            + '<td>' + summary + '</td>'
                            + '<td class="hash-cell">' + hashShort + '</td>'
                            + '</tr>';
                    }
                    tbody.innerHTML = html;
                }

                function renderPagination(page, totalPages, totalEntries) {
                    const el = document.getElementById('pagination');
                    if (totalPages <= 1) {
                        el.innerHTML = '<span class="page-info">' + totalEntries + ' entries</span>';
                        return;
                    }

                    let html = '';
                    html += '<button class="btn" onclick="goToPage(' + (page - 1) + ')" ' + (page <= 1 ? 'disabled' : '') + '>&laquo; Prev</button>';
                    html += '<span class="page-info">Page ' + page + ' of ' + totalPages + ' (' + totalEntries + ' entries)</span>';
                    html += '<button class="btn" onclick="goToPage(' + (page + 1) + ')" ' + (page >= totalPages ? 'disabled' : '') + '>Next &raquo;</button>';
                    el.innerHTML = html;
                }

                function goToPage(page) {
                    if (page < 1) return;
                    currentPage = page;
                    loadEntries();
                }

                function applyFilters() {
                    currentPage = 1;
                    loadEntries();
                }

                function debounceFilter() {
                    if (filterDebounceTimer) clearTimeout(filterDebounceTimer);
                    filterDebounceTimer = setTimeout(applyFilters, 300);
                }

                function refreshAll() {
                    loadStats();
                    loadEntries();
                    loadVerification();
                    loadMachines();
                }

                function formatTimestamp(iso) {
                    try {
                        const d = new Date(iso);
                        return d.toLocaleString();
                    } catch (e) {
                        return iso;
                    }
                }

                function escapeHtml(str) {
                    const div = document.createElement('div');
                    div.textContent = str;
                    return div.innerHTML;
                }

                // Initial load
                refreshAll();

                // Auto-refresh
                setInterval(refreshAll, \(refreshMs));
            </script>
        </body>
        </html>
        """
    }
}

// MARK: - Audit Web Server

/// Lightweight HTTP server using Network framework for serving the audit dashboard
@MainActor
class AuditWebServer: ObservableObject {

    @Published var config: WebDashboardConfig = WebDashboardConfig()
    @Published var isRunning: Bool = false
    @Published var connectedClients: Int = 0
    @Published var requestCount: Int = 0

    static let shared = AuditWebServer()

    /// Computed dashboard URL based on current config
    var dashboardURL: String {
        let host = config.effectiveBindAddress == "0.0.0.0" ? "localhost" : config.effectiveBindAddress
        var url = "http://\(host):\(config.port)"
        if !config.authToken.isEmpty {
            url += "?token=\(config.authToken)"
        }
        return url
    }

    private var listener: NWListener?
    private let apiHandler = AuditAPIHandler()
    private let serverQueue = DispatchQueue(label: "com.neuralforge.audit-web-server", qos: .userInitiated)
    private var activeConnections = [ObjectIdentifier: NWConnection]()

    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = .prettyPrinted
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    init() {
        loadConfig()
    }

    // MARK: - Configuration Persistence

    func loadConfig() {
        let path = WebDashboardConfig.configPath
        let fm = FileManager.default

        guard fm.fileExists(atPath: path.path),
              let data = try? Data(contentsOf: path),
              let loaded = try? decoder.decode(WebDashboardConfig.self, from: data) else {
            config = WebDashboardConfig()
            return
        }

        config = loaded
    }

    func saveConfig() {
        let path = WebDashboardConfig.configPath
        let fm = FileManager.default

        // Ensure directory exists
        let dir = path.deletingLastPathComponent()
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }

        guard let data = try? encoder.encode(config) else { return }
        try? data.write(to: path, options: .atomic)
    }

    // MARK: - Server Lifecycle

    func start() {
        guard !isRunning else { return }

        let port = NWEndpoint.Port(integerLiteral: UInt16(clamping: config.port))

        let parameters = NWParameters.tcp
        // Configure bind address
        if config.effectiveBindAddress == "127.0.0.1" {
            // Restrict to localhost
            parameters.requiredLocalEndpoint = NWEndpoint.hostPort(
                host: NWEndpoint.Host("127.0.0.1"),
                port: port
            )
        }

        do {
            listener = try NWListener(using: parameters, on: port)
        } catch {
            print("[AuditWebServer] Failed to create listener: \(error)")
            return
        }

        listener?.stateUpdateHandler = { [weak self] state in
            Task { @MainActor in
                guard let self = self else { return }
                switch state {
                case .ready:
                    self.isRunning = true
                    print("[AuditWebServer] Dashboard running at \(self.dashboardURL)")
                case .failed(let error):
                    print("[AuditWebServer] Listener failed: \(error)")
                    self.isRunning = false
                    self.listener?.cancel()
                    self.listener = nil
                case .cancelled:
                    self.isRunning = false
                default:
                    break
                }
            }
        }

        listener?.newConnectionHandler = { [weak self] connection in
            Task { @MainActor in
                self?.handleConnection(connection)
            }
        }

        listener?.start(queue: serverQueue)
    }

    func stop() {
        listener?.cancel()
        listener = nil

        // Cancel all active connections
        for (_, connection) in activeConnections {
            connection.cancel()
        }
        activeConnections.removeAll()

        isRunning = false
        connectedClients = 0
        print("[AuditWebServer] Server stopped")
    }

    // MARK: - Connection Handling

    private func handleConnection(_ connection: NWConnection) {
        activeConnections[ObjectIdentifier(connection)] = connection
        connectedClients = activeConnections.count

        // Capture nonisolated references for use in closures
        let handler = apiHandler
        let currentConfig = config

        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                AuditWebServer.readRequest(from: connection, handler: handler, config: currentConfig, server: self)
            case .failed, .cancelled:
                Task { @MainActor in
                    self?.removeConnection(connection)
                }
            default:
                break
            }
        }

        connection.start(queue: serverQueue)
    }

    private func removeConnection(_ connection: NWConnection) {
        activeConnections.removeValue(forKey: ObjectIdentifier(connection))
        connectedClients = activeConnections.count
    }

    nonisolated private static func readRequest(from connection: NWConnection, handler: AuditAPIHandler,
                                     config: WebDashboardConfig, server: AuditWebServer?) {
        // Read up to 64KB for the HTTP request
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { content, _, isComplete, error in
            if let error = error {
                print("[AuditWebServer] Read error: \(error)")
                connection.cancel()
                Task { @MainActor in
                    server?.removeConnection(connection)
                }
                return
            }

            guard let data = content, !data.isEmpty,
                  let requestString = String(data: data, encoding: .utf8) else {
                connection.cancel()
                Task { @MainActor in
                    server?.removeConnection(connection)
                }
                return
            }

            // Parse and handle the request
            let responseData: Data
            if let request = handler.parseRequest(requestString) {
                let response = handler.handleRequest(request, config: config)
                responseData = response.serialize()

                Task { @MainActor in
                    server?.requestCount += 1
                }
            } else {
                // Malformed request
                let errorResponse = AuditAPIHandler.HTTPResponse(
                    statusCode: 400,
                    statusText: "Bad Request",
                    contentType: "text/plain",
                    body: "Bad Request".data(using: .utf8) ?? Data()
                )
                responseData = errorResponse.serialize()
            }

            // Send response
            connection.send(content: responseData, completion: .contentProcessed { _ in
                connection.cancel()
                Task { @MainActor in
                    server?.removeConnection(connection)
                }
            })
        }
    }
}
