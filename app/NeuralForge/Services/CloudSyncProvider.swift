// CloudSyncProvider.swift — Cloud-based checkpoint sync (S3 + CloudKit)
// Extends the local SyncService with remote cloud backends.
// S3-compatible storage uses presigned URLs with AWS Signature V4.
// CloudKit uses CKContainer for iCloud-based sync.

import Foundation
import Security
import CommonCrypto

#if canImport(CloudKit)
import CloudKit
#endif

// MARK: - Cloud Sync Provider Protocol

/// Interface for cloud storage backends. All methods are async and throw on failure.
protocol CloudSyncProvider {
    var providerName: String { get }

    /// Upload a local file to a remote path.
    func upload(localPath: String, remotePath: String) async throws

    /// Download a remote file to a local path.
    func download(remotePath: String, localPath: String) async throws

    /// List remote files matching a prefix.
    func list(prefix: String) async throws -> [RemoteFileEntry]

    /// Delete a remote file.
    func delete(remotePath: String) async throws

    /// Test connectivity and credentials. Throws on failure.
    func testConnection() async throws
}

// MARK: - Remote File Entry

struct RemoteFileEntry: Identifiable, Codable {
    var id: String { remotePath }
    let name: String
    let remotePath: String
    let sizeBytes: Int64
    let lastModified: Date
    let provider: String
}

// MARK: - Cloud Sync Configuration

enum CloudProvider: String, Codable, CaseIterable {
    case none
    case s3
    case cloudkit
}

struct CloudSyncConfig: Codable {
    var provider: CloudProvider = .none
    var s3Endpoint: String = ""          // e.g. "https://s3.amazonaws.com"
    var s3Bucket: String = ""
    var s3Region: String = "us-east-1"
    // s3AccessKey and s3SecretKey are NOT stored here — Keychain only
    var cloudKitContainerID: String = "" // e.g. "iCloud.com.neuralforge.app"

    static var configPath: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("NeuralForge/cloud_sync_config.json")
    }
}

// MARK: - Cloud Sync Errors

enum CloudSyncError: LocalizedError {
    case notConfigured
    case credentialsMissing
    case uploadFailed(String)
    case downloadFailed(String)
    case listFailed(String)
    case deleteFailed(String)
    case connectionFailed(String)
    case invalidResponse(Int)
    case fileNotFound(String)

    var errorDescription: String? {
        switch self {
        case .notConfigured: return "Cloud sync not configured"
        case .credentialsMissing: return "Cloud credentials not found in Keychain"
        case .uploadFailed(let msg): return "Upload failed: \(msg)"
        case .downloadFailed(let msg): return "Download failed: \(msg)"
        case .listFailed(let msg): return "List failed: \(msg)"
        case .deleteFailed(let msg): return "Delete failed: \(msg)"
        case .connectionFailed(let msg): return "Connection failed: \(msg)"
        case .invalidResponse(let code): return "Server returned HTTP \(code)"
        case .fileNotFound(let path): return "File not found: \(path)"
        }
    }
}

// MARK: - Keychain Helpers (private to this file)

/// Internal keychain helpers for cloud sync credentials.
/// Uses the same keychain service as NFKeychain ("com.neuralforge.app")
/// with cloud-specific account identifiers.
private enum CloudKeychain {
    static let service = "com.neuralforge.app"

    static func save(account: String, value: String) -> Bool {
        guard let data = value.data(using: .utf8) else { return false }
        let delQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        SecItemDelete(delQuery as CFDictionary)
        let addQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String: data
        ]
        return SecItemAdd(addQuery as CFDictionary, nil) == errSecSuccess
    }

    static func load(account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data,
              let val = String(data: data, encoding: .utf8) else { return nil }
        return val
    }

    static func delete(account: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        return SecItemDelete(query as CFDictionary) == errSecSuccess
    }
}

// MARK: - S3 Sync Provider

/// S3-compatible cloud storage using presigned URLs with AWS Signature V4 (HMAC-SHA256).
/// Credentials are stored in Keychain via NFKeychain service ("com.neuralforge.app"),
/// with accounts "s3-access-key" and "s3-secret-key".
class S3SyncProvider: CloudSyncProvider {
    let providerName = "S3"

    private let endpoint: String   // e.g. "https://s3.us-east-1.amazonaws.com"
    private let bucket: String
    private let region: String
    private let session: URLSession

    private static let accessKeyAccount = "s3-access-key"
    private static let secretKeyAccount = "s3-secret-key"

    init(endpoint: String, bucket: String, region: String) {
        self.endpoint = endpoint.hasSuffix("/") ? String(endpoint.dropLast()) : endpoint
        self.bucket = bucket
        self.region = region

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        config.timeoutIntervalForResource = 600
        self.session = URLSession(configuration: config)
    }

    // MARK: - Credential Management

    /// Save S3 credentials to Keychain.
    static func saveCredentials(accessKey: String, secretKey: String) -> Bool {
        let a = CloudKeychain.save(account: accessKeyAccount, value: accessKey)
        let b = CloudKeychain.save(account: secretKeyAccount, value: secretKey)
        return a && b
    }

    /// Load the S3 access key from Keychain.
    static func loadAccessKey() -> String? {
        CloudKeychain.load(account: accessKeyAccount)
    }

    /// Load the S3 secret key from Keychain.
    static func loadSecretKey() -> String? {
        CloudKeychain.load(account: secretKeyAccount)
    }

    /// Delete all S3 credentials from Keychain.
    static func deleteCredentials() {
        _ = CloudKeychain.delete(account: accessKeyAccount)
        _ = CloudKeychain.delete(account: secretKeyAccount)
    }

    private func loadCredentials() throws -> (accessKey: String, secretKey: String) {
        guard let ak = S3SyncProvider.loadAccessKey(),
              let sk = S3SyncProvider.loadSecretKey(),
              !ak.isEmpty, !sk.isEmpty else {
            throw CloudSyncError.credentialsMissing
        }
        return (ak, sk)
    }

    // MARK: - CloudSyncProvider

    func upload(localPath: String, remotePath: String) async throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: localPath) else {
            throw CloudSyncError.fileNotFound(localPath)
        }

        let fileData = try Data(contentsOf: URL(fileURLWithPath: localPath))
        let creds = try loadCredentials()

        let cleanPath = remotePath.hasPrefix("/") ? String(remotePath.dropFirst()) : remotePath
        let encodedPath = cleanPath.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? cleanPath
        guard let url = URL(string: "\(endpoint)/\(bucket)/\(encodedPath)") else {
            throw CloudSyncError.uploadFailed("Invalid URL for path: \(remotePath)")
        }

        var request = try signedRequest(
            method: "PUT",
            url: url,
            headers: ["Content-Type": "application/octet-stream"],
            payloadHash: sha256Hex(fileData),
            credentials: creds
        )
        request.httpBody = fileData

        let (_, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw CloudSyncError.uploadFailed("Invalid response")
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            throw CloudSyncError.invalidResponse(httpResponse.statusCode)
        }
    }

    func download(remotePath: String, localPath: String) async throws {
        let creds = try loadCredentials()

        let cleanPath = remotePath.hasPrefix("/") ? String(remotePath.dropFirst()) : remotePath
        let encodedPath = cleanPath.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? cleanPath
        guard let url = URL(string: "\(endpoint)/\(bucket)/\(encodedPath)") else {
            throw CloudSyncError.downloadFailed("Invalid URL for path: \(remotePath)")
        }

        // Empty payload hash for GET requests
        let emptyHash = sha256Hex(Data())

        let request = try signedRequest(
            method: "GET",
            url: url,
            headers: [:],
            payloadHash: emptyHash,
            credentials: creds
        )

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw CloudSyncError.downloadFailed("Invalid response")
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            throw CloudSyncError.invalidResponse(httpResponse.statusCode)
        }

        // Ensure parent directory exists
        let parentDir = (localPath as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: parentDir, withIntermediateDirectories: true)

        try data.write(to: URL(fileURLWithPath: localPath), options: .atomic)
    }

    func list(prefix: String) async throws -> [RemoteFileEntry] {
        let creds = try loadCredentials()

        let cleanPrefix = prefix.hasPrefix("/") ? String(prefix.dropFirst()) : prefix
        var components = URLComponents(string: "\(endpoint)/\(bucket)")!
        components.queryItems = [
            URLQueryItem(name: "list-type", value: "2"),
            URLQueryItem(name: "prefix", value: cleanPrefix)
        ]
        guard let url = components.url else {
            throw CloudSyncError.listFailed("Invalid URL")
        }

        let emptyHash = sha256Hex(Data())

        let request = try signedRequest(
            method: "GET",
            url: url,
            headers: [:],
            payloadHash: emptyHash,
            credentials: creds
        )

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw CloudSyncError.listFailed("Invalid response")
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            throw CloudSyncError.invalidResponse(httpResponse.statusCode)
        }

        return parseListBucketResult(data)
    }

    func delete(remotePath: String) async throws {
        let creds = try loadCredentials()

        let cleanPath = remotePath.hasPrefix("/") ? String(remotePath.dropFirst()) : remotePath
        let encodedPath = cleanPath.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? cleanPath
        guard let url = URL(string: "\(endpoint)/\(bucket)/\(encodedPath)") else {
            throw CloudSyncError.deleteFailed("Invalid URL for path: \(remotePath)")
        }

        let emptyHash = sha256Hex(Data())

        let request = try signedRequest(
            method: "DELETE",
            url: url,
            headers: [:],
            payloadHash: emptyHash,
            credentials: creds
        )

        let (_, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw CloudSyncError.deleteFailed("Invalid response")
        }
        // S3 returns 204 on successful delete
        guard (200..<300).contains(httpResponse.statusCode) else {
            throw CloudSyncError.invalidResponse(httpResponse.statusCode)
        }
    }

    func testConnection() async throws {
        let creds = try loadCredentials()

        // HEAD request on the bucket to verify access
        guard let url = URL(string: "\(endpoint)/\(bucket)") else {
            throw CloudSyncError.connectionFailed("Invalid endpoint URL")
        }

        let emptyHash = sha256Hex(Data())

        let request = try signedRequest(
            method: "HEAD",
            url: url,
            headers: [:],
            payloadHash: emptyHash,
            credentials: creds
        )

        let (_, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw CloudSyncError.connectionFailed("Invalid response")
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            throw CloudSyncError.connectionFailed("HTTP \(httpResponse.statusCode)")
        }
    }

    // MARK: - AWS Signature V4

    /// Build a signed URLRequest using AWS Signature V4 (HMAC-SHA256).
    private func signedRequest(
        method: String,
        url: URL,
        headers: [String: String],
        payloadHash: String,
        credentials: (accessKey: String, secretKey: String)
    ) throws -> URLRequest {
        let now = Date()
        let dateFormatter = DateFormatter()
        dateFormatter.locale = Locale(identifier: "en_US_POSIX")
        dateFormatter.timeZone = TimeZone(identifier: "UTC")

        dateFormatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        let amzDate = dateFormatter.string(from: now)

        dateFormatter.dateFormat = "yyyyMMdd"
        let dateStamp = dateFormatter.string(from: now)

        let host = url.host ?? ""
        let path = url.path.isEmpty ? "/" : url.path
        let query = url.query ?? ""

        // Canonical query string: parameters sorted by name
        let canonicalQueryString = query
            .split(separator: "&")
            .sorted()
            .joined(separator: "&")

        // Build signed headers
        var allHeaders = headers
        allHeaders["Host"] = host
        allHeaders["x-amz-date"] = amzDate
        allHeaders["x-amz-content-sha256"] = payloadHash

        let signedHeaderKeys = allHeaders.keys
            .map { $0.lowercased() }
            .sorted()
        let signedHeadersString = signedHeaderKeys.joined(separator: ";")

        let canonicalHeaders = signedHeaderKeys
            .map { key in
                let value = allHeaders.first(where: { $0.key.lowercased() == key })!.value
                return "\(key):\(value.trimmingCharacters(in: .whitespaces))\n"
            }
            .joined()

        // Canonical request
        let canonicalRequest = [
            method,
            path,
            canonicalQueryString,
            canonicalHeaders,
            signedHeadersString,
            payloadHash
        ].joined(separator: "\n")

        let canonicalRequestHash = sha256Hex(canonicalRequest.data(using: .utf8)!)

        // String to sign
        let credentialScope = "\(dateStamp)/\(region)/s3/aws4_request"
        let stringToSign = [
            "AWS4-HMAC-SHA256",
            amzDate,
            credentialScope,
            canonicalRequestHash
        ].joined(separator: "\n")

        // Derive signing key: HMAC chain
        let kDate = hmacSHA256(
            key: "AWS4\(credentials.secretKey)".data(using: .utf8)!,
            data: dateStamp.data(using: .utf8)!
        )
        let kRegion = hmacSHA256(key: kDate, data: region.data(using: .utf8)!)
        let kService = hmacSHA256(key: kRegion, data: "s3".data(using: .utf8)!)
        let kSigning = hmacSHA256(key: kService, data: "aws4_request".data(using: .utf8)!)

        // Compute signature
        let signatureData = hmacSHA256(key: kSigning, data: stringToSign.data(using: .utf8)!)
        let signatureHex = signatureData.map { String(format: "%02x", $0) }.joined()

        // Authorization header
        let authorization = "AWS4-HMAC-SHA256 Credential=\(credentials.accessKey)/\(credentialScope), SignedHeaders=\(signedHeadersString), Signature=\(signatureHex)"

        // Build URLRequest
        var request = URLRequest(url: url)
        request.httpMethod = method
        for (key, value) in allHeaders {
            request.setValue(value, forHTTPHeaderField: key)
        }
        request.setValue(authorization, forHTTPHeaderField: "Authorization")

        return request
    }

    // MARK: - Crypto Helpers

    private func hmacSHA256(key: Data, data: Data) -> Data {
        var digest = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        key.withUnsafeBytes { keyBytes in
            data.withUnsafeBytes { dataBytes in
                CCHmac(
                    CCHmacAlgorithm(kCCHmacAlgSHA256),
                    keyBytes.baseAddress, key.count,
                    dataBytes.baseAddress, data.count,
                    &digest
                )
            }
        }
        return Data(digest)
    }

    private func sha256Hex(_ data: Data) -> String {
        var digest = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        data.withUnsafeBytes { bytes in
            _ = CC_SHA256(bytes.baseAddress, CC_LONG(data.count), &digest)
        }
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - S3 XML Parsing

    /// Parses the S3 ListBucketResult XML to extract file entries.
    private func parseListBucketResult(_ data: Data) -> [RemoteFileEntry] {
        let parser = S3ListParser(data: data)
        return parser.parse()
    }
}

// MARK: - S3 XML List Parser

/// Simple XML parser for S3 ListBucketResult responses.
/// Extracts <Contents> entries with Key, Size, and LastModified fields.
private class S3ListParser: NSObject, XMLParserDelegate {
    private let data: Data
    private var entries: [RemoteFileEntry] = []
    private var currentElement = ""
    private var currentText = ""

    // Per-entry accumulator
    private var entryKey = ""
    private var entrySize: Int64 = 0
    private var entryLastModified = Date()

    private var insideContents = false

    private let iso8601Formatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    private let iso8601FallbackFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    init(data: Data) {
        self.data = data
    }

    func parse() -> [RemoteFileEntry] {
        let parser = XMLParser(data: data)
        parser.delegate = self
        parser.parse()
        return entries
    }

    func parser(_ parser: XMLParser, didStartElement elementName: String,
                namespaceURI: String?, qualifiedName: String?,
                attributes: [String: String] = [:]) {
        // Strip namespace prefix if present (e.g. "s3:Contents" -> "Contents")
        let localName = elementName.components(separatedBy: ":").last ?? elementName
        currentElement = localName
        currentText = ""
        if localName == "Contents" {
            insideContents = true
            entryKey = ""
            entrySize = 0
            entryLastModified = Date()
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        currentText += string
    }

    func parser(_ parser: XMLParser, didEndElement elementName: String,
                namespaceURI: String?, qualifiedName: String?) {
        let localName = elementName.components(separatedBy: ":").last ?? elementName

        if insideContents {
            let trimmed = currentText.trimmingCharacters(in: .whitespacesAndNewlines)
            switch localName {
            case "Key":
                entryKey = trimmed
            case "Size":
                entrySize = Int64(trimmed) ?? 0
            case "LastModified":
                entryLastModified = iso8601Formatter.date(from: trimmed)
                    ?? iso8601FallbackFormatter.date(from: trimmed)
                    ?? Date()
            case "Contents":
                let name = (entryKey as NSString).lastPathComponent
                let entry = RemoteFileEntry(
                    name: name,
                    remotePath: entryKey,
                    sizeBytes: entrySize,
                    lastModified: entryLastModified,
                    provider: "S3"
                )
                entries.append(entry)
                insideContents = false
            default:
                break
            }
        }
        currentText = ""
    }
}

// MARK: - CloudKit Sync Provider

#if canImport(CloudKit)

/// CloudKit-based sync using CKContainer and CKAsset for file storage.
/// Uses the private database and a "Checkpoints" record type.
/// Each record stores the file as a CKAsset along with metadata.
class CloudKitSyncProvider: CloudSyncProvider {
    let providerName = "CloudKit"

    private let container: CKContainer
    private let database: CKDatabase
    private let recordType = "Checkpoints"

    init(containerID: String? = nil) {
        if let id = containerID, !id.isEmpty {
            self.container = CKContainer(identifier: id)
        } else {
            self.container = CKContainer.default()
        }
        self.database = container.privateCloudDatabase
    }

    func upload(localPath: String, remotePath: String) async throws {
        let fileURL = URL(fileURLWithPath: localPath)
        guard FileManager.default.fileExists(atPath: localPath) else {
            throw CloudSyncError.fileNotFound(localPath)
        }

        let recordID = CKRecord.ID(recordName: sanitizeRecordName(remotePath))
        let record = CKRecord(recordType: recordType, recordID: recordID)

        let asset = CKAsset(fileURL: fileURL)
        record["fileData"] = asset
        record["remotePath"] = remotePath as CKRecordValue
        record["fileName"] = (remotePath as NSString).lastPathComponent as CKRecordValue

        let attrs = try? FileManager.default.attributesOfItem(atPath: localPath)
        let fileSize = attrs?[.size] as? Int64 ?? 0
        record["sizeBytes"] = fileSize as CKRecordValue

        do {
            _ = try await database.save(record)
        } catch let error as CKError where error.code == .serverRecordChanged {
            // Record already exists — fetch the server version, update it, and save again
            let existing = try await database.record(for: recordID)
            existing["fileData"] = asset
            existing["remotePath"] = remotePath as CKRecordValue
            existing["fileName"] = (remotePath as NSString).lastPathComponent as CKRecordValue
            existing["sizeBytes"] = fileSize as CKRecordValue
            _ = try await database.save(existing)
        } catch {
            throw CloudSyncError.uploadFailed(error.localizedDescription)
        }
    }

    func download(remotePath: String, localPath: String) async throws {
        let recordID = CKRecord.ID(recordName: sanitizeRecordName(remotePath))

        let record: CKRecord
        do {
            record = try await database.record(for: recordID)
        } catch {
            throw CloudSyncError.downloadFailed(error.localizedDescription)
        }

        guard let asset = record["fileData"] as? CKAsset,
              let assetURL = asset.fileURL else {
            throw CloudSyncError.downloadFailed("No file data in CloudKit record")
        }

        // Ensure parent directory exists
        let parentDir = (localPath as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: parentDir, withIntermediateDirectories: true)

        let destURL = URL(fileURLWithPath: localPath)
        if FileManager.default.fileExists(atPath: localPath) {
            try FileManager.default.removeItem(at: destURL)
        }
        try FileManager.default.copyItem(at: assetURL, to: destURL)
    }

    func list(prefix: String) async throws -> [RemoteFileEntry] {
        let predicate: NSPredicate
        if prefix.isEmpty {
            predicate = NSPredicate(value: true)
        } else {
            predicate = NSPredicate(format: "remotePath BEGINSWITH %@", prefix)
        }

        let query = CKQuery(recordType: recordType, predicate: predicate)
        query.sortDescriptors = [NSSortDescriptor(key: "modificationDate", ascending: false)]

        var entries: [RemoteFileEntry] = []

        do {
            let (results, _) = try await database.records(matching: query)
            for (_, result) in results {
                switch result {
                case .success(let record):
                    let name = record["fileName"] as? String ?? "unknown"
                    let path = record["remotePath"] as? String ?? ""
                    let size = record["sizeBytes"] as? Int64 ?? 0
                    let modified = record.modificationDate ?? Date()

                    entries.append(RemoteFileEntry(
                        name: name,
                        remotePath: path,
                        sizeBytes: size,
                        lastModified: modified,
                        provider: "CloudKit"
                    ))
                case .failure:
                    continue
                }
            }
        } catch {
            throw CloudSyncError.listFailed(error.localizedDescription)
        }

        return entries
    }

    func delete(remotePath: String) async throws {
        let recordID = CKRecord.ID(recordName: sanitizeRecordName(remotePath))
        do {
            try await database.deleteRecord(withID: recordID)
        } catch {
            throw CloudSyncError.deleteFailed(error.localizedDescription)
        }
    }

    func testConnection() async throws {
        do {
            let status = try await container.accountStatus()
            guard status == .available else {
                throw CloudSyncError.connectionFailed(
                    "iCloud account not available (status: \(status.rawValue))"
                )
            }
        } catch let error as CloudSyncError {
            throw error
        } catch {
            throw CloudSyncError.connectionFailed(error.localizedDescription)
        }
    }

    // MARK: - Helpers

    /// Sanitize a remote path into a valid CKRecord name.
    private func sanitizeRecordName(_ path: String) -> String {
        path.replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: " ", with: "_")
            .trimmingCharacters(in: CharacterSet(charactersIn: "."))
    }
}

#endif

// MARK: - Cloud Sync Status

enum CloudSyncStatus: Equatable {
    case idle
    case uploading(progress: String)
    case downloading(progress: String)
    case error(String)

    var description: String {
        switch self {
        case .idle: return "Idle"
        case .uploading(let msg): return "Uploading: \(msg)"
        case .downloading(let msg): return "Downloading: \(msg)"
        case .error(let msg): return "Error: \(msg)"
        }
    }

    var isActive: Bool {
        switch self {
        case .uploading, .downloading: return true
        default: return false
        }
    }
}

// MARK: - Cloud Sync Manager

/// Singleton manager for cloud sync operations. Handles configuration persistence,
/// provider instantiation, and orchestrates upload/download/sync workflows.
/// Loads and saves configuration from ~/Library/Application Support/NeuralForge/cloud_sync_config.json.
@MainActor
class CloudSyncManager: ObservableObject {
    @Published var config = CloudSyncConfig()
    @Published var status: CloudSyncStatus = .idle
    @Published var remoteFiles: [RemoteFileEntry] = []

    static let shared = CloudSyncManager()

    private let fm = FileManager.default

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
        let path = CloudSyncConfig.configPath
        guard fm.fileExists(atPath: path.path),
              let data = try? Data(contentsOf: path),
              let cfg = try? decoder.decode(CloudSyncConfig.self, from: data) else {
            return
        }
        config = cfg
    }

    func saveConfig() {
        let dir = CloudSyncConfig.configPath.deletingLastPathComponent()
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        if let data = try? encoder.encode(config) {
            try? data.write(to: CloudSyncConfig.configPath, options: .atomic)
        }
    }

    // MARK: - Provider Factory

    /// Creates the appropriate CloudSyncProvider based on the current configuration.
    func makeProvider() throws -> CloudSyncProvider {
        switch config.provider {
        case .none:
            throw CloudSyncError.notConfigured
        case .s3:
            guard !config.s3Endpoint.isEmpty, !config.s3Bucket.isEmpty else {
                throw CloudSyncError.notConfigured
            }
            return S3SyncProvider(
                endpoint: config.s3Endpoint,
                bucket: config.s3Bucket,
                region: config.s3Region
            )
        case .cloudkit:
            #if canImport(CloudKit)
            return CloudKitSyncProvider(containerID: config.cloudKitContainerID)
            #else
            throw CloudSyncError.notConfigured
            #endif
        }
    }

    // MARK: - Upload

    /// Upload a local checkpoint file to the cloud, organized by project name.
    /// Remote path format: checkpoints/<project_name>/<filename>
    func uploadCheckpoint(localPath: String, projectName: String) async {
        guard !status.isActive else { return }

        let fileName = (localPath as NSString).lastPathComponent
        let safeName = projectName
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: " ", with: "_")
        let remotePath = "checkpoints/\(safeName)/\(fileName)"

        status = .uploading(progress: fileName)

        do {
            let provider = try makeProvider()
            try await provider.upload(localPath: localPath, remotePath: remotePath)
            status = .idle
        } catch {
            status = .error(error.localizedDescription)
        }
    }

    // MARK: - Download

    /// Download a remote checkpoint file to a local path.
    func downloadCheckpoint(remotePath: String, localPath: String) async {
        guard !status.isActive else { return }

        let fileName = (remotePath as NSString).lastPathComponent
        status = .downloading(progress: fileName)

        do {
            let provider = try makeProvider()
            try await provider.download(remotePath: remotePath, localPath: localPath)
            status = .idle
        } catch {
            status = .error(error.localizedDescription)
        }
    }

    // MARK: - List Remote Files

    /// List all remote files (or those matching a prefix) and update remoteFiles.
    func listRemote(prefix: String = "") async {
        do {
            let provider = try makeProvider()
            let files = try await provider.list(prefix: prefix)
            remoteFiles = files.sorted { $0.lastModified > $1.lastModified }
        } catch {
            status = .error(error.localizedDescription)
            remoteFiles = []
        }
    }

    // MARK: - Sync

    /// Upload all new local checkpoints that are not yet on the remote.
    /// Scans ~/Library/Application Support/NeuralForge/Projects/ for .bin
    /// checkpoint files and uploads any that do not already exist remotely.
    func sync() async {
        guard !status.isActive else { return }

        status = .uploading(progress: "Scanning...")

        do {
            let provider = try makeProvider()

            // Get current remote file list
            let existing = try await provider.list(prefix: "checkpoints/")
            let existingPaths = Set(existing.map { $0.remotePath })

            // Scan local projects for checkpoints
            let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            let projectsDir = appSupport.appendingPathComponent("NeuralForge/Projects")

            guard let projectDirs = try? fm.contentsOfDirectory(
                at: projectsDir, includingPropertiesForKeys: [.isDirectoryKey]) else {
                status = .idle
                return
            }

            var uploadCount = 0

            for dir in projectDirs {
                var isDir: ObjCBool = false
                guard fm.fileExists(atPath: dir.path, isDirectory: &isDir),
                      isDir.boolValue else { continue }

                // Read project name from project.json if available
                let projectFile = dir.appendingPathComponent("project.json")
                var projectName = dir.lastPathComponent
                if let data = try? Data(contentsOf: projectFile),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let name = json["name"] as? String {
                    projectName = name
                }

                let safeName = projectName
                    .replacingOccurrences(of: "/", with: "_")
                    .replacingOccurrences(of: " ", with: "_")

                // Scan checkpoint files
                let checkpointDir = dir.appendingPathComponent("checkpoints")
                guard let files = try? fm.contentsOfDirectory(
                    at: checkpointDir,
                    includingPropertiesForKeys: [.fileSizeKey]
                ) else { continue }

                for file in files where file.pathExtension == "bin" {
                    let remotePath = "checkpoints/\(safeName)/\(file.lastPathComponent)"

                    if !existingPaths.contains(remotePath) {
                        uploadCount += 1
                        status = .uploading(progress: "\(file.lastPathComponent) (\(uploadCount))")
                        try await provider.upload(
                            localPath: file.path,
                            remotePath: remotePath
                        )
                    }
                }
            }

            // Refresh the remote listing
            let updated = try await provider.list(prefix: "checkpoints/")
            remoteFiles = updated.sorted { $0.lastModified > $1.lastModified }

            status = .idle
        } catch {
            status = .error(error.localizedDescription)
        }
    }

    // MARK: - Test Connection

    /// Test connectivity to the configured cloud provider.
    /// Returns true on success, sets status to error on failure.
    func testConnection() async -> Bool {
        do {
            let provider = try makeProvider()
            try await provider.testConnection()
            return true
        } catch {
            status = .error(error.localizedDescription)
            return false
        }
    }

    // MARK: - Delete Remote

    /// Delete a remote file and remove it from the local listing.
    func deleteRemote(remotePath: String) async {
        do {
            let provider = try makeProvider()
            try await provider.delete(remotePath: remotePath)
            remoteFiles.removeAll { $0.remotePath == remotePath }
        } catch {
            status = .error(error.localizedDescription)
        }
    }

    // MARK: - Helpers

    /// Format a byte count for display (e.g., "42 MB").
    static func formatSize(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
