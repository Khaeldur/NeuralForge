// DocumentImporter.swift — Extract text from PDFs and text files

import Foundation
import PDFKit

struct DocumentImporter {

    /// Extract text from a file (supports .txt, .pdf, .md, .csv, .json, .docx)
    static func extractText(from url: URL) throws -> String {
        let ext = url.pathExtension.lowercased()

        switch ext {
        case "pdf":
            return try extractPDF(url)
        case "docx":
            return try extractDOCX(url)
        case "txt", "md", "csv", "json", "swift", "py", "js", "c", "m", "h":
            return try String(contentsOf: url, encoding: .utf8)
        default:
            // Try reading as UTF-8 text
            return try String(contentsOf: url, encoding: .utf8)
        }
    }

    /// Extract all text from a PDF using PDFKit
    private static func extractPDF(_ url: URL) throws -> String {
        guard let doc = PDFDocument(url: url) else {
            throw ImportError.cannotOpenPDF
        }

        var text = ""
        for i in 0..<doc.pageCount {
            if let page = doc.page(at: i), let pageText = page.string {
                text += pageText
                text += "\n\n"
            }
        }

        guard !text.isEmpty else {
            throw ImportError.noPDFText
        }

        return text
    }

    /// Extract text from a DOCX file (ZIP containing word/document.xml)
    private static func extractDOCX(_ url: URL) throws -> String {
        let fm = FileManager.default
        let tempDir = fm.temporaryDirectory.appendingPathComponent("nf_docx_\(UUID().uuidString)")
        try fm.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempDir) }

        // Unzip using /usr/bin/ditto (built-in macOS)
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        proc.arguments = ["-xk", url.path, tempDir.path]
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        proc.waitUntilExit()

        guard proc.terminationStatus == 0 else {
            throw ImportError.cannotOpenDOCX
        }

        let docXML = tempDir.appendingPathComponent("word/document.xml")
        guard fm.fileExists(atPath: docXML.path) else {
            throw ImportError.cannotOpenDOCX
        }

        let xmlData = try Data(contentsOf: docXML)
        let parser = DOCXTextParser()
        let xmlParser = XMLParser(data: xmlData)
        xmlParser.delegate = parser
        xmlParser.parse()

        let text = parser.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            throw ImportError.noDOCXText
        }
        return text
    }

    /// Merge multiple files into a single text, separated by newlines
    static func mergeFiles(_ urls: [URL]) throws -> String {
        var combined = ""
        for url in urls {
            let text = try extractText(from: url)
            combined += text
            combined += "\n\n"
        }
        return combined
    }

    /// Write merged text to a temporary file for tokenization
    static func writeToTempFile(_ text: String) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("nf_import_\(UUID().uuidString).txt")
        try text.write(to: tempFile, atomically: true, encoding: .utf8)
        return tempFile
    }

    enum ImportError: LocalizedError {
        case cannotOpenPDF
        case noPDFText
        case cannotOpenDOCX
        case noDOCXText

        var errorDescription: String? {
            switch self {
            case .cannotOpenPDF: "Cannot open PDF file"
            case .noPDFText: "PDF contains no extractable text (may be scanned/image-only)"
            case .cannotOpenDOCX: "Cannot open DOCX file (invalid or corrupted)"
            case .noDOCXText: "DOCX contains no extractable text"
            }
        }
    }
}

/// XMLParser delegate that extracts text from <w:t> elements in DOCX XML
private class DOCXTextParser: NSObject, XMLParserDelegate {
    var text = ""
    private var insideTextRun = false
    private var insideParagraph = false
    private var currentParagraph = ""

    func parser(_ parser: XMLParser, didStartElement elementName: String,
                namespaceURI: String?, qualifiedName qName: String?,
                attributes attributeDict: [String: String] = [:]) {
        if elementName == "w:p" {
            insideParagraph = true
            currentParagraph = ""
        } else if elementName == "w:t" {
            insideTextRun = true
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        if insideTextRun {
            currentParagraph += string
        }
    }

    func parser(_ parser: XMLParser, didEndElement elementName: String,
                namespaceURI: String?, qualifiedName qName: String?) {
        if elementName == "w:t" {
            insideTextRun = false
        } else if elementName == "w:p" {
            insideParagraph = false
            if !currentParagraph.isEmpty {
                text += currentParagraph + "\n"
            }
        }
    }
}
