// ComplianceReportView.swift — Generate and preview HIPAA/SOX compliance reports
import SwiftUI

struct ComplianceReportView: View {
    @Binding var project: NFProject
    @StateObject private var generator = ComplianceReportGenerator()
    @State private var selectedFramework: ComplianceFramework = .general
    @State private var useDateRange = false
    @State private var startDate = Calendar.current.date(byAdding: .month, value: -1, to: Date()) ?? Date()
    @State private var endDate = Date()
    @State private var showExportSuccess = false

    var body: some View {
        VStack(spacing: 0) {
            // Config bar
            configBar

            Divider()

            if generator.isGenerating {
                Spacer()
                ProgressView("Generating \(selectedFramework.rawValue) report...")
                Spacer()
            } else if let report = generator.report {
                reportPreview(report)
            } else {
                emptyState
            }

            if let error = generator.errorMessage {
                Label(error, systemImage: "exclamationmark.triangle")
                    .foregroundColor(.red)
                    .font(.caption)
                    .padding()
            }
        }
    }

    // MARK: - Config Bar

    private var configBar: some View {
        VStack(spacing: 12) {
            HStack {
                Label("Compliance Reports", systemImage: "doc.text.magnifyingglass")
                    .font(.headline)
                Spacer()
            }

            HStack(spacing: 16) {
                // Framework picker
                Picker("Framework", selection: $selectedFramework) {
                    ForEach(ComplianceFramework.allCases) { fw in
                        Label(fw.rawValue, systemImage: fw.icon).tag(fw)
                    }
                }
                .frame(width: 200)

                Divider().frame(height: 24)

                // Date range toggle
                Toggle("Date range", isOn: $useDateRange)
                    .toggleStyle(.checkbox)

                if useDateRange {
                    DatePicker("From", selection: $startDate, displayedComponents: .date)
                        .labelsHidden()
                        .frame(width: 120)
                    Text("—")
                        .foregroundColor(.secondary)
                    DatePicker("To", selection: $endDate, displayedComponents: .date)
                        .labelsHidden()
                        .frame(width: 120)
                }

                Spacer()

                Button(action: generateReport) {
                    Label("Generate", systemImage: "doc.badge.gearshape")
                }
                .buttonStyle(.borderedProminent)
                .disabled(generator.isGenerating)
            }

            // Framework description
            HStack {
                Image(systemName: selectedFramework.icon)
                    .foregroundColor(.blue)
                Text(selectedFramework.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
            }
        }
        .padding()
    }

    // MARK: - Report Preview

    private func reportPreview(_ report: ComplianceReport) -> some View {
        VStack(spacing: 0) {
            // Report header
            reportHeader(report)

            Divider()

            // Sections
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(report.sections) { section in
                        sectionCard(section)
                    }
                }
                .padding()
            }

            Divider()

            // Export bar
            exportBar(report)
        }
    }

    private func reportHeader(_ report: ComplianceReport) -> some View {
        HStack(spacing: 16) {
            Image(systemName: report.overallStatus.icon)
                .font(.system(size: 32))
                .foregroundColor(statusColor(report.overallStatus))

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(report.framework.rawValue)
                        .font(.title3.bold())
                    Text("Compliance Report")
                        .font(.title3)
                        .foregroundColor(.secondary)
                }

                Text(report.summary)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                statusBadge(report.overallStatus)

                Text("Generated \(report.generatedAt, style: .date)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(statusColor(report.overallStatus).opacity(0.05))
    }

    private func sectionCard(_ section: ReportSection) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: section.icon)
                    .foregroundColor(severityColor(section.severity))
                Text(section.title)
                    .font(.subheadline.bold())
                Spacer()
                severityBadge(section.severity)
            }

            Text(section.content)
                .font(.caption.monospaced())
                .foregroundColor(.primary.opacity(0.85))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8)
            .fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(severityColor(section.severity).opacity(0.3), lineWidth: 1)
        )
    }

    private func exportBar(_ report: ComplianceReport) -> some View {
        HStack {
            Text("Report ID: \(report.id.uuidString.prefix(8))...")
                .font(.caption2.monospaced())
                .foregroundColor(.secondary)

            Spacer()

            if showExportSuccess {
                Label("Saved!", systemImage: "checkmark.circle.fill")
                    .foregroundColor(.green)
                    .font(.caption)
            }

            Button(action: exportText) {
                Label("Export Text", systemImage: "doc.text")
            }
            .buttonStyle(.bordered)

            Button(action: exportPDF) {
                Label("Export PDF", systemImage: "doc.richtext")
            }
            .buttonStyle(.bordered)
        }
        .padding()
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "doc.text.magnifyingglass")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            Text("Generate a Compliance Report")
                .font(.headline)
                .foregroundColor(.secondary)
            Text("Select a framework and click Generate to produce\na compliance report from your audit trail.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            HStack(spacing: 24) {
                frameworkCard(.general)
                frameworkCard(.hipaa)
                frameworkCard(.sox)
            }
            .padding(.top, 8)

            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private func frameworkCard(_ fw: ComplianceFramework) -> some View {
        Button(action: {
            selectedFramework = fw
            generateReport()
        }) {
            VStack(spacing: 8) {
                Image(systemName: fw.icon)
                    .font(.system(size: 24))
                    .foregroundColor(.blue)
                Text(fw.rawValue)
                    .font(.caption.bold())
                Text(fw == .general ? "Full audit summary" :
                     fw == .hipaa ? "Data & access controls" :
                     "Change management")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            .frame(width: 140, height: 100)
            .background(RoundedRectangle(cornerRadius: 8)
                .fill(Color(nsColor: .controlBackgroundColor)))
            .overlay(RoundedRectangle(cornerRadius: 8)
                .strokeBorder(Color.blue.opacity(0.2), lineWidth: 1))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Badges & Colors

    private func statusBadge(_ status: ComplianceReport.ReportStatus) -> some View {
        Text(status.rawValue)
            .font(.caption.bold())
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(statusColor(status).opacity(0.15))
            .foregroundColor(statusColor(status))
            .clipShape(Capsule())
    }

    private func severityBadge(_ severity: ReportSection.SectionSeverity) -> some View {
        let label: String
        switch severity {
        case .info: label = "INFO"
        case .pass: label = "PASS"
        case .warning: label = "REVIEW"
        case .critical: label = "CRITICAL"
        }

        return Text(label)
            .font(.caption2.bold())
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(severityColor(severity).opacity(0.15))
            .foregroundColor(severityColor(severity))
            .clipShape(Capsule())
    }

    private func statusColor(_ status: ComplianceReport.ReportStatus) -> Color {
        switch status {
        case .compliant: return .green
        case .needsReview: return .orange
        case .nonCompliant: return .red
        }
    }

    private func severityColor(_ severity: ReportSection.SectionSeverity) -> Color {
        switch severity {
        case .info: return .blue
        case .pass: return .green
        case .warning: return .orange
        case .critical: return .red
        }
    }

    // MARK: - Actions

    private func generateReport() {
        generator.generateReport(
            framework: selectedFramework,
            startDate: useDateRange ? startDate : nil,
            endDate: useDateRange ? endDate : nil
        )
    }

    private func exportText() {
        guard let text = generator.exportAsText() else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = "neuralforge_\(selectedFramework.rawValue.lowercased())_report.txt"

        if panel.runModal() == .OK, let url = panel.url {
            try? text.write(to: url, atomically: true, encoding: .utf8)
            showExportSuccess = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { showExportSuccess = false }
        }
    }

    private func exportPDF() {
        guard let text = generator.exportAsText() else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.pdf]
        panel.nameFieldStringValue = "neuralforge_\(selectedFramework.rawValue.lowercased())_report.pdf"

        if panel.runModal() == .OK, let url = panel.url {
            // Use NSAttributedString to create PDF
            let font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
            let attrs: [NSAttributedString.Key: Any] = [
                .font: font,
                .foregroundColor: NSColor.textColor
            ]
            let attrStr = NSAttributedString(string: text, attributes: attrs)

            let printInfo = NSPrintInfo.shared
            printInfo.horizontalPagination = .fit
            printInfo.verticalPagination = .automatic
            printInfo.leftMargin = 50
            printInfo.rightMargin = 50
            printInfo.topMargin = 50
            printInfo.bottomMargin = 50

            let pageSize = printInfo.paperSize
            let textRect = NSRect(
                x: printInfo.leftMargin,
                y: printInfo.bottomMargin,
                width: pageSize.width - printInfo.leftMargin - printInfo.rightMargin,
                height: pageSize.height - printInfo.topMargin - printInfo.bottomMargin
            )

            let textView = NSTextView(frame: textRect)
            textView.textStorage?.setAttributedString(attrStr)

            let pdfData = NSMutableData()
            let printOp = NSPrintOperation.pdfOperation(
                with: textView, inside: textRect, to: pdfData, printInfo: printInfo)
            printOp.showsPrintPanel = false
            printOp.showsProgressPanel = false
            printOp.run()

            try? pdfData.write(to: url, options: .atomic)
            showExportSuccess = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { showExportSuccess = false }
        }
    }
}
