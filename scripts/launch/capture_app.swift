#!/usr/bin/env swift
// NeuralForge Screenshot Capture Tool
// Captures NeuralForge app windows using ScreenCaptureKit
// Usage: swift scripts/launch/capture_app.swift
//
// NOTE: First run will prompt for Screen Recording permission.
//       Grant it in System Settings → Privacy & Security → Screen Recording
//       Then run again.

import Cocoa
import ScreenCaptureKit

let ROOT = ProcessInfo.processInfo.environment["NF_ROOT"]
    ?? FileManager.default.currentDirectoryPath
let screenshotsDir = "\(ROOT)/assets/screenshots"

// Ensure directory exists
try? FileManager.default.createDirectory(
    atPath: screenshotsDir,
    withIntermediateDirectories: true
)

func captureWindow(window: SCWindow, name: String) async throws {
    let filter = SCContentFilter(desktopIndependentWindow: window)
    let config = SCStreamConfiguration()
    config.width = Int(window.frame.width) * 2  // Retina 2x
    config.height = Int(window.frame.height) * 2
    config.showsCursor = false

    let image = try await SCScreenshotManager.captureImage(
        contentFilter: filter,
        configuration: config
    )

    let bitmapRep = NSBitmapImageRep(cgImage: image)
    guard let pngData = bitmapRep.representation(using: .png, properties: [:]) else {
        print("  ❌ Failed to create PNG for \(name)")
        return
    }

    let path = "\(screenshotsDir)/\(name).png"
    try pngData.write(to: URL(fileURLWithPath: path))
    let sizeKB = pngData.count / 1024
    print("  ✅ Captured \(name).png (\(sizeKB) KB, \(Int(window.frame.width))×\(Int(window.frame.height)))")
}

print("")
print("╔══════════════════════════════════════════════╗")
print("║   📸 NeuralForge Screenshot Capture          ║")
print("╚══════════════════════════════════════════════╝")
print("")

let semaphore = DispatchSemaphore(value: 0)

Task {
    do {
        // Get all shareable content
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: true
        )

        // Find NeuralForge windows
        let nfWindows = content.windows.filter {
            $0.owningApplication?.applicationName == "NeuralForge"
        }

        if nfWindows.isEmpty {
            print("  ⚠️  NeuralForge is not running!")
            print("  Please launch it first: open app/NeuralForge.xcodeproj → ⌘R")
            print("")
            semaphore.signal()
            return
        }

        print("  Found \(nfWindows.count) NeuralForge window(s)")
        print("")

        // Capture each window
        for (i, window) in nfWindows.enumerated() {
            let title = window.title ?? "window_\(i + 1)"
            let safeName = title
                .lowercased()
                .replacingOccurrences(of: " ", with: "_")
                .replacingOccurrences(of: "/", with: "_")
                .prefix(40)
            let name = String(format: "%02d_%@", i + 1, String(safeName))
            try await captureWindow(window: window, name: name)
        }

        // Also capture the full screen with NeuralForge visible
        print("")
        print("  Capturing full screen...")
        if let display = content.displays.first {
            let filter = SCContentFilter(display: display, excludingWindows: [])
            let config = SCStreamConfiguration()
            config.width = Int(display.width) * 2
            config.height = Int(display.height) * 2
            config.showsCursor = false

            let image = try await SCScreenshotManager.captureImage(
                contentFilter: filter,
                configuration: config
            )
            let bitmapRep = NSBitmapImageRep(cgImage: image)
            if let pngData = bitmapRep.representation(using: .png, properties: [:]) {
                let path = "\(screenshotsDir)/00_full_screen.png"
                try pngData.write(to: URL(fileURLWithPath: path))
                print("  ✅ Captured full screen (\(pngData.count / 1024) KB)")
            }
        }

        print("")
        print("  📂 Screenshots saved to: \(screenshotsDir)/")
        print("")
        print("  Next: Navigate NeuralForge to different tabs and run again,")
        print("  or use 'bash scripts/launch/update_readme.sh' to embed in README.")

    } catch {
        let nsError = error as NSError
        if nsError.domain == "com.apple.ScreenCaptureKit.SCStreamErrorDomain"
            && nsError.code == -3801 {
            print("  ⚠️  Screen Recording permission required!")
            print("")
            print("  To fix:")
            print("  1. Open System Settings → Privacy & Security → Screen Recording")
            print("  2. Enable 'Terminal' (or your terminal app)")
            print("  3. Run this script again")
            print("")
            print("  Opening System Settings...")
            NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")!)
        } else {
            print("  ❌ Error: \(error)")
        }
    }
    semaphore.signal()
}

semaphore.wait()
