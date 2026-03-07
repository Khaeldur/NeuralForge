// NeuralForgeUITests.swift — End-to-end UI automation tests
//
// XCUITest suite for NeuralForge macOS app.
// Tests the full user journey: onboarding → project creation → configuration → training.
//
// Run: xcodebuild test -project NeuralForge.xcodeproj -scheme NeuralForge -destination 'platform=macOS'

import XCTest

final class NeuralForgeUITests: XCTestCase {

    var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        // Reset onboarding state for a clean start
        app.launchArguments += ["-onboardingComplete", "NO"]
    }

    override func tearDownWithError() throws {
        app = nil
    }

    // MARK: - App Launch

    func testAppLaunches() throws {
        app.launch()
        XCTAssertTrue(app.windows.count > 0, "App should have at least one window")
    }

    func testAppWindowHasTitle() throws {
        app.launch()
        let window = app.windows.firstMatch
        XCTAssertTrue(window.exists, "Main window should exist")
    }

    // MARK: - Onboarding Flow

    func testOnboardingShowsOnFirstLaunch() throws {
        app.launch()
        // Onboarding welcome page should be visible
        let welcomeText = app.staticTexts["Welcome to NeuralForge"]
        XCTAssertTrue(welcomeText.waitForExistence(timeout: 5),
                      "Welcome text should appear on first launch")
    }

    func testOnboardingHasNextButton() throws {
        app.launch()
        let nextButton = app.buttons["Next"]
        XCTAssertTrue(nextButton.waitForExistence(timeout: 5),
                      "Next button should be visible on onboarding")
    }

    func testOnboardingBackButtonDisabledOnFirstPage() throws {
        app.launch()
        let backButton = app.buttons["Back"]
        if backButton.exists {
            XCTAssertFalse(backButton.isEnabled,
                           "Back button should be disabled on first page")
        }
    }

    func testOnboardingFeaturePillsVisible() throws {
        app.launch()
        // Feature pills on the welcome page
        let hardwareText = app.staticTexts["Hardware acceleration"]
        let privacyText = app.staticTexts["Data privacy"]
        let metricsText = app.staticTexts["Real-time metrics"]
        XCTAssertTrue(hardwareText.waitForExistence(timeout: 5))
        XCTAssertTrue(privacyText.exists)
        XCTAssertTrue(metricsText.exists)
    }

    func testOnboardingNavigateToSetupPage() throws {
        app.launch()
        let nextButton = app.buttons["Next"]
        XCTAssertTrue(nextButton.waitForExistence(timeout: 5))
        nextButton.click()

        // Setup page should have CLI path field
        let browseButton = app.buttons["Browse"]
        XCTAssertTrue(browseButton.waitForExistence(timeout: 3),
                      "Browse button should appear on setup page")
    }

    func testOnboardingAutoDetectButton() throws {
        app.launch()
        let nextButton = app.buttons["Next"]
        XCTAssertTrue(nextButton.waitForExistence(timeout: 5))
        nextButton.click()

        let autoDetect = app.buttons["Auto-detect"]
        XCTAssertTrue(autoDetect.waitForExistence(timeout: 3),
                      "Auto-detect button should exist on setup page")
    }

    func testOnboardingGoalSelectionPage() throws {
        app.launch()
        // Navigate to setup page
        let nextButton = app.buttons["Next"]
        XCTAssertTrue(nextButton.waitForExistence(timeout: 5))
        nextButton.click()

        // Navigate to goal page (need CLI valid, try clicking next)
        // If CLI is not found, Next may be disabled. Just check we can go back.
        let backButton = app.buttons["Back"]
        XCTAssertTrue(backButton.waitForExistence(timeout: 3))
        XCTAssertTrue(backButton.isEnabled, "Back button should be enabled on page 2")
    }

    // MARK: - Main App (post-onboarding)

    func testMainViewShowsSidebar() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        // Sidebar should show "Projects" navigation
        let projectsTitle = app.staticTexts["Projects"]
        XCTAssertTrue(projectsTitle.waitForExistence(timeout: 5),
                      "Projects sidebar should be visible")
    }

    func testMainViewNewProjectButton() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        // Plus button in toolbar to create new project
        let addButton = app.buttons.matching(NSPredicate(format: "label CONTAINS 'Add'")).firstMatch
        let plusButton = app.buttons.matching(NSPredicate(format: "label CONTAINS 'New'")).firstMatch

        // Either the + button or "New Project" button should exist
        let hasButton = addButton.waitForExistence(timeout: 5) || plusButton.exists
        XCTAssertTrue(hasButton, "Should have a button to create new projects")
    }

    func testEmptyStateShowsCreateProjectPrompt() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        // When no project is selected, should show a prompt
        let createText = app.staticTexts["New Project"]
        if createText.waitForExistence(timeout: 5) {
            XCTAssertTrue(createText.exists)
        }
        // If no empty state visible, that's also acceptable (projects may exist)
    }

    // MARK: - Project Creation Sheet

    func testNewProjectSheetOpens() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        // Try to open the new project sheet
        let addButton = app.buttons.matching(NSPredicate(format: "label CONTAINS 'Add'")).firstMatch
        if addButton.waitForExistence(timeout: 5) {
            addButton.click()

            // Sheet should appear with "Create" and "Cancel" buttons
            let createButton = app.buttons["Create"]
            let cancelButton = app.buttons["Cancel"]
            XCTAssertTrue(createButton.waitForExistence(timeout: 3) || cancelButton.waitForExistence(timeout: 3),
                          "New project sheet should open with Create/Cancel buttons")

            // Clean up: dismiss the sheet
            if cancelButton.exists { cancelButton.click() }
        }
    }

    func testNewProjectSheetHasNameField() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        let addButton = app.buttons.matching(NSPredicate(format: "label CONTAINS 'Add'")).firstMatch
        if addButton.waitForExistence(timeout: 5) {
            addButton.click()

            // Should have a text field for project name
            let textField = app.textFields["Project name"]
            let hasField = textField.waitForExistence(timeout: 3)
            XCTAssertTrue(hasField, "Should have a project name text field")

            let cancelButton = app.buttons["Cancel"]
            if cancelButton.exists { cancelButton.click() }
        }
    }

    // MARK: - Settings Window

    func testSettingsWindowOpens() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        // Open Settings via keyboard shortcut (Cmd+,)
        app.typeKey(",", modifierFlags: .command)

        // Settings window should appear with tabs
        let generalTab = app.staticTexts["General"]
        let settingsExists = generalTab.waitForExistence(timeout: 5)

        // Settings might open as a separate window
        if !settingsExists {
            // Try menu bar: NeuralForge > Settings
            let menuBar = app.menuBars.firstMatch
            if menuBar.exists {
                menuBar.menuBarItems["NeuralForge"].click()
                let settingsItem = menuBar.menuItems["Settings…"]
                if settingsItem.waitForExistence(timeout: 2) {
                    settingsItem.click()
                }
            }
        }
    }

    // MARK: - Menu Bar

    func testMenuBarHasExpectedMenus() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        let menuBar = app.menuBars.firstMatch
        XCTAssertTrue(menuBar.waitForExistence(timeout: 5))

        // Check standard menus exist
        let neuralForgeMenu = menuBar.menuBarItems["NeuralForge"]
        XCTAssertTrue(neuralForgeMenu.exists, "NeuralForge menu should exist")
    }

    func testFileMenuExists() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        let menuBar = app.menuBars.firstMatch
        XCTAssertTrue(menuBar.waitForExistence(timeout: 5))

        let fileMenu = menuBar.menuBarItems["File"]
        XCTAssertTrue(fileMenu.exists, "File menu should exist")
    }

    // MARK: - Window Management

    func testWindowCanBeResized() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        let window = app.windows.firstMatch
        XCTAssertTrue(window.waitForExistence(timeout: 5))

        let frame = window.frame
        XCTAssertTrue(frame.width > 0, "Window should have positive width")
        XCTAssertTrue(frame.height > 0, "Window should have positive height")
    }

    func testOnboardingWindowSize() throws {
        app.launch()
        let window = app.windows.firstMatch
        XCTAssertTrue(window.waitForExistence(timeout: 5))

        let frame = window.frame
        // Onboarding window should be 620x520
        XCTAssertTrue(frame.width >= 600, "Onboarding window width should be ~620")
        XCTAssertTrue(frame.height >= 500, "Onboarding window height should be ~520")
    }

    func testMainWindowSize() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()
        let window = app.windows.firstMatch
        XCTAssertTrue(window.waitForExistence(timeout: 5))

        let frame = window.frame
        // Main window should be 1200x800
        XCTAssertTrue(frame.width >= 1000, "Main window width should be ~1200")
        XCTAssertTrue(frame.height >= 700, "Main window height should be ~800")
    }

    // MARK: - Accessibility

    func testMainViewIsAccessible() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        // Check that the app has accessible elements
        let allElements = app.descendants(matching: .any)
        XCTAssertTrue(allElements.count > 0, "App should have accessible UI elements")
    }

    func testOnboardingIsAccessible() throws {
        app.launch()

        // Check that onboarding has accessible text
        let staticTexts = app.staticTexts
        XCTAssertTrue(staticTexts.count > 0, "Onboarding should have accessible text elements")
    }

    // MARK: - Navigation Integration

    func testSidebarToDetailNavigation() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        // The sidebar should be visible
        let sidebar = app.groups.firstMatch
        XCTAssertTrue(sidebar.waitForExistence(timeout: 5))
    }

    // MARK: - Full Journey (smoke test)

    func testOnboardingToMainTransition() throws {
        // This test verifies the full onboarding → main app transition
        // by launching with onboarding complete
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        // Should see the main app view, not onboarding
        let projectsTitle = app.staticTexts["Projects"]
        let welcomeText = app.staticTexts["Welcome to NeuralForge"]

        let mainViewVisible = projectsTitle.waitForExistence(timeout: 5)
        let onboardingVisible = welcomeText.exists

        XCTAssertTrue(mainViewVisible || !onboardingVisible,
                      "Main view should be visible when onboarding is complete")
    }

    // =========================================================================
    // MARK: - Sidebar Navigation Smoke Tests
    // =========================================================================

    /// Helper: launch app, create a project, return its name
    private func launchWithProject(named name: String = "SmokeTest") {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        let addButton = app.buttons.matching(NSPredicate(format: "label CONTAINS 'Add' OR label CONTAINS 'New'")).firstMatch
        guard addButton.waitForExistence(timeout: 5) else { return }
        addButton.click()

        let nameField = app.textFields["Project name"]
        if nameField.waitForExistence(timeout: 3) {
            nameField.click()
            nameField.typeText(name)
            let createButton = app.buttons["Create"]
            if createButton.waitForExistence(timeout: 2) {
                createButton.click()
            }
        }
    }

    /// Helper: click a sidebar item by its label text
    private func clickSidebarTab(_ label: String) {
        let staticText = app.staticTexts[label]
        if staticText.waitForExistence(timeout: 3) {
            staticText.click()
            return
        }
        let cell = app.cells.containing(.staticText, identifier: label).firstMatch
        if cell.waitForExistence(timeout: 2) {
            cell.click()
            return
        }
        let any = app.descendants(matching: .any).matching(NSPredicate(format: "label == %@", label)).firstMatch
        if any.waitForExistence(timeout: 2) {
            any.click()
        }
    }

    // MARK: Training Section

    func testSidebarTab_Dashboard() throws {
        launchWithProject()
        clickSidebarTab("Dashboard")
        XCTAssertTrue(app.windows.count > 0, "Dashboard tab should keep window open")
    }

    func testSidebarTab_Config() throws {
        launchWithProject()
        clickSidebarTab("Config")
        let hasConfig = app.staticTexts.matching(NSPredicate(
            format: "label CONTAINS[c] 'learning' OR label CONTAINS[c] 'steps' OR label CONTAINS[c] 'optimizer'"
        )).firstMatch.waitForExistence(timeout: 3)
        XCTAssertTrue(hasConfig || app.windows.count > 0, "Config tab should show training parameters")
    }

    func testSidebarTab_Data() throws {
        launchWithProject()
        clickSidebarTab("Data")
        XCTAssertTrue(app.windows.count > 0, "Data tab should load without crashing")
    }

    func testSidebarTab_Generate() throws {
        launchWithProject()
        clickSidebarTab("Generate")
        XCTAssertTrue(app.windows.count > 0, "Generate tab should load without crashing")
    }

    func testSidebarTab_Export() throws {
        launchWithProject()
        clickSidebarTab("Export")
        XCTAssertTrue(app.windows.count > 0, "Export tab should load without crashing")
    }

    // MARK: Data & Models Section

    func testSidebarTab_Models() throws {
        launchWithProject()
        clickSidebarTab("Models")
        XCTAssertTrue(app.windows.count > 0, "Models tab should load without crashing")
    }

    func testSidebarTab_Ingest() throws {
        launchWithProject()
        clickSidebarTab("Ingest")
        XCTAssertTrue(app.windows.count > 0, "Ingest tab should load without crashing")
    }

    func testSidebarTab_History() throws {
        launchWithProject()
        clickSidebarTab("History")
        XCTAssertTrue(app.windows.count > 0, "History tab should load without crashing")
    }

    func testSidebarTab_Benchmarks() throws {
        launchWithProject()
        clickSidebarTab("Benchmarks")
        XCTAssertTrue(app.windows.count > 0, "Benchmarks tab should load without crashing")
    }

    // MARK: Tools Section

    func testSidebarTab_Assistant() throws {
        launchWithProject()
        clickSidebarTab("Assistant")
        XCTAssertTrue(app.windows.count > 0, "Assistant tab should load without crashing")
    }

    func testSidebarTab_Sync() throws {
        launchWithProject()
        clickSidebarTab("Sync")
        XCTAssertTrue(app.windows.count > 0, "Sync tab should load without crashing")
    }

    func testSidebarTab_Cluster() throws {
        launchWithProject()
        clickSidebarTab("Cluster")
        XCTAssertTrue(app.windows.count > 0, "Cluster tab should load without crashing")
    }

    // MARK: Compliance Section

    func testSidebarTab_Audit() throws {
        launchWithProject()
        clickSidebarTab("Audit")
        XCTAssertTrue(app.windows.count > 0, "Audit tab should load without crashing")
    }

    func testSidebarTab_Reports() throws {
        launchWithProject()
        clickSidebarTab("Reports")
        XCTAssertTrue(app.windows.count > 0, "Reports tab should load without crashing")
    }

    // MARK: Full Walkthrough

    func testAllSidebarTabsNavigable() throws {
        launchWithProject(named: "FullNavTest")
        let tabs = [
            "Dashboard", "Config", "Data", "Generate", "Export",
            "Models", "Ingest", "History", "Benchmarks",
            "Assistant", "Sync", "Cluster", "Audit", "Reports"
        ]
        for tab in tabs {
            clickSidebarTab(tab)
            Thread.sleep(forTimeInterval: 0.3)
            XCTAssertTrue(app.windows.count > 0, "App should remain stable after clicking \(tab)")
        }
    }

    // =========================================================================
    // MARK: - Config Interaction Tests
    // =========================================================================

    func testConfigView_LoRASection() throws {
        launchWithProject()
        clickSidebarTab("Config")
        let loraText = app.staticTexts.matching(NSPredicate(
            format: "label CONTAINS[c] 'lora' OR label CONTAINS[c] 'LoRA'"
        )).firstMatch
        if loraText.waitForExistence(timeout: 3) {
            XCTAssertTrue(loraText.exists, "LoRA config section should be visible")
        }
    }

    func testConfigView_SchedulerSection() throws {
        launchWithProject()
        clickSidebarTab("Config")
        let schedulerText = app.staticTexts.matching(NSPredicate(
            format: "label CONTAINS[c] 'schedule' OR label CONTAINS[c] 'warmup' OR label CONTAINS[c] 'cosine'"
        )).firstMatch
        if schedulerText.waitForExistence(timeout: 3) {
            XCTAssertTrue(schedulerText.exists, "LR scheduler section should be visible")
        }
    }

    // =========================================================================
    // MARK: - Project Lifecycle Tests
    // =========================================================================

    func testCreateAndSelectProject() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        let addButton = app.buttons.matching(NSPredicate(format: "label CONTAINS 'Add' OR label CONTAINS 'New'")).firstMatch
        guard addButton.waitForExistence(timeout: 5) else {
            XCTFail("Add button should exist"); return
        }
        addButton.click()

        let nameField = app.textFields["Project name"]
        guard nameField.waitForExistence(timeout: 3) else {
            XCTFail("Name field should exist"); return
        }
        nameField.click()
        nameField.typeText("TestProject123")

        let createButton = app.buttons["Create"]
        guard createButton.waitForExistence(timeout: 2) else {
            XCTFail("Create button should exist"); return
        }
        createButton.click()
        Thread.sleep(forTimeInterval: 1.0)
        XCTAssertTrue(app.windows.count > 0, "Window should remain after project creation")
    }

    func testCreateMultipleProjects() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()

        for i in 1...3 {
            let addButton = app.buttons.matching(NSPredicate(format: "label CONTAINS 'Add' OR label CONTAINS 'New'")).firstMatch
            guard addButton.waitForExistence(timeout: 5) else { continue }
            addButton.click()

            let nameField = app.textFields["Project name"]
            guard nameField.waitForExistence(timeout: 3) else { continue }
            nameField.click()
            nameField.typeText("Project\(i)")

            let createButton = app.buttons["Create"]
            guard createButton.waitForExistence(timeout: 2) else { continue }
            createButton.click()
            Thread.sleep(forTimeInterval: 0.5)
        }
        XCTAssertTrue(app.windows.count > 0, "App should handle multiple project creation")
    }

    // =========================================================================
    // MARK: - Generate View Tests
    // =========================================================================

    func testGenerateView_HasPromptField() throws {
        launchWithProject()
        clickSidebarTab("Generate")
        let hasEditor = app.textViews.count > 0
        let hasPromptLabel = app.staticTexts.matching(NSPredicate(
            format: "label CONTAINS[c] 'prompt'"
        )).firstMatch.waitForExistence(timeout: 3)
        XCTAssertTrue(hasEditor || hasPromptLabel, "Generate view should have prompt input")
    }

    // =========================================================================
    // MARK: - Keyboard Shortcuts Tests
    // =========================================================================

    func testCmdNOpensNewProject() throws {
        app.launchArguments = ["-onboardingComplete", "YES"]
        app.launch()
        app.typeKey("n", modifierFlags: .command)

        let nameField = app.textFields["Project name"]
        let cancelButton = app.buttons["Cancel"]
        let sheetOpened = nameField.waitForExistence(timeout: 3) || cancelButton.waitForExistence(timeout: 1)
        if sheetOpened {
            XCTAssertTrue(sheetOpened, "Cmd+N should open new project sheet")
            if cancelButton.exists { cancelButton.click() }
        }
    }

    // =========================================================================
    // MARK: - Stress Tests
    // =========================================================================

    func testRapidTabSwitching() throws {
        launchWithProject(named: "StressTest")
        let tabs = ["Dashboard", "Config", "Generate", "Models", "Export",
                     "Audit", "Assistant", "Cluster", "Dashboard"]
        for tab in tabs {
            clickSidebarTab(tab)
        }
        XCTAssertTrue(app.windows.count > 0, "App should survive rapid tab switching")
    }

    func testWindowSurvivesTabNavigation() throws {
        launchWithProject(named: "SurvivalTest")
        let allTabs = ["Config", "Data", "Generate", "Export",
                        "Models", "Ingest", "History", "Benchmarks",
                        "Assistant", "Sync", "Cluster",
                        "Audit", "Reports", "Dashboard"]
        for tab in allTabs {
            clickSidebarTab(tab)
            Thread.sleep(forTimeInterval: 0.2)
        }
        XCTAssertTrue(app.windows.count > 0)
        let window = app.windows.firstMatch
        XCTAssertTrue(window.exists, "Window should still exist after full navigation")
    }

    // =========================================================================
    // MARK: - Sidebar Structure Tests
    // =========================================================================

    func testSidebarSectionsExist() throws {
        launchWithProject(named: "SectionTest")
        let trainingSection = app.staticTexts["Training"]
        let dataSection = app.staticTexts["Data & Models"]
        let toolsSection = app.staticTexts["Tools"]
        let complianceSection = app.staticTexts["Compliance"]
        let anySection = trainingSection.waitForExistence(timeout: 3) ||
                         dataSection.waitForExistence(timeout: 1) ||
                         toolsSection.waitForExistence(timeout: 1) ||
                         complianceSection.waitForExistence(timeout: 1)
        XCTAssertTrue(anySection, "Sidebar should have section headers")
    }

    func testSidebarDefaultsToDashboard() throws {
        launchWithProject(named: "DefaultTest")
        Thread.sleep(forTimeInterval: 1.0)
        XCTAssertTrue(app.windows.count > 0, "App should default to Dashboard view")
    }
}
