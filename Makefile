# NeuralForge — Root Makefile
# Run `make test-all` for one-click full test suite

.PHONY: build test test-cli test-swift test-all test-quick test-integration build-app verify clean help launch launch-posts metrics campaign-status campaign-hn screenshots

# ── One-Click Test Launchers ──

# Full test suite: unit + build + integration (~3 min)
test-all:
	@./test_all.sh --full

# Quick: unit tests only (~5 sec)
test-quick:
	@./test_all.sh --quick

# Integration tests only (real training on hardware)
test-integration:
	@cd cli && ./integration_test.sh

# XCUITests (needs Accessibility permission)
test-ui:
	@./test_all.sh --ui

# ── Individual Targets ──

# Build CLI binary
build:
	@echo "=== Building CLI ==="
	cd cli && make

# Run CLI tests (152 tests)
test-cli:
	@echo "=== CLI Tests ==="
	cd cli && make test

# Run Swift tests (416 tests)
test-swift:
	@echo "=== Swift Tests ==="
	cd app/Tests && swiftc -o test_swift -framework Foundation NeuralForgeTests.swift && ./test_swift

# Build macOS app via Xcode
build-app:
	@echo "=== Building App ==="
	cd app && xcodebuild -project NeuralForge.xcodeproj -scheme NeuralForge build 2>&1 | tail -5

# Run unit tests (CLI + Swift)
test: test-cli test-swift

# Full verification: build + test + app build
verify: build test build-app
	@echo ""
	@echo "✅ All checks passed"

# Quick check (just tests, no app build)
check: test
	@echo ""
	@echo "✅ All tests passed"

# Clean all build artifacts
clean:
	cd cli && make clean
	rm -f app/Tests/test_swift
	rm -rf /tmp/NF_TestBuild /tmp/NF_DerivedData

# ── Launch & Marketing ──

launch:
	@bash scripts/launch/launch.sh all

launch-posts:
	@bash scripts/launch/generate_posts.sh

metrics:
	@bash scripts/launch/track_metrics.sh

# Marketing campaign (scheduled posting)
campaign-status:
	@bash scripts/launch/campaign.sh status

campaign-hn:
	@bash scripts/launch/campaign.sh hn

campaign-reddit:
	@bash scripts/launch/campaign.sh reddit-localllama

campaign-twitter:
	@bash scripts/launch/campaign.sh twitter

campaign-devto:
	@bash scripts/launch/campaign.sh devto

campaign-ph:
	@bash scripts/launch/campaign.sh producthunt

# Screenshot capture (needs Screen Recording permission)
screenshots:
	@swift scripts/launch/capture_app.swift

# ── Help ──
help:
	@echo "NeuralForge Targets:"
	@echo ""
	@echo "  ${GREEN}make test-all${NC}     — One-click full suite: unit + build + integration (~3 min)"
	@echo "  make test-quick   — Unit tests only (~5 sec)"
	@echo "  make test-integration — Real training/generation on hardware"
	@echo "  make test-ui      — XCUITests (needs Accessibility permission)"
	@echo ""
	@echo "Individual:"
	@echo "  make test         — Unit tests (CLI + Swift)"
	@echo "  make test-cli     — CLI tests only (152 tests)"
	@echo "  make test-swift   — Swift tests only (416 tests)"
	@echo "  make build-app    — Build macOS app via Xcode"
	@echo "  make verify       — Full verification (build + tests + app)"
	@echo "  make clean        — Remove build artifacts"
	@echo ""
	@echo "Launch & Marketing:"
	@echo "  make launch       — Full launch preparation (GIF, posts, browser tabs)"
	@echo "  make campaign-status — View campaign posting schedule and status"
	@echo "  make campaign-hn  — Post to Hacker News"
	@echo "  make campaign-reddit — Post to Reddit r/LocalLLaMA"
	@echo "  make campaign-twitter — Post Twitter thread"
	@echo "  make campaign-devto — Post Dev.to article"
	@echo "  make campaign-ph  — Launch on Product Hunt"
	@echo "  make metrics      — Track GitHub metrics"
	@echo "  make screenshots  — Capture app screenshots (needs Screen Recording)"
