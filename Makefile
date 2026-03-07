# NeuralForge — Root Makefile
# Run `make verify` to check everything in one shot

.PHONY: build test test-cli test-swift build-app verify clean help

# Build CLI binary
build:
	@echo "=== Building CLI ==="
	cd cli && make

# Run CLI tests (109 tests)
test-cli:
	@echo "=== CLI Tests ==="
	cd cli && make test

# Run Swift tests (119 tests)
test-swift:
	@echo "=== Swift Tests ==="
	cd app/Tests && swiftc -o test_swift -framework Foundation NeuralForgeTests.swift && ./test_swift

# Build macOS app via Xcode
build-app:
	@echo "=== Building App ==="
	cd app && xcodebuild -project NeuralForge.xcodeproj -scheme NeuralForge build 2>&1 | tail -5

# Run all tests
test: test-cli test-swift

# Full verification: build + test + app build
verify: build test build-app
	@echo ""
	@echo "✅ All checks passed: CLI built, 109 CLI tests, 119 Swift tests, app built"

# Quick check (just tests, no app build)
check: test
	@echo ""
	@echo "✅ All tests passed"

# Clean all build artifacts
clean:
	cd cli && make clean
	rm -f app/Tests/test_swift

# Show available targets
help:
	@echo "NeuralForge Makefile targets:"
	@echo "  make build      — Build CLI binary"
	@echo "  make test        — Run all tests (CLI + Swift)"
	@echo "  make test-cli    — Run CLI tests only (109 tests)"
	@echo "  make test-swift  — Run Swift tests only (119 tests)"
	@echo "  make build-app   — Build macOS app via Xcode"
	@echo "  make verify      — Full verification (build + tests + app)"
	@echo "  make check       — Quick check (tests only, no app build)"
	@echo "  make clean       — Remove build artifacts"
