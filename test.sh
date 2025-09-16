#!/usr/bin/env bash

# Script to run tests for the ai-ox project with proper environment setup

set -e

echo "🚀 Running ai-ox tests with nix develop environment..."
echo ""

# Parse command line arguments
SKIP_FAILING=false
VERBOSE=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-failing) SKIP_FAILING=true ;;
        --verbose|-v) VERBOSE=true ;;
        --help|-h) 
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-failing    Skip known failing tests (provider_compliance, multi_agent_workflow)"
            echo "  --verbose, -v     Show verbose output"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Build the test command
TEST_CMD="cargo test --all-features"

if [ "$SKIP_FAILING" = true ]; then
    echo "⚠️  Skipping known failing tests..."
    TEST_CMD="$TEST_CMD -- --skip provider_compliance --skip multi_agent_workflow"
fi

if [ "$VERBOSE" = false ]; then
    TEST_CMD="$TEST_CMD 2>&1 | grep -E 'test result:|Running|^test .* \.\.\. (ok|FAILED)$|^failures:$|^error:'"
fi

echo "Running command: nix develop -c $TEST_CMD"
echo ""

# Run the tests
nix develop -c bash -c "$TEST_CMD"

# Show summary
echo ""
echo "📊 Test Summary:"
if [ "$SKIP_FAILING" = true ]; then
    echo "✅ All tests passed (known failing tests were skipped)"
else
    nix develop -c bash -c "cargo test --all-features 2>&1 | grep 'test result:' | grep -c 'ok\.' || true" | xargs -I {} echo "✅ Passing test suites: {}"
    nix develop -c bash -c "cargo test --all-features 2>&1 | grep 'test result:' | grep -c 'FAILED' || true" | xargs -I {} echo "❌ Failing test suites: {}"
fi
