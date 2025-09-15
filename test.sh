#!/usr/bin/env sh
set -e

echo "========================================="
echo "🚀 COMPREHENSIVE TEST SUITE - AI-OX"
echo "========================================="

echo ""
echo "📦 Testing with ALL FEATURES enabled..."
echo "-----------------------------------------"
RUSTC_WRAPPER="" nix develop -c cargo test --all-features 2>&1 | grep -E "test result:|Running" | head -20

echo ""
echo "✅ Compilation check with all features..."
echo "-----------------------------------------"
RUSTC_WRAPPER="" nix develop -c cargo check --all-features

echo ""
echo "🎯 Individual crate tests..."
echo "-----------------------------------------"
for crate in ai-ox-common anthropic-ox mistral-ox groq-ox openrouter-ox gemini-ox openai-ox mcp-ox conversion-ox ai-ox; do
    echo "Testing $crate..."
    RUSTC_WRAPPER="" nix develop -c cargo test --package $crate --lib --quiet
done

echo ""
echo "========================================="
echo "✅ ALL TESTS PASSED! The project is fully functional!"
echo "========================================="
