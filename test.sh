#!/usr/bin/env bash
set -e

echo "Running ai-ox workspace tests with proper features..."

echo "=== Testing ai-ox-common ==="
cargo test -p ai-ox-common

echo "=== Testing anthropic-ox ==="
cargo test -p anthropic-ox --features full

echo "=== Testing gemini-ox ==="
cargo test -p gemini-ox

echo "=== Testing openrouter-ox ==="
cargo test -p openrouter-ox

echo "=== Testing openai-ox ==="  
cargo test -p openai-ox

echo "=== Testing mistral-ox ==="
cargo test -p mistral-ox

echo "=== Testing groq-ox ==="
cargo test -p groq-ox

echo "=== Testing conversion-ox ==="
cargo test -p conversion-ox --features "anthropic-gemini anthropic-openrouter"

echo "=== Testing ai-ox ==="
cargo test -p ai-ox --features "anthropic gemini openai openrouter mistral groq bedrock"

echo "All tests completed!"