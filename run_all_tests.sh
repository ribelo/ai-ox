#!/usr/bin/env sh
set -e

echo "Running all tests without video feature..."
export RUSTC_WRAPPER=""

# Test all packages except gemini-ox with all features
echo "Testing all crates except gemini-ox with all features..."
cargo test --workspace --exclude gemini-ox --all-features

# Test gemini-ox without video feature  
echo "Testing gemini-ox without video feature..."
cargo test --package gemini-ox --no-default-features --features "leaky-bucket,audio,audio-output"

echo "All tests passed successfully!"
