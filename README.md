# ai-ox

A Rust workspace for AI model integrations with unified abstractions for multiple providers.

## Overview

ai-ox provides a unified interface for working with different AI models and providers through a common abstraction layer. The workspace currently supports Anthropic, Google Gemini and OpenRouter, with extensible architecture for additional providers.

## Workspace Structure

- **`ai-ox`** - Core library with unified abstractions for AI models, agents, workflows, and tools
- **`anthropic-ox`** - Anthropic API client for text generation, and structured outputs.
- **`gemini-ox`** - Google Gemini API client with support for text generation, multimodal content, live sessions, and embeddings
- **`openrouter-ox`** - OpenRouter API client for accessing multiple AI models through a single interface
- **`ai-ox-macros`** - Procedural macros and code generation utilities

## Features

- **Unified Agent Interface** - Common abstractions for different AI providers
- **Workflow System** - Graph-based workflow execution with node management
- **Tool Integration** - Structured tool calling with type safety
- **Multimodal Support** - Text, image, audio, and video content handling
- **Streaming Support** - Real-time content generation and live sessions
- **Content Management** - Delta updates and message handling
- **Usage Tracking** - Monitor API usage across providers

## Getting Started

Add to your `Cargo.toml`:

```toml
[dependencies]
ai-ox = "0.1.0"
```

Enable specific providers:

```toml
[dependencies]
ai-ox = { version = "0.1.0", features = ["anthropic", "gemini", "openrouter"] }
```

## Usage

```rust
use ai-ox::agent::Agent;
use ai-ox::content::Message;

// Example usage would go here
```

## Development

This project uses Cargo workspaces. To build all crates:

```bash
cargo build
```

To run tests:

```bash
cargo test
```

## License

[License information not specified]