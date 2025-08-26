# ai-ox

A Rust workspace for AI model integrations with unified abstractions for multiple providers.

## Overview

ai-ox provides a unified interface for working with different AI models and providers through a common abstraction layer. The workspace supports a growing list of providers, including Anthropic, Google Gemini, Groq, Mistral, and OpenRouter, with an extensible architecture for adding more.

## Workspace Structure

- **`ai-ox`** - Core library with unified abstractions for AI models, agents, workflows, and tools.
- **`ai-ox-macros`** - Procedural macros and code generation utilities.
- **`anthropic-ox`** - Standalone client for the Anthropic API.
- **`gemini-ox`** - Standalone client for the Google Gemini API.
- **`groq-ox`** - Standalone client for the Groq API.
- **`mistral-ox`** - Standalone client for the Mistral API.
- **`openrouter-ox`** - Standalone client for the OpenRouter API.

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
ai-ox = { version = "0.1.0", features = ["anthropic", "gemini", "groq", "mistral", "openrouter"] }
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