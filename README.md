# ai-ox

A Rust workspace for AI model integrations with unified abstractions for multiple providers.

## Overview

`ai-ox` provides a unified interface for working with different AI models and providers through a common abstraction layer. The workspace is designed to be highly extensible, allowing developers to easily add new providers while maintaining a consistent API.

The project currently supports the following providers:
- **Anthropic** (`anthropic-ox`)
- **Google Gemini** (`gemini-ox`)
- **OpenAI** (`openai-ox`)
- **Mistral** (`mistral-ox`)
- **Groq** (`groq-ox`)
- **OpenRouter** (`openrouter-ox`)

## Shared Infrastructure: The `RequestBuilder` Pattern

A core design principle of `ai-ox` is to share as much infrastructure as possible between provider crates. This is achieved through the `ai-ox-common` crate, which provides a generic `RequestBuilder`.

This pattern eliminates thousands of lines of boilerplate code by centralizing:
- **HTTP Request Logic:** No more manual `reqwest` client setup for each API call.
- **Authentication:** A unified `AuthMethod` enum handles API keys, bearer tokens, and more.
- **Error Handling:** API errors are parsed into a common `CommonRequestError` type.
- **Streaming (SSE):** A shared `SseParser` handles Server-Sent Events for all providers.
- **Multipart Payloads:** A `MultipartForm` helper simplifies file uploads.

This approach makes the provider crates significantly leaner and easier to maintain. For more details, see the [`ai-ox-common` README](./crates/ai-ox-common/README.md).

## Workspace Structure

- **`ai-ox`** - Core library with unified abstractions for AI models, agents, workflows, and tools.
- **`ai-ox-common`** - Shared infrastructure for building provider clients (e.g., `RequestBuilder`).
- **`anthropic-ox`** - Anthropic API client.
- **`gemini-ox`** - Google Gemini API client.
- **`openai-ox`** - OpenAI API client.
- **`mistral-ox`** - Mistral API client.
- **`groq-ox`** - Groq API client.
- **`openrouter-ox`** - OpenRouter API client.
- **`ai-ox-macros`** - Procedural macros and code generation utilities.

## Features

- **Unified Agent Interface** - Common abstractions for different AI providers.
- **Shared RequestBuilder** - Centralized HTTP logic for all clients.
- **Workflow System** - Graph-based workflow execution with node management.
- **Tool Integration** - Structured tool calling with type safety.
- **Multimodal Support** - Text, image, audio, and video content handling.
- **Streaming Support** - Real-time content generation and live sessions.
- **Content Management** - Delta updates and message handling.
- **Usage Tracking** - Monitor API usage across providers.

## Getting Started

Add to your `Cargo.toml`:

```toml
[dependencies]
ai-ox = { version = "0.1.0", features = ["anthropic", "gemini", "openai"] }
```

## Usage Example

Here's a high-level example of how you might use the `ai-ox` crate to interact with a provider.

```rust
use ai_ox::agent::{Agent, AgentCompletion, AgentStream};
use ai_ox::content::Content;
use futures_util::stream::StreamExt;

// This example is conceptual and may require a specific provider implementation.
async fn run_agent_example() {
    // 1. Create a client for a specific provider (e.g., OpenAI)
    // let client = openai_ox::Client::new().with_api_key("YOUR_API_KEY");

    // 2. Create an agent from the client
    // let agent = Agent::from(client);

    // 3. Create a message to send to the agent
    // let message = Content::text("Hello, world!");

    // 4. Get a completion
    // let completion = agent.completion(vec![message]).await.unwrap();
    // println!("Response: {:?}", completion.content);

    // 5. Or, get a stream
    // let mut stream = agent.stream(vec![message]).await.unwrap();
    // while let Some(chunk) = stream.next().await {
    //     println!("Chunk: {:?}", chunk);
    // }
}
```

## Development

This project uses Cargo workspaces. To build all crates:

```bash
cargo build --all-targets
```

To run tests:

```bash
cargo test --all-targets
```

## License

This project is licensed under the terms of the MIT license. See `LICENSE.md` for more details.