# OpenAI-ox

OpenAI AI API client for Rust.

## Features

- **Chat Completions**: Send messages and receive AI responses
- **Streaming**: Real-time response streaming
- **Tool Calling**: Function/tool integration
- **Rate Limiting**: Built-in rate limiting support (optional)
- **Error Handling**: Comprehensive error types
- **Type Safety**: Full Rust type safety with serde
- **Async/Await**: Built on tokio for async programming

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
openai-ox = "0.1.0"

# Optional features
openai-ox = { version = "0.1.0", features = ["leaky-bucket", "schema"] }
```

## Quick Start

```rust
use openai_ox::OpenAI;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client with API key
    let client = OpenAI::new("your-api-key-here");

    // Or load from environment variable OPENAI_API_KEY
    let client = OpenAI::from_env()?;

    // Create a chat request
    let request = client
        .chat()
        .model("gpt-3.5-turbo")
        .user("Hello, world!")
        .build();

    // Send the request
    let response = client.send(&request).await?;

    // Print the response
    if let Some(content) = response.content() {
        println!("Assistant: {}", content);
    }

    Ok(())
}
```

## Streaming Responses

```rust
use openai_ox::OpenAI;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAI::from_env()?;

    let request = client
        .chat()
        .model("gpt-3.5-turbo")
        .user("Write a haiku about Rust")
        .build();

    let mut stream = client.stream(&request).await?;

    while let Some(result) = stream.next().await {
        match result {
            Ok(response) => {
                if let Some(content) = response.content() {
                    print!("{}", content);
                }
            }
            Err(e) => eprintln!("Stream error: {}", e),
        }
    }

    Ok(())
}
```

## Tool/Function Calling

```rust
use openai_ox::{OpenAI, Tool};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAI::from_env()?;

    // Define a tool
    let weather_tool = Tool::function_with_params(
        "get_weather",
        "Get current weather for a location",
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        })
    );

    let request = client
        .chat()
        .model("gpt-3.5-turbo")
        .user("What's the weather like in New York?")
        .tool(weather_tool)
        .build();

    let response = client.send(&request).await?;
    if let Some(tool_call) = response.choices[0].message.tool_calls.as_ref().and_then(|t| t.first()) {
        println!("Tool call: {:?}", tool_call);
    } else {
        println!("No tool call made.");
    }

    Ok(())
}
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key

### Features

- `leaky-bucket` - Enable rate limiting support
- `schema` - Enable JSON schema generation for tools

## Error Handling

The client provides detailed error types:

```rust
use openai_ox::{OpenAI, OpenAIRequestError};

match client.send(&request).await {
    Ok(response) => {
        if let Some(content) = response.content() {
            println!("Success: {}", content);
        } else {
            println!("Success, but no content.");
        }
    }
    Err(OpenAIRequestError::RateLimit) => {
        println!("Rate limited, please wait");
    }
    Err(OpenAIRequestError::InvalidRequestError { message, .. }) => {
        println!("Invalid request: {}", message);
    }
    Err(e) => println!("Other error: {}", e),
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.