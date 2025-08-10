# {{Provider}}-ox

{{Provider}} AI API client for Rust.

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
{{provider}}-ox = "0.1.0"

# Optional features
{{provider}}-ox = { version = "0.1.0", features = ["leaky-bucket", "schema"] }
```

## Quick Start

```rust
use {{provider}}_ox::{{Provider}};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client with API key
    let client = {{Provider}}::new("your-api-key-here");
    
    // Or load from environment variable {{ENV_VAR}}
    let client = {{Provider}}::from_env()?;
    
    // Create a chat request
    let request = client
        .chat()
        .model("{{default_model}}")
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
use {{provider}}_ox::{{Provider}};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = {{Provider}}::from_env()?;
    
    let request = client
        .chat()
        .model("{{default_model}}")
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
use {{provider}}_ox::{{{Provider}}, Tool};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = {{Provider}}::from_env()?;
    
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
        .model("{{default_model}}")
        .user("What's the weather like in New York?")
        .tool(weather_tool)
        .build();
    
    let response = client.send(&request).await?;
    println!("{:#?}", response);
    
    Ok(())
}
```

## Configuration

### Environment Variables

- `{{ENV_VAR}}` - Your {{Provider}} API key

### Features

- `leaky-bucket` - Enable rate limiting support
- `schema` - Enable JSON schema generation for tools

## Error Handling

The client provides detailed error types:

```rust
use {{provider}}_ox::{{{Provider}}, {{Provider}}RequestError};

match client.send(&request).await {
    Ok(response) => println!("Success: {:#?}", response),
    Err({{Provider}}RequestError::RateLimit) => {
        println!("Rate limited, please wait");
    }
    Err({{Provider}}RequestError::InvalidRequestError { message, .. }) => {
        println!("Invalid request: {}", message);
    }
    Err(e) => println!("Other error: {}", e),
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.