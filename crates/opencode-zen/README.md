# opencode-zen

Rust client for the Opencode Zen OpenAI-compatible API. This crate mirrors the
`chat/completions` surface used across other `ai-ox` providers while reusing the
shared OpenAI-format types from `ai-ox-common`.

```rust,no_run
use opencode-zen::{Message, OpencodeZen};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpencodeZen::new().with_api_key("your-api-key");

    let request = client
        .chat()
        .messages([
            Message::system("You are a helpful assistant."),
            Message::user("Say hello"),
        ])
        .model("grok-code")
        .build();

    let response = client.send(&request).await?;
    println!("{}", response.choices[0].message.content.as_deref().unwrap_or(""));
    Ok(())
}
```
