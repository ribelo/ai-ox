#![cfg(feature = "anthropic")]

use anthropic_ox::message::{Content, Message, Messages, Role, Text};
use anthropic_ox::response::ContentBlockDelta;
use anthropic_ox::{Anthropic, ChatRequest, Model, StreamEvent};

#[tokio::test]
async fn test_anthropic_client_initialization() {
    // Test that we can create an Anthropic client with manual API key
    let client = Anthropic::new("test-api-key");

    // Basic structure tests
    assert!(!format!("{:?}", client).contains("test-api-key")); // Should be redacted in debug

    println!("✅ Anthropic client initialization works");
}

#[tokio::test]
async fn test_anthropic_load_from_env() {
    // Test loading from environment - should gracefully handle missing key
    match Anthropic::load_from_env() {
        Ok(_client) => {
            println!("✅ ANTHROPIC_API_KEY found, client loaded successfully");
        }
        Err(_) => {
            println!("ℹ️  ANTHROPIC_API_KEY not found, which is expected for CI/CD");
        }
    }
}

#[tokio::test]
async fn test_model_enum_conversion() {
    // Test that our Model enum converts to strings correctly
    assert_eq!(
        Model::Claude35Sonnet20241022.to_string(),
        "claude-3-5-sonnet-20241022"
    );
    assert_eq!(
        Model::Claude35SonnetLatest.to_string(),
        "claude-3-5-sonnet-latest"
    );
    assert_eq!(
        Model::Claude35Haiku20241022.to_string(),
        "claude-3-5-haiku-20241022"
    );
    assert_eq!(
        Model::Claude3Opus20240229.to_string(),
        "claude-3-opus-20240229"
    );
    assert_eq!(
        Model::Claude3Haiku20240307.to_string(),
        "claude-3-haiku-20240307"
    );

    // Test string conversion
    let model_string: String = Model::Claude35Sonnet20241022.into();
    assert_eq!(model_string, "claude-3-5-sonnet-20241022");

    println!("✅ Model enum conversions work correctly");
}

#[tokio::test]
async fn test_chat_request_builder() {
    use anthropic_ox::{
        ChatRequest,
        message::{Content, Message, Messages, Role, StringOrContents, Text},
    };

    // Create test messages
    let mut messages = Messages::new();
    messages.push(Message::new(
        Role::User,
        vec![Content::Text(Text::new("Hello".to_string()))],
    ));

    // Test building a chat request
    let request = ChatRequest::builder()
        .model("claude-3-5-sonnet-20241022")
        .messages(messages.clone())
        .max_tokens(1000)
        .maybe_system(Some(StringOrContents::String(
            "You are a helpful assistant".to_string(),
        )))
        .build();

    assert_eq!(request.model, "claude-3-5-sonnet-20241022");
    assert_eq!(request.max_tokens, 1000);
    assert_eq!(
        request.system,
        Some(StringOrContents::String(
            "You are a helpful assistant".to_string()
        ))
    );
    assert_eq!(request.messages.len(), 1);

    // Test without system message
    let request_no_system = ChatRequest::builder()
        .model("claude-3-haiku-20240307")
        .messages(messages)
        .max_tokens(500)
        .build();

    assert_eq!(request_no_system.model, "claude-3-haiku-20240307");
    assert_eq!(request_no_system.max_tokens, 500);
    assert_eq!(request_no_system.system, None);

    println!("✅ ChatRequest builder works correctly");
}

#[tokio::test]
async fn test_message_structures() {
    use anthropic_ox::message::{Content, ImageSource, Message, Messages, Role, Text};

    // Test text content
    let text_content = Content::Text(Text::new("Hello, world!".to_string()));
    assert!(matches!(text_content, Content::Text { .. }));

    // Test image content
    let image_source = ImageSource::Base64 {
        media_type: "image/png".to_string(),
        data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==".to_string(),
    };
    let image_content = Content::Image {
        source: image_source,
    };
    assert!(matches!(image_content, Content::Image { .. }));

    // Test message creation
    let user_message = Message::new(
        Role::User,
        vec![Content::Text(Text::new("Hello".to_string()))],
    );
    assert_eq!(user_message.role, Role::User);
    assert_eq!(user_message.len(), 1);

    let assistant_message = Message::new(
        Role::Assistant,
        vec![Content::Text(Text::new("Hi there!".to_string()))],
    );
    assert_eq!(assistant_message.role, Role::Assistant);
    assert_eq!(assistant_message.len(), 1);

    // Test messages collection
    let mut messages = Messages::new();
    messages.push(user_message);
    messages.push(assistant_message);
    assert_eq!(messages.len(), 2);
    assert!(!messages.is_empty());

    println!("✅ Message structures work correctly");
}

#[tokio::test]
async fn test_error_types() {
    use anthropic_ox::{AnthropicRequestError, error::ErrorInfo};

    // Test error conversion from ErrorInfo
    let error_info = ErrorInfo {
        r#type: "invalid_request_error".to_string(),
        message: "Test error message".to_string(),
    };

    let error: AnthropicRequestError = error_info.into();
    match error {
        AnthropicRequestError::InvalidRequestError { message, .. } => {
            assert_eq!(message, "Test error message");
        }
        _ => panic!("Expected InvalidRequestError"),
    }

    // Test rate limit error
    let rate_limit_info = ErrorInfo {
        r#type: "rate_limit_error".to_string(),
        message: "Rate limit exceeded".to_string(),
    };

    let rate_error: AnthropicRequestError = rate_limit_info.into();
    assert!(matches!(rate_error, AnthropicRequestError::RateLimit));

    println!("✅ Error types work correctly");
}

// Real integration tests using actual API calls
mod real_api_tests {
    use super::*;
    use anthropic_ox::{
        ChatRequest,
        message::{Content, Message, Messages, Role, Text},
        tool::Tool,
    };

    fn get_client() -> Anthropic {
        Anthropic::load_from_env().expect("ANTHROPIC_API_KEY must be set for integration tests")
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY and makes real API calls"]
    async fn test_chat_completion() {
        let client = get_client();

        let mut messages = Messages::new();
        messages.push(Message::new(
            Role::User,
            vec![Content::Text(Text::new(
                "Say 'hello' in one word".to_string(),
            ))],
        ));

        let request = ChatRequest::builder()
            .model("claude-3-haiku-20240307") // Cheapest Claude model
            .messages(messages)
            .max_tokens(5)
            .build();

        let response = client.send(&request).await;
        assert!(response.is_ok());

        let chat_response = response.unwrap();
        assert_eq!(chat_response.model, "claude-3-haiku-20240307");
        assert!(!chat_response.content.is_empty());
        assert!(
            chat_response.usage.input_tokens.is_some()
                || chat_response.usage.output_tokens.is_some()
        );
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY and makes real API calls"]
    async fn test_streaming_chat() {
        let client = get_client();

        let mut messages = Messages::new();
        messages.push(Message::new(
            Role::User,
            vec![Content::Text(Text::new("Count from 1 to 3".to_string()))],
        ));

        let request = ChatRequest::builder()
            .model("claude-3-haiku-20240307")
            .messages(messages)
            .max_tokens(20)
            .build();

        let mut stream = client.stream(&request);
        use futures_util::StreamExt;

        let mut chunks_received = 0;
        while let Some(chunk_result) = stream.next().await {
            assert!(chunk_result.is_ok());
            chunks_received += 1;
            if chunks_received > 10 {
                break; // Prevent infinite loops
            }
        }

        assert!(
            chunks_received > 0,
            "Should have received at least one chunk"
        );
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY and makes real API calls"]
    async fn test_system_message() {
        let client = get_client();

        let mut messages = Messages::new();
        messages.push(Message::new(
            Role::User,
            vec![Content::Text(Text::new("What is 2+2?".to_string()))],
        ));

        let request = ChatRequest::builder()
            .model("claude-3-haiku-20240307")
            .system("You are a helpful assistant that responds very briefly.".into())
            .messages(messages)
            .max_tokens(10)
            .build();

        let response = client.send(&request).await;
        assert!(response.is_ok());

        let chat_response = response.unwrap();
        assert!(!chat_response.content.is_empty());
        assert!(
            chat_response.usage.input_tokens.is_some()
                || chat_response.usage.output_tokens.is_some()
        );
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY and makes real API calls"]
    async fn test_tool_calling() {
        let client = get_client();

        let weather_tool = Tool::Custom(tool::CustomTool {
            object_type: "custom".to_string(),
            name: "get_weather".to_string(),
            description: "Get the current weather for a location".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g., 'Paris, France'"
                    }
                },
                "required": ["location"]
            }),
        });

        let mut messages = Messages::new();
        messages.push(Message::new(
            Role::User,
            vec![Content::Text(Text::new(
                "What's the weather like in Tokyo?".to_string(),
            ))],
        ));

        let request = ChatRequest::builder()
            .model("claude-3-haiku-20240307")
            .messages(messages)
            .tools(vec![weather_tool])
            .max_tokens(100)
            .build();

        let response = client.send(&request).await;
        assert!(response.is_ok());

        let chat_response = response.unwrap();
        // Claude might or might not call the tool, both are valid responses
        assert!(!chat_response.content.is_empty() || chat_response.has_tool_use());
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY and makes real API calls"]
    async fn test_error_handling() {
        let client = get_client();

        let mut messages = Messages::new();
        messages.push(Message::new(
            Role::User,
            vec![Content::Text(Text::new("Hello".to_string()))],
        ));

        // Test with invalid model name
        let request = ChatRequest::builder()
            .model("invalid-claude-model")
            .messages(messages)
            .build();

        let result = client.send(&request).await;
        assert!(result.is_err(), "Expected error for invalid model");
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY and makes real API calls"]
    async fn test_multiple_messages() {
        let client = get_client();

        let mut messages = Messages::new();
        messages.push(Message::new(
            Role::User,
            vec![Content::Text(Text::new(
                "What is the capital of France?".to_string(),
            ))],
        ));
        messages.push(Message::new(
            Role::Assistant,
            vec![Content::Text(Text::new(
                "The capital of France is Paris.".to_string(),
            ))],
        ));
        messages.push(Message::new(
            Role::User,
            vec![Content::Text(Text::new("What about Italy?".to_string()))],
        ));

        let request = ChatRequest::builder()
            .model("claude-3-haiku-20240307")
            .messages(messages)
            .max_tokens(50)
            .build();

        let response = client.send(&request).await;
        assert!(response.is_ok());

        let chat_response = response.unwrap();
        assert!(!chat_response.content.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY and makes real API calls"]
    async fn test_temperature_control() {
        let client = get_client();

        let mut messages = Messages::new();
        messages.push(Message::new(
            Role::User,
            vec![Content::Text(Text::new("Say hello".to_string()))],
        ));

        let request = ChatRequest::builder()
            .model("claude-3-haiku-20240307")
            .messages(messages)
            .temperature(0.0) // Deterministic
            .max_tokens(10)
            .build();

        let response = client.send(&request).await;
        assert!(response.is_ok());

        let chat_response = response.unwrap();
        assert!(!chat_response.content.is_empty());
    }
}

/// Helper to get test client
fn get_test_client() -> Result<Anthropic, Box<dyn std::error::Error>> {
    Anthropic::load_from_env().map_err(|e| {
        format!(
            "Failed to load Anthropic API key: {}. Set ANTHROPIC_API_KEY environment variable.",
            e
        )
        .into()
    })
}

#[tokio::test]
async fn test_basic_chat() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;

    let mut messages = Messages::new();
    messages.push(Message::new(
        Role::User,
        vec![Content::Text(Text::new(
            "What is 2+2? Reply with just the number.".to_string(),
        ))],
    ));

    let request = ChatRequest::builder()
        .model("claude-3-haiku-20240307")
        .messages(messages)
        .max_tokens(10)
        .build();

    let response = client.send(&request).await?;

    // Verify response structure
    assert!(!response.id.is_empty());
    assert_eq!(response.r#type, "message");
    assert!(!response.content.is_empty());

    // Check usage if present
    if let Some(usage) = &response.usage {
        assert!(usage.input_tokens > 0);
        assert!(usage.output_tokens > 0);
    }

    println!("Basic chat test passed");
    Ok(())
}

#[tokio::test]
async fn test_streaming() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;

    let mut messages = Messages::new();
    messages.push(Message::new(
        Role::User,
        vec![Content::Text(Text::new(
            "Count from 1 to 5, one number per line.".to_string(),
        ))],
    ));

    let request = ChatRequest::builder()
        .model("claude-3-haiku-20240307")
        .messages(messages)
        .build();

    let mut stream = client.stream(&request);
    use futures_util::StreamExt;
    let mut chunks_received = 0;
    let mut content = String::new();
    let mut finish_reason_received = false;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        chunks_received += 1;

        match chunk {
            StreamEvent::ContentBlockDelta { delta, .. } => {
                if let ContentBlockDelta::TextDelta { text } = delta {
                    content.push_str(&text);
                }
            }
            StreamEvent::MessageStop => {
                finish_reason_received = true;
            }
            _ => {}
        }
    }

    assert!(chunks_received > 0, "No chunks received from stream");
    assert!(!content.is_empty(), "No content received from stream");
    assert!(finish_reason_received, "No stop event received");

    println!("Streamed content ({} chunks): {}", chunks_received, content);
    println!("Streaming test passed");
    Ok(())
}
