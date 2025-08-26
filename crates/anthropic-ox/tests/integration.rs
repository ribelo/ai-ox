use anthropic_ox::{Anthropic, Model};

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
    assert_eq!(Model::Claude35Sonnet20241022.to_string(), "claude-3-5-sonnet-20241022");
    assert_eq!(Model::Claude35SonnetLatest.to_string(), "claude-3-5-sonnet-latest");
    assert_eq!(Model::Claude35Haiku20241022.to_string(), "claude-3-5-haiku-20241022");
    assert_eq!(Model::Claude3Opus20240229.to_string(), "claude-3-opus-20240229");
    assert_eq!(Model::Claude3Haiku20240307.to_string(), "claude-3-haiku-20240307");
    
    // Test string conversion
    let model_string: String = Model::Claude35Sonnet20241022.into();
    assert_eq!(model_string, "claude-3-5-sonnet-20241022");
    
    println!("✅ Model enum conversions work correctly");
}

#[tokio::test]
async fn test_chat_request_builder() {
    use anthropic_ox::{ChatRequest, message::{Message, Messages, Role, Content, StringOrContents, Text}};
    
    // Create test messages
    let mut messages = Messages::new();
    messages.push(Message::new(Role::User, vec![Content::Text(Text::new("Hello".to_string()))]));
    
    // Test building a chat request
    let request = ChatRequest::builder()
        .model("claude-3-5-sonnet-20241022")
        .messages(messages.clone())
        .max_tokens(1000)
        .maybe_system(Some(StringOrContents::String("You are a helpful assistant".to_string())))
        .build();
    
    assert_eq!(request.model, "claude-3-5-sonnet-20241022");
    assert_eq!(request.max_tokens, 1000);
    assert_eq!(request.system, Some(StringOrContents::String("You are a helpful assistant".to_string())));
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
    use anthropic_ox::message::{Message, Messages, Role, Content, ImageSource, Text};
    
    // Test text content
    let text_content = Content::from("Hello, world!");
    if let Content::Text(text) = &text_content {
        assert_eq!(text.text, "Hello, world!");
    } else {
        panic!("Expected Content::Text");
    }
    
    // Test image content
    let image_source = ImageSource::Base64 {
        media_type: "image/png".to_string(),
        data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==".to_string(),
    };
    let image_content = Content::Image { source: image_source };
    assert!(matches!(image_content, Content::Image { .. }));
    
    // Test message creation
    let user_message = Message::user(vec![Content::from("Hello")]);
    assert_eq!(user_message.role, Role::User);
    assert_eq!(user_message.len(), 1);
    
    let assistant_message = Message::assistant(vec![Content::from("Hi there!")]);
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

// Integration test that requires actual API key - will be skipped if not available
#[tokio::test]
async fn test_actual_api_call() {
    use anthropic_ox::{ChatRequest, message::{Message, Messages, Role, Content, Text}};
    
    // Skip test if no API key is available
    let client = match Anthropic::load_from_env() {
        Ok(client) => client,
        Err(_) => {
            println!("ℹ️  Skipping API test: ANTHROPIC_API_KEY not found");
            return;
        }
    };
    
    // Create a simple chat request
    let mut messages = Messages::new();
    messages.push(Message::new(Role::User, vec![Content::Text(Text::new("Say 'Hello from Anthropic!'".to_string()))]));
    
    let request = ChatRequest::builder()
        .model("claude-3-haiku-20240307")
        .messages(messages)
        .max_tokens(50)
        .build();
    
    // Make the API call
    match client.send(&request).await {
        Ok(response) => {
            println!("✅ API call successful!");
            println!("   Response ID: {}", response.id);
            println!("   Model: {}", response.model);
            assert_eq!(response.role, Role::Assistant);
            assert!(!response.content.is_empty());
            
            // Print first bit of response text
            if let Some(Content::Text(text)) = response.content.first() {
                println!("   Content preview: {}...", text.text.chars().take(50).collect::<String>());
            }
        }
        Err(e) => {
            println!("⚠️  API call failed (this might be expected): {}", e);
            // Don't fail the test for API errors as they might be due to invalid keys, rate limits, etc.
        }
    }
}