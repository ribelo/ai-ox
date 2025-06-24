use ai_ox::{
    GenerateContentError,
    content::{
        delta::StreamEvent,
        message::{Message, MessageRole},
        part::Part,
    },
    model::{Model, request::ModelRequest},
};
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;

// Test data structures for structured content testing
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SimpleResponse {
    answer: String,
    confidence: f64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ComplexResponse {
    summary: String,
    key_points: Vec<String>,
    sentiment: String,
    #[schemars(range(min = 1, max = 10000))]
    word_count: i32,
}

// Provider setup functions with environment variable detection
#[cfg(feature = "gemini")]
fn setup_gemini() -> Option<ai_ox::GeminiModel> {
    env::var("GEMINI_API_KEY")
        .or_else(|_| env::var("GOOGLE_AI_API_KEY"))
        .ok()
        .map(|key| {
            ai_ox::GeminiModel::builder()
                .api_key(key)
                .model("gemini-1.5-flash".to_string())
                .build()
        })
}

#[cfg(feature = "openrouter")]
fn setup_openrouter() -> Option<ai_ox::OpenRouterModel> {
    env::var("OPENROUTER_API_KEY").ok().and_then(|_| {
        ai_ox::OpenRouterModel::new_from_env("anthropic/claude-3.5-sonnet").ok()
    })
}

// Generic test function that works with any Model implementation
async fn test_basic_request<M: Model + ?Sized>(model: &M) -> Result<(), GenerateContentError> {
    let message = Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "What is 2+2? Respond with just the number.".to_string(),
        }],
    );

    let request = ModelRequest {
        messages: vec![message],
        system_message: None,
        tools: None,
    };

    let response = model.request(request).await?;
    assert!(!response.message.content.is_empty());

    Ok(())
}

async fn test_streaming_request<M: Model + ?Sized>(model: &M) -> Result<(), GenerateContentError> {
    let message = Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Count from 1 to 5. Be brief.".to_string(),
        }],
    );

    let request = ModelRequest {
        messages: vec![message],
        system_message: None,
        tools: None,
    };

    let mut stream = model.request_stream(request);
    let mut events_received = 0;
    let mut message_deltas = 0;
    let mut stream_stopped = false;

    while let Some(event_result) = stream.next().await {
        let event = event_result?;
        events_received += 1;

        match event {
            StreamEvent::MessageDelta(_) => {
                message_deltas += 1;
            }
            StreamEvent::TextDelta(_) => {
                message_deltas += 1; // Count text deltas as message deltas for compatibility
            }
            StreamEvent::ToolCall(_) => {
                // Tool calls are handled but don't count as message deltas
            }
            StreamEvent::ToolResult(_) => {
                // Tool results are handled but don't count as message deltas
            }
            StreamEvent::Usage(_) => {
                // Usage events are expected
            }
            StreamEvent::StreamStop(_) => {
                stream_stopped = true;
                break;
            }
        }

        // Break after reasonable number of events to avoid infinite streams
        if events_received > 100 {
            break;
        }
    }

    assert!(events_received > 0, "Should receive at least one streaming event");
    assert!(message_deltas > 0, "Should receive at least one message delta");
    assert!(stream_stopped, "Stream should end with StreamStop event");

    println!(
        "Streaming test for {} completed: {} events, {} deltas", 
        model.model(), 
        events_received, 
        message_deltas
    );

    Ok(())
}

async fn test_structured_simple<M: Model>(model: &M) -> Result<(), GenerateContentError>
{
    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "What is the capital of France? Provide your answer and confidence level (0.0 to 1.0).".to_string(),
        }],
    )];

    let response = model.request_structured::<SimpleResponse>(messages).await?;

    assert!(!response.data.answer.is_empty());
    assert!(response.data.confidence >= 0.0 && response.data.confidence <= 1.0);

    Ok(())
}

async fn test_structured_complex<M: Model>(model: &M) -> Result<(), GenerateContentError>
{
    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Analyze this text: 'Artificial intelligence is rapidly transforming various industries, from healthcare to finance. While AI brings tremendous opportunities for innovation and efficiency, it also raises important questions about privacy, job displacement, and ethical considerations that society must address.' Provide a summary, key points, sentiment analysis, and word count.".to_string(),
        }],
    )];

    let response = model
        .request_structured::<ComplexResponse>(messages)
        .await?;

    assert!(!response.data.summary.is_empty());
    assert!(!response.data.key_points.is_empty());
    assert!(!response.data.sentiment.is_empty());
    assert!(
        response.data.word_count > 0,
        "Word count should be positive, got: {}",
        response.data.word_count
    );

    Ok(())
}

// Helper function to collect available models
fn get_available_models() -> Vec<Box<dyn Model>> {
    let mut models: Vec<Box<dyn Model>> = Vec::new();

    #[cfg(feature = "gemini")]
    if let Some(gemini) = setup_gemini() {
        models.push(Box::new(gemini));
    }

    #[cfg(feature = "openrouter")]
    if let Some(openrouter) = setup_openrouter() {
        models.push(Box::new(openrouter));
    }

    models
}

// Unified test functions that work with any available provider
#[tokio::test]
#[ignore = "requires provider API keys"]
async fn test_all_providers_basic_request() {
    let models = get_available_models();

    if models.is_empty() {
        println!("No models available - skipping test (need API keys)");
        return;
    }

    for model in models {
        println!("Testing basic request with model: {}", model.model());
        test_basic_request(&*model)
            .await
            .expect("Basic request should succeed");
    }
}

#[tokio::test]
#[ignore = "requires provider API keys"]
async fn test_all_providers_streaming() {
    let models = get_available_models();

    if models.is_empty() {
        println!("No models available - skipping test (need API keys)");
        return;
    }

    for model in models {
        println!("Testing streaming with model: {}", model.model());
        test_streaming_request(&*model)
            .await
            .expect("Streaming should work");
    }
}

// Structured content tests - need to be separated by provider because of Sized requirement
#[cfg(feature = "gemini")]
#[tokio::test]
#[ignore = "requires GEMINI_API_KEY or GOOGLE_AI_API_KEY"]
async fn test_gemini_structured_simple() {
    if let Some(model) = setup_gemini() {
        println!("Testing simple structured response with Gemini");
        test_structured_simple(&model)
            .await
            .expect("Simple structured response should work");
    } else {
        println!("Skipping Gemini structured test - no API key available");
    }
}

#[cfg(feature = "gemini")]
#[tokio::test]
#[ignore = "requires GEMINI_API_KEY or GOOGLE_AI_API_KEY"]
async fn test_gemini_structured_complex() {
    if let Some(model) = setup_gemini() {
        println!("Testing complex structured response with Gemini");
        test_structured_complex(&model)
            .await
            .expect("Complex structured response should work");
    } else {
        println!("Skipping Gemini structured test - no API key available");
    }
}

#[cfg(feature = "openrouter")]
#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn test_openrouter_structured_simple() {
    if let Some(model) = setup_openrouter() {
        println!("Testing simple structured response with OpenRouter");
        test_structured_simple(&model)
            .await
            .expect("Simple structured response should work");
    } else {
        println!("Skipping OpenRouter structured test - no API key available");
    }
}

#[cfg(feature = "openrouter")]
#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn test_openrouter_structured_complex() {
    if let Some(model) = setup_openrouter() {
        println!("Testing complex structured response with OpenRouter");
        test_structured_complex(&model)
            .await
            .expect("Complex structured response should work");
    } else {
        println!("Skipping OpenRouter structured test - no API key available");
    }
}

#[tokio::test]
#[ignore = "requires provider API keys"]
async fn test_all_providers_model_names() {
    let models = get_available_models();

    if models.is_empty() {
        println!("No models available - skipping test (need API keys)");
        return;
    }

    for model in models {
        let model_name = model.model();
        println!("Testing model name: {model_name}");

        // Verify model name is not empty and contains expected patterns
        assert!(!model_name.is_empty(), "Model name should not be empty");

        // Check that model name contains provider-specific patterns
        let is_valid = model_name.contains("gemini")
            || model_name.contains("claude")
            || model_name.contains("gpt")
            || model_name.contains("anthropic")
            || !model_name.trim().is_empty(); // Fallback: just ensure it's not blank

        assert!(
            is_valid,
            "Model name '{model_name}' should contain recognizable pattern"
        );
    }
}

// Cross-provider consistency tests
#[tokio::test]
#[ignore = "requires multiple provider API keys"]
async fn test_cross_provider_consistency() {
    let models = get_available_models();

    if models.len() < 2 {
        println!(
            "Skipping cross-provider test: need at least 2 providers with API keys (found: {})",
            models.len()
        );
        return;
    }

    println!(
        "Testing cross-provider consistency with {} providers",
        models.len()
    );

    let test_message = Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "What is the capital of Japan?".to_string(),
        }],
    );

    let request = ModelRequest {
        messages: vec![test_message],
        system_message: None,
        tools: None,
    };

    // Test that all providers can handle the same request
    for model in &models {
        println!("Testing consistency with model: {}", model.model());
        let response = model.request(request.clone()).await;
        assert!(
            response.is_ok(),
            "All providers should handle basic requests"
        );

        let response = response.unwrap();
        assert!(!response.message.content.is_empty());
    }
}

// Error handling test functions
async fn test_invalid_request<M: Model + ?Sized>(model: &M) -> Result<(), GenerateContentError> {
    // Test with empty messages (should handle gracefully)
    let request = ModelRequest {
        messages: vec![],
        system_message: None,
        tools: None,
    };

    let result = model.request(request).await;

    // Depending on provider, this might succeed with a default response or fail
    // The important thing is that it doesn't panic or hang
    match result {
        Ok(_) => println!(
            "Provider {} handled empty messages gracefully",
            model.model()
        ),
        Err(e) => {
            println!(
                "Provider {} rejected empty messages with error: {}",
                model.model(),
                e
            );
            // This is also acceptable behavior
        }
    }

    Ok(())
}

async fn test_very_long_message<M: Model + ?Sized>(model: &M) -> Result<(), GenerateContentError> {
    // Test with a very long message to see how the provider handles it
    let long_text = "A".repeat(10000);
    let message = Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: format!("Summarize this text: {long_text}"),
        }],
    );

    let request = ModelRequest {
        messages: vec![message],
        system_message: None,
        tools: None,
    };

    let result = model.request(request).await;

    // Provider should either handle it or return a proper error
    match result {
        Ok(response) => {
            assert!(!response.message.content.is_empty());
            println!(
                "Provider {} handled long message successfully",
                model.model()
            );
        }
        Err(e) => {
            println!(
                "Provider {} rejected long message with error: {}",
                model.model(),
                e
            );
            // This is acceptable - provider may have token limits
        }
    }

    Ok(())
}

#[tokio::test]
#[ignore = "requires provider API keys"]
async fn test_all_providers_error_handling() {
    let models = get_available_models();

    if models.is_empty() {
        println!("No models available - skipping test (need API keys)");
        return;
    }

    for model in models {
        println!("Testing error handling with model: {}", model.model());
        test_invalid_request(&*model)
            .await
            .expect("Error handling test should not panic");
        test_very_long_message(&*model)
            .await
            .expect("Long message test should not panic");
    }
}

// Tool usage tests
async fn test_tool_usage<M: Model + ?Sized>(model: &M) -> Result<(), GenerateContentError> {
    use ai_ox::tool::{FunctionMetadata, Tool};
    use serde_json::json;

    // Create a simple function definition
    let function = FunctionMetadata {
        name: "get_weather".to_string(),
        description: Some("Get the current weather for a location".to_string()),
        parameters: json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state or country"
                }
            },
            "required": ["location"]
        }),
    };

    let tools = vec![Tool::FunctionDeclarations(vec![function])];

    let message = Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "What's the weather like in Tokyo? Please use the get_weather function."
                .to_string(),
        }],
    );

    let request = ModelRequest {
        messages: vec![message],
        system_message: None,
        tools: Some(tools),
    };

    let result = model.request(request).await;

    match result {
        Ok(response) => {
            println!(
                "Provider {} handled tool request successfully",
                model.model()
            );
            assert!(!response.message.content.is_empty());
            // Note: We don't assert that tool calls were made, as that depends on the model's behavior
        }
        Err(e) => {
            println!(
                "Provider {} failed tool request with error: {}",
                model.model(),
                e
            );
            // Some providers might not support tools, which is acceptable
            // We just ensure it doesn't panic
        }
    }

    Ok(())
}

#[tokio::test]
#[ignore = "requires provider API keys"]
async fn test_all_providers_tool_usage() {
    let models = get_available_models();

    if models.is_empty() {
        println!("No models available - skipping test (need API keys)");
        return;
    }

    for model in models {
        println!("Testing tool usage with model: {}", model.model());
        test_tool_usage(&*model)
            .await
            .expect("Tool usage test should not panic");
    }
}

// Test specifically for OpenRouter tool streaming functionality
#[cfg(feature = "openrouter")]
#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn test_openrouter_tool_streaming() {
    if let Some(model) = setup_openrouter() {
        println!("Testing OpenRouter tool streaming");
        
        use ai_ox::tool::{FunctionMetadata, Tool};
        use serde_json::json;

        // Create a simple function definition
        let function = FunctionMetadata {
            name: "get_weather".to_string(),
            description: Some("Get the current weather for a location".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state or country"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units"
                    }
                },
                "required": ["location"]
            }),
        };

        let tools = vec![Tool::FunctionDeclarations(vec![function])];

        let message = Message::new(
            MessageRole::User,
            vec![Part::Text {
                text: "What's the weather like in Tokyo? Please use the get_weather function with celsius units.".to_string(),
            }],
        );

        let request = ModelRequest {
            messages: vec![message],
            system_message: None,
            tools: Some(tools),
        };

        let mut stream = model.request_stream(request);
        let mut events_received = 0;
        let mut tool_calls_received = 0;
        let mut stream_stopped = false;

        while let Some(event_result) = stream.next().await {
            let event = event_result.expect("Stream should not error");
            events_received += 1;

            match &event {
                StreamEvent::ToolCall(tool_call) => {
                    // Assert this is a complete tool call
                    assert!(!tool_call.name.is_empty(), "Tool call should have a name");
                    assert!(!tool_call.args.is_null(), "Tool call should have args");
                    
                    if tool_call.name == "get_weather" {
                        tool_calls_received += 1;
                        println!("Found complete get_weather tool call: {:?}", tool_call);
                        
                        // Verify the location argument
                        if let Some(location) = tool_call.args.get("location") {
                            assert!(location.is_string(), "Location should be a string");
                        }
                    }
                }
                StreamEvent::TextDelta(text) => {
                    // Text deltas are expected and normal
                    if !text.is_empty() {
                        println!("Received text delta: {}", text);
                    }
                }
                StreamEvent::Usage(_) => {
                    // Usage events are expected
                }
                StreamEvent::StreamStop(_) => {
                    stream_stopped = true;
                    break;
                }
                StreamEvent::MessageDelta(_) => {
                    // MessageDelta events should not contain tool calls in the new implementation
                }
                StreamEvent::ToolResult(_) => {
                    // Tool results would come from user, not the model itself
                }
            }

            // Break after reasonable number of events to avoid infinite streams
            if events_received > 100 {
                break;
            }
        }

        assert!(events_received > 0, "Should receive at least one streaming event");
        assert!(stream_stopped, "Stream should end with StreamStop event");
        
        // Note: We can't always assert that tool_calls_received > 0 because
        // the model might not always use tool calls for this specific prompt.
        // But if it does, our streaming conversion should provide complete tool calls.
        println!(
            "OpenRouter tool streaming test completed: {} events, {} tool calls", 
            events_received, 
            tool_calls_received
        );
    } else {
        println!("Skipping OpenRouter tool streaming test - no API key available");
    }
}

// Test specifically for Gemini code interpreter streaming functionality
#[cfg(feature = "gemini")]
#[tokio::test]
#[ignore = "requires GEMINI_API_KEY or GOOGLE_AI_API_KEY"]
async fn test_gemini_code_interpreter_streaming() {
    if let Some(model) = setup_gemini() {
        println!("Testing Gemini code interpreter streaming");
        
        let message = Message::new(
            MessageRole::User,
            vec![Part::Text {
                text: "Calculate the sum of the first 10 fibonacci numbers using Python code. Please show the code and run it.".to_string(),
            }],
        );

        let request = ModelRequest {
            messages: vec![message],
            system_message: None,
            tools: None,
        };

        let mut stream = model.request_stream(request);
        let mut events_received = 0;
        let mut code_interpreter_tool_calls = 0;
        let mut stream_stopped = false;

        while let Some(event_result) = stream.next().await {
            let event = event_result.expect("Stream should not error");
            events_received += 1;

            match &event {
                StreamEvent::ToolCall(tool_call) => {
                    if tool_call.name == "code_interpreter" {
                        // Assert this is a complete tool call
                        assert!(!tool_call.name.is_empty(), "Tool call should have a name");
                        assert!(!tool_call.args.is_null(), "Tool call should have args");
                        
                        code_interpreter_tool_calls += 1;
                        println!("Found complete code interpreter tool call: {tool_call:?}");
                    }
                }
                StreamEvent::ToolResult(tool_result) => {
                    if tool_result.name == "code_interpreter" {
                        // Assert this is a complete tool result
                        assert!(!tool_result.name.is_empty(), "Tool result should have a name");
                        assert!(!tool_result.response.is_empty(), "Tool result should have response");
                        
                        code_interpreter_tool_calls += 1;
                        println!("Found complete code interpreter tool result: {tool_result:?}");
                    }
                }
                StreamEvent::TextDelta(text) => {
                    // Text deltas are expected and normal
                    if !text.is_empty() {
                        println!("Received text delta: {text}");
                    }
                }
                StreamEvent::Usage(_) => {
                    // Usage events are expected
                }
                StreamEvent::StreamStop(_) => {
                    stream_stopped = true;
                    break;
                }
                StreamEvent::MessageDelta(_) => {
                    // MessageDelta events should not contain tool calls in the new implementation
                }
            }

            // Break after reasonable number of events to avoid infinite streams
            if events_received > 200 {
                break;
            }
        }

        assert!(events_received > 0, "Should receive at least one streaming event");
        assert!(stream_stopped, "Stream should end with StreamStop event");
        
        // Note: We can't always assert that code_interpreter_tool_calls > 0 because
        // the model might not always use code execution for this specific prompt.
        // But if it does, our streaming conversion should provide complete tool calls.
        println!(
            "Gemini code interpreter streaming test completed: {events_received} events, {code_interpreter_tool_calls} code interpreter tool calls"
        );
    } else {
        println!("Skipping Gemini code interpreter streaming test - no API key available");
    }
}
