#![cfg(feature = "openrouter")]

#[cfg(test)]
mod tests {
    use openrouter_ox::message::Message;
    use openrouter_ox::tool::{FunctionMetadata, Tool, ToolChoice};
    use openrouter_ox::*;
    use openrouter_ox::{ChatRequest, OpenRouter};

    fn get_client() -> OpenRouter {
        OpenRouter::load_from_env().expect("OPENROUTER_API_KEY must be set for integration tests")
    }

    #[tokio::test]
    #[ignore = "requires OPENROUTER_API_KEY and makes real API calls"]
    async fn test_list_models() {
        let client = get_client();
        let response = client.list_models().await;

        assert!(response.is_ok());
        let models = response.unwrap();
        assert_eq!(models.object, "list");
        assert!(!models.data.is_empty());

        // Verify we have at least some basic models
        let model_ids: Vec<&str> = models.data.iter().map(|m| m.id.as_str()).collect();
        assert!(
            model_ids
                .iter()
                .any(|id| id.contains("gpt") || id.contains("claude") || id.contains("llama"))
        );
    }

    #[tokio::test]
    #[ignore = "requires OPENROUTER_API_KEY and makes real API calls"]
    async fn test_chat_completion() {
        let client = get_client();

        let request = ChatRequest::builder()
            .model("openrouter/auto") // Let OpenRouter choose cheapest available
            .messages(vec![Message::user("Say 'hello' in one word")])
            .max_tokens(5)
            .temperature(0.0) // Deterministic
            .build();

        let response = client.send(&request).await;
        assert!(response.is_ok());

        let chat_response = response.unwrap();
        assert!(!chat_response.choices.is_empty());
        if chat_response.usage.prompt_tokens > 0 {
            // Usage is present - good
        }
    }

    #[tokio::test]
    #[ignore = "requires OPENROUTER_API_KEY and makes real API calls"]
    async fn test_streaming_chat() {
        let client = get_client();

        let request = ChatRequest::builder()
            .model("openrouter/auto")
            .messages(vec![Message::user("Count from 1 to 3")])
            .max_tokens(20)
            .temperature(0.0)
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
    #[ignore = "requires OPENROUTER_API_KEY and makes real API calls"]
    async fn test_specific_model() {
        let client = get_client();

        // Use a specific free/cheap model if available
        let request = ChatRequest::builder()
            .model("meta-llama/llama-3.2-1b-instruct:free") // Free Llama model
            .messages(vec![Message::user(
                "What is 2+2? Reply with just the number.",
            )])
            .max_tokens(5)
            .build();

        let response = client.send(&request).await;
        // This might fail if the specific model is not available
        // In that case, we just verify the error handling works
        match response {
            Ok(chat_response) => {
                assert!(!chat_response.choices.is_empty());
                println!("Successfully used specific model: {}", chat_response.model);
            }
            Err(e) => {
                println!("Model not available (expected): {}", e);
                // This is acceptable as model availability changes
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires OPENROUTER_API_KEY and makes real API calls"]
    async fn test_provider_preferences() {
        let client = get_client();

        // Test with provider preferences
        let mut request = ChatRequest::builder()
            .model("openrouter/auto")
            .messages(vec![Message::user("Hello")])
            .max_tokens(10)
            .build();

        // Add provider preferences in the request
        use openrouter_ox::provider_preference::{Provider, ProviderPreferences};
        request.provider = Some(ProviderPreferences {
            allow_fallbacks: None,
            require_parameters: None,
            data_collection: None,
            order: Some(vec![
                Provider::OpenAI,
                Provider::Anthropic,
                Provider::Google,
            ]),
            only: None,
            ignore: None,
            quantizations: None,
            sort: None,
            max_price: None,
        });

        let response = client.send(&request).await;
        assert!(response.is_ok());

        let chat_response = response.unwrap();
        assert!(!chat_response.choices.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires OPENROUTER_API_KEY and makes real API calls"]
    async fn test_generation_info() {
        let client = get_client();

        let generation_id = "test-generation-id".to_string();
        let response = client.get_generation(&generation_id).await;

        // This will likely fail with a not found error, which is expected
        // We're just testing that the endpoint is callable
        match response {
            Ok(_info) => {
                println!("Generation info retrieved successfully");
            }
            Err(e) => {
                println!("Generation not found (expected): {}", e);
                // This is expected behavior for a random generation ID
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires OPENROUTER_API_KEY and makes real API calls"]
    async fn test_error_handling() {
        let client = get_client();

        // Test with invalid model name
        let request = ChatRequest::builder()
            .model("invalid/model-name-that-does-not-exist")
            .messages(vec![Message::user("Hello")])
            .build();

        let result = client.send(&request).await;
        assert!(result.is_err(), "Expected error for invalid model");
    }

    #[tokio::test]
    #[ignore = "requires OPENROUTER_API_KEY and makes real API calls"]
    async fn test_tool_calling() {
        let client = get_client();

        let weather_tool = Tool {
            tool_type: "function".to_string(),
            function: FunctionMetadata {
                name: "get_weather".to_string(),
                description: Some("Get the current weather for a location".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and country, e.g., 'Paris, France'"
                        }
                    },
                    "required": ["location"]
                }),
            },
        };

        let request = ChatRequest::builder()
            .model("openrouter/auto")
            .messages(vec![Message::user("What's the weather like in Tokyo?")])
            .tools(vec![weather_tool])
            .tool_choice(ToolChoice::Auto)
            .max_tokens(100)
            .build();

        let response = client.send(&request).await;
        assert!(response.is_ok());

        let chat_response = response.unwrap();
        assert!(!chat_response.choices.is_empty());

        // Check if the model called the tool or provided a text response
        let choice = &chat_response.choices[0];
        let has_tool_call = choice.message.tool_calls.is_some();
        let has_content = !choice.message.content.is_empty();

        assert!(
            has_tool_call || has_content,
            "Response should have either tool calls or content"
        );
    }
}

/// Helper to get test client
fn get_test_client() -> Result<openrouter_ox::OpenRouter, Box<dyn std::error::Error>> {
    use openrouter_ox::message::Message;
    use openrouter_ox::{ChatRequest, OpenRouter};
    OpenRouter::load_from_env().map_err(|e| {
        format!(
            "Failed to load OpenRouter API key: {}. Set OPENROUTER_API_KEY environment variable.",
            e
        )
        .into()
    })
}

#[tokio::test]
async fn test_basic_chat() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;

    use openrouter_ox::ChatRequest;
    use openrouter_ox::message::Message;

    let request = ChatRequest::builder()
        .model("openrouter/auto")
        .messages(vec![Message::user(
            "What is 2+2? Reply with just the number.",
        )])
        .max_tokens(10)
        .build();

    let response = client.send(&request).await?;

    // Verify response structure
    assert!(!response.id.is_empty());
    assert_eq!(response.object, "chat.completion");
    assert!(!response.choices.is_empty());

    let choice = &response.choices[0];
    assert_eq!(choice.index, 0);

    assert!(!choice.message.content.is_empty());

    // Check usage if present
    let usage = &response.usage;
    assert!(usage.prompt_tokens > 0);
    assert!(usage.completion_tokens > 0);
    assert!(usage.total_tokens > 0);

    println!("Basic chat test passed");
    Ok(())
}

#[tokio::test]
async fn test_streaming() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;

    use openrouter_ox::ChatRequest;
    use openrouter_ox::message::Message;

    let request = ChatRequest::builder()
        .model("openrouter/auto")
        .messages(vec![Message::user(
            "Count from 1 to 5, one number per line.",
        )])
        .build();

    let mut stream = client.stream(&request);
    use futures_util::StreamExt;
    let mut chunks_received = 0;
    let mut content = String::new();
    let mut finish_reason_received = false;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        chunks_received += 1;

        if let Some(choice) = chunk.choices.first() {
            if let Some(delta_content) = &choice.delta.content {
                content.push_str(delta_content);
            }

            if choice.finish_reason.is_some() {
                finish_reason_received = true;
            }
        }
    }

    assert!(chunks_received > 0, "No chunks received from stream");
    assert!(!content.is_empty(), "No content received from stream");
    assert!(finish_reason_received, "No finish reason received");

    println!("Streamed content ({} chunks): {}", chunks_received, content);
    println!("Streaming test passed");
    Ok(())
}
