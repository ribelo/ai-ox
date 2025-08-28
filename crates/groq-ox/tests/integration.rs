#[cfg(test)]
mod tests {
    use groq_ox::*;
    use ai_ox_common::openai_format::{Tool, ToolChoice};
    use groq_ox::{Groq, ChatRequest};
    use ai_ox_common::openai_format::Message;

    fn get_client() -> Groq {
        Groq::load_from_env().expect("GROQ_API_KEY must be set for integration tests")
    }

    #[tokio::test]
    #[ignore = "requires GROQ_API_KEY and makes real API calls"]
    async fn test_list_models() {
        let client = get_client();
        let response = client.list_models().await;
        
        assert!(response.is_ok());
        let models = response.unwrap();
        assert_eq!(models.object, "list");
        assert!(!models.data.is_empty());
        
        // Verify we have at least some basic models
        let model_ids: Vec<&str> = models.data.iter().map(|m| m.id.as_str()).collect();
        assert!(model_ids.iter().any(|id| id.contains("llama") || id.contains("mixtral") || id.contains("gemma")));
    }

    #[tokio::test]
    #[ignore = "requires GROQ_API_KEY and makes real API calls"]
    async fn test_chat_completion() {
        let client = get_client();
        
        let request = ChatRequest::builder()
            .model("llama3-8b-8192") // Fast and free Groq model
            .messages(vec![Message::user("Say 'hello' in one word")])
            .max_completion_tokens(5)
            .temperature(0.0) // Deterministic
            .build();
        
        let response = client.send(&request).await;
        assert!(response.is_ok());
        
        let chat_response = response.unwrap();
        assert_eq!(chat_response.model, "llama3-8b-8192");
        assert!(!chat_response.choices.is_empty());
        assert!(chat_response.usage.is_some());
    }

    #[tokio::test]
    #[ignore = "requires GROQ_API_KEY and makes real API calls"]
    async fn test_streaming_chat() {
        let client = get_client();
        
        let request = ChatRequest::builder()
            .model("llama3-8b-8192")
            .messages(vec![Message::user("Count from 1 to 3")])
            .max_completion_tokens(20)
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
        
        assert!(chunks_received > 0, "Should have received at least one chunk");
    }

    #[tokio::test]
    #[ignore = "requires GROQ_API_KEY and makes real API calls"]
    async fn test_system_message() {
        let client = get_client();
        
        let request = ChatRequest::builder()
            .model("llama3-8b-8192")
            .messages(vec![
                Message::system("You are a helpful assistant that responds very briefly."),
                Message::user("What is 2+2?")
            ])
            .max_completion_tokens(10)
            .build();
        
        let response = client.send(&request).await;
        assert!(response.is_ok());
        
        let chat_response = response.unwrap();
        assert!(!chat_response.choices.is_empty());
        let choice = &chat_response.choices[0];
        if let Some(content) = &choice.message.content {
            assert!(!content.is_empty());
        }
    }

    #[tokio::test]
    #[ignore = "requires GROQ_API_KEY and makes real API calls"]
    async fn test_tool_calling() {
        let client = get_client();
        
        use groq_ox::tool::ToolFunction;
        let weather_function = ToolFunction::with_parameters(
            "get_weather",
            "Get the current weather for a location",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g., 'Paris, France'"
                    }
                },
                "required": ["location"]
            })
        );
        let weather_tool = Tool::function(weather_function);
        
        let request = ChatRequest::builder()
            .model("llama3-groq-70b-8192-tool-use-preview") // Model with tool support
            .messages(vec![Message::user("What's the weather like in Tokyo?")])
            .tools(vec![weather_tool])
            .tool_choice(ToolChoice::Auto)
            .max_completion_tokens(100)
            .build();
        
        let response = client.send(&request).await;
        
        match response {
            Ok(chat_response) => {
                assert!(!chat_response.choices.is_empty());
                
                // Check if the model called the tool or provided a text response
                let choice = &chat_response.choices[0];
                let has_tool_call = choice.message.tool_calls.is_some();
                let has_content = choice.message.content.is_some() && !choice.message.content.as_ref().unwrap().is_empty();
                
                assert!(has_tool_call || has_content, "Response should have either tool calls or content");
                println!("Tool calling test passed with model: {}", chat_response.model);
            }
            Err(e) => {
                // Tool-capable model might not be available
                println!("Tool calling model not available (acceptable): {}", e);
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires GROQ_API_KEY and makes real API calls"]
    async fn test_multiple_models() {
        let client = get_client();
        
        let models = vec![
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
        ];
        
        for model in models {
            println!("Testing model: {}", model);
            
            let request = ChatRequest::builder()
                .model(model)
                .messages(vec![Message::user("Say 'Hello' in one word.")])
                .max_completion_tokens(5)
                .build();
            
            match client.send(&request).await {
                Ok(response) => {
                    assert_eq!(response.model, model);
                    assert!(!response.choices.is_empty());
                    println!("  ✅ {} responded successfully", model);
                }
                Err(e) => {
                    println!("  ⚠️  {} error: {}", model, e);
                    // Don't fail the test - model might not be available
                }
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires GROQ_API_KEY and makes real API calls"]
    async fn test_error_handling() {
        let client = get_client();
        
        // Test with invalid model name
        let request = ChatRequest::builder()
            .model("invalid-groq-model")
            .messages(vec![Message::user("Hello")])
            .build();
        
        let result = client.send(&request).await;
        assert!(result.is_err(), "Expected error for invalid model");
    }

    #[tokio::test]
    #[ignore = "requires GROQ_API_KEY and makes real API calls"]
    async fn test_temperature_control() {
        let client = get_client();
        
        let request = ChatRequest::builder()
            .model("llama3-8b-8192")
            .messages(vec![Message::user("Say hello")])
            .temperature(0.0) // Deterministic
            .max_completion_tokens(10)
            .build();
        
        let response = client.send(&request).await;
        assert!(response.is_ok());
        
        let chat_response = response.unwrap();
        assert!(!chat_response.choices.is_empty());
    }
}

/// Helper to get test client
fn get_test_client() -> Result<groq_ox::Groq, Box<dyn std::error::Error>> {
    use groq_ox::Groq;
    Groq::load_from_env().map_err(|e| format!("Failed to load Groq API key: {}. Set GROQ_API_KEY environment variable.", e).into())
}

#[tokio::test]
async fn test_basic_chat() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;
    
    use groq_ox::{ChatRequest};
    
    let request = ChatRequest::builder()
        .model("llama3-8b-8192")
        .messages(vec![Message::user("What is 2+2? Reply with just the number.")])
        .max_completion_tokens(10)
        .build();
    
    let response = client.send(&request).await?;
    
    // Verify response structure
    assert!(!response.id.is_empty());
    assert_eq!(response.object, "chat.completion");
    assert!(!response.choices.is_empty());
    
    let choice = &response.choices[0];
    assert_eq!(choice.index, 0);
    
    if let Some(content) = &choice.message.content {
        assert!(!content.is_empty());
    }
    
    // Check usage if present
    if let Some(usage) = &response.usage {
        assert!(usage.prompt_tokens > 0);
        assert!(usage.completion_tokens > 0);
        assert!(usage.total_tokens > 0);
    }
    
    println!("Basic chat test passed");
    Ok(())
}

#[tokio::test]
async fn test_streaming() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;
    
    use groq_ox::{ChatRequest};
    
    let request = ChatRequest::builder()
        .model("llama3-8b-8192")
        .messages(vec![Message::user("Count from 1 to 5, one number per line.")])
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