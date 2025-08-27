#[cfg(test)]
mod tests {
    use gemini_ox::*;
    use gemini_ox::message::{Message, Part, Content, Text};

    fn get_client() -> Gemini {
        Gemini::from_env().expect("GEMINI_API_KEY must be set for integration tests")
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY and makes real API calls"]
    async fn test_list_models() {
        let client = get_client();
        let response = client.list_models().await;
        
        assert!(response.is_ok());
        let models = response.unwrap();
        assert!(!models.models.is_empty());
        
        // Verify we have at least some basic models
        let model_names: Vec<&str> = models.models.iter().map(|m| m.name.as_str()).collect();
        assert!(model_names.iter().any(|name| name.contains("gemini")));
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY and makes real API calls"]
    async fn test_generate_content() {
        let client = get_client();
        
        let request = GenerateContentRequest::builder()
            .model("gemini-1.5-flash") // Fast and cheap Gemini model
            .contents(vec![Content {
                parts: vec![Part::Text(Text {
                    text: "Say 'hello' in one word".to_string(),
                })],
                role: Some("user".to_string()),
            }])
            .generation_config(Some(GenerationConfig {
                max_output_tokens: Some(5),
                temperature: Some(0.0), // Deterministic
                ..Default::default()
            }))
            .build();
        
        let response = client.generate_content(&request).await;
        assert!(response.is_ok());
        
        let generate_response = response.unwrap();
        assert!(!generate_response.candidates.is_empty());
        
        let candidate = &generate_response.candidates[0];
        assert!(!candidate.content.parts.is_empty());
        
        if let Some(usage) = &generate_response.usage_metadata {
            assert!(usage.prompt_token_count > 0);
            assert!(usage.candidates_token_count > 0);
        }
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY and makes real API calls"]
    async fn test_streaming_generate_content() {
        let client = get_client();
        
        let request = GenerateContentRequest::builder()
            .model("gemini-1.5-flash")
            .contents(vec![Content {
                parts: vec![Part::Text(Text {
                    text: "Count from 1 to 3".to_string(),
                })],
                role: Some("user".to_string()),
            }])
            .generation_config(Some(GenerationConfig {
                max_output_tokens: Some(20),
                temperature: Some(0.0),
                ..Default::default()
            }))
            .build();
        
        let mut stream = client.stream_generate_content(&request);
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
    #[ignore = "requires GEMINI_API_KEY and makes real API calls"]
    async fn test_chat_session() {
        let client = get_client();
        
        // Start a chat session
        let chat_request = ChatRequest::builder()
            .model("gemini-1.5-flash")
            .messages(vec![Message::user("Hello, I'm starting a conversation.")])
            .build();
        
        let response = client.send(&chat_request).await;
        assert!(response.is_ok());
        
        let chat_response = response.unwrap();
        assert!(!chat_response.candidates.is_empty());
        
        let candidate = &chat_response.candidates[0];
        assert!(!candidate.content.parts.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY and makes real API calls"]
    async fn test_function_calling() {
        let client = get_client();
        
        let weather_tool = Tool::new("get_weather")
            .with_function_declaration(FunctionDeclaration {
                name: "get_weather".to_string(),
                description: "Get the current weather for a location".to_string(),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and country, e.g., 'Paris, France'"
                        }
                    },
                    "required": ["location"]
                })),
            });
        
        let request = GenerateContentRequest::builder()
            .model("gemini-1.5-flash")
            .contents(vec![Content {
                parts: vec![Part::Text(Text {
                    text: "What's the weather like in Tokyo?".to_string(),
                })],
                role: Some("user".to_string()),
            }])
            .tools(vec![weather_tool])
            .build();
        
        let response = client.generate_content(&request).await;
        assert!(response.is_ok());
        
        let generate_response = response.unwrap();
        assert!(!generate_response.candidates.is_empty());
        
        let candidate = &generate_response.candidates[0];
        // Gemini might or might not call the function, both are valid responses
        assert!(!candidate.content.parts.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY and makes real API calls"]
    async fn test_embed_content() {
        let client = get_client();
        
        let request = EmbedContentRequest::builder()
            .model("text-embedding-004") // Gemini embedding model
            .content(Content {
                parts: vec![Part::Text(Text {
                    text: "Hello, world! This is a test embedding.".to_string(),
                })],
                role: None,
            })
            .build();
        
        let response = client.embed_content(&request).await;
        
        match response {
            Ok(embed_response) => {
                assert!(!embed_response.embedding.values.is_empty());
                println!("Embedding generated successfully with {} dimensions", embed_response.embedding.values.len());
            }
            Err(e) => {
                // Embedding model might not be available in all regions
                println!("Embedding model not available (acceptable): {}", e);
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY and makes real API calls"]
    async fn test_batch_embed_contents() {
        let client = get_client();
        
        let request = BatchEmbedContentsRequest::builder()
            .model("text-embedding-004")
            .requests(vec![
                EmbedContentRequest::builder()
                    .model("text-embedding-004")
                    .content(Content {
                        parts: vec![Part::Text(Text {
                            text: "First text to embed".to_string(),
                        })],
                        role: None,
                    })
                    .build(),
                EmbedContentRequest::builder()
                    .model("text-embedding-004")
                    .content(Content {
                        parts: vec![Part::Text(Text {
                            text: "Second text to embed".to_string(),
                        })],
                        role: None,
                    })
                    .build(),
            ])
            .build();
        
        let response = client.batch_embed_contents(&request).await;
        
        match response {
            Ok(batch_response) => {
                assert_eq!(batch_response.embeddings.len(), 2);
                for embedding in &batch_response.embeddings {
                    assert!(!embedding.values.is_empty());
                }
                println!("Batch embedding successful");
            }
            Err(e) => {
                // Batch embedding might not be available
                println!("Batch embedding not available (acceptable): {}", e);
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY and makes real API calls"]
    async fn test_error_handling() {
        let client = get_client();
        
        // Test with invalid model name
        let request = GenerateContentRequest::builder()
            .model("invalid-gemini-model")
            .contents(vec![Content {
                parts: vec![Part::Text(Text {
                    text: "Hello".to_string(),
                })],
                role: Some("user".to_string()),
            }])
            .build();
        
        let result = client.generate_content(&request).await;
        assert!(result.is_err(), "Expected error for invalid model");
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY and makes real API calls"]
    async fn test_safety_settings() {
        let client = get_client();
        
        let request = GenerateContentRequest::builder()
            .model("gemini-1.5-flash")
            .contents(vec![Content {
                parts: vec![Part::Text(Text {
                    text: "Tell me about artificial intelligence".to_string(),
                })],
                role: Some("user".to_string()),
            }])
            .safety_settings(vec![SafetySetting {
                category: HarmCategory::HarassmentHate,
                threshold: HarmBlockThreshold::BlockOnlyHigh,
            }])
            .build();
        
        let response = client.generate_content(&request).await;
        assert!(response.is_ok());
        
        let generate_response = response.unwrap();
        assert!(!generate_response.candidates.is_empty());
    }
}

/// Helper to get test client
fn get_test_client() -> Result<Gemini, Box<dyn std::error::Error>> {
    Gemini::from_env().map_err(|e| format!("Failed to load Gemini API key: {}. Set GEMINI_API_KEY environment variable.", e).into())
}

#[tokio::test]
async fn test_basic_generate() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;
    
    let request = GenerateContentRequest::builder()
        .model("gemini-1.5-flash")
        .contents(vec![Content {
            parts: vec![Part::Text(Text {
                text: "What is 2+2? Reply with just the number.".to_string(),
            })],
            role: Some("user".to_string()),
        }])
        .generation_config(Some(GenerationConfig {
            max_output_tokens: Some(10),
            ..Default::default()
        }))
        .build();
    
    let response = client.generate_content(&request).await?;
    
    // Verify response structure
    assert!(!response.candidates.is_empty());
    
    let candidate = &response.candidates[0];
    assert!(!candidate.content.parts.is_empty());
    
    // Check usage if present
    if let Some(usage) = &response.usage_metadata {
        assert!(usage.prompt_token_count > 0);
        assert!(usage.candidates_token_count > 0);
    }
    
    println!("Basic generate test passed");
    Ok(())
}

#[tokio::test]
async fn test_streaming() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;
    
    let request = GenerateContentRequest::builder()
        .model("gemini-1.5-flash")
        .contents(vec![Content {
            parts: vec![Part::Text(Text {
                text: "Count from 1 to 5, one number per line.".to_string(),
            })],
            role: Some("user".to_string()),
        }])
        .build();
    
    let mut stream = client.stream_generate_content(&request);
    use futures_util::StreamExt;
    let mut chunks_received = 0;
    let mut content = String::new();
    let mut finish_reason_received = false;
    
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        chunks_received += 1;
        
        if let Some(candidate) = chunk.candidates.first() {
            for part in &candidate.content.parts {
                if let Part::Text(text) = part {
                    content.push_str(&text.text);
                }
            }
            
            if candidate.finish_reason.is_some() {
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