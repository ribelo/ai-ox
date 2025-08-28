use gemini_ox::prelude::*;
use futures_util::StreamExt;
use gemini_ox::content::{Content, Part, Role};
use gemini_ox::tool::{Tool, FunctionMetadata};
use gemini_ox::generate_content::request::GenerateContentRequest;
use gemini_ox::embedding::request::EmbedContentRequest;
use gemini_ox::Gemini;
use gemini_ox::generate_content::{SafetySettings, HarmCategory, HarmBlockThreshold};

#[cfg(test)]
mod tests {
    use super::*;

    fn get_client() -> Gemini {
        Gemini::load_from_env().expect("GEMINI_API_KEY must be set for integration tests")
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY and makes real API calls"]
    async fn test_list_models() {
        let client = get_client();
        let response = client.list_models(None, None).await;
        
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
            .model("gemini-1.5-flash".to_string()) // Fast and cheap Gemini model
            .content(Content::new(Role::User, vec![Part::from("Say 'hello' in one word")]))
            .generation_config(GenerationConfig {
                max_output_tokens: Some(5),
                temperature: Some(0.0), // Deterministic
                ..Default::default()
            })
            .build();
        
        let response = request.send(&client).await;
        assert!(response.is_ok());
        
        let generate_response = response.unwrap();
        assert!(!generate_response.candidates.is_empty());
        
        let candidate = &generate_response.candidates[0];
        assert!(!candidate.content.parts.is_empty());
        
        if let Some(usage) = &generate_response.usage_metadata {
            assert!(usage.prompt_token_count > 0);
            assert!(usage.candidates_token_count > Some(0));
        }
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY and makes real API calls"]
    async fn test_streaming_generate_content() {
        let client = get_client();
        
        let request = GenerateContentRequest::builder()
            .model("gemini-1.5-flash".to_string())
            .content(Content::new(Role::User, vec![Part::from("Count from 1 to 3")]))
            .build();
        
        let mut stream = request.stream(&client);
        
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
    async fn test_function_calling() {
        let client = get_client();
        
        let weather_tool = Tool::FunctionDeclarations(vec![FunctionMetadata {
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
        }]);
        
        let request = GenerateContentRequest::builder()
            .model("gemini-1.5-flash".to_string())
            .content(Content::new(Role::User, vec![Part::from("What's the weather like in Tokyo?")]))
            .tools(vec![weather_tool])
            .build();
        
        let response = request.send(&client).await;
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
            .gemini(client.clone())
            .model("text-embedding-004".to_string()) // Gemini embedding model
            .content(Content::new(Role::User, vec![Part::from("Hello, world! This is a test embedding.")]))
            .build();
        
        let response = request.send().await;
        
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

    /*
    /*
    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY and makes real API calls"]
    async fn test_batch_embed_contents() {
        let client = get_client();
        
        let request = client.batch_embed_contents(vec![
                EmbedContentRequest::builder()
                    .gemini(client.clone())
                    .model("text-embedding-004".to_string())
                    .content(Content::new(Role::User, vec![Part::from("First text to embed")]))
                    .build(),
                EmbedContentRequest::builder()
                    .gemini(client.clone())
                    .model("text-embedding-004".to_string())
                    .content(Content::new(Role::User, vec![Part::from("Second text to embed")]))
                    .build(),
            ]);
        
        let response = request.await;
        
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
    */
    */

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY and makes real API calls"]
    async fn test_error_handling() {
        let client = get_client();
        
        // Test with invalid model name
        let request = GenerateContentRequest::builder()
            .model("invalid-gemini-model".to_string())
            .content(Content::new(Role::User, vec![Part::from("Hello")]))
            .build();
        
        let result = request.send(&client).await;
        assert!(result.is_err(), "Expected error for invalid model");
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY and makes real API calls"]
    async fn test_safety_settings() {
        let client = get_client();
        
        let request = GenerateContentRequest::builder()
            .model("gemini-1.5-flash".to_string())
            .content(Content::new(Role::User, vec![Part::from("Tell me about artificial intelligence")]))
            .safety_settings(SafetySettings::default().with_category(HarmCategory::HarmCategoryHarassment, HarmBlockThreshold::BlockOnlyHigh))
            .build();
        
        let response = request.send(&client).await;
        assert!(response.is_ok());
        
        let generate_response = response.unwrap();
        assert!(!generate_response.candidates.is_empty());
    }
}

/// Helper to get test client
fn get_test_client() -> Result<Gemini, Box<dyn std::error::Error>> {
    Gemini::load_from_env().map_err(|e| format!("Failed to load Gemini API key: {}. Set GEMINI_API_KEY environment variable.", e).into())
}

#[tokio::test]
async fn test_basic_generate() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;
    
    let request = GenerateContentRequest::builder()
        .model("gemini-1.5-flash".to_string())
        .content(Content::new(Role::User, vec![Part::from("What is 2+2? Reply with just the number.")]))
        .generation_config(GenerationConfig {
            max_output_tokens: Some(10),
            ..Default::default()
        })
        .build();
    
    let response = request.send(&client).await?;
    
    // Verify response structure
    assert!(!response.candidates.is_empty());
    
    let candidate = &response.candidates[0];
    assert!(!candidate.content.parts.is_empty());
    
    // Check usage if present
    if let Some(usage) = &response.usage_metadata {
        assert!(usage.prompt_token_count > 0);
        assert!(usage.candidates_token_count > Some(0));
    }
    
    println!("Basic generate test passed");
    Ok(())
}

#[tokio::test]
async fn test_streaming() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;
    
    let request = GenerateContentRequest::builder()
        .model("gemini-1.5-flash".to_string())
        .content(Content::new(Role::User, vec![Part::from("Count from 1 to 5, one number per line.")]))
        .build();
    
    let mut stream = request.stream(&client);
    let mut chunks_received = 0;
    let mut content = String::new();
    let mut finish_reason_received = false;
    
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        chunks_received += 1;
        
        if let Some(candidate) = chunk.candidates.first() {
            for part in &candidate.content.parts {
                if let Some(text) = part.as_text() {
                    content.push_str(&text.to_string());
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