#[cfg(test)]
mod tests {
    use openai_ox::*;

    fn get_client() -> OpenAI {
        OpenAI::from_env().expect("OPENAI_API_KEY must be set for integration tests")
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and makes real API calls"]
    async fn test_list_models() {
        let client = get_client();
        let response = client.list_models().await;
        
        assert!(response.is_ok());
        let models = response.unwrap();
        assert_eq!(models.object, "list");
        assert!(!models.data.is_empty());
        
        // Verify we have at least one of the basic models
        let model_ids: Vec<&str> = models.data.iter().map(|m| m.id.as_str()).collect();
        assert!(model_ids.contains(&"gpt-3.5-turbo") || model_ids.contains(&"gpt-4"));
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and makes real API calls"]
    async fn test_create_embeddings() {
        let client = get_client();
        
        let request = EmbeddingsRequest::builder()
            .model("text-embedding-3-small".to_string()) // Cheapest embedding model
            .input(EmbeddingInput::Single("Hello, world!".to_string()))
            .build();
        
        let response = client.create_embeddings(&request).await;
        assert!(response.is_ok());
        
        let embeddings = response.unwrap();
        assert_eq!(embeddings.model, "text-embedding-3-small");
        assert_eq!(embeddings.data.len(), 1);
        assert!(!embeddings.data[0].embedding.is_empty());
        assert!(embeddings.usage.prompt_tokens > 0);
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and makes real API calls"]
    async fn test_create_moderation() {
        let client = get_client();
        
        let request = ModerationRequest::builder()
            .input(ModerationInput::Single("Hello, this is a test message.".to_string()))
            .build();
        
        let response = client.create_moderation(&request).await;
        assert!(response.is_ok());
        
        let moderation = response.unwrap();
        assert_eq!(moderation.results.len(), 1);
        assert!(!moderation.results[0].flagged); // Should not be flagged for benign content
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and makes real API calls"]
    async fn test_chat_completion() {
        let client = get_client();
        
        let request = client.chat()
            .model("gpt-3.5-turbo") // Cheapest chat model
            .user_message("Say 'hello' in one word")
            .max_tokens(5)
            .temperature(0.0) // Deterministic
            .build();
        
        let response = client.send(&request).await;
        assert!(response.is_ok());
        
        let chat_response = response.unwrap();
        assert_eq!(chat_response.model, "gpt-3.5-turbo");
        assert!(!chat_response.choices.is_empty());
        assert!(chat_response.content().is_some());
        assert!(chat_response.usage.is_some());
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and makes real API calls"]
    async fn test_list_files() {
        let client = get_client();
        
        let response = client.list_files().await;
        // This should succeed even if there are no files
        assert!(response.is_ok());
        
        let files = response.unwrap();
        assert_eq!(files.object, "list");
        // files.data can be empty, that's fine
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and makes real API calls"]
    async fn test_list_fine_tuning_jobs() {
        let client = get_client();
        
        let response = client.list_fine_tuning_jobs().await;
        // This should succeed even if there are no jobs
        assert!(response.is_ok());
        
        let jobs = response.unwrap();
        assert_eq!(jobs.object, "list");
        // jobs.data can be empty, that's fine
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and makes real API calls"]
    async fn test_list_assistants() {
        let client = get_client();
        
        let response = client.list_assistants().await;
        // This should succeed even if there are no assistants
        assert!(response.is_ok());
        
        let assistants = response.unwrap();
        assert_eq!(assistants.object, "list");
        // assistants.data can be empty, that's fine
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and makes real API calls"] 
    async fn test_streaming_chat() {
        let client = get_client();
        
        let request = client.chat()
            .model("gpt-3.5-turbo")
            .user_message("Count from 1 to 3")
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
        
        assert!(chunks_received > 0, "Should have received at least one chunk");
    }

    // Test embeddings with multiple inputs to verify batch processing
    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and makes real API calls"]
    async fn test_embeddings_multiple_inputs() {
        let client = get_client();
        
        let request = EmbeddingsRequest::builder()
            .model("text-embedding-3-small".to_string())
            .input(EmbeddingInput::Multiple(vec![
                "First text".to_string(),
                "Second text".to_string(),
            ]))
            .build();
        
        let response = client.create_embeddings(&request).await;
        assert!(response.is_ok());
        
        let embeddings = response.unwrap();
        assert_eq!(embeddings.data.len(), 2);
        assert_eq!(embeddings.data[0].index, 0);
        assert_eq!(embeddings.data[1].index, 1);
    }
}