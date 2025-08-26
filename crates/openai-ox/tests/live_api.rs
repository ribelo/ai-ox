//! Live API integration tests
//!
//! These tests require a valid API key and make real API calls.
//! They are ignored by default - run with `cargo test -- --ignored` to execute.

#[cfg(test)]
mod tests {
use openai_ox::{OpenAI, OpenAIRequestError};
    use futures_util::StreamExt;

async fn create_client() -> Result<OpenAI, OpenAIRequestError> {
    OpenAI::from_env()
    }

    #[tokio::test]
    #[ignore = "requires API key and makes live API calls"]
    async fn test_simple_chat() {
        let client = create_client().await.expect("Failed to create client");

        let request = client
            .chat()
            .model("gpt-3.5-turbo")
            .user("Say hello in exactly 3 words")
            .max_tokens(10)
            .build();

        let response = client.send(&request).await.expect("Failed to send request");

        assert!(!response.choices.is_empty());
        assert!(response.content().is_some());
        assert!(response.usage.is_some());

        let usage = response.usage.unwrap();
        assert!(usage.total_tokens > 0);
        assert!(usage.prompt_tokens > 0);
        assert!(usage.completion_tokens > 0);
    }

    #[tokio::test]
    #[ignore = "requires API key and makes live API calls"]
    async fn test_streaming_chat() {
        let client = create_client().await.expect("Failed to create client");

        let request = client
            .chat()
            .model("gpt-3.5-turbo")
            .user("Count from 1 to 5")
            .max_tokens(50)
            .build();

        let mut stream = client.stream(&request).await.expect("Failed to create stream");

        let mut responses = Vec::new();
        while let Some(result) = stream.next().await {
            match result {
                Ok(response) => {
                    responses.push(response);
                }
                Err(e) => panic!("Stream error: {}", e),
            }
        }

        assert!(!responses.is_empty());
        // At least some responses should have content
        assert!(responses.iter().any(|r| r.content().is_some()));
    }

    #[tokio::test]
    #[ignore = "requires API key and makes live API calls"]
    async fn test_conversation() {
        let client = create_client().await.expect("Failed to create client");

        let request = client
            .chat()
            .model("gpt-3.5-turbo")
            .system("You are a helpful math tutor. Be concise.")
            .user("What is 2 + 2?")
            .build();

        let response = client.send(&request).await.expect("Failed to send request");

        assert!(response.content().is_some());
        let first_response = response.content().unwrap();

        // Continue the conversation
        let request2 = client
            .chat()
            .model("gpt-3.5-turbo")
            .system("You are a helpful math tutor. Be concise.")
            .user("What is 2 + 2?")
            .assistant(first_response)
            .user("Now what is 3 + 3?")
            .build();

        let response2 = client.send(&request2).await.expect("Failed to send second request");
        assert!(response2.content().is_some());
    }

    #[tokio::test]
    #[ignore = "requires API key and makes live API calls"]
    async fn test_temperature_effects() {
        let client = create_client().await.expect("Failed to create client");

        // Test with temperature 0 (deterministic)
        let request_deterministic = client
            .chat()
            .model("gpt-3.5-turbo")
            .user("Say exactly: 'This is a test'")
            .temperature(0.0)
            .max_tokens(10)
            .build();

        let response1 = client.send(&request_deterministic).await.expect("Failed to send first request");
        let response2 = client.send(&request_deterministic).await.expect("Failed to send second request");

        // Responses should be identical or very similar with temperature 0
        assert!(response1.content().is_some());
        assert!(response2.content().is_some());
    }

    #[tokio::test]
    #[ignore = "requires API key and makes live API calls"]
    async fn test_max_tokens_limit() {
        let client = create_client().await.expect("Failed to create client");

        let request = client
            .chat()
            .model("gpt-3.5-turbo")
            .user("Write a very long story about a dragon")
            .max_tokens(5) // Very small limit
            .build();

        let response = client.send(&request).await.expect("Failed to send request");

        assert!(response.content().is_some());
        if let Some(usage) = response.usage {
            assert!(usage.completion_tokens <= 5);
        }
    }

    #[tokio::test]
    #[ignore = "requires API key and makes live API calls"]
    async fn test_invalid_model_error() {
        let client = create_client().await.expect("Failed to create client");

        let request = client
            .chat()
            .model("invalid-model-name-that-does-not-exist")
            .user("Hello")
            .build();

        let result = client.send(&request).await;
        assert!(result.is_err());

        // Should get an invalid request error
        if let Err(OpenAIRequestError::InvalidRequestError { message, .. }) = result {
            assert!(message.to_lowercase().contains("model") || message.to_lowercase().contains("invalid"));
        } else {
            panic!("Expected InvalidRequestError for invalid model");
        }
    }

    #[tokio::test]
    #[ignore = "requires API key and makes live API calls"]
    async fn test_empty_message_error() {
        let client = create_client().await.expect("Failed to create client");

        let request = client
            .chat()
            .model("gpt-3.5-turbo")
            .build(); // No messages

        let result = client.send(&request).await;
        assert!(result.is_err());
    }
}