use bedrock_ox::prelude::*;
use bedrock_ox::request::ChatRequest;
use bedrock_ox::message::{Message, Role};
use futures_util::StreamExt;

#[cfg(test)]
mod tests {
    use super::*;

    fn get_client() -> Bedrock {
        Bedrock::new()
    }

    #[tokio::test]
    #[ignore = "requires AWS credentials and makes real API calls"]
    async fn test_chat_completion() {
        let client = get_client();

        let request = ChatRequest {
            model: "anthropic.claude-3-haiku-20240307-v1:0".to_string(),
            messages: vec![Message {
                role: Role::User,
                content: "Say 'hello' in one word".to_string(),
            }],
            max_tokens: Some(5),
            temperature: Some(0.0),
        };

        let response = client.send(&request).await;
        assert!(response.is_ok());

        let chat_response = response.unwrap();
        assert_eq!(chat_response.model, "anthropic.claude-3-haiku-20240307-v1:0");
        assert!(!chat_response.content.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires AWS credentials and makes real API calls"]
    async fn test_streaming_chat() {
        let client = get_client();

        let request = ChatRequest {
            model: "anthropic.claude-3-haiku-20240307-v1:0".to_string(),
            messages: vec![Message {
                role: Role::User,
                content: "Count from 1 to 3".to_string(),
            }],
            max_tokens: Some(20),
            temperature: Some(0.0),
        };

        let mut stream = client.stream(&request);

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
}
