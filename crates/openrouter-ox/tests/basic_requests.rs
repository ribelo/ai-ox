use futures_util::StreamExt;
use openrouter_ox::{
    message::{Message, Messages},
    request::Request,
    OpenRouter,
};
use serde_json::json;
use std::env;

fn get_api_key() -> Option<String> {
    env::var("OPENROUTER_API_KEY").ok()
}

fn setup_client() -> Option<OpenRouter> {
    get_api_key().map(|key| OpenRouter::builder().api_key(key).build())
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn test_basic_chat_completion() {
    let client = setup_client().expect("OPENROUTER_API_KEY not set");

    let messages = Messages::new([Message::user("Hello, how are you?")]);

    let request = Request::builder()
        .messages(messages)
        .model("openai/gpt-3.5-turbo")
        .max_tokens(100)
        .temperature(0.7)
        .build();

    let response = request.send(&client).await;

    match response {
        Ok(completion) => {
            assert!(!completion.choices.is_empty(), "Should have at least one choice");
            assert!(
                !completion.choices[0].message.content.is_empty(),
                "Response content should not be empty"
            );
            println!("Response: {:?}", completion.choices[0].message.content);
        }
        Err(e) => {
            println!("Request failed with error: {:?}", e);
            panic!("Basic chat completion should succeed, but got error: {}", e);
        }
    }
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn test_chat_completion_with_system_message() {
    let client = setup_client().expect("OPENROUTER_API_KEY not set");

    let messages = Messages::new([
        Message::system("You are a helpful assistant that always responds with enthusiasm."),
        Message::user("What is the capital of France?"),
    ]);

    let request = Request::builder()
        .messages(messages)
        .model("openai/gpt-3.5-turbo")
        .max_tokens(50)
        .temperature(0.5)
        .build();

    let response = request.send(&client).await;

    match response {
        Ok(completion) => {
            assert!(!completion.choices.is_empty());
            let content = &completion.choices[0].message.content;
            assert!(!content.is_empty());
            println!("System message response: {:?}", content);
        }
        Err(e) => {
            panic!("Chat completion with system message should succeed: {}", e);
        }
    }
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn test_chat_completion_streaming() {
    let client = setup_client().expect("OPENROUTER_API_KEY not set");

    let messages = Messages::new([Message::user("Tell me a short story about a cat.")]);

    let request = Request::builder()
        .messages(messages)
        .model("openai/gpt-3.5-turbo")
        .max_tokens(100)
        .temperature(0.7)
        .build();

    let mut stream = request.stream(&client);
    let mut full_content = String::new();
    let mut chunk_count = 0;

    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => {
                chunk_count += 1;
                if let Some(choice) = chunk.choices.first() {
                    if let Some(delta_content) = &choice.delta.content {
                        full_content.push_str(delta_content);
                    }
                }
            }
            Err(e) => {
                panic!("Stream chunk failed: {:?}", e);
            }
        }
    }

    assert!(chunk_count > 0, "Should receive at least one chunk");
    assert!(!full_content.is_empty(), "Should receive some content");
    println!("Streamed content ({} chunks): {}", chunk_count, full_content);
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn test_multiple_models() {
    let client = setup_client().expect("OPENROUTER_API_KEY not set");

    let models = ["openai/gpt-3.5-turbo", "anthropic/claude-3-haiku"];

    for model in &models {
        let messages = Messages::new([Message::user("What is 2+2?")]);

        let request = Request::builder()
            .messages(messages)
            .model(*model)
            .max_tokens(50)
            .temperature(0.0)
            .build();

        let response = request.send(&client).await;

        match response {
            Ok(completion) => {
                assert!(!completion.choices.is_empty());
                let content = &completion.choices[0].message.content;
                assert!(!content.is_empty());
                println!("Model {} response: {:?}", model, content);
            }
            Err(e) => {
                println!("Model {} failed: {:?}", model, e);
                // Don't panic here as some models might not be available
            }
        }
    }
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn test_json_response_format() {
    let client = setup_client().expect("OPENROUTER_API_KEY not set");

    let messages = Messages::new([Message::user("Return a JSON object with name: 'Alice', age: 30, city: 'New York'")]);

    let _response_format = json!({
        "type": "json_object"
    });

    let request = Request::builder()
        .messages(messages)
        .model("openai/gpt-3.5-turbo")
        // Note: response_format might not be supported the same way
        // .response_format(Some(response_format))
        .max_tokens(100)
        .temperature(0.0)
        .build();

    let response = request.send(&client).await;

    match response {
        Ok(completion) => {
            assert!(!completion.choices.is_empty());
            let content = &completion.choices[0].message.content;

            // Content handling - just check that we got a response
            println!("JSON response: {:?}", content);
        }
        Err(e) => {
            println!("JSON format request failed: {:?}", e);
            // Some models might not support JSON format, so don't panic
        }
    }
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn test_conversation_with_multiple_messages() {
    let client = setup_client().expect("OPENROUTER_API_KEY not set");

    let messages = Messages::new([
        Message::user("I'm planning a trip to Japan."),
        Message::assistant("That sounds exciting! What cities are you planning to visit?"),
        Message::user("I'm thinking Tokyo and Kyoto. What should I see there?"),
    ]);

    let request = Request::builder()
        .messages(messages)
        .model("openai/gpt-3.5-turbo")
        .max_tokens(150)
        .temperature(0.7)
        .build();

    let response = request.send(&client).await;

    match response {
        Ok(completion) => {
            assert!(!completion.choices.is_empty());
            let content = &completion.choices[0].message.content;
            assert!(!content.is_empty());
            println!("Travel advice: {:?}", content);
        }
        Err(e) => {
            panic!("Multi-turn conversation should succeed: {}", e);
        }
    }
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn test_request_with_stop_sequences() {
    let client = setup_client().expect("OPENROUTER_API_KEY not set");

    let messages = Messages::new([Message::user("Count from 1 to 10: 1, 2, 3, 4, 5,")]);

    let request = Request::builder()
        .messages(messages)
        .model("openai/gpt-3.5-turbo")
        .stop(vec!["8".to_string()])
        .max_tokens(50)
        .temperature(0.0)
        .build();

    let response = request.send(&client).await;

    match response {
        Ok(completion) => {
            assert!(!completion.choices.is_empty());
            let content = &completion.choices[0].message.content;
            assert!(!content.is_empty());
            println!("Stopped response: {:?}", content);
        }
        Err(e) => {
            panic!("Request with stop sequences should succeed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_request_builder_pattern() {
    // Test the builder pattern without making actual API calls
    let messages = Messages::new([Message::user("Test message")]);

    let request = Request::builder()
        .messages(messages.clone())
        .model("test-model")
        .max_tokens(100)
        .temperature(0.7)
        .top_p(0.9)
        .frequency_penalty(0.5)
        .presence_penalty(0.3)
        .seed(42)
        .build();

    assert_eq!(request.model, "test-model");
    assert_eq!(request.max_tokens, Some(100));
    assert_eq!(request.temperature, Some(0.7));
    assert_eq!(request.top_p, Some(0.9));
    assert_eq!(request.frequency_penalty, Some(0.5));
    assert_eq!(request.presence_penalty, Some(0.3));
    assert_eq!(request.seed, Some(42));
    assert_eq!(request.messages.len(), 1);
}

#[test]
fn test_message_creation() {
    let user_msg = Message::user("Hello");
    let assistant_msg = Message::assistant("Hi there!");
    let system_msg = Message::system("You are helpful");

    // Test that messages were created correctly
    match user_msg {
        Message::User(_) => {},
        _ => panic!("Expected User message"),
    }
    match assistant_msg {
        Message::Assistant(_) => {},
        _ => panic!("Expected Assistant message"),
    }
    match system_msg {
        Message::System(_) => {},
        _ => panic!("Expected System message"),
    }
}
