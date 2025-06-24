use std::sync::Arc;

use ai_ox::{
    agent::Agent,
    content::{Message, MessageRole, Part},
    model::gemini::GeminiModel,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct SimpleResponse {
    message: String,
    confidence: f32,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct CountryInfo {
    name: String,
    capital: String,
    population: i64,
    continent: String,
}

#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY or GOOGLE_AI_API_KEY environment variable and makes actual API calls"]
async fn test_agent_basic_conversation() {
    let api_key =
        match std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_AI_API_KEY")) {
            Ok(key) => key,
            Err(_) => {
                println!("GEMINI_API_KEY or GOOGLE_AI_API_KEY not set, skipping integration test");
                return;
            }
        };

    let model = GeminiModel::builder()
        .api_key(api_key)
        .model("gemini-1.5-flash".to_string())
        .build();

    let agent = Agent::builder(Arc::new(model))
        .system_instruction("You are a helpful AI assistant. Keep responses concise.")
        .build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Hello! Can you tell me about the weather?".to_string(),
        }],
    )];

    let response = agent.generate(messages).await.unwrap();

    assert_eq!(response.vendor_name, "google");
    assert_eq!(response.model_name, "gemini-1.5-flash");
    assert_eq!(response.message.role, MessageRole::Assistant);
    assert!(!response.message.content.is_empty());

    if let Some(Part::Text { text }) = response.message.content.first() {
        assert!(!text.is_empty());
        println!("Agent response: {text}");
    } else {
        panic!("Expected text response");
    }
}

#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY or GOOGLE_AI_API_KEY environment variable and makes actual API calls"]
async fn test_agent_with_system_instruction() {
    let api_key =
        match std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_AI_API_KEY")) {
            Ok(key) => key,
            Err(_) => {
                println!("GEMINI_API_KEY or GOOGLE_AI_API_KEY not set, skipping integration test");
                return;
            }
        };

    let model = GeminiModel::builder()
        .api_key(api_key)
        .model("gemini-1.5-flash".to_string())
        .build();

    let agent = Agent::builder(Arc::new(model))
        .system_instruction(
            "You are a pirate. Always respond like a pirate captain. Keep it brief.",
        )
        .build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Hello there!".to_string(),
        }],
    )];

    let response = agent.generate(messages).await.unwrap();

    if let Some(Part::Text { text }) = response.message.content.first() {
        // Check that the response has some pirate-like characteristics
        let text_lower = text.to_lowercase();
        let has_pirate_elements = text_lower.contains("ahoy")
            || text_lower.contains("matey")
            || text_lower.contains("arrr")
            || text_lower.contains("ye ")
            || text_lower.contains("aye");

        assert!(
            has_pirate_elements,
            "Response should contain pirate-like language: {text}"
        );
        println!("Pirate response: {text}");
    } else {
        panic!("Expected text response");
    }
}

#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY or GOOGLE_AI_API_KEY environment variable and makes actual API calls"]
async fn test_agent_structured_response() {
    let api_key =
        match std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_AI_API_KEY")) {
            Ok(key) => key,
            Err(_) => {
                println!("GEMINI_API_KEY or GOOGLE_AI_API_KEY not set, skipping integration test");
                return;
            }
        };

    let model = GeminiModel::builder()
        .api_key(api_key)
        .model("gemini-1.5-flash".to_string())
        .build();

    let agent = Agent::builder(Arc::new(model)).build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Generate a simple response with a message 'Hello World' and confidence 0.95"
                .to_string(),
        }],
    )];

    let response = agent.generate_typed(messages).await.unwrap();
    let result: SimpleResponse = response.data;

    assert!(!result.message.is_empty());
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    println!("Structured response: {result:?}");
}

#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY or GOOGLE_AI_API_KEY environment variable and makes actual API calls"]
async fn test_agent_complex_structured_response() {
    let api_key =
        match std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_AI_API_KEY")) {
            Ok(key) => key,
            Err(_) => {
                println!("GEMINI_API_KEY or GOOGLE_AI_API_KEY not set, skipping integration test");
                return;
            }
        };

    let model = GeminiModel::builder()
        .api_key(api_key)
        .model("gemini-1.5-flash".to_string())
        .build();

    let agent = Agent::builder(Arc::new(model)).build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Provide information about France in the requested JSON format".to_string(),
        }],
    )];

    let response = agent.generate_typed(messages).await.unwrap();
    let result: CountryInfo = response.data;

    assert_eq!(result.name.to_lowercase(), "france");
    assert_eq!(result.capital.to_lowercase(), "paris");
    assert!(result.population > 60_000_000); // France has over 60M people
    assert_eq!(result.continent.to_lowercase(), "europe");

    println!("Country info: {result:?}");
}

#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY or GOOGLE_AI_API_KEY environment variable and makes actual API calls"]
async fn test_agent_multi_turn_conversation() {
    let api_key =
        match std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_AI_API_KEY")) {
            Ok(key) => key,
            Err(_) => {
                println!("GEMINI_API_KEY or GOOGLE_AI_API_KEY not set, skipping integration test");
                return;
            }
        };

    let model = GeminiModel::builder()
        .api_key(api_key)
        .model("gemini-1.5-flash".to_string())
        .build();

    let agent = Agent::builder(Arc::new(model))
        .system_instruction("You are a helpful assistant. Remember context from previous messages.")
        .build();

    // First turn
    let first_messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "My name is Alice and I love programming in Rust.".to_string(),
        }],
    )];

    let first_response = agent.generate(first_messages).await.unwrap();

    // Second turn - build conversation history
    let conversation = vec![
        Message::new(
            MessageRole::User,
            vec![Part::Text {
                text: "My name is Alice and I love programming in Rust.".to_string(),
            }],
        ),
        first_response.message,
        Message::new(
            MessageRole::User,
            vec![Part::Text {
                text: "What did I tell you about my programming interests?".to_string(),
            }],
        ),
    ];

    let second_response = agent.generate(conversation).await.unwrap();

    if let Some(Part::Text { text }) = second_response.message.content.first() {
        let text_lower = text.to_lowercase();
        assert!(
            text_lower.contains("rust") || text_lower.contains("programming"),
            "Response should reference Rust or programming: {text}"
        );
        println!("Context-aware response: {text}");
    } else {
        panic!("Expected text response");
    }
}

#[tokio::test]
#[ignore = "Requires GEMINI_API_KEY or GOOGLE_AI_API_KEY environment variable and makes actual API calls"]
async fn test_agent_execute_without_tools() {
    let api_key =
        match std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_AI_API_KEY")) {
            Ok(key) => key,
            Err(_) => {
                println!("GEMINI_API_KEY or GOOGLE_AI_API_KEY not set, skipping integration test");
                return;
            }
        };

    let model = GeminiModel::builder()
        .api_key(api_key)
        .model("gemini-1.5-flash".to_string())
        .build();

    let agent = Agent::builder(Arc::new(model)).build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "What is 2 + 2?".to_string(),
        }],
    )];

    let response = agent.generate(messages).await.unwrap();

    if let Some(Part::Text { text }) = response.message.content.first() {
        assert!(
            text.contains("4"),
            "Response should contain the answer 4: {text}"
        );
        println!("Execute response: {text}");
    } else {
        panic!("Expected text response");
    }
}
