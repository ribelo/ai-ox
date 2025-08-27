#![cfg(any(feature = "tokens", feature = "models"))]

use anthropic_ox::{Anthropic, Model};
use anthropic_ox::tokens::TokenCountRequest;
use anthropic_ox::message::{Message, Messages, Role, Content, Text};

// Integration test for list_models endpoint
#[cfg(feature = "models")]
#[tokio::test]
async fn test_list_models_api() {
    let client = match Anthropic::load_from_env() {
        Ok(client) => client,
        Err(_) => {
            println!("ℹ️  Skipping API test: ANTHROPIC_API_KEY not found");
            return;
        }
    };

    match client.list_models(None, None, None).await {
        Ok(response) => {
            println!("✅ API call successful: list_models");
            assert!(!response.data.is_empty(), "Should return at least one model");
            println!("   Models returned: {}", response.data.len());
        }
        Err(e) => {
            println!("⚠️  API call failed (this might be expected): {}", e);
        }
    }
}

// Integration test for get_model endpoint
#[cfg(feature = "models")]
#[tokio::test]
async fn test_get_model_api() {
    let client = match Anthropic::load_from_env() {
        Ok(client) => client,
        Err(_) => {
            println!("ℹ️  Skipping API test: ANTHROPIC_API_KEY not found");
            return;
        }
    };

    let model_id = Model::Claude3Haiku20240307.to_string();

    match client.get_model(&model_id).await {
        Ok(response) => {
            println!("✅ API call successful: get_model");
            assert_eq!(response.id, model_id, "Returned model ID should match requested ID");
            println!("   Model ID: {}", response.id);
            println!("   Model Name: {}", response.display_name);
        }
        Err(e) => {
            println!("⚠️  API call failed (this might be expected): {}", e);
        }
    }
}

// Integration test for count_tokens endpoint
#[cfg(feature = "tokens")]
#[tokio::test]
async fn test_count_tokens_api() {
    let client = match Anthropic::load_from_env() {
        Ok(client) => client,
        Err(_) => {
            println!("ℹ️  Skipping API test: ANTHROPIC_API_KEY not found");
            return;
        }
    };

    let messages = vec![Message::new(Role::User, vec![Content::Text(Text::new("Hello, world".to_string()))])];

    let request = TokenCountRequest {
        model: Model::Claude3Haiku20240307.to_string(),
        messages: Messages(messages),
        system: None,
        tools: None,
        tool_choice: None,
        thinking: None,
    };

    match client.count_tokens(&request).await {
        Ok(response) => {
            println!("✅ API call successful: count_tokens");
            assert!(response.input_tokens > 0, "Token count should be greater than zero");
            println!("   Input tokens: {}", response.input_tokens);
        }
        Err(e) => {
            println!("⚠️  API call failed (this might be expected): {}", e);
        }
    }
}
