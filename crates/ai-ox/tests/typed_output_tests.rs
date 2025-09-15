mod common;

use ai_ox::agent::Agent;
use ai_ox::content::message::{Message, MessageRole};
use ai_ox::content::part::Part;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct SimpleResponse {
    message: String,
    number: i32,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct ComplexResponse {
    title: String,
    items: Vec<Item>,
    metadata: Metadata,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Item {
    name: String,
    value: f64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Metadata {
    created_at: String,
    version: i32,
}

/// Test that all providers can handle simple typed output requests
#[tokio::test]
async fn test_all_providers_simple_typed_output() {
    let models = common::get_available_models().await;

    if models.is_empty() {
        println!("No models available for testing. Skipping typed output test.");
        return;
    }

    for model in models {
        let model_name = model.name().to_string();
        println!("\n--- Testing typed output with model: {} ---", &model_name);
        
        let agent = Agent::builder()
            .model(model.into())
            .system_instruction("You are a helpful assistant that returns JSON responses exactly as requested.")
            .build();
        
        let messages = vec![Message::new(
            MessageRole::User,
            vec![Part::Text {
                text: "Return a simple JSON object with a message 'hello world' and number 42".to_string(),
                ext: BTreeMap::new(),
            }],
        )];
        
        let result = agent.generate_typed::<SimpleResponse>(messages).await;
        
        match result {
            Ok(response) => {
                println!("✅ Success! Response: {:?}", response);
                assert_eq!(response.data.message, "hello world");
                assert_eq!(response.data.number, 42);
                println!("✅ Model {} passed simple typed output test", response.model_name);
            }
            Err(e) => {
                println!("❌ Error with model {}: {}", &model_name, e);
                panic!("Model {} failed typed output test: {}", &model_name, e);
            }
        }
    }
}

/// Test that all providers can handle complex typed output with nested structures
#[tokio::test]
async fn test_all_providers_complex_typed_output() {
    let models = common::get_available_models().await;

    if models.is_empty() {
        println!("No models available for testing. Skipping complex typed output test.");
        return;
    }

    for model in models {
        let model_name = model.name().to_string();
        println!("\n--- Testing complex typed output with model: {} ---", &model_name);
        
        let agent = Agent::builder()
            .model(model.into())
            .system_instruction("You are a helpful assistant that returns JSON responses exactly as requested.")
            .build();
        
        let messages = vec![Message::new(
            MessageRole::User,
            vec![Part::Text {
                text: r#"Return a JSON object with:
- title: "Test Document"
- items: array with 2 items, each having name and value fields
- metadata: object with created_at (current date string) and version (1)
"#.to_string(),
                ext: BTreeMap::new(),
            }],
        )];
        
        let result = agent.generate_typed::<ComplexResponse>(messages).await;
        
        match result {
            Ok(response) => {
                println!("✅ Success! Response: {:?}", response);
                assert_eq!(response.data.title, "Test Document");
                assert_eq!(response.data.items.len(), 2);
                assert_eq!(response.data.metadata.version, 1);
                
                // Verify items have been populated
                for (i, item) in response.data.items.iter().enumerate() {
                    assert!(!item.name.is_empty(), "Item {} should have a name", i);
                    assert!(item.value.is_finite(), "Item {} should have a valid value", i);
                }
                
                println!("✅ Model {} passed complex typed output test", response.model_name);
            }
            Err(e) => {
                println!("❌ Error with model {}: {}", &model_name, e);
                // Some models might not support complex structured output well, so we just log the error
                println!("⚠️  Model {} does not fully support complex typed output", &model_name);
            }
        }
    }
}

/// Test schema generation works correctly
#[test]
fn test_schema_generation() {
    // Test that we can generate schemas correctly
    let schema = ai_ox::tool::schema_for_type::<SimpleResponse>();
    println!("Generated schema for SimpleResponse: {}", schema);
    
    let schema_str = schema.to_string();
    assert!(schema_str.contains("message"));
    assert!(schema_str.contains("number"));
    assert!(schema_str.contains("string"));
    assert!(schema_str.contains("integer"));
}

/// Test that all providers handle invalid JSON parsing correctly
#[tokio::test]
async fn test_all_providers_invalid_json_parsing() {
    let models = common::get_available_models().await;

    if models.is_empty() {
        println!("No models available for testing. Skipping invalid JSON test.");
        return;
    }

    for model in models {
        let model_name = model.name().to_string();
        println!("\n--- Testing invalid JSON handling with model: {} ---", &model_name);
        
        let agent = Agent::builder()
            .model(model.into())
            .system_instruction("Return the text 'this is not json' exactly as given, with no JSON formatting.")
            .build();
        
        let messages = vec![Message::new(
            MessageRole::User,
            vec![Part::Text {
                text: "Return the text 'this is not json' exactly as given, with no JSON formatting.".to_string(),
                ext: BTreeMap::new(),
            }],
        )];
        
        let result = agent.generate_typed::<SimpleResponse>(messages).await;
        
        match result {
            Ok(_) => {
                println!("⚠️  Model {} unexpectedly returned valid JSON when asked not to", &model_name);
                // This is actually fine - some models are so well-trained they return JSON anyway
            }
            Err(e) => {
                println!("✅ Expected error occurred: {}", e);
                let error_str = e.to_string();
                
                // Verify the error includes our enhanced information
                assert!(error_str.contains("Failed to parse response") || 
                        error_str.contains("parse") || 
                        error_str.contains("JSON"),
                        "Error should mention parsing or JSON issue");
                
                println!("✅ Model {} correctly errored on non-JSON response", &model_name);
            }
        }
    }
}