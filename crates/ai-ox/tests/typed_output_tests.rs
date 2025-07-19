use ai_ox::agent::Agent;
use ai_ox::content::message::{Message, MessageRole};
use ai_ox::content::part::Part;
use ai_ox::model::openrouter::OpenRouterModel;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;

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

async fn create_test_agent() -> Agent {
    let api_key = env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set");
    let model = OpenRouterModel::builder()
        .api_key(api_key)
        .model("google/gemini-2.0-flash-exp")
        .build();
    
    Agent::model(model)
        .system_instruction("You are a helpful assistant that returns JSON responses exactly as requested.")
        .build()
}

#[tokio::test]
#[ignore] // Remove this to run the test
async fn test_simple_typed_output() {
    let agent = create_test_agent().await;
    
    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Return a simple JSON object with a message 'hello world' and number 42".to_string(),
        }],
    )];
    
    let result = agent.generate_typed::<SimpleResponse>(messages).await;
    
    match result {
        Ok(response) => {
            println!("Success! Response: {:?}", response);
            assert_eq!(response.data.message, "hello world");
            assert_eq!(response.data.number, 42);
        }
        Err(e) => {
            println!("Error: {}", e);
            panic!("Failed to generate typed output: {}", e);
        }
    }
}

#[tokio::test]
#[ignore] // Remove this to run the test
async fn test_complex_typed_output() {
    let agent = create_test_agent().await;
    
    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: r#"Return a JSON object with:
- title: "Test Document"
- items: array with 2 items, each having name and value fields
- metadata: object with created_at (current date string) and version (1)
"#.to_string(),
        }],
    )];
    
    let result = agent.generate_typed::<ComplexResponse>(messages).await;
    
    match result {
        Ok(response) => {
            println!("Success! Response: {:?}", response);
            assert_eq!(response.data.title, "Test Document");
            assert_eq!(response.data.items.len(), 2);
            assert_eq!(response.data.metadata.version, 1);
        }
        Err(e) => {
            println!("Error: {}", e);
            panic!("Failed to generate complex typed output: {}", e);
        }
    }
}

#[tokio::test]
#[ignore] // Remove this to run the test
async fn test_schema_generation() {
    // Test that we can generate schemas correctly
    let schema = ai_ox::tool::schema_for_type::<SimpleResponse>();
    println!("Generated schema for SimpleResponse: {}", schema);
    
    let schema_str = schema.to_string();
    assert!(schema_str.contains("message"));
    assert!(schema_str.contains("number"));
    assert!(schema_str.contains("string"));
    assert!(schema_str.contains("integer"));
}

#[tokio::test]
#[ignore] // Remove this to run the test
async fn test_invalid_json_parsing() {
    let agent = create_test_agent().await;
    
    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Return the text 'this is not json' without any JSON formatting".to_string(),
        }],
    )];
    
    let result = agent.generate_typed::<SimpleResponse>(messages).await;
    
    match result {
        Ok(_) => panic!("Expected parsing to fail but it succeeded"),
        Err(e) => {
            println!("Expected error occurred: {}", e);
            let error_str = e.to_string();
            assert!(error_str.contains("Failed to parse response"));
            assert!(error_str.contains("Expected schema"));
            assert!(error_str.contains("message"));
            assert!(error_str.contains("number"));
        }
    }
}

#[tokio::test]
#[ignore] // Remove this to run the test
async fn test_request_structure() {
    // Test that we can inspect what's being sent to the model
    let agent = create_test_agent().await;
    
    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Return a simple JSON with message and number".to_string(),
        }],
    )];
    
    // First test regular generation to see what happens
    let regular_result = agent.generate(messages.clone()).await;
    println!("Regular generation result: {:?}", regular_result);
    
    // Then test typed generation
    let typed_result = agent.generate_typed::<SimpleResponse>(messages).await;
    println!("Typed generation result: {:?}", typed_result);
}

#[tokio::test]
#[ignore] // Remove this to run the test
async fn test_different_models() {
    let api_key = env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set");
    
    let models = vec![
        "google/gemini-2.0-flash-exp",
        "google/gemini-2.5-flash-preview",
        "openai/gpt-4",
    ];
    
    for model_name in models {
        println!("\n--- Testing model: {} ---", model_name);
        
        let model = OpenRouterModel::builder()
            .api_key(api_key.clone())
            .model(model_name)
            .build();
        
        let agent = Agent::model(model).build();
        
        let messages = vec![Message::new(
            MessageRole::User,
            vec![Part::Text {
                text: "Return JSON: {\"message\": \"hello\", \"number\": 42}".to_string(),
            }],
        )];
        
        match agent.generate_typed::<SimpleResponse>(messages).await {
            Ok(response) => {
                println!("✓ Success with {}: {:?}", model_name, response.data);
            }
            Err(e) => {
                println!("✗ Failed with {}: {}", model_name, e);
            }
        }
    }
}

#[tokio::test]
#[ignore] // Remove this to run the test
async fn test_debug_request_content() {
    use ai_ox::tool::schema_for_type;
    
    // Test what the schema looks like
    let schema = schema_for_type::<SimpleResponse>();
    println!("Schema for SimpleResponse:");
    println!("{}", serde_json::to_string_pretty(&schema).unwrap());
    
    // Test the actual request building
    let agent = create_test_agent().await;
    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Return a JSON object with message 'test' and number 123".to_string(),
        }],
    )];
    
    // We need to check what's actually being sent to the model
    // This requires inspecting the model request
    println!("\nTesting typed generation...");
    let result = agent.generate_typed::<SimpleResponse>(messages).await;
    
    match result {
        Ok(response) => {
            println!("Success: {:?}", response);
        }
        Err(e) => {
            println!("Error details: {}", e);
            // Print the full error to see if it includes the schema
            println!("Error debug: {:?}", e);
        }
    }
}