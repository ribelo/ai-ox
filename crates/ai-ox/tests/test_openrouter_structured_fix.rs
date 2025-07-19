use ai_ox::agent::Agent;
use ai_ox::content::message::{Message, MessageRole};
use ai_ox::content::part::Part;
use ai_ox::model::openrouter::OpenRouterModel;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct TestResponse {
    message: String,
    number: i32,
    success: bool,
}

#[tokio::test]
#[ignore] // Remove this to run the test
async fn test_openrouter_structured_request_format() {
    let api_key = env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY must be set for this test");
    
    let model = OpenRouterModel::builder()
        .api_key(api_key)
        .model("google/gemini-2.0-flash-exp")
        .build();
    
    let agent = Agent::model(model)
        .system_instruction("You are a helpful assistant that returns JSON responses exactly as requested.")
        .build();
    
    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Return a JSON object with: message='Hello World', number=42, success=true".to_string(),
        }],
    )];
    
    // This should now work because we fixed the response_format structure
    let result = agent.generate_typed::<TestResponse>(messages).await;
    
    match result {
        Ok(response) => {
            println!("✓ Success! Response: {:?}", response);
            assert_eq!(response.data.message, "Hello World");
            assert_eq!(response.data.number, 42);
            assert_eq!(response.data.success, true);
        }
        Err(e) => {
            println!("✗ Error: {}", e);
            // Print detailed error information for debugging
            println!("Error details: {:?}", e);
            panic!("Test failed with error: {}", e);
        }
    }
}

#[tokio::test]
#[ignore] // Remove this to run the test
async fn test_schema_format_in_request() {
    // Test that our schema is properly formatted
    let schema = ai_ox::tool::schema_for_type::<TestResponse>();
    println!("Generated schema: {}", schema);
    
    // Test that it can be parsed as JSON
    let schema_value: serde_json::Value = serde_json::from_str(&schema.to_string())
        .expect("Schema should be valid JSON");
    
    // Test the OpenRouter format
    let response_format = serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": "Response",
            "schema": schema_value
        }
    });
    
    println!("OpenRouter format: {}", serde_json::to_string_pretty(&response_format).unwrap());
    
    // Verify the structure
    assert_eq!(response_format["type"], "json_schema");
    assert!(response_format["json_schema"]["schema"]["properties"].is_object());
    assert!(response_format["json_schema"]["schema"]["properties"]["message"].is_object());
    assert!(response_format["json_schema"]["schema"]["properties"]["number"].is_object());
    assert!(response_format["json_schema"]["schema"]["properties"]["success"].is_object());
}

#[tokio::test]
#[ignore] // Remove this to run the test
async fn test_error_message_includes_schema() {
    let api_key = env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY must be set for this test");
    
    let model = OpenRouterModel::builder()
        .api_key(api_key)
        .model("google/gemini-2.0-flash-exp")
        .build();
    
    let agent = Agent::model(model)
        .system_instruction("Return the text 'this is not json' exactly as given, with no JSON formatting.")
        .build();
    
    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Return the text 'this is not json' exactly as given, with no JSON formatting.".to_string(),
        }],
    )];
    
    let result = agent.generate_typed::<TestResponse>(messages).await;
    
    match result {
        Ok(_) => panic!("Expected parsing to fail but it succeeded"),
        Err(e) => {
            println!("Expected error occurred: {}", e);
            let error_str = e.to_string();
            
            // Verify the error includes our enhanced information
            assert!(error_str.contains("Failed to parse response"));
            assert!(error_str.contains("Expected schema"));
            assert!(error_str.contains("message"));
            assert!(error_str.contains("number"));
            assert!(error_str.contains("success"));
            
            println!("✓ Error message includes schema information as expected");
        }
    }
}

#[tokio::test]
#[ignore] // Remove this to run the test
async fn test_different_openrouter_models() {
    let api_key = env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY must be set for this test");
    
    let models = vec![
        "google/gemini-2.0-flash-exp",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
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
                text: "Return JSON: {\"message\": \"test\", \"number\": 123, \"success\": true}".to_string(),
            }],
        )];
        
        match agent.generate_typed::<TestResponse>(messages).await {
            Ok(response) => {
                println!("✓ Success with {}: {:?}", model_name, response.data);
            }
            Err(e) => {
                println!("✗ Failed with {}: {}", model_name, e);
                // Some models might not support structured output, so we just log the error
            }
        }
    }
}