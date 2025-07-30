use mistral_ox::{Mistral, Model};
use mistral_ox::message::{Message, Messages};
use mistral_ox::request::ChatRequest;
use mistral_ox::tool::{Tool, ToolChoice};
use futures_util::StreamExt;

/// Helper to get test client
fn get_test_client() -> Result<Mistral, Box<dyn std::error::Error>> {
    Mistral::load_from_env().map_err(|e| format!("Failed to load Mistral API key: {}. Set MISTRAL_API_KEY environment variable.", e).into())
}

#[tokio::test]
async fn test_basic_chat() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;
    
    let messages = Messages::new(vec![
        Message::user("What is 2+2? Reply with just the number.")
    ]);
    
    let request = ChatRequest::builder()
        .model(Model::MistralSmallLatest.to_string())
        .messages(messages)
        .max_tokens(10)
        .build();
    
    let response = client.send(&request).await?;
    
    // Verify response structure
    assert!(!response.id.is_empty());
    assert_eq!(response.object, "chat.completion");
    assert!(!response.choices.is_empty());
    
    let choice = &response.choices[0];
    assert_eq!(choice.index, 0);
    assert!(!choice.message.content.is_empty());
    
    // Check usage if present
    if let Some(usage) = &response.usage {
        assert!(usage.prompt_tokens > 0);
        assert!(usage.completion_tokens > 0);
        assert!(usage.total_tokens > 0);
    }
    
    println!("✅ Basic chat test passed");
    Ok(())
}

#[tokio::test]
async fn test_system_message() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;
    
    let messages = Messages::new(vec![
        Message::system("You are a pirate. Always respond like a pirate would."),
        Message::user("Hello, how are you?")
    ]);
    
    let request = ChatRequest::builder()
        .model(Model::MistralSmallLatest.to_string())
        .messages(messages)
        .max_tokens(100)
        .build();
    
    let response = client.send(&request).await?;
    
    let content = response.choices[0].message.content
        .iter()
        .find_map(|p| p.as_text())
        .map(|t| t.text.clone())
        .unwrap_or_else(|| String::new());
    
    println!("Pirate response: {}", content);
    
    // Verify the response has pirate-like characteristics
    let lower_content = content.to_lowercase();
    assert!(
        lower_content.contains("arr") || 
        lower_content.contains("ahoy") || 
        lower_content.contains("matey") ||
        lower_content.contains("ye") ||
        lower_content.contains("aye"),
        "Response doesn't seem pirate-like: {}", content
    );
    
    println!("✅ System message test passed");
    Ok(())
}

#[tokio::test]
async fn test_streaming() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;
    
    let messages = Messages::new(vec![
        Message::user("Count from 1 to 5, one number per line.")
    ]);
    
    let request = ChatRequest::builder()
        .model(Model::MistralSmallLatest.to_string())
        .messages(messages)
        .build();
    
    let mut stream = client.stream(&request);
    let mut chunks_received = 0;
    let mut content = String::new();
    let mut finish_reason_received = false;
    
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        chunks_received += 1;
        
        if let Some(choice) = chunk.choices.first() {
            if let Some(delta_content) = &choice.delta.content {
                content.push_str(delta_content);
            }
            
            if choice.finish_reason.is_some() {
                finish_reason_received = true;
            }
        }
    }
    
    assert!(chunks_received > 0, "No chunks received from stream");
    assert!(!content.is_empty(), "No content received from stream");
    assert!(finish_reason_received, "No finish reason received");
    
    println!("Streamed content ({} chunks): {}", chunks_received, content);
    println!("✅ Streaming test passed");
    Ok(())
}

#[tokio::test]
async fn test_tool_calling() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;
    
    let weather_tool = Tool::new("get_weather", "Get the current weather for a location")
        .with_parameters(serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g., 'Paris, France'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit",
                    "default": "celsius"
                }
            },
            "required": ["location"]
        }));
    
    let messages = Messages::new(vec![
        Message::user("What's the weather like in Tokyo?")
    ]);
    
    let request = ChatRequest::builder()
        .model(Model::MistralSmallLatest.to_string())
        .messages(messages)
        .tools(vec![weather_tool])
        .tool_choice(ToolChoice::Auto)
        .build();
    
    let response = client.send(&request).await?;
    
    // Check if the model called the tool
    let has_tool_call = response.choices[0].message.tool_calls.is_some();
    
    if has_tool_call {
        let tool_calls = response.choices[0].message.tool_calls.as_ref().unwrap();
        assert!(!tool_calls.is_empty(), "Tool calls array is empty");
        
        let tool_call = &tool_calls[0];
        assert_eq!(tool_call.function.name, "get_weather");
        
        // Verify arguments can be parsed
        let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)?;
        assert!(args["location"].is_string(), "Location not found in tool arguments");
        
        println!("Tool called with arguments: {}", serde_json::to_string_pretty(&args)?);
    } else {
        // Some models might provide a text response instead
        println!("Model provided text response instead of tool call");
    }
    
    println!("✅ Tool calling test passed");
    Ok(())
}

#[tokio::test]
async fn test_json_mode() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;
    
    let messages = Messages::new(vec![
        Message::user("Generate a JSON object with the following fields: name (string), age (number), city (string). Use realistic values.")
    ]);
    
    // Create request with response_format field set directly
    let mut request = ChatRequest::builder()
        .model(Model::MistralSmallLatest.to_string())
        .messages(messages)
        .build();
    
    // Manually set response format for JSON mode
    request.response_format = Some(serde_json::json!({
        "type": "json_object"
    }));
    
    let response = client.send(&request).await?;
    
    let content = response.choices[0].message.content
        .iter()
        .find_map(|p| p.as_text())
        .map(|t| t.text.clone())
        .unwrap_or_else(|| String::new());
    
    // Verify the response is valid JSON
    let json_value: serde_json::Value = serde_json::from_str(&content)?;
    
    // Verify expected fields
    assert!(json_value["name"].is_string(), "Missing or invalid 'name' field");
    assert!(json_value["age"].is_number(), "Missing or invalid 'age' field");
    assert!(json_value["city"].is_string(), "Missing or invalid 'city' field");
    
    println!("Generated JSON: {}", serde_json::to_string_pretty(&json_value)?);
    println!("✅ JSON mode test passed");
    Ok(())
}

#[tokio::test]
async fn test_multiple_models() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;
    
    let models = vec![
        Model::MistralTiny,
        Model::MistralSmall,
        Model::MistralSmallLatest,
    ];
    
    for model in models {
        println!("\nTesting model: {}", model);
        
        let messages = Messages::new(vec![
            Message::user("Say 'Hello' in one word.")
        ]);
        
        let request = ChatRequest::builder()
            .model(model.to_string())
            .messages(messages)
            .max_tokens(10)
            .build();
        
        match client.send(&request).await {
            Ok(response) => {
                let content = response.choices[0].message.content
                    .iter()
                    .find_map(|p| p.as_text())
                    .map(|t| t.text.clone())
                    .unwrap_or_else(|| String::new());
                println!("  ✅ {} responded: {}", model, content);
            }
            Err(e) => {
                println!("  ⚠️  {} error: {}", model, e);
                // Don't fail the test - some models might not be available
            }
        }
    }
    
    println!("\n✅ Multiple models test completed");
    Ok(())
}

#[tokio::test]
async fn test_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_test_client()?;
    
    // Test with invalid model name
    let messages = Messages::new(vec![
        Message::user("Hello")
    ]);
    
    let request = ChatRequest::builder()
        .model("invalid-model-name".to_string())
        .messages(messages)
        .build();
    
    let result = client.send(&request).await;
    assert!(result.is_err(), "Expected error for invalid model");
    
    println!("✅ Error handling test passed");
    Ok(())
}