use anthropic_ox::{
    message::{Content as AnthropicContent, Message as AnthropicMessage, Role as AnthropicRole, Text},
    request::ChatRequest as AnthropicRequest,
    tool::{Tool as AnthropicTool, ToolUse, ToolResult},
};
use ai_ox::model::request::ModelRequest;
use conversion_ox::anthropic_openai::{
    anthropic_to_openai_request, openai_to_anthropic_request,
};
use conversion_ox::anthropic_openrouter::{
    anthropic_to_openrouter_request, openrouter_to_anthropic_request,
};
use gemini_ox::generate_content::request::GenerateContentRequest as GeminiRequest;
use openai_ox::request::ChatRequest as OpenAIChatRequest;
use serde_json::json;

/// Test roundtrip conversion: Anthropic -> OpenAI -> Anthropic
/// The first and last Anthropic representations should be functionally equivalent.
#[tokio::test]
async fn test_anthropic_openai_roundtrip() {
    let original_request = create_test_anthropic_request();
    
    println!("Testing: Anthropic -> OpenAI -> Anthropic");
    
    // Step 1: Anthropic -> OpenAI
    let openai_request = anthropic_to_openai_request(original_request.clone())
        .expect("Failed to convert Anthropic to OpenAI");
    
    // Step 2: OpenAI -> Anthropic
    let final_request = openai_to_anthropic_request(openai_request)
        .expect("Failed to convert OpenAI back to Anthropic");
    
    // Verify essential properties are preserved
    assert_eq!(original_request.messages.len(), final_request.messages.len());
    
    // Verify we have tools (they might be transformed but should exist)
    let original_has_tools = original_request.tools.as_ref().map_or(0, |t| t.len()) > 0;
    let final_has_tools = final_request.tools.as_ref().map_or(0, |t| t.len()) > 0;
    assert_eq!(original_has_tools, final_has_tools, "Tool presence should be preserved");
    
    println!("✅ Anthropic -> OpenAI -> Anthropic roundtrip passed!");
}

/// Test roundtrip conversion: Anthropic -> OpenRouter -> Anthropic
#[tokio::test]
async fn test_anthropic_openrouter_roundtrip() {
    let original_request = create_test_anthropic_request();
    
    println!("Testing: Anthropic -> OpenRouter -> Anthropic");
    
    // Step 1: Anthropic -> OpenRouter
    let openrouter_request = anthropic_to_openrouter_request(original_request.clone())
        .expect("Failed to convert Anthropic to OpenRouter");
    
    // Step 2: OpenRouter -> Anthropic
    let final_request = openrouter_to_anthropic_request(openrouter_request)
        .expect("Failed to convert OpenRouter back to Anthropic");
    
    // Verify essential properties are preserved
    assert_eq!(original_request.messages.len(), final_request.messages.len());
    
    // Verify tool presence
    let original_has_tools = original_request.tools.as_ref().map_or(0, |t| t.len()) > 0;
    let final_has_tools = final_request.tools.as_ref().map_or(0, |t| t.len()) > 0;
    assert_eq!(original_has_tools, final_has_tools, "Tool presence should be preserved");
    
    println!("✅ Anthropic -> OpenRouter -> Anthropic roundtrip passed!");
}

/// Test full multi-provider roundtrip with multiple conversions
#[tokio::test]
async fn test_full_multi_provider_roundtrip() {
    let original_request = create_complex_anthropic_request();
    
    println!("Testing: Anthropic -> OpenAI -> Anthropic -> OpenRouter -> Anthropic");
    
    // Round 1: Anthropic -> OpenAI -> Anthropic
    let openai_request = anthropic_to_openai_request(original_request.clone())
        .expect("Failed: Anthropic -> OpenAI");
    
    let after_openai = openai_to_anthropic_request(openai_request)
        .expect("Failed: OpenAI -> Anthropic");
    
    // Round 2: Anthropic -> OpenRouter -> Anthropic
    let openrouter_request = anthropic_to_openrouter_request(after_openai.clone())
        .expect("Failed: Anthropic -> OpenRouter");
    
    let final_request = openrouter_to_anthropic_request(openrouter_request)
        .expect("Failed: OpenRouter -> Anthropic");
    
    // Verify core structure is preserved
    assert_eq!(original_request.messages.len(), final_request.messages.len());
    
    // Check that we have at least some tools (they might be modified but not lost)
    let original_has_tools = original_request.tools.as_ref().map_or(0, |t| t.len()) > 0;
    let final_has_tools = final_request.tools.as_ref().map_or(0, |t| t.len()) > 0;
    if original_has_tools {
        assert!(final_has_tools, "Tools were completely lost in conversion");
    }
    
    println!("✅ Full multi-provider roundtrip passed!");
}

/// Test with tool usage conversation
#[tokio::test]
async fn test_tool_usage_roundtrip() {
    let original_request = create_tool_usage_anthropic_request();
    
    println!("Testing tool usage: Anthropic -> OpenAI -> Anthropic");
    
    let openai_request = anthropic_to_openai_request(original_request.clone())
        .expect("Failed to convert tool usage to OpenAI");
    
    let final_request = openai_to_anthropic_request(openai_request)
        .expect("Failed to convert tool usage back to Anthropic");
    
    // Verify we still have the right number of messages
    assert_eq!(original_request.messages.len(), final_request.messages.len());
    
    // Verify tools are preserved
    let original_tool_count = original_request.tools.as_ref().map_or(0, |t| t.len());
    let final_tool_count = final_request.tools.as_ref().map_or(0, |t| t.len());
    assert_eq!(original_tool_count, final_tool_count);
    
    println!("✅ Tool usage roundtrip passed!");
}

/// RED test capturing the desired cross-provider roundtrip without yet
/// implementing the required conversions. This should fail until the
/// conversion helpers are available.
#[tokio::test]
async fn test_anthropic_ai_ox_multi_provider_roundtrip() {
    let original_request = create_complex_anthropic_request();

    println!(
        "Testing: Anthropic -> ai-ox -> OpenAI -> ai-ox -> Gemini -> ai-ox -> Anthropic"
    );

    // Step 1: Anthropic -> ai-ox
    let ai_ox_request_after_anthropic =
        convert_anthropic_request_to_ai_ox(original_request.clone());

    // Step 2: ai-ox -> OpenAI
    let openai_request = convert_ai_ox_request_to_openai(&ai_ox_request_after_anthropic);

    // Step 3: OpenAI -> ai-ox
    let ai_ox_after_openai = convert_openai_request_to_ai_ox(&openai_request);

    // Step 4: ai-ox -> Gemini
    let gemini_request = convert_ai_ox_request_to_gemini(&ai_ox_after_openai);

    // Step 5: Gemini -> ai-ox
    let ai_ox_after_gemini = convert_gemini_request_to_ai_ox(&gemini_request);

    // Step 6: ai-ox -> Anthropic
    let final_anthropic =
        convert_ai_ox_request_to_anthropic(&ai_ox_after_gemini, &original_request);

    let original_json = serde_json::to_value(&original_request).expect("serialize original");
    let final_json = serde_json::to_value(&final_anthropic).expect("serialize final");

    assert_eq!(
        original_json, final_json,
        "Anthropic request should survive the multi-provider roundtrip"
    );
}

fn create_test_anthropic_request() -> AnthropicRequest {
    AnthropicRequest::builder()
        .model("claude-3-sonnet-20240229")
        .messages(vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicContent::Text(Text {
                    text: "What's the weather like in Paris?".to_string(),
                    cache_control: None,
                })].into(),
            },
        ])
        .tools(vec![
            AnthropicTool::Function {
                name: "get_weather".to_string(),
                description: "Get current weather for a location".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }),
                cache_control: None,
            },
        ])
        .system("You are a helpful weather assistant.".into())
        .build()
}

fn create_complex_anthropic_request() -> AnthropicRequest {
    AnthropicRequest::builder()
        .model("claude-3-sonnet-20240229")
        .messages(vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicContent::Text(Text {
                    text: "Help me with weather and calculations.".to_string(),
                    cache_control: None,
                })].into(),
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![
                    AnthropicContent::Text(Text {
                        text: "I can help with both weather and calculations.".to_string(),
                        cache_control: None,
                    }),
                ].into(),
            },
        ])
        .tools(vec![
            AnthropicTool::Function {
                name: "get_weather".to_string(),
                description: "Get current weather".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }),
                cache_control: None,
            },
            AnthropicTool::Function {
                name: "calculate".to_string(),
                description: "Perform calculations".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }),
                cache_control: None,
            },
        ])
        .system("You are a helpful assistant.".into())
        .build()
}

fn create_tool_usage_anthropic_request() -> AnthropicRequest {
    AnthropicRequest::builder()
        .model("claude-3-sonnet-20240229")
        .messages(vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicContent::Text(Text {
                    text: "What's the weather in Tokyo?".to_string(),
                    cache_control: None,
                })].into(),
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![
                    AnthropicContent::Text(Text {
                        text: "I'll check the weather in Tokyo for you.".to_string(),
                        cache_control: None,
                    }),
                    AnthropicContent::ToolUse(ToolUse {
                        id: "call_123".to_string(),
                        name: "get_weather".to_string(),
                        input: json!({"location": "Tokyo", "unit": "celsius"}),
                        cache_control: None,
                    }),
                ].into(),
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicContent::ToolResult(ToolResult {
                    tool_use_id: "call_123".to_string(),
                    content: vec![AnthropicContent::Text(Text {
                        text: "Temperature: 25°C, Condition: Sunny".to_string(),
                        cache_control: None,
                    })],
                    is_error: Some(false),
                    cache_control: None,
                })].into(),
            },
        ])
        .tools(vec![
            AnthropicTool::Function {
                name: "get_weather".to_string(),
                description: "Get current weather for a location".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }),
                cache_control: None,
            },
        ])
        .build()
}

fn convert_anthropic_request_to_ai_ox(_request: AnthropicRequest) -> ModelRequest {
    todo!("convert_anthropic_request_to_ai_ox");
}

fn convert_ai_ox_request_to_openai(_request: &ModelRequest) -> OpenAIChatRequest {
    todo!("convert_ai_ox_request_to_openai");
}

fn convert_openai_request_to_ai_ox(_request: &OpenAIChatRequest) -> ModelRequest {
    todo!("convert_openai_request_to_ai_ox");
}

fn convert_ai_ox_request_to_gemini(_request: &ModelRequest) -> GeminiRequest {
    todo!("convert_ai_ox_request_to_gemini");
}

fn convert_gemini_request_to_ai_ox(_request: &GeminiRequest) -> ModelRequest {
    todo!("convert_gemini_request_to_ai_ox");
}

fn convert_ai_ox_request_to_anthropic(
    _request: &ModelRequest,
    _original: &AnthropicRequest,
) -> AnthropicRequest {
    todo!("convert_ai_ox_request_to_anthropic");
}
