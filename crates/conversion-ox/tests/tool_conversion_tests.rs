use std::fs;
use serde_json;
use conversion_ox::anthropic_gemini::{anthropic_to_gemini_request, anthropic_tool_to_gemini_tool};
use anthropic_ox::{
    message::{Content, Message, Messages, Role, StringOrContents, CacheControl},
    request::ChatRequest,
    tool::{Tool, ToolUse, ToolResult, ToolResultContent},
};
use gemini_ox::tool::FunctionMetadata;

#[test]
fn test_tool_conversion_from_resource_file() {
    // Load tool definition from test resources
    let tool_json = fs::read_to_string("tests/resources/tool_with_cache.json")
        .expect("Failed to read tool_with_cache.json");
    
    let tool: Tool = serde_json::from_str(&tool_json)
        .expect("Failed to deserialize tool from JSON");
    
    // Convert to Gemini format
    let gemini_tool = anthropic_tool_to_gemini_tool(tool.clone());
    
    // Verify conversion
    match gemini_tool {
        gemini_ox::tool::Tool::FunctionDeclarations(functions) => {
            assert_eq!(functions.len(), 1);
            let func = &functions[0];
            assert_eq!(func.name, "Task");
            assert!(func.description.is_some());
            match &tool {
                Tool::Custom(custom) => {
                    assert_eq!(func.description.as_ref().unwrap(), &custom.description);
                    // Verify input schema is preserved
                    assert_eq!(func.parameters, custom.input_schema);
                }
                Tool::Computer(_) => {
                    assert_eq!(func.description.as_ref().unwrap(), "Computer tool");
                }
            }
        }
        _ => panic!("Expected FunctionDeclarations"),
    }
}

#[test]
fn test_tool_use_with_cache_control_parsing() {
    // Load tool_use with cache_control from test resources
    let tool_use_json = fs::read_to_string("tests/resources/tool_use_with_cache.json")
        .expect("Failed to read tool_use_with_cache.json");
    
    // Parse as Content (since tool_use is a Content variant)
    let content: Content = serde_json::from_str(&tool_use_json)
        .expect("Failed to deserialize tool_use with cache_control");
    
    // Verify it parsed correctly with cache_control
    match content {
        Content::ToolUse(tool_use) => {
            assert_eq!(tool_use.name, "Task");
            assert_eq!(tool_use.id, "toolu_01T6x4J8DqKVfPqz3UVL5Z");
            assert!(tool_use.cache_control.is_some());
            
            let cache_control = tool_use.cache_control.unwrap();
            assert_eq!(cache_control.cache_type, "ephemeral");
        }
        _ => panic!("Expected ToolUse content"),
    }
}

#[test]
fn test_tool_result_with_cache_control_parsing() {
    // Load tool_result with cache_control from test resources  
    let tool_result_json = fs::read_to_string("tests/resources/tool_result_with_cache.json")
        .expect("Failed to read tool_result_with_cache.json");
    
    // Parse as Content (since tool_result is a Content variant)
    let content: Content = serde_json::from_str(&tool_result_json)
        .expect("Failed to deserialize tool_result with cache_control");
    
    // Verify it parsed correctly with cache_control
    match content {
        Content::ToolResult(tool_result) => {
            assert_eq!(tool_result.tool_use_id, "toolu_01T6x4J8DqKVfPqz3UVL5Z");
            assert_eq!(tool_result.content.len(), 1);
            assert!(tool_result.cache_control.is_some());
            
            let cache_control = tool_result.cache_control.unwrap();
            assert_eq!(cache_control.cache_type, "ephemeral");
        }
        _ => panic!("Expected ToolResult content"),
    }
}

#[test]
fn test_cache_control_is_dropped_in_conversion() {
    // Create a ChatRequest with tool_use containing cache_control
    let tool_use_with_cache = ToolUse {
        id: "test_id".to_string(),
        name: "TestTool".to_string(),
        input: serde_json::json!({"param": "value"}),
        cache_control: Some(CacheControl {
            cache_type: "ephemeral".to_string(),
        }),
    };
    
    let messages = Messages(vec![
        Message {
            role: Role::User,
            content: StringOrContents::Contents(vec![
                Content::ToolUse(tool_use_with_cache)
            ]),
        }
    ]);
    
    let chat_request = ChatRequest {
        model: "claude-3-sonnet".to_string(),
        max_tokens: 100,
        messages,
        system: None,
        metadata: None,
        temperature: None,
        top_p: None,
        top_k: None,
        stream: None,
        stop_sequences: None,
        tools: None,
        tool_choice: None,
        thinking: None,
    };
    
    // Convert to Gemini request
    let gemini_request = anthropic_to_gemini_request(chat_request);
    
    // Verify the tool_use was converted but cache_control was dropped
    assert_eq!(gemini_request.contents.len(), 1);
    
    let parts = &gemini_request.contents[0].parts;
    assert_eq!(parts.len(), 1);
    
    match &parts[0].data {
        gemini_ox::content::PartData::FunctionCall(function_call) => {
            assert_eq!(function_call.name, "TestTool");
            assert_eq!(function_call.id.as_ref().unwrap(), "test_id");
            // Note: cache_control should not exist in Gemini format
            // The FunctionCall struct in gemini_ox doesn't have cache_control
        }
        _ => panic!("Expected FunctionCall"),
    }
}

#[test]
fn test_anthropic_request_with_tools_and_cache_control() {
    // This test verifies that a complex Anthropic request with tools and cache_control
    // can be parsed and converted without errors
    
    let complex_anthropic_json = r#"{
        "model": "claude-3-sonnet",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": "Use the Task tool to extract information"
            },
            {
                "role": "assistant", 
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "Task",
                        "input": {
                            "description": "Extract data",
                            "prompt": "Extract the information from the logs",
                            "subagent_type": "general-purpose"
                        },
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extracted information successfully"
                            }
                        ],
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        ]
    }"#;
    
    // This should parse successfully now
    let chat_request: ChatRequest = serde_json::from_str(complex_anthropic_json)
        .expect("Failed to parse complex Anthropic request with cache_control");
    
    // Convert to Gemini format - should succeed and drop cache_control
    let gemini_request = anthropic_to_gemini_request(chat_request);
    
    // Verify it has the expected number of contents (3 messages)
    assert_eq!(gemini_request.contents.len(), 3);
    
    // Verify function call was converted properly (cache_control dropped)
    let assistant_content = &gemini_request.contents[1];
    assert_eq!(assistant_content.parts.len(), 1);
    match &assistant_content.parts[0].data {
        gemini_ox::content::PartData::FunctionCall(fc) => {
            assert_eq!(fc.name, "Task");
            assert_eq!(fc.id.as_ref().unwrap(), "toolu_123");
        }
        _ => panic!("Expected FunctionCall in assistant content"),
    }
    
    // Verify function response was converted properly (cache_control dropped)
    let user_response_content = &gemini_request.contents[2];
    assert_eq!(user_response_content.parts.len(), 1);
    match &user_response_content.parts[0].data {
        gemini_ox::content::PartData::FunctionResponse(fr) => {
            assert_eq!(fr.id.as_ref().unwrap(), "toolu_123");
        }
        _ => panic!("Expected FunctionResponse in user response content"),
    }
}