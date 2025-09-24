#![cfg(feature = "anthropic-gemini")]

use anthropic_ox::{
    message::{CacheControl, Content, Message, Messages, Role, StringOrContents},
    request::ChatRequest,
    tool::{CustomTool, Tool, ToolChoice, ToolResult, ToolUse},
};
use conversion_ox::anthropic_gemini::{
    anthropic_to_gemini_request, anthropic_tool_to_gemini_tool, gemini_to_anthropic_request,
    gemini_tool_to_anthropic_tool,
};
use gemini_ox::{content::PartData, tool::config::Mode};
use serde_json;
use std::fs;

#[test]
fn test_tool_conversion_from_resource_file() {
    // Load tool definition from test resources
    let tool_json = fs::read_to_string("tests/resources/tool_with_cache.json")
        .expect("Failed to read tool_with_cache.json");

    let tool: Tool =
        serde_json::from_str(&tool_json).expect("Failed to deserialize tool from JSON");

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

    let messages = Messages(vec![Message {
        role: Role::User,
        content: StringOrContents::Contents(vec![Content::ToolUse(tool_use_with_cache)]),
    }]);

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
fn test_tool_choice_roundtrip_preserves_selection() {
    let tool_name = "Task".to_string();
    let messages = Messages(vec![Message {
        role: Role::User,
        content: StringOrContents::String("Use the Task tool".to_string()),
    }]);

    let tool = Tool::Custom(
        CustomTool::new(tool_name.clone(), "Does work".to_string())
            .with_schema(serde_json::json!({"type": "object"})),
    );

    let chat_request = ChatRequest {
        model: "gemini-2.0-flash".to_string(),
        max_tokens: 512,
        messages,
        system: None,
        metadata: None,
        temperature: None,
        top_p: None,
        top_k: None,
        stream: None,
        stop_sequences: None,
        tools: Some(vec![tool]),
        tool_choice: Some(ToolChoice::Tool {
            name: tool_name.clone(),
        }),
        thinking: None,
    };

    let gemini_request = anthropic_to_gemini_request(chat_request);
    let roundtrip = gemini_to_anthropic_request(gemini_request).expect("roundtrip succeeds");

    assert_eq!(
        roundtrip.tool_choice,
        Some(ToolChoice::Tool { name: tool_name }),
        "forced tool selection should round-trip"
    );
}

#[test]
fn test_schema_roundtrip_restores_original_property_names() {
    let original_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "-leading": {"type": "string"},
            "normal": {"type": "integer"}
        },
        "required": ["-leading"]
    });

    let anthropic_tool = Tool::Custom(
        CustomTool::new("Task".to_string(), "Does work".to_string())
            .with_schema(original_schema.clone()),
    );

    let gemini_tool = anthropic_tool_to_gemini_tool(anthropic_tool.clone());
    let restored_tool = gemini_tool_to_anthropic_tool(gemini_tool);

    let restored_schema = match restored_tool {
        Tool::Custom(custom) => custom.input_schema,
        _ => panic!("expected custom tool"),
    };

    assert_eq!(restored_schema, original_schema);
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

#[test]
fn test_tool_choice_is_preserved_in_conversion() {
    fn base_request() -> ChatRequest {
        ChatRequest {
            model: "claude-3-sonnet".to_string(),
            max_tokens: 100,
            messages: Messages(vec![Message {
                role: Role::User,
                content: StringOrContents::String("Hello".to_string()),
            }]),
            system: None,
            metadata: None,
            stop_sequences: None,
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            tools: None,
            tool_choice: None,
            thinking: None,
        }
    }

    let mut auto_request = base_request();
    auto_request.tool_choice = Some(ToolChoice::Auto);
    let auto_converted = anthropic_to_gemini_request(auto_request);
    let auto_config = auto_converted
        .tool_config
        .expect("expected tool config for auto choice");
    let auto_function_config = auto_config
        .function_calling_config
        .expect("expected function calling config");
    assert_eq!(auto_function_config.mode, Some(Mode::Auto));
    assert!(auto_function_config.allowed_function_names.is_none());

    let mut any_request = base_request();
    any_request.tool_choice = Some(ToolChoice::Any);
    let any_converted = anthropic_to_gemini_request(any_request);
    let any_config = any_converted
        .tool_config
        .expect("expected tool config for any choice");
    let any_function_config = any_config
        .function_calling_config
        .expect("expected function calling config");
    assert_eq!(any_function_config.mode, Some(Mode::Any));
    assert!(any_function_config.allowed_function_names.is_none());

    let mut specific_request = base_request();
    let tool_name = "Task".to_string();
    specific_request.tools = Some(vec![Tool::Custom(CustomTool::new(
        tool_name.clone(),
        "Test tool".to_string(),
    ))]);
    specific_request.tool_choice = Some(ToolChoice::Tool {
        name: tool_name.clone(),
    });
    let specific_converted = anthropic_to_gemini_request(specific_request);
    let specific_config = specific_converted
        .tool_config
        .expect("expected tool config for specific tool choice");
    let specific_function_config = specific_config
        .function_calling_config
        .expect("expected function calling config");
    assert_eq!(specific_function_config.mode, Some(Mode::Any));
    assert_eq!(
        specific_function_config.allowed_function_names,
        Some(vec![tool_name])
    );
}

#[test]
fn test_tool_result_conversion_preserves_name_and_handles_empty_content() {
    use anthropic_ox::message::Role as AnthropicRole;
    use anthropic_ox::response::{ChatResponse as AnthropicResponse, Usage};
    use conversion_ox::anthropic_gemini::anthropic_to_gemini_response;

    let tool_use = ToolUse {
        id: "toolu_123".to_string(),
        name: "call_weather".to_string(),
        input: serde_json::json!({ "location": "SF" }),
        cache_control: None,
    };

    let tool_result = ToolResult {
        tool_use_id: tool_use.id.clone(),
        content: Vec::new(),
        is_error: None,
        cache_control: None,
    };

    let anthropic_response = AnthropicResponse {
        id: "resp_123".to_string(),
        r#type: "message".to_string(),
        role: AnthropicRole::Assistant,
        content: vec![
            Content::ToolUse(tool_use.clone()),
            Content::ToolResult(tool_result.clone()),
        ],
        model: "claude-3-sonnet".to_string(),
        stop_reason: None,
        stop_sequence: None,
        usage: Usage::default(),
    };

    let gemini_response =
        anthropic_to_gemini_response(anthropic_response).expect("conversion should succeed");

    let parts = &gemini_response
        .candidates
        .first()
        .expect("expected candidate")
        .content
        .parts;

    assert_eq!(
        parts.len(),
        2,
        "expected function call and function response parts"
    );

    let response_part = parts
        .iter()
        .find_map(|part| match &part.data {
            PartData::FunctionResponse(resp) => Some(resp),
            _ => None,
        })
        .expect("expected function response part");

    assert_eq!(response_part.name, tool_use.name);
    assert_eq!(response_part.id.as_ref().unwrap(), &tool_result.tool_use_id);
    assert_eq!(response_part.response, serde_json::json!([]));
}
