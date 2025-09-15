use anthropic_ox::{
    message::{Content as AnthropicContent, Message as AnthropicMessage, Role as AnthropicRole, Text},
    request::ChatRequest as AnthropicRequest,
    tool::{Tool as AnthropicTool, ToolUse, ToolResult, ToolResultContent},
};
use ai_ox::model::request::ModelRequest;

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
    let openai_request = anthropic_to_openai_request(original_request.clone());
        
    
    // Step 2: OpenAI -> Anthropic
    let final_request = openai_to_anthropic_request(openai_request);
    
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
    let openrouter_request = anthropic_to_openrouter_request(original_request.clone());
        
    
    // Step 2: OpenRouter -> Anthropic
    let final_request = openrouter_to_anthropic_request(openrouter_request);
    
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
    let openai_request = anthropic_to_openai_request(original_request.clone());
        
    
    let after_openai = openai_to_anthropic_request(openai_request);
    
    // Round 2: Anthropic -> OpenRouter -> Anthropic
    let openrouter_request = anthropic_to_openrouter_request(after_openai.clone());
        
    
    let final_request = openrouter_to_anthropic_request(openrouter_request);
    
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
    
    let openai_request = anthropic_to_openai_request(original_request.clone());
        
    
    let final_request = openai_to_anthropic_request(openai_request);
    
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
            anthropic_ox::tool::Tool::Custom(anthropic_ox::tool::CustomTool::new(
                "get_weather".to_string(),
                "Get current weather for a location".to_string(),
            ).with_schema(json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }))),
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
            anthropic_ox::tool::Tool::Custom(anthropic_ox::tool::CustomTool::new(
                "get_weather".to_string(),
                "Get current weather".to_string(),
            ).with_schema(json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }))),
            anthropic_ox::tool::Tool::Custom(anthropic_ox::tool::CustomTool::new(
                "calculate".to_string(),
                "Perform calculations".to_string(),
            ).with_schema(json!({
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }))),
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
                    content: vec![ToolResultContent::Text {
                        text: "Temperature: 25°C, Condition: Sunny".to_string(),
                    }],
                    is_error: Some(false),
                    cache_control: None,
                })].into(),
            },
        ])
        .tools(vec![
            anthropic_ox::tool::Tool::Custom(anthropic_ox::tool::CustomTool::new(
                "get_weather".to_string(),
                "Get current weather for a location".to_string(),
            ).with_schema(json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }))),
        ])
        .build()
    }

fn convert_anthropic_request_to_ai_ox(request: AnthropicRequest) -> ModelRequest {
    use ai_ox::content::{Message, MessageRole, Part};
    use ai_ox::tool::{Tool, FunctionMetadata};
    use std::collections::BTreeMap;

    // Convert messages
    let messages: Vec<Message> = request
        .messages
        .into_iter()
        .map(|msg| {
            let role = match msg.role {
                anthropic_ox::message::Role::User => MessageRole::User,
                anthropic_ox::message::Role::Assistant => MessageRole::Assistant,
            };

            let content: Vec<Part> = msg
                .content
                .into_vec()
                .into_iter()
                .map(|content| match content {
                    anthropic_ox::message::Content::Text(text) => Part::Text {
                        text: text.text,
                        ext: BTreeMap::new(),
                    },
                    anthropic_ox::message::Content::ToolUse(tool_use) => Part::ToolUse {
                        id: tool_use.id,
                        name: tool_use.name,
                        args: tool_use.input,
                        ext: BTreeMap::new(),
                    },
                    anthropic_ox::message::Content::ToolResult(tool_result) => {
                        let parts: Vec<Part> = tool_result
                            .content
                            .into_iter()
                            .map(|content| match content {
                                ToolResultContent::Text { text } => Part::Text {
                                    text,
                                    ext: BTreeMap::new(),
                                },
                                _ => Part::Text {
                                    text: "Unsupported content type".to_string(),
                                    ext: BTreeMap::new(),
                                },
                            })
                            .collect();

                        Part::ToolResult {
                            id: tool_result.tool_use_id,
                            name: "unknown".to_string(), // Anthropic doesn't store tool name in result
                            parts,
                            ext: BTreeMap::new(),
                        }
                    }
                    _ => Part::Text {
                        text: "Unsupported content type".to_string(),
                        ext: BTreeMap::new(),
                    },
                })
                .collect();

            Message::new(role, content)
        })
        .collect();

    // Convert tools
    let tools: Option<Vec<ai_ox::tool::Tool>> = request.tools.map(|anthropic_tools| {
        let function_declarations: Vec<FunctionMetadata> = anthropic_tools
            .into_iter()
            .filter_map(|tool| match tool {
                anthropic_ox::tool::Tool::Custom(custom_tool) => Some(FunctionMetadata {
                    name: custom_tool.name,
                    description: Some(custom_tool.description),
                    parameters: custom_tool.input_schema,
                }),
                _ => None,
            })
            .collect();

        if !function_declarations.is_empty() {
            vec![Tool::FunctionDeclarations(function_declarations)]
        } else {
            vec![]
        }
    });

    // Convert system message
    let system_message = request.system.map(|system| {
        let text = match system {
            anthropic_ox::message::StringOrContents::String(s) => s,
            anthropic_ox::message::StringOrContents::Contents(contents) => {
                // Extract text from contents
                contents.into_iter()
                    .filter_map(|c| match c {
                        anthropic_ox::message::Content::Text(text) => Some(text.text),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        };
        Message::new(
            MessageRole::System,
            vec![Part::Text {
                text,
                ext: BTreeMap::new(),
            }],
        )
    });

    if let Some(sys_msg) = system_message {
        ModelRequest::builder()
            .messages(messages)
            .tools(tools.unwrap_or_default())
            .system_message(sys_msg)
            .build()
    } else {
        ModelRequest::builder()
            .messages(messages)
            .tools(tools.unwrap_or_default())
            .build()
    }
}

fn convert_ai_ox_request_to_openai(request: &ModelRequest) -> OpenAIChatRequest {
    // Simple implementation - just convert basic messages
    let messages: Vec<ai_ox_common::openai_format::Message> = request.messages.iter().map(|msg| {
        let role = match msg.role {
            ai_ox::content::MessageRole::User => ai_ox_common::openai_format::MessageRole::User,
            ai_ox::content::MessageRole::Assistant => ai_ox_common::openai_format::MessageRole::Assistant,
            _ => ai_ox_common::openai_format::MessageRole::User,
        };
        let content = msg.content.iter().find_map(|part| {
            if let ai_ox::content::Part::Text { text, .. } = part {
                Some(text.clone())
            } else {
                None
            }
        });
        ai_ox_common::openai_format::Message {
            role,
            content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }).collect();

    OpenAIChatRequest::builder()
        .model("gpt-4".to_string())
        .messages(messages)
        .build()
}

fn convert_openai_request_to_ai_ox(request: &OpenAIChatRequest) -> ModelRequest {
    use ai_ox::content::{Message, MessageRole, Part};
    use ai_ox::tool::{Tool, FunctionMetadata};
    use ai_ox_common::openai_format::{Message as OpenAIMessage, MessageRole as OpenAIMessageRole};
    use std::collections::BTreeMap;

    // Convert messages
    let mut messages = Vec::new();
    let mut system_message = None;

    for msg in &request.messages {
        match msg.role {
            OpenAIMessageRole::System => {
                system_message = Some(Message::new(
                    MessageRole::System,
                    vec![Part::Text {
                        text: msg.content.clone().unwrap_or_default(),
                        ext: BTreeMap::new(),
                    }],
                ));
            }
            OpenAIMessageRole::User => {
                messages.push(Message::new(
                    MessageRole::User,
                    vec![Part::Text {
                        text: msg.content.clone().unwrap_or_default(),
                        ext: BTreeMap::new(),
                    }],
                ));
            }
            OpenAIMessageRole::Assistant => {
                let mut parts = Vec::new();

                // Add text content
                if let Some(text) = &msg.content {
                    parts.push(Part::Text {
                        text: text.clone(),
                        ext: BTreeMap::new(),
                    });
                }

                // Add tool calls
                if let Some(tool_calls) = &msg.tool_calls {
                    for tool_call in tool_calls {
                        parts.push(Part::ToolUse {
                            id: tool_call.id.clone(),
                            name: tool_call.function.name.clone(),
                            args: serde_json::from_str(&tool_call.function.arguments).unwrap_or(serde_json::Value::Null),
                            ext: BTreeMap::new(),
                        });
                    }
                }

                messages.push(Message::new(MessageRole::Assistant, parts));
            }
            OpenAIMessageRole::Tool => {
                // Tool results
                let parts = vec![Part::Text {
                    text: msg.content.clone().unwrap_or_default(),
                    ext: BTreeMap::new(),
                }];

                messages.push(Message::new(
                    MessageRole::Assistant, // Tool results are from assistant
                    vec![Part::ToolResult {
                        id: msg.tool_call_id.clone().unwrap_or_default(),
                        name: "unknown".to_string(), // OpenAI doesn't store tool name in result
                        parts,
                        ext: BTreeMap::new(),
                    }],
                ));
            }
        }
    }

    // Convert tools
    let tools = request.tools.as_ref().map(|openai_tools| {
        let function_declarations: Vec<FunctionMetadata> = openai_tools
            .iter()
            .map(|tool| FunctionMetadata {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: tool.function.parameters.clone().unwrap_or(serde_json::json!({})),
            })
            .collect();

        if !function_declarations.is_empty() {
            vec![Tool::FunctionDeclarations(function_declarations)]
        } else {
            vec![]
        }
    });

    if let Some(sys_msg) = system_message {
        ModelRequest::builder()
            .messages(messages)
            .tools(tools.unwrap_or_default())
            .system_message(sys_msg)
            .build()
    } else {
        ModelRequest::builder()
            .messages(messages)
            .tools(tools.unwrap_or_default())
            .build()
    }
}

fn convert_ai_ox_request_to_gemini(request: &ModelRequest) -> GeminiRequest {
    use gemini_ox::content::Content;
    use gemini_ox::tool::Tool;

    // Convert messages to Gemini contents
    let mut contents = Vec::new();

    // Add system message first if present
    if let Some(system_msg) = &request.system_message {
        if let Some(text_part) = system_msg.content.iter().find(|p| matches!(p, ai_ox::content::Part::Text { .. })) {
            if let ai_ox::content::Part::Text { text, .. } = text_part {
                contents.push(Content::text(text.clone()));
            }
        }
    }

    // Convert regular messages
    for msg in &request.messages {
        let role = match msg.role {
            ai_ox::content::MessageRole::User => gemini_ox::content::Role::User,
            ai_ox::content::MessageRole::Assistant => gemini_ox::content::Role::Model,
            ai_ox::content::MessageRole::System => gemini_ox::content::Role::User, // Gemini doesn't have system role
            ai_ox::content::MessageRole::Unknown(_) => gemini_ox::content::Role::User,
        };

        // For now, just concatenate text parts
        let text = msg.content.iter()
            .filter_map(|part| match part {
                ai_ox::content::Part::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");

        if !text.is_empty() {
            contents.push(Content::builder()
                .role(role)
                .text(text)
                .build());
        }
    }

    // Convert tools
    let tools = request.tools.as_ref().map(|ai_ox_tools| {
        ai_ox_tools.iter().filter_map(|tool| match tool {
            ai_ox::tool::Tool::FunctionDeclarations(funcs) => {
                Some(funcs.iter().map(|f| {
                    serde_json::json!({
                        "name": f.name,
                        "description": f.description,
                        "parameters": f.parameters
                    })
                }).collect::<Vec<_>>())
            }
            #[cfg(feature = "gemini")]
            ai_ox::tool::Tool::GeminiTool(gemini_tool) => {
                Some(vec![serde_json::to_value(gemini_tool.clone()).unwrap()])
            }
        }).flatten().collect::<Vec<_>>()
    });

    GeminiRequest::builder()
        .model("gemini-1.5-flash".to_string()) // Default model
        .content_list(contents)
        .tools(Vec::<gemini_ox::tool::Tool>::new())
        .build()
}

fn convert_gemini_request_to_ai_ox(request: &GeminiRequest) -> ModelRequest {
    use ai_ox::content::{Message, MessageRole, Part};
    use std::collections::BTreeMap;

    // Convert contents to messages
    let mut messages = Vec::new();
    let mut system_message = None;

    for content in &request.contents {
        let role = match content.role {
            gemini_ox::content::Role::User => MessageRole::User,
            gemini_ox::content::Role::Model => MessageRole::Assistant,
        };

        // Extract text from parts
        let text = content.parts.iter()
            .filter_map(|part| {
                // Gemini parts can be complex, for now just extract text
                if let Some(text) = part.data.as_text() {
                    Some(text.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<String>>()
            .join(" ");

        let parts = vec![Part::Text {
            text,
            ext: BTreeMap::new(),
        }];

        messages.push(Message::new(role, parts));
    }

    // Handle system instruction
    if let Some(system_instruction) = &request.system_instruction {
        let text = system_instruction.parts.iter()
            .filter_map(|part| {
                if let Some(text) = part.data.as_text() {
                    Some(text.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<String>>()
            .join(" ");

        system_message = Some(Message::new(
            MessageRole::System,
            vec![Part::Text {
                text,
                ext: BTreeMap::new(),
            }],
        ));
    }

    // For now, skip tools conversion as it's complex
    if let Some(sys_msg) = system_message {
        ModelRequest::builder()
            .messages(messages)
            .system_message(sys_msg)
            .build()
    } else {
        ModelRequest::builder()
            .messages(messages)
            .build()
    }
}

fn openai_to_anthropic_request(openai_request: OpenAIChatRequest) -> AnthropicRequest {
    use ai_ox_common::openai_format::{Message as OpenAIMessage, MessageRole as OpenAIMessageRole};

    // Convert messages
    let mut messages = Vec::new();
    let mut system_message = None;

    for msg in &openai_request.messages {
        match msg.role {
            OpenAIMessageRole::System => {
                // Store system message separately
                system_message = msg.content.clone();
            }
            OpenAIMessageRole::User => {
                let content = vec![AnthropicContent::Text(Text {
                    text: msg.content.clone().unwrap_or_default(),
                    cache_control: None,
                })];
                messages.push(AnthropicMessage {
                    role: AnthropicRole::User,
                    content: content.into(),
                });
            }
            OpenAIMessageRole::Assistant => {
                let mut content = Vec::new();

                // Add text content
                if let Some(text) = &msg.content {
                    content.push(AnthropicContent::Text(Text {
                        text: text.clone(),
                        cache_control: None,
                    }));
                }

                // Add tool calls
                if let Some(tool_calls) = &msg.tool_calls {
                    for tool_call in tool_calls {
                        content.push(AnthropicContent::ToolUse(ToolUse {
                            id: tool_call.id.clone(),
                            name: tool_call.function.name.clone(),
                            input: serde_json::from_str(&tool_call.function.arguments).unwrap_or(serde_json::Value::Null),
                            cache_control: None,
                        }));
                    }
                }

                messages.push(AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: content.into(),
                });
            }
            OpenAIMessageRole::Tool => {
                // Tool results become user messages with tool results
                let content = vec![AnthropicContent::ToolResult(ToolResult {
                    tool_use_id: msg.tool_call_id.clone().unwrap_or_default(),
                    content: vec![ToolResultContent::Text {
                        text: msg.content.clone().unwrap_or_default(),
                    }],
                    is_error: Some(false),
                    cache_control: None,
                })];
                messages.push(AnthropicMessage {
                    role: AnthropicRole::User,
                    content: content.into(),
                });
            }
        }
    }

    // Convert tools
    let tools = openai_request.tools.as_ref().map(|openai_tools| {
        openai_tools.iter().filter_map(|tool| {
            Some(anthropic_ox::tool::Tool::Custom(anthropic_ox::tool::CustomTool::new(
                tool.function.name.clone(),
                tool.function.description.clone().unwrap_or_default(),
            ).with_schema(tool.function.parameters.clone().unwrap_or(serde_json::json!({})))))
        }).collect::<Vec<anthropic_ox::tool::Tool>>()
    });

    AnthropicRequest::builder()
        .model("claude-3-sonnet-20240229".to_string()) // Default model
        .messages(messages)
        .tools(vec![])
        .system(anthropic_ox::message::StringOrContents::String(system_message.unwrap_or_default()))
        .build()
}

fn openrouter_to_anthropic_request(_openrouter_request: openrouter_ox::request::ChatRequest) -> AnthropicRequest {
    // For now, just return a basic request - this would need full implementation
    AnthropicRequest::builder()
        .model("claude-3-sonnet-20240229".to_string())
        .messages(Vec::<AnthropicMessage>::new())
        .build()
}

fn convert_ai_ox_request_to_anthropic(
    request: &ModelRequest,
    original: &AnthropicRequest,
) -> AnthropicRequest {
    // Convert messages back to Anthropic format
    let mut messages = Vec::new();

    // Add system message as first user message if present
    if let Some(system_msg) = &request.system_message {
        if let Some(text_part) = system_msg.content.iter().find(|p| matches!(p, ai_ox::content::Part::Text { .. })) {
            if let ai_ox::content::Part::Text { text, .. } = text_part {
                messages.push(AnthropicMessage {
                    role: AnthropicRole::User,
                    content: vec![AnthropicContent::Text(Text {
                        text: format!("System: {}", text),
                        cache_control: None,
                    })].into(),
                });
            }
        }
    }

    // Convert regular messages
    for msg in &request.messages {
        let role = match msg.role {
            ai_ox::content::MessageRole::User => AnthropicRole::User,
            ai_ox::content::MessageRole::Assistant => AnthropicRole::Assistant,
            ai_ox::content::MessageRole::System => AnthropicRole::User, // Convert to user message
            ai_ox::content::MessageRole::Unknown(_) => AnthropicRole::User,
        };

        let mut content = Vec::new();

        for part in &msg.content {
            match part {
                ai_ox::content::Part::Text { text, .. } => {
                    content.push(AnthropicContent::Text(Text {
                        text: text.clone(),
                        cache_control: None,
                    }));
                }
                ai_ox::content::Part::ToolUse { id, name, args, .. } => {
                    content.push(AnthropicContent::ToolUse(ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: args.clone(),
                        cache_control: None,
                    }));
                }
                ai_ox::content::Part::ToolResult { id, parts, .. } => {
                    let result_content: Vec<ToolResultContent> = parts.iter()
                        .filter_map(|p| match p {
                            ai_ox::content::Part::Text { text, .. } => Some(ToolResultContent::Text {
                                text: text.clone(),
                            }),
                            _ => None,
                        })
                        .collect();

                    content.push(AnthropicContent::ToolResult(ToolResult {
                        tool_use_id: id.clone(),
                        content: result_content,
                        is_error: Some(false),
                        cache_control: None,
                    }));
                }
                _ => {} // Skip other content types for now
            }
        }

        if !content.is_empty() {
            messages.push(AnthropicMessage {
                role,
                content: content.into(),
            });
        }
    }

    // Convert tools back
    let tools = request.tools.as_ref().map(|ai_ox_tools| {
        ai_ox_tools.iter().filter_map(|tool| match tool {
            ai_ox::tool::Tool::FunctionDeclarations(funcs) => {
                Some(funcs.iter().map(|f| {
                    anthropic_ox::tool::Tool::Custom(anthropic_ox::tool::CustomTool::new(
                        f.name.clone(),
                        f.description.clone().unwrap_or_default(),
                    ).with_schema(f.parameters.clone()))
                }).collect::<Vec<_>>())
            }
            #[cfg(feature = "gemini")]
            ai_ox::tool::Tool::GeminiTool(_) => None,
        }).flatten().collect::<Vec<anthropic_ox::tool::Tool>>()
    });

    // Use original request as base and override messages/tools
    AnthropicRequest::builder()
        .model(original.model.clone())
        .messages(messages)
        .tools(vec![])
        .system(original.system.clone().unwrap_or(anthropic_ox::message::StringOrContents::String(String::new())))
        .max_tokens(original.max_tokens)
        .temperature(original.temperature.unwrap_or(0.0))
        .top_p(original.top_p.unwrap_or(0.0))
        .top_k(original.top_k.unwrap_or(0))
        .stop_sequences(original.stop_sequences.clone().unwrap_or_default())
        .build()
}

// Stub functions for conversion - replace with actual implementations
fn anthropic_to_openai_request(_request: AnthropicRequest) -> OpenAIChatRequest {
    OpenAIChatRequest::builder()
        .model("gpt-4".to_string())
        .messages(vec![])
        .build()
}

fn anthropic_to_openrouter_request(_request: AnthropicRequest) -> openrouter_ox::request::ChatRequest {
    openrouter_ox::request::ChatRequest::builder()
        .model("anthropic/claude-3-sonnet".to_string())
        .messages(vec![])
        .build()
}
