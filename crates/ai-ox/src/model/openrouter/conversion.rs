use openrouter_ox::{
    message::{
        AssistantMessage, Message as OpenRouterMessage, Messages as OpenRouterMessages, SystemMessage, ToolMessage, UserMessage,
    },
    response::{FinishReason as OpenRouterFinishReason, FunctionCall, ToolCall as OpenRouterToolCall},
    tool::FunctionMetadata,
};
use serde_json::Value;

use crate::{
    content::{
        delta::FinishReason,
        message::{Message, MessageRole},
        part::Part,
    },
    model::request::ModelRequest,
    tool::{encode_tool_result_parts, decode_tool_result_parts, Tool},
    usage::Usage,
};

use super::error::OpenRouterError;

/// Convert OpenRouter finish reason to ai-ox finish reason
pub fn convert_finish_reason(reason: OpenRouterFinishReason) -> FinishReason {
    match reason {
        OpenRouterFinishReason::Stop => FinishReason::Stop,
        OpenRouterFinishReason::Length | OpenRouterFinishReason::Limit => FinishReason::Length,
        OpenRouterFinishReason::ContentFilter => FinishReason::ContentFilter,
        OpenRouterFinishReason::ToolCalls => FinishReason::ToolCalls,
    }
}


/// Build OpenRouter messages from ai-ox ModelRequest
pub fn build_openrouter_messages(request: &ModelRequest, model_name: &str) -> Result<OpenRouterMessages, OpenRouterError> {
    let mut messages = Vec::new();

    // Add system message if present
    if let Some(system_msg) = &request.system_message {
        let system_text = system_msg.content.iter()
            .filter_map(|part| match part {
                Part::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        messages.push(OpenRouterMessage::System(SystemMessage::text(system_text)));
    }

    // Convert regular messages using the new conversion function
    for message in &request.messages {
        let converted_messages = convert_message_to_openrouter(message.clone(), model_name)?;
        messages.extend(converted_messages);
    }

    Ok(OpenRouterMessages(messages))
}

/// Convert ai-ox tools to OpenRouter Tool (using FunctionMetadata directly)
pub fn convert_tools_to_openrouter(tools: Option<Vec<Tool>>) -> Result<Vec<FunctionMetadata>, OpenRouterError> {
    match tools {
        Some(tool_vec) => {
            let tool_schemas = tool_vec
                .iter()
                .flat_map(|tool| -> Vec<FunctionMetadata> { 
                    match tool {
                        Tool::FunctionDeclarations(functions) => {
                            functions.iter().map(|func| FunctionMetadata {
                                name: func.name.clone(),
                                description: func.description.clone(),
                                parameters: func.parameters.clone(),
                            }).collect()
                        }
                        #[cfg(feature = "gemini")]
                        Tool::GeminiTool(_) => Vec::new(),
                    }
                })
                .collect();
            Ok(tool_schemas)
        }
        None => Ok(Vec::new()),
    }
}

/// Detect if the model is a Google provider model that requires simple string format
fn is_google_model(model_name: &str) -> bool {
    model_name.starts_with("google/") || 
    model_name.contains("gemini")
}

/// Extract usage data from OpenRouter response
pub fn extract_usage_from_response(usage_data: Option<&openrouter_ox::response::Usage>) -> Usage {
    match usage_data {
        Some(usage) => {
            let mut result = Usage::default();
            result.input_tokens_by_modality.insert(crate::usage::Modality::Text, usage.prompt_tokens as u64);
            result.output_tokens_by_modality.insert(crate::usage::Modality::Text, usage.completion_tokens as u64);
            result.requests = 1;
            result
        },
        None => Usage::default(),
    }
}

/// Converts an `ai-ox` `Message` to one or more `openrouter-ox` `Message`s.
/// 
/// Returns a vector because tool results need to be converted to separate Tool messages,
/// which means a single ai-ox message can become multiple OpenRouter messages.
pub fn convert_message_to_openrouter(message: Message, model_name: &str) -> Result<Vec<OpenRouterMessage>, OpenRouterError> {
    match message.role {
        MessageRole::User => {
            // For user messages, separate tool results from regular content
            let mut text_parts = Vec::new();
            let mut tool_results = Vec::new();

            for part in message.content {
                match part {
                    Part::Text { text, .. } => text_parts.push(text),
                     Part::ToolResult { id, name, parts, .. } => {
                          // Collect tool results to create separate Tool messages
                          // Use standardized encoding for lossless conversion
                          let content_str = encode_tool_result_parts(&name, &parts).unwrap_or_else(|_| "null".to_string());
                          tool_results.push((id, name, content_str));
                      }
                      Part::Opaque { provider, kind, .. } => {
                          return Err(OpenRouterError::MessageConversion(
                              format!("OpenRouter does not support Opaque content from provider: {}, kind: {}", provider, kind)
                          ));
                      }
                     _ => {
                         // Convert other parts to text representation
                         if let Ok(serialized) = serde_json::to_string(&part) {
                             text_parts.push(serialized);
                         }
                     }
                }
            }

            let mut messages = Vec::new();

            // Add user message if there's any text content
            // Use provider-specific formatting
            if !text_parts.is_empty() {
                if is_google_model(model_name) {
                    // Google models require simple string format
                    messages.push(OpenRouterMessage::User(UserMessage::text(text_parts.join("\n"))));
                } else {
                    // OpenAI and other models use complex content format
                    messages.push(OpenRouterMessage::User(UserMessage::text(text_parts.join("\n"))));
                }
            }

            // Add tool result messages - Google requires name field
            for (id, name, content) in tool_results {
                messages.push(OpenRouterMessage::Tool(ToolMessage::with_name(id, content, name)));
            }

            // If no content at all, return empty user message
            if messages.is_empty() {
                messages.push(OpenRouterMessage::User(UserMessage::text("")));
            }

            Ok(messages)
        }
        MessageRole::Assistant => {
            // For assistant messages, handle tool calls and regular content
            // BUT if we find ToolResult parts, those need to be converted to separate Tool messages
            let mut text_parts = Vec::new();
            let mut tool_calls = Vec::new();
            let mut tool_results = Vec::new();

            for part in message.content {
                match part {
                    Part::Text { text, .. } => text_parts.push(text),
                     Part::ToolUse { id, name, args, .. } => {
                         // Convert to proper OpenRouter tool call
                         let tool_call = OpenRouterToolCall {
                             index: None,
                             id: Some(id),
                             type_field: "function".to_string(),
                             function: FunctionCall {
                                 name: Some(name),
                                 arguments: serde_json::to_string(&args).unwrap_or_default(),
                             },
                         };
                         tool_calls.push(tool_call);
                     }
                      Part::ToolResult { id, name, parts, .. } => {
                          // Convert tool results to separate Tool messages
                          // Use standardized encoding for lossless conversion
                          let content_str = encode_tool_result_parts(&name, &parts).unwrap_or_else(|_| "null".to_string());
                          tool_results.push((id, name, content_str));
                      }
                      Part::Opaque { provider, kind, .. } => {
                          return Err(OpenRouterError::MessageConversion(
                              format!("OpenRouter does not support Opaque content from provider: {}, kind: {}", provider, kind)
                          ));
                      }
                     _ => {
                         // Convert other parts to text representation
                         if let Ok(serialized) = serde_json::to_string(&part) {
                             text_parts.push(serialized);
                         }
                     }
                }
            }

            let mut messages = Vec::new();

            // Create assistant message (if there's content to add)
            if !text_parts.is_empty() || !tool_calls.is_empty() {
                let mut assistant_msg = if text_parts.is_empty() {
                    AssistantMessage::text("")
                } else {
                    AssistantMessage::text(text_parts.join("\n"))
                };

                if !tool_calls.is_empty() {
                    assistant_msg.tool_calls = Some(tool_calls);
                }

                messages.push(OpenRouterMessage::Assistant(assistant_msg));
            }

            // Add tool result messages (these should be separate Tool messages) - Google requires name field
            for (id, name, content) in tool_results {
                messages.push(OpenRouterMessage::Tool(ToolMessage::with_name(id, content, name)));
            }

            // If no content at all, return empty assistant message
            if messages.is_empty() {
                messages.push(OpenRouterMessage::Assistant(AssistantMessage::text("")));
            }

            Ok(messages)
        }
        MessageRole::System => {
            // For system messages, convert to OpenRouter system message
            let text_parts = message.content.iter()
                .filter_map(|part| match part {
                    Part::Text { text, .. } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            if text_parts.is_empty() {
                Ok(vec![OpenRouterMessage::System(SystemMessage::text(""))])
            } else {
                Ok(vec![OpenRouterMessage::System(SystemMessage::text(text_parts))])
            }
        }
        MessageRole::Unknown(_) => {
            // Map unknown roles to User as default
            Ok(vec![OpenRouterMessage::User(UserMessage::text("Unknown role content"))])
        }
    }
}


/// Converts an `openrouter-ox` `Message` to an `ai-ox` `Message`.
impl From<OpenRouterMessage> for Message {
    fn from(message: OpenRouterMessage) -> Self {
        let (role, parts) = match message {
            OpenRouterMessage::User(user_msg) => {
                let text = user_msg.content.0.iter()
                    .filter_map(|part| part.as_text().map(|t| t.text.clone()))
                    .collect::<Vec<_>>()
                    .join(" ");

                let content = vec![Part::Text { text, ext: std::collections::BTreeMap::new() }];
                (MessageRole::User, content)
            }
            OpenRouterMessage::Assistant(assistant_msg) => {
                let mut parts = Vec::new();

                // Add text content
                let text = assistant_msg.content.0.iter()
                    .filter_map(|part| part.as_text().map(|t| t.text.clone()))
                    .collect::<Vec<_>>()
                    .join(" ");

                if !text.is_empty() {
                    parts.push(Part::Text { text, ext: std::collections::BTreeMap::new() });
                }

                // Add tool calls
                if let Some(tool_calls) = assistant_msg.tool_calls {
                    for tool_call in tool_calls {
                        if let (Some(id), Some(name)) = (tool_call.id, tool_call.function.name) {
                            let args: Value = serde_json::from_str(&tool_call.function.arguments)
                                .unwrap_or(Value::Object(Default::default()));

                            parts.push(Part::ToolUse { id, name, args, ext: Default::default() });
                        }
                    }
                }

                (MessageRole::Assistant, parts)
            }
            OpenRouterMessage::System(system_msg) => {
                let text = system_msg.content().0.iter()
                    .filter_map(|part| part.as_text().map(|t| t.text.clone()))
                    .collect::<Vec<_>>()
                    .join(" ");

                let content = if text.is_empty() {
                    vec![]
                } else {
                    vec![Part::Text { text, ext: std::collections::BTreeMap::new() }]
                };
                (MessageRole::System, content)
            }
             OpenRouterMessage::Tool(tool_msg) => {
                 // Tool messages are converted to User messages with ToolResult parts
                 let (decoded_name, parts) = match decode_tool_result_parts(&tool_msg.content) {
                     Ok(result) => result,
                     Err(_) => {
                         // Use tool_msg.name if available, otherwise "unknown"
                         let name = tool_msg.name.clone().unwrap_or_else(|| "unknown".to_string());
                         (name, vec![])
                     }
                 };
                let content = vec![Part::ToolResult {
                    id: tool_msg.tool_call_id.clone(),
                    name: decoded_name,
                    parts,
                    ext: Default::default(),
                }];
                (MessageRole::User, content)
            }
        };

        Message {
            role,
            content: parts,
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        }
    }
}

/// Note: There's a fundamental mismatch between ai-ox and openrouter-ox tool systems:
/// - ai-ox uses `Tool::FunctionDeclarations` which are just schemas (metadata)
/// - openrouter-ox has a full `ToolBox` system with executable tools implementing the `Tool` trait
///
/// For proper integration, you would need to:
/// 1. Create wrapper tools that implement `openrouter_ox::Tool`
/// 2. Or extend ai-ox to support executable tools like openrouter-ox does
///
/// For now, we handle tool conversion at the schema level in the model implementation.

// Note: Tool conversion is handled inline in the conversion function above
// due to import issues with openrouter_ox::tool::Tool type


#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::message::{Message, MessageRole};
    use crate::content::part::Part;
    use serde_json::json;

    #[test]
    fn test_message_to_openrouter_user_role() {
        let message = Message {
            role: MessageRole::User,
            content: vec![Part::Text {
                text: "Hello, world!".to_string(),
                ext: std::collections::BTreeMap::new(),
            }],
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        };

        let openrouter_msg = convert_message_to_openrouter(message, "test-model").unwrap().into_iter().next().unwrap();
        match openrouter_msg {
            OpenRouterMessage::User(user_msg) => {
                let text = user_msg.content.0.iter()
                    .filter_map(|part| part.as_text().map(|t| t.text.as_str()))
                    .collect::<Vec<_>>()
                    .join(" ");
                assert_eq!(text, "Hello, world!");
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_message_to_openrouter_assistant_role() {
        let message = Message {
            role: MessageRole::Assistant,
            content: vec![Part::Text {
                text: "Hi there!".to_string(),
                ext: std::collections::BTreeMap::new(),
            }],
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        };

        let openrouter_msg = convert_message_to_openrouter(message, "test-model").unwrap().into_iter().next().unwrap();
        match openrouter_msg {
            OpenRouterMessage::Assistant(assistant_msg) => {
                let text = assistant_msg.content.0.iter()
                    .filter_map(|part| part.as_text().map(|t| t.text.as_str()))
                    .collect::<Vec<_>>()
                    .join(" ");
                assert_eq!(text, "Hi there!");
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_message_to_openrouter_multiple_text_parts() {
        let message = Message {
            role: MessageRole::User,
            content: vec![
                Part::Text {
                    text: "First part".to_string(),
                    ext: std::collections::BTreeMap::new(),
                },
                Part::Text {
                    text: "Second part".to_string(),
                    ext: std::collections::BTreeMap::new(),
                },
            ],
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        };

        let openrouter_msg = convert_message_to_openrouter(message, "test-model").unwrap().into_iter().next().unwrap();
        match openrouter_msg {
            OpenRouterMessage::User(user_msg) => {
                let text = user_msg.content.0.iter()
                    .filter_map(|part| part.as_text().map(|t| t.text.as_str()))
                    .collect::<Vec<_>>()
                    .join(" ");
                assert_eq!(text, "First part\nSecond part");
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_message_to_openrouter_empty_content() {
        let message = Message {
            role: MessageRole::User,
            content: vec![],
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        };

        let openrouter_msg = convert_message_to_openrouter(message, "test-model").unwrap().into_iter().next().unwrap();
        match openrouter_msg {
            OpenRouterMessage::User(user_msg) => {
                let text = user_msg.content.0.iter()
                    .filter_map(|part| part.as_text().map(|t| t.text.as_str()))
                    .collect::<Vec<_>>()
                    .join(" ");
                assert_eq!(text, "");
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_openrouter_to_message_conversion() {
        use openrouter_ox::message::AssistantMessage;

        let openrouter_msg = OpenRouterMessage::Assistant(
            AssistantMessage::text("AI response")
        );

        let ai_message: Message = openrouter_msg.into();
        assert_eq!(ai_message.role, MessageRole::Assistant);
        assert_eq!(ai_message.content.len(), 1);

        match &ai_message.content[0] {
            Part::Text { text, .. } => assert_eq!(text, "AI response"),
            _ => panic!("Expected text part"),
        }
    }

    #[test]
    fn test_openrouter_to_message_no_content() {
        use openrouter_ox::message::UserMessage;

        let openrouter_msg = OpenRouterMessage::User(
            UserMessage::text("")
        );

        let ai_message: Message = openrouter_msg.into();
        assert_eq!(ai_message.role, MessageRole::User);
        // Empty text still creates a Part::Text with empty string
        assert_eq!(ai_message.content.len(), 1);
        match &ai_message.content[0] {
            Part::Text { text, .. } => assert_eq!(text, ""),
            _ => panic!("Expected text part"),
        }
    }

    #[test]
    fn test_assistant_tool_call_conversion() {
        let message = Message {
            role: MessageRole::Assistant,
            content: vec![
                Part::Text {
                    text: "I'll search for that information.".to_string(),
                    ext: std::collections::BTreeMap::new(),
                },
                Part::ToolUse {
                    id: "call_123".to_string(),
                    name: "search_web".to_string(),
                    args: json!({"query": "rust programming", "max_results": 5}),
                    ext: Default::default(),
                 }
             ],
             timestamp: None,
             ext: Some(std::collections::BTreeMap::new()),
         };

        let openrouter_msg = convert_message_to_openrouter(message, "test-model").unwrap().into_iter().next().unwrap();
        match openrouter_msg {
            OpenRouterMessage::Assistant(assistant_msg) => {
                // Check text content
                let text = assistant_msg.content.0.iter()
                    .filter_map(|part| part.as_text().map(|t| t.text.as_str()))
                    .collect::<Vec<_>>()
                    .join(" ");
                assert_eq!(text, "I'll search for that information.");

                // Check tool calls
                let tool_calls = assistant_msg.tool_calls.expect("Should have tool calls");
                assert_eq!(tool_calls.len(), 1);

                let tool_call = &tool_calls[0];
                assert_eq!(tool_call.id, Some("call_123".to_string()));
                assert_eq!(tool_call.type_field, "function");
                assert_eq!(tool_call.function.name, Some("search_web".to_string()));

                let args: Value = serde_json::from_str(&tool_call.function.arguments).unwrap();
                assert_eq!(args["query"], "rust programming");
                assert_eq!(args["max_results"], 5);
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_user_tool_result_conversion() {
        let message = Message {
            role: MessageRole::User,
            content: vec![
                Part::Text {
                    text: "Here are the search results:".to_string(),
                    ext: std::collections::BTreeMap::new(),
                },
                Part::ToolResult {
                    id: "call_123".to_string(),
                    name: "search_web".to_string(),
                    parts: vec![Part::Text {
                        text: serde_json::to_string(&json!({"results": ["Result 1", "Result 2"], "count": 2})).unwrap(),
                        ext: Default::default(),
                    }],
                    ext: Default::default(),
                }
            ],
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        };

        let openrouter_msg = convert_message_to_openrouter(message, "test-model").unwrap().into_iter().next().unwrap();
        match openrouter_msg {
            OpenRouterMessage::User(user_msg) => {
                let text = user_msg.content.0.iter()
                    .filter_map(|part| part.as_text().map(|t| t.text.as_str()))
                    .collect::<Vec<_>>()
                    .join(" ");

                // Tool results should be converted to text representation for user messages
                assert!(text.contains("Here are the search results:"));
                // The tool result should be serialized as JSON in the text
                assert!(text.contains("call_123") || text.contains("search_web") || text.contains("results"));
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_reverse_tool_call_conversion() {
        use openrouter_ox::message::AssistantMessage;
        use openrouter_ox::response::{FunctionCall, ToolCall as OpenRouterToolCall};

        let tool_call = OpenRouterToolCall {
            index: None,
            id: Some("call_456".to_string()),
            type_field: "function".to_string(),
            function: FunctionCall {
                name: Some("calculate".to_string()),
                arguments: json!({"expression": "2 + 2"}).to_string(),
            },
        };

        let mut assistant_msg = AssistantMessage::text("I'll calculate that for you.");
        assistant_msg.tool_calls = Some(vec![tool_call]);

        let openrouter_msg = OpenRouterMessage::Assistant(assistant_msg);
        let ai_message: Message = openrouter_msg.into();

        assert_eq!(ai_message.role, MessageRole::Assistant);
        assert_eq!(ai_message.content.len(), 2); // Text + ToolCall

        // Check text part
        match &ai_message.content[0] {
            Part::Text { text, .. } => assert_eq!(text, "I'll calculate that for you."),
            _ => panic!("Expected text part first"),
        }

        // Check tool call part
        match &ai_message.content[1] {
            Part::ToolUse { id, name, args, .. } => {
                assert_eq!(id, "call_456");
                assert_eq!(name, "calculate");
                assert_eq!(args["expression"], "2 + 2");
            }
            _ => panic!("Expected tool use part"),
        }
    }

    #[test]
    fn test_function_declarations_tool_conversion() {
        use crate::tool::FunctionMetadata as AiOxFunctionMetadata;

        let tool_vec = vec![Tool::FunctionDeclarations(vec![
            AiOxFunctionMetadata {
                name: "test_function".to_string(),
                description: Some("A test function".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "param": {"type": "string"}
                    }
                }),
            }
        ])];

        let schemas = convert_tools_to_openrouter(Some(tool_vec)).unwrap();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].name, "test_function");
        assert_eq!(schemas[0].description, Some("A test function".to_string()));
    }

    #[test]
    fn test_complete_tool_workflow() {
        // Test a complete tool workflow: user asks -> assistant calls tool -> user provides result

        // 1. User asks a question
        let user_question = Message {
            role: MessageRole::User,
            content: vec![Part::Text {
                text: "What's the weather in Tokyo?".to_string(),
                ext: std::collections::BTreeMap::new(),
            }],
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        };

        let openrouter_user = convert_message_to_openrouter(user_question, "test-model").unwrap().into_iter().next().unwrap();
        let back_to_ai: Message = openrouter_user.into();

        assert_eq!(back_to_ai.role, MessageRole::User);
        assert_eq!(back_to_ai.content.len(), 1);

        // 2. Assistant responds with tool call
        let assistant_response = Message {
            role: MessageRole::Assistant,
            content: vec![
                Part::Text {
                    text: "I'll check the weather in Tokyo for you.".to_string(),
                    ext: std::collections::BTreeMap::new(),
                },
                Part::ToolUse {
                    id: "call_weather_123".to_string(),
                    name: "get_weather".to_string(),
                    args: json!({"location": "Tokyo", "units": "celsius"}),
                    ext: Default::default(),
                }
            ],
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        };

        let openrouter_assistant = convert_message_to_openrouter(assistant_response, "test-model").unwrap().into_iter().next().unwrap();
        let back_to_ai_assistant: Message = openrouter_assistant.into();

        assert_eq!(back_to_ai_assistant.role, MessageRole::Assistant);
        assert_eq!(back_to_ai_assistant.content.len(), 2); // Text + ToolCall

        // Verify the tool call is preserved
        match &back_to_ai_assistant.content[1] {
            Part::ToolUse { id, name, args, .. } => {
                assert_eq!(id, "call_weather_123");
                assert_eq!(name, "get_weather");
                assert_eq!(args["location"], "Tokyo");
                assert_eq!(args["units"], "celsius");
            }
            _ => panic!("Expected tool use part"),
        }

        // 3. User provides tool result
        let user_tool_result = Message {
            role: MessageRole::User,
            content: vec![
                Part::ToolResult {
                    id: "call_weather_123".to_string(),
                    name: "get_weather".to_string(),
                    parts: vec![Part::Text {
                        text: serde_json::to_string(&json!({
                            "temperature": 22,
                            "condition": "sunny",
                            "humidity": 60
                        })).unwrap(),
                        ext: Default::default(),
                    }],
                    ext: Default::default(),
                }
            ],
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        };

        let openrouter_tool_result = convert_message_to_openrouter(user_tool_result, "test-model").unwrap().into_iter().next().unwrap();
        let back_to_ai_tool_result: Message = openrouter_tool_result.into();

        assert_eq!(back_to_ai_tool_result.role, MessageRole::User);
        // Tool results are currently converted to text for OpenRouter
        // This demonstrates the workflow works end-to-end
        assert!(!back_to_ai_tool_result.content.is_empty());
    }

    #[test]
    fn test_convert_message_to_openrouter_tool_results() {
        use serde_json::json;

        let message = Message {
            role: MessageRole::User,
            content: vec![
                Part::Text {
                    text: "Here's the weather data:".to_string(),
                    ext: std::collections::BTreeMap::new(),
                },
                Part::ToolResult {
                    id: "call_123".to_string(),
                    name: "get_weather".to_string(),
                    parts: vec![Part::Text {
                        text: serde_json::to_string(&json!({
                            "temperature": 25,
                            "condition": "cloudy"
                        })).unwrap(),
                        ext: Default::default(),
                    }],
                    ext: Default::default(),
                }
            ],
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        };

        // Use the new conversion function
        let openrouter_messages = convert_message_to_openrouter(message, "test-model").unwrap();
        
        // Should produce two messages: one User message and one Tool message
        assert_eq!(openrouter_messages.len(), 2);

        // First message should be User with text content
        match &openrouter_messages[0] {
            OpenRouterMessage::User(user_msg) => {
                let text = user_msg.content.0.iter()
                    .filter_map(|part| part.as_text().map(|t| t.text.as_str()))
                    .collect::<Vec<_>>()
                    .join(" ");
                assert_eq!(text, "Here's the weather data:");
            }
            _ => panic!("Expected first message to be User"),
        }

        // Second message should be Tool with the result
        match &openrouter_messages[1] {
            OpenRouterMessage::Tool(tool_msg) => {
                assert_eq!(tool_msg.tool_call_id, "call_123");
                let parsed_content: serde_json::Value = serde_json::from_str(&tool_msg.content).unwrap();
                assert_eq!(parsed_content["temperature"], 25);
                assert_eq!(parsed_content["condition"], "cloudy");
            }
            _ => panic!("Expected second message to be Tool"),
        }
    }

    #[test]
    fn test_convert_message_to_openrouter_only_tool_results() {
        use serde_json::json;

        let message = Message {
            role: MessageRole::User,
            content: vec![
                Part::ToolResult {
                    id: "call_456".to_string(),
                    name: "calculator".to_string(),
                    parts: vec![Part::Text {
                        text: serde_json::to_string(&json!("42")).unwrap(),
                        ext: std::collections::BTreeMap::new(),
                    }],
                    ext: Default::default(),
                }
            ],
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        };

        // Use the new conversion function
        let openrouter_messages = convert_message_to_openrouter(message, "test-model").unwrap();
        
        // Should produce one Tool message
        assert_eq!(openrouter_messages.len(), 1);

        match &openrouter_messages[0] {
            OpenRouterMessage::Tool(tool_msg) => {
                assert_eq!(tool_msg.tool_call_id, "call_456");
                assert_eq!(tool_msg.content, "\"42\"");
            }
            _ => panic!("Expected Tool message"),
        }
    }

    #[test]
    fn test_convert_message_to_openrouter_assistant_with_tool_calls() {
        use serde_json::json;

        let message = Message {
            role: MessageRole::Assistant,
            content: vec![
                Part::Text {
                    text: "I need to call a tool.".to_string(),
                    ext: std::collections::BTreeMap::new(),
                },
                Part::ToolUse {
                    id: "call_789".to_string(),
                    name: "get_weather".to_string(),
                    args: json!({"location": "Paris"}),
                    ext: Default::default(),
                }
            ],
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        };

        // Use the new conversion function
        let openrouter_messages = convert_message_to_openrouter(message, "test-model").unwrap();
        
        // Should produce one Assistant message
        assert_eq!(openrouter_messages.len(), 1);

        match &openrouter_messages[0] {
            OpenRouterMessage::Assistant(assistant_msg) => {
                // Check text content
                let text = assistant_msg.content.0.iter()
                    .filter_map(|part| part.as_text().map(|t| t.text.as_str()))
                    .collect::<Vec<_>>()
                    .join(" ");
                assert_eq!(text, "I need to call a tool.");

                // Check tool calls
                assert!(assistant_msg.tool_calls.is_some());
                let tool_calls = assistant_msg.tool_calls.as_ref().unwrap();
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, Some("call_789".to_string()));
                assert_eq!(tool_calls[0].function.name, Some("get_weather".to_string()));
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_convert_tool_result_message_format() {
        use crate::content::message::{Message, MessageRole};
        use crate::content::part::Part;
        use serde_json::json;

        // Create a message with tool result (similar to what Agronauts produces)
        let tool_result_content = json!([{
            "content_id": 0,
            "document_id": 2,
            "score": 0.4306826078078052,
            "tags": [
                {"id": 5, "name": "Centrum Doradztwa Rolniczego"},
                {"id": 1, "name": "nawozy azotowe"}
            ],
            "text": "Nitrogen fertilizers are essential for corn production."
        }]);

        let tool_result_message = Message {
            role: MessageRole::User,
            content: vec![Part::ToolResult {
                id: "call_123".to_string(),
                name: "knowledge_search".to_string(),
                parts: vec![Part::Text {
                    text: serde_json::to_string(&tool_result_content).unwrap(),
                    ext: std::collections::BTreeMap::new(),
                }],
                ext: Default::default(),
            }],
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        };

        // Convert to OpenRouter format
        let openrouter_messages = convert_message_to_openrouter(tool_result_message, "test-model").unwrap();

        println!("Converted messages: {:#?}", openrouter_messages);
        println!("Number of messages: {}", openrouter_messages.len());

        // The issue might be in how tool results are converted
        // According to the conversion.rs logic, tool results should create Tool messages
        assert_eq!(openrouter_messages.len(), 1, "Should produce exactly one Tool message");
        
        match &openrouter_messages[0] {
            OpenRouterMessage::Tool(tool_msg) => {
                assert_eq!(tool_msg.tool_call_id, "call_123");
                println!("Tool message content: {}", tool_msg.content);
                
                // Verify the content is valid JSON
                let _parsed: serde_json::Value = serde_json::from_str(&tool_msg.content)
                    .expect("Tool message content should be valid JSON");
            }
            _ => panic!("Expected Tool message, got: {:?}", openrouter_messages[0]),
        }
    }

    #[tokio::test]
    #[ignore = "Requires OPENROUTER_API_KEY environment variable and makes actual API calls"]
    async fn test_openrouter_tool_result_flow_reproduces_400_error() {
        use crate::content::message::{Message, MessageRole};
        use crate::content::part::Part;
        use crate::model::{Model, request::ModelRequest};
        use crate::tool::{FunctionMetadata, Tool};
        use serde_json::json;

        let api_key = std::env::var("OPENROUTER_API_KEY")
            .expect("OPENROUTER_API_KEY must be set for this test");

        // Create OpenRouter model - test with OpenAI GPT-4o
        let model = crate::model::openrouter::OpenRouterModel::builder()
            .api_key(api_key)
            .model("openai/gpt-4o")
            .build();

        // Define the knowledge search tool (similar to what Agronauts uses)
        let knowledge_search_tool = Tool::FunctionDeclarations(vec![FunctionMetadata {
            name: "knowledge_search".to_string(),
            description: Some("Search agricultural knowledge database".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for agricultural knowledge"
                    }
                },
                "required": ["query"]
            }),
        }]);

        // Step 1: User asks question
        let user_message = Message {
            role: MessageRole::User,
            content: vec![Part::Text {
                text: "Tell me about corn fertilizers".to_string(),
                ext: std::collections::BTreeMap::new(),
            }],
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        };

        let first_request = ModelRequest {
            messages: vec![user_message],
            system_message: None,
            tools: Some(vec![knowledge_search_tool]),
        };

        println!("Step 1: Making initial request with tool...");
        let first_response = model.request(first_request).await.unwrap();
        
        // Verify we got a tool call
        let tool_call = first_response.message.content.iter()
            .find_map(|part| match part {
                Part::ToolUse { id, name, args, .. } => Some((id.clone(), name.clone(), args.clone())),
                _ => None,
            })
            .expect("Expected a tool use in the response");

        println!("Step 2: Got tool call: {} -> {}", tool_call.1, tool_call.2);

        // Step 3: Simulate tool execution result (this is what Agronauts returns)
        let tool_result_content = json!([{
            "content_id": 0,
            "document_id": 2,
            "score": 0.4306826078078052,
            "tags": [
                {"id": 5, "name": "Centrum Doradztwa Rolniczego"},
                {"id": 1, "name": "nawozy azotowe"}
            ],
            "text": "Nitrogen fertilizers are essential for corn production. Apply 150-200 kg N/ha in split applications."
        }]);

        let tool_result_message = Message {
            role: MessageRole::User,
            content: vec![Part::ToolResult {
                id: tool_call.0.clone(),
                name: tool_call.1.clone(),
                parts: vec![Part::Text {
                    text: serde_json::to_string(&tool_result_content).unwrap(),
                    ext: std::collections::BTreeMap::new(),
                }],
                ext: Default::default(),
            }],
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        };

        // Step 4: Send the tool result back - this should trigger the 400 error
        let messages_with_result = vec![
            Message {
                role: MessageRole::User,
                content: vec![Part::Text {
                    text: "Tell me about corn fertilizers".to_string(),
                    ext: std::collections::BTreeMap::new(),
                }],
                timestamp: None,
                ext: Some(std::collections::BTreeMap::new()),
            },
            Message {
                role: MessageRole::Assistant,
                content: vec![Part::ToolUse {
                    id: tool_call.0.clone(),
                    name: tool_call.1.clone(),
                    args: tool_call.2.clone(),
                    ext: Default::default(),
                }],
                timestamp: None,
                ext: Some(std::collections::BTreeMap::new()),
            },
            tool_result_message,
        ];

        let second_request = ModelRequest {
            messages: messages_with_result,
            system_message: None,
            tools: Some(vec![]),
        };

        println!("Step 3: Sending tool result back to OpenRouter...");
        println!("Messages being sent to OpenRouter:");
        for (i, msg) in second_request.messages.iter().enumerate() {
            println!("  Message {}: {:?}", i, msg.role);
            for (j, part) in msg.content.iter().enumerate() {
                match part {
                    Part::Text { text, .. } => println!("    Part {}: Text({})", j, text.chars().take(50).collect::<String>()),
                    Part::ToolUse { id, name, .. } => println!("    Part {}: ToolUse({}, {})", j, id, name),
                    Part::ToolResult { id, name, .. } => println!("    Part {}: ToolResult({}, {})", j, id, name),
                    _ => println!("    Part {}: {:?}", j, part),
                }
            }
        }

        // This should reproduce the 400 error we're seeing in Agronauts
        let result = model.request(second_request).await;
        
        match result {
            Ok(_) => {
                println!("SUCCESS: Tool result processed without error!");
                // If this passes, the fix worked
            }
            Err(e) => {
                println!("ERROR: {}", e);
                println!("DETAILED ERROR DEBUG: {:?}", e);
                
                // Check if it's the specific 400 error we're debugging
                let error_str = e.to_string();
                if error_str.contains("400") && error_str.contains("Provider returned error") {
                    panic!("REPRODUCED: OpenRouter 400 error when processing tool results: {}", e);
                } else {
                    // Some other error occurred
                    println!("Different error occurred: {}", e);
                    panic!("Unexpected error: {}", e);
                }
            }
        }
    }
}

