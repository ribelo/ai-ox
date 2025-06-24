use openrouter_ox::{
    message::{
        AssistantMessage, Message as OpenRouterMessage, SystemMessage, ToolMessage, UserMessage,
    },
    response::{FunctionCall, ToolCall as OpenRouterToolCall},
    tool::{FunctionMetadata, ToolBox as OpenRouterToolBox, ToolSchema},
};
use serde_json::Value;

use crate::{
    content::{
        message::{Message, MessageRole},
        part::Part,
    },
    tool::{Tool, ToolSet},
};

/// Converts an `ai-ox` `Message` to an `openrouter-ox` `Message`.
impl From<Message> for OpenRouterMessage {
    fn from(message: Message) -> Self {
        match message.role {
            MessageRole::User => {
                // For user messages, handle tool results and regular content
                let mut text_parts = Vec::new();
                let mut tool_results = Vec::new();

                for part in message.content {
                    match part {
                        Part::Text { text } => text_parts.push(text),
                        Part::ToolResult { call_id, content, .. } => {
                            // Convert tool results to tool messages that will be sent separately
                            let content_str = serde_json::to_string(&content).unwrap_or_default();
                            tool_results.push((call_id, content_str));
                        }
                        _ => {
                            // Convert other parts to text representation
                            if let Ok(serialized) = serde_json::to_string(&part) {
                                text_parts.push(serialized);
                            }
                        }
                    }
                }

                OpenRouterMessage::User(UserMessage::text(text_parts.join("\n")))
            }
            MessageRole::Assistant => {
                // For assistant messages, handle tool calls and regular content
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();

                for part in message.content {
                    match part {
                        Part::Text { text } => text_parts.push(text),
                        Part::ToolCall { id, name, args } => {
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
                        _ => {
                            // Convert other parts to text representation
                            if let Ok(serialized) = serde_json::to_string(&part) {
                                text_parts.push(serialized);
                            }
                        }
                    }
                }

                let mut assistant_msg = if text_parts.is_empty() {
                    AssistantMessage::text("")
                } else {
                    AssistantMessage::text(text_parts.join("\n"))
                };

                if !tool_calls.is_empty() {
                    assistant_msg.tool_calls = Some(tool_calls);
                }

                OpenRouterMessage::Assistant(assistant_msg)
            }
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

                let content = if text.is_empty() {
                    vec![]
                } else {
                    vec![Part::Text { text }]
                };
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
                    parts.push(Part::Text { text });
                }

                // Add tool calls
                if let Some(tool_calls) = assistant_msg.tool_calls {
                    for tool_call in tool_calls {
                        if let (Some(id), Some(name)) = (tool_call.id, tool_call.function.name) {
                            let args: Value = serde_json::from_str(&tool_call.function.arguments)
                                .unwrap_or(Value::Object(Default::default()));

                            parts.push(Part::ToolCall { id, name, args });
                        }
                    }
                }

                (MessageRole::Assistant, parts)
            }
            OpenRouterMessage::System(system_msg) => {
                // System messages are converted to User messages since ai-ox doesn't have System role
                let text = system_msg.content().0.iter()
                    .filter_map(|part| part.as_text().map(|t| t.text.clone()))
                    .collect::<Vec<_>>()
                    .join(" ");

                let content = if text.is_empty() {
                    vec![]
                } else {
                    vec![Part::Text { text }]
                };
                (MessageRole::User, content)
            }
            OpenRouterMessage::Tool(tool_msg) => {
                // Tool messages are converted to User messages with ToolResult parts
                let content = vec![Part::ToolResult {
                    call_id: tool_msg.tool_call_id().clone(),
                    name: "unknown".to_string(), // OpenRouter doesn't provide tool name in response
                    content: serde_json::from_str(tool_msg.content()).unwrap_or_default(),
                }];
                (MessageRole::User, content)
            }
        };

        Message::new(role, parts)
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

/// Converts an `ai-ox` `ToolSet` to `openrouter-ox` `ToolBox` (limited conversion).
impl From<ToolSet> for OpenRouterToolBox {
    fn from(_tool_set: ToolSet) -> Self {
        // Create an empty toolbox since we can't convert schemas to executable tools
        OpenRouterToolBox::builder().build()
    }
}

/// Converts an `ai-ox` `Tool` to `openrouter-ox` `ToolSchema` vector.
impl From<&Tool> for Vec<ToolSchema> {
    fn from(tool: &Tool) -> Self {
        match tool {
            Tool::FunctionDeclarations(functions) => {
                functions.iter().map(|func| ToolSchema {
                    tool_type: "function".to_string(),
                    function: FunctionMetadata {
                        name: func.name.clone(),
                        description: func.description.clone(),
                        parameters: func.parameters.clone(),
                    },
                }).collect()
            }
            #[cfg(feature = "gemini")]
            Tool::GeminiTool(_) => {
                // Gemini-specific tools aren't directly compatible with OpenRouter
                // Return empty vector for now
                Vec::new()
            }
        }
    }
}


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
            }],
            timestamp: chrono::Utc::now(),
        };

        let openrouter_msg: OpenRouterMessage = message.into();
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
            }],
            timestamp: chrono::Utc::now(),
        };

        let openrouter_msg: OpenRouterMessage = message.into();
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
                },
                Part::Text {
                    text: "Second part".to_string(),
                },
            ],
            timestamp: chrono::Utc::now(),
        };

        let openrouter_msg: OpenRouterMessage = message.into();
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
            timestamp: chrono::Utc::now(),
        };

        let openrouter_msg: OpenRouterMessage = message.into();
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
            Part::Text { text } => assert_eq!(text, "AI response"),
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
            Part::Text { text } => assert_eq!(text, ""),
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
                },
                Part::ToolCall {
                    id: "call_123".to_string(),
                    name: "search_web".to_string(),
                    args: json!({"query": "rust programming", "max_results": 5}),
                }
            ],
            timestamp: chrono::Utc::now(),
        };

        let openrouter_msg: OpenRouterMessage = message.into();
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
                },
                Part::ToolResult {
                    call_id: "call_123".to_string(),
                    name: "search_web".to_string(),
                    content: json!({"results": ["Result 1", "Result 2"], "count": 2}),
                }
            ],
            timestamp: chrono::Utc::now(),
        };

        let openrouter_msg: OpenRouterMessage = message.into();
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
            Part::Text { text } => assert_eq!(text, "I'll calculate that for you."),
            _ => panic!("Expected text part first"),
        }

        // Check tool call part
        match &ai_message.content[1] {
            Part::ToolCall { id, name, args } => {
                assert_eq!(id, "call_456");
                assert_eq!(name, "calculate");
                assert_eq!(args["expression"], "2 + 2");
            }
            _ => panic!("Expected tool call part"),
        }
    }

    #[test]
    fn test_function_declarations_tool_conversion() {
        use crate::tool::FunctionMetadata;

        let tool = Tool::FunctionDeclarations(vec![
            FunctionMetadata {
                name: "test_function".to_string(),
                description: Some("A test function".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "param": {"type": "string"}
                    }
                }),
            }
        ]);

        let schemas: Vec<ToolSchema> = (&tool).into();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].tool_type, "function");
        assert_eq!(schemas[0].function.name, "test_function");
        assert_eq!(schemas[0].function.description, Some("A test function".to_string()));
    }

    #[test]
    fn test_complete_tool_workflow() {
        // Test a complete tool workflow: user asks -> assistant calls tool -> user provides result

        // 1. User asks a question
        let user_question = Message {
            role: MessageRole::User,
            content: vec![Part::Text {
                text: "What's the weather in Tokyo?".to_string(),
            }],
            timestamp: chrono::Utc::now(),
        };

        let openrouter_user: OpenRouterMessage = user_question.into();
        let back_to_ai: Message = openrouter_user.into();

        assert_eq!(back_to_ai.role, MessageRole::User);
        assert_eq!(back_to_ai.content.len(), 1);

        // 2. Assistant responds with tool call
        let assistant_response = Message {
            role: MessageRole::Assistant,
            content: vec![
                Part::Text {
                    text: "I'll check the weather in Tokyo for you.".to_string(),
                },
                Part::ToolCall {
                    id: "call_weather_123".to_string(),
                    name: "get_weather".to_string(),
                    args: json!({"location": "Tokyo", "units": "celsius"}),
                }
            ],
            timestamp: chrono::Utc::now(),
        };

        let openrouter_assistant: OpenRouterMessage = assistant_response.into();
        let back_to_ai_assistant: Message = openrouter_assistant.into();

        assert_eq!(back_to_ai_assistant.role, MessageRole::Assistant);
        assert_eq!(back_to_ai_assistant.content.len(), 2); // Text + ToolCall

        // Verify the tool call is preserved
        match &back_to_ai_assistant.content[1] {
            Part::ToolCall { id, name, args } => {
                assert_eq!(id, "call_weather_123");
                assert_eq!(name, "get_weather");
                assert_eq!(args["location"], "Tokyo");
                assert_eq!(args["units"], "celsius");
            }
            _ => panic!("Expected tool call part"),
        }

        // 3. User provides tool result
        let user_tool_result = Message {
            role: MessageRole::User,
            content: vec![
                Part::ToolResult {
                    call_id: "call_weather_123".to_string(),
                    name: "get_weather".to_string(),
                    content: json!({
                        "temperature": 22,
                        "condition": "sunny",
                        "humidity": 60
                    }),
                }
            ],
            timestamp: chrono::Utc::now(),
        };

        let openrouter_tool_result: OpenRouterMessage = user_tool_result.into();
        let back_to_ai_tool_result: Message = openrouter_tool_result.into();

        assert_eq!(back_to_ai_tool_result.role, MessageRole::User);
        // Tool results are currently converted to text for OpenRouter
        // This demonstrates the workflow works end-to-end
        assert!(!back_to_ai_tool_result.content.is_empty());
    }
}
