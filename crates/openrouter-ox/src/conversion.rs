//! Conversions between Anthropic and OpenRouter formats
//!
//! This module provides `From` trait implementations and helper functions to convert
//! between Anthropic and OpenRouter API formats. Some conversions are lossless, while
//! others may lose information due to structural differences between the APIs.
//!
//! ## Limitations
//!
//! - System messages: Anthropic has dedicated system field, OpenRouter uses message chain
//! - Tool results: Different representations (content vs separate messages)
//! - Tool names in responses: OpenRouter doesn't preserve tool names in some cases
//! - Streaming events: Completely different architectures

use anthropic_ox::{
    message::{
        Content as AnthropicContent, Message as AnthropicMessage, Messages as AnthropicMessages,
        Role as AnthropicRole,
    },
    request::ChatRequest as AnthropicRequest,
    response::{ChatResponse as AnthropicResponse, StopReason as AnthropicStopReason},
    tool::Tool as AnthropicTool,
};

use crate::{
    message::{
        AssistantMessage, ContentPart, Message as OpenRouterMessage,
        Messages as OpenRouterMessages, SystemMessage, ToolMessage, UserMessage,
    },
    request::ChatRequest as OpenRouterRequest,
    response::{
        ChatCompletionResponse as OpenRouterResponse, ChatCompletionChunk, Choice, 
        FinishReason as OpenRouterFinishReason,
    },
    tool::{FunctionMetadata, Tool as OpenRouterTool},
};

/// Error type for conversion failures
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("Unable to convert content: {0}")]
    ContentConversion(String),
    #[error("Missing required data: {0}")]
    MissingData(String),
    #[error("Unsupported conversion: {0}")]
    UnsupportedConversion(String),
}

/// Convert Anthropic ChatRequest to OpenRouter ChatRequest
impl From<AnthropicRequest> for OpenRouterRequest {
    fn from(anthropic_request: AnthropicRequest) -> Self {
        let mut openrouter_messages = Vec::new();

        // Handle system message: convert from dedicated system field to first SystemMessage
        if let Some(system) = anthropic_request.system {
            let system_content = match system {
                anthropic_ox::message::StringOrContents::String(s) => s,
                anthropic_ox::message::StringOrContents::Contents(contents) => {
                    // Extract text from contents
                    contents
                        .iter()
                        .filter_map(|content| match content {
                            AnthropicContent::Text(text) => Some(text.text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            };
            if !system_content.is_empty() {
                openrouter_messages.push(OpenRouterMessage::System(SystemMessage::text(
                    system_content,
                )));
            }
        }

        // Convert messages using helper function
        let converted_messages = convert_anthropic_messages_to_openrouter(anthropic_request.messages.0);
        openrouter_messages.extend(converted_messages);

        // Convert tools
        let tools = anthropic_request.tools.map(|anthropic_tools| {
            anthropic_tools
                .into_iter()
                .map(OpenRouterTool::from)
                .collect()
        });

        // Build OpenRouter request
        OpenRouterRequest::builder()
            .model(anthropic_request.model)
            .messages(OpenRouterMessages(openrouter_messages))
            .maybe_max_tokens(Some(anthropic_request.max_tokens))
            .maybe_temperature(anthropic_request.temperature.map(|t| t as f64))
            .maybe_top_p(anthropic_request.top_p.map(|tp| tp as f64))
            .maybe_top_k(anthropic_request.top_k.map(|tk| tk as u32))
            .maybe_tools(tools)
            .maybe_stop(anthropic_request.stop_sequences)
            .build()
    }
}

/// Convert OpenRouter ChatRequest to Anthropic ChatRequest
impl From<OpenRouterRequest> for AnthropicRequest {
    fn from(openrouter_request: OpenRouterRequest) -> Self {
        let mut anthropic_messages = Vec::new();
        let mut system_message: Option<String> = None;

        // Extract system message from first SystemMessage (if any)
        let messages_iter = openrouter_request.messages.0.into_iter();

        for message in messages_iter {
            match message {
                OpenRouterMessage::System(sys_msg) => {
                    // Extract text from system message
                    let system_text = sys_msg
                        .content()
                        .0
                        .iter()
                        .filter_map(|part| match part {
                            ContentPart::Text(text) => Some(text.text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    if system_message.is_none() && !system_text.is_empty() {
                        system_message = Some(system_text);
                    }
                }
                other => {
                    // Convert other messages
                    if let Some(anthropic_msg) = try_convert_openrouter_message_to_anthropic(other) {
                        anthropic_messages.push(anthropic_msg);
                    }
                }
            }
        }

        // Convert tools
        let tools = openrouter_request.tools.map(|openrouter_tools| {
            openrouter_tools
                .into_iter()
                .map(AnthropicTool::from)
                .collect()
        });

        // Build Anthropic request
        AnthropicRequest::builder()
            .model(openrouter_request.model)
            .messages(AnthropicMessages::new())
            .max_tokens(openrouter_request.max_tokens.unwrap_or(4096))
            .maybe_system(
                system_message.map(anthropic_ox::message::StringOrContents::String),
            )
            .maybe_temperature(openrouter_request.temperature.map(|t| t as f32))
            .maybe_top_p(openrouter_request.top_p.map(|tp| tp as f32))
            .maybe_top_k(openrouter_request.top_k.map(|tk| tk as i32))
            .maybe_tools(tools)
            .maybe_stop_sequences(openrouter_request.stop)
            .build()
    }
}

/// Convert Anthropic Response to OpenRouter Response
impl From<AnthropicResponse> for OpenRouterResponse {
    fn from(anthropic_response: AnthropicResponse) -> Self {
        // Clone content to avoid borrow checker issues
        let content_clone = anthropic_response.content.clone();
        
        // Convert content to OpenRouter format
        let content_parts: Vec<ContentPart> = content_clone
            .iter()
            .filter_map(|content| match content {
                AnthropicContent::Text(text) => Some(ContentPart::Text(text.text.clone().into())),
                AnthropicContent::Image { source } => {
                    // Convert image source to URL format
                    match source {
                        anthropic_ox::message::ImageSource::Base64 { data, media_type } => {
                            let data_url = format!("data:{};base64,{}", media_type, data);
                            Some(ContentPart::ImageUrl(
                                crate::message::ImageContent::new(data_url),
                            ))
                        }
                    }
                }
                AnthropicContent::ToolUse(_) => {
                    // Tool calls will be handled separately in assistant message
                    None
                }
                AnthropicContent::ToolResult(_) => {
                    // Tool results are not part of assistant message content in OpenRouter
                    None
                }
            })
            .collect();

        // Extract tool calls
        let tool_calls: Vec<crate::response::ToolCall> = anthropic_response
            .content
            .iter()
            .filter_map(|content| match content {
                AnthropicContent::ToolUse(tool_use) => {
                    Some(crate::response::ToolCall {
                        index: None,
                        id: Some(tool_use.id.clone()),
                        type_field: "function".to_string(),
                        function: crate::response::FunctionCall {
                            name: Some(tool_use.name.clone()),
                            arguments: serde_json::to_string(&tool_use.input).unwrap_or_default(),
                        },
                    })
                }
                _ => None,
            })
            .collect();

        // Create assistant message
        let mut assistant_message = AssistantMessage::new(content_parts);
        assistant_message.tool_calls = if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        };

        // Create choice
        let choice = Choice {
            index: 0,
            message: assistant_message,
            logprobs: None,
            finish_reason: anthropic_response
                .stop_reason
                .map(OpenRouterFinishReason::from)
                .unwrap_or(OpenRouterFinishReason::Stop),
            native_finish_reason: None,
        };

        OpenRouterResponse {
            id: anthropic_response.id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: anthropic_response.model,
            choices: vec![choice],
            system_fingerprint: None,
            usage: crate::response::Usage {
                prompt_tokens: anthropic_response.usage.input_tokens.unwrap_or(0),
                completion_tokens: anthropic_response.usage.output_tokens.unwrap_or(0),
                total_tokens: anthropic_response.usage.input_tokens.unwrap_or(0)
                    + anthropic_response.usage.output_tokens.unwrap_or(0),
            },
        }
    }
}

/// Convert OpenRouter Response to Anthropic Response
impl From<OpenRouterResponse> for AnthropicResponse {
    fn from(openrouter_response: OpenRouterResponse) -> Self {
        let first_choice = openrouter_response
            .choices
            .into_iter()
            .next()
            .unwrap_or(Choice {
                index: 0,
                message: AssistantMessage::new(Vec::<ContentPart>::new()),
                logprobs: None,
                finish_reason: OpenRouterFinishReason::Stop,
                native_finish_reason: None,
            });

        let mut content = Vec::new();

        // Convert text content
        for part in first_choice.message.content.0 {
            match part {
                ContentPart::Text(text) => {
                    content.push(AnthropicContent::Text(
                        anthropic_ox::message::Text::new(text.text),
                    ));
                }
                ContentPart::ImageUrl(image) => {
                    // Convert data URL back to base64 format
                    if let Some(data_url) = image.image_url.url.strip_prefix("data:") {
                        if let Some((media_part, data_part)) = data_url.split_once(";base64,") {
                            content.push(AnthropicContent::Image {
                                source: anthropic_ox::message::ImageSource::Base64 {
                                    media_type: media_part.to_string(),
                                    data: data_part.to_string(),
                                },
                            });
                        }
                    }
                }
            }
        }

        // Convert tool calls
        if let Some(tool_calls) = first_choice.message.tool_calls {
            for tool_call in tool_calls {
                if let (Some(id), Some(name)) = (tool_call.id, tool_call.function.name) {
                    let input: serde_json::Value =
                        serde_json::from_str(&tool_call.function.arguments)
                            .unwrap_or(serde_json::Value::Object(Default::default()));

                    content.push(AnthropicContent::ToolUse(
                        anthropic_ox::tool::ToolUse::new(id, name, input),
                    ));
                }
            }
        }

        AnthropicResponse {
            id: openrouter_response.id,
            r#type: "message".to_string(),
            role: AnthropicRole::Assistant,
            content,
            model: openrouter_response.model,
            stop_reason: Some(AnthropicStopReason::from(first_choice.finish_reason)),
            stop_sequence: None,
            usage: anthropic_ox::response::Usage {
                input_tokens: Some(openrouter_response.usage.prompt_tokens),
                output_tokens: Some(openrouter_response.usage.completion_tokens),
            },
        }
    }
}

/// Convert Anthropic StopReason to OpenRouter FinishReason
impl From<AnthropicStopReason> for OpenRouterFinishReason {
    fn from(reason: AnthropicStopReason) -> Self {
        match reason {
            AnthropicStopReason::EndTurn => OpenRouterFinishReason::Stop,
            AnthropicStopReason::MaxTokens => OpenRouterFinishReason::Length,
            AnthropicStopReason::StopSequence => OpenRouterFinishReason::Stop,
            AnthropicStopReason::ToolUse => OpenRouterFinishReason::ToolCalls,
        }
    }
}

/// Convert OpenRouter FinishReason to Anthropic StopReason
impl From<OpenRouterFinishReason> for AnthropicStopReason {
    fn from(reason: OpenRouterFinishReason) -> Self {
        match reason {
            OpenRouterFinishReason::Stop => AnthropicStopReason::EndTurn,
            OpenRouterFinishReason::Length | OpenRouterFinishReason::Limit => {
                AnthropicStopReason::MaxTokens
            }
            OpenRouterFinishReason::ContentFilter => AnthropicStopReason::EndTurn,
            OpenRouterFinishReason::ToolCalls => AnthropicStopReason::ToolUse,
        }
    }
}

/// Convert Anthropic Tool to OpenRouter Tool
impl From<AnthropicTool> for OpenRouterTool {
    fn from(anthropic_tool: AnthropicTool) -> Self {
        OpenRouterTool {
            tool_type: "function".to_string(),
            function: FunctionMetadata {
                name: anthropic_tool.name,
                description: Some(anthropic_tool.description),
                parameters: anthropic_tool.input_schema,
            }
        }
    }
}

/// Convert OpenRouter Tool to Anthropic Tool
impl From<OpenRouterTool> for AnthropicTool {
    fn from(openrouter_tool: OpenRouterTool) -> Self {
        {
            AnthropicTool::new(
                openrouter_tool.function.name,
                openrouter_tool.function.description.unwrap_or_default(),
            )
            .with_schema(openrouter_tool.function.parameters)
        }
    }
}

/// Helper function to convert Anthropic messages to OpenRouter messages
fn convert_anthropic_messages_to_openrouter(
    messages: Vec<AnthropicMessage>,
) -> Vec<OpenRouterMessage> {
    let mut result = Vec::new();

    for message in messages {
        match message.role {
            AnthropicRole::User => {
                // Separate regular content from tool results
                let mut text_parts = Vec::new();
                let mut tool_results = Vec::new();

                for content in message.content.as_vec() {
                    match content {
                        AnthropicContent::Text(text) => {
                            text_parts.push(ContentPart::Text(text.text.into()));
                        }
                        AnthropicContent::Image { source } => {
                            match source {
                                anthropic_ox::message::ImageSource::Base64 {
                                    media_type,
                                    data,
                                } => {
                                    let data_url = format!("data:{};base64,{}", media_type, data);
                                    text_parts.push(ContentPart::ImageUrl(
                                        crate::message::ImageContent::new(data_url),
                                    ));
                                }
                            }
                        }
                        AnthropicContent::ToolResult(tool_result) => {
                            // Tool results become separate ToolMessage
                            let content_str = match &tool_result.content[0] {
                                anthropic_ox::tool::ToolResultContent::Text { text } => text.clone(),
                                anthropic_ox::tool::ToolResultContent::Image { .. } => {
                                    "[Image content]".to_string()
                                }
                            };

                            tool_results.push(ToolMessage::with_name(
                                tool_result.tool_use_id,
                                content_str,
                                "unknown".to_string(), // OpenRouter doesn't preserve tool names
                            ));
                        }
                        AnthropicContent::ToolUse(_) => {
                            // Tool use should not appear in user messages
                        }
                    }
                }

                // Add user message if there's content
                if !text_parts.is_empty() {
                    result.push(OpenRouterMessage::User(UserMessage::new(text_parts)));
                }

                // Add tool result messages
                result.extend(tool_results.into_iter().map(OpenRouterMessage::Tool));
            }
            AnthropicRole::Assistant => {
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();

                for content in message.content.as_vec() {
                    match content {
                        AnthropicContent::Text(text) => {
                            text_parts.push(ContentPart::Text(text.text.into()));
                        }
                        AnthropicContent::Image { source } => {
                            match source {
                                anthropic_ox::message::ImageSource::Base64 {
                                    media_type,
                                    data,
                                } => {
                                    let data_url = format!("data:{};base64,{}", media_type, data);
                                    text_parts.push(ContentPart::ImageUrl(
                                        crate::message::ImageContent::new(data_url),
                                    ));
                                }
                            }
                        }
                        AnthropicContent::ToolUse(tool_use) => {
                            tool_calls.push(crate::response::ToolCall {
                                index: None,
                                id: Some(tool_use.id),
                                type_field: "function".to_string(),
                                function: crate::response::FunctionCall {
                                    name: Some(tool_use.name),
                                    arguments: serde_json::to_string(&tool_use.input)
                                        .unwrap_or_default(),
                                },
                            });
                        }
                        AnthropicContent::ToolResult(_) => {
                            // Tool results should not appear in assistant messages
                        }
                    }
                }

                let mut assistant_msg = AssistantMessage::new(text_parts);
                assistant_msg.tool_calls = if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                };
                result.push(OpenRouterMessage::Assistant(assistant_msg));
            }
        }
    }

    result
}

/// Helper function to try converting OpenRouter message to Anthropic message
fn try_convert_openrouter_message_to_anthropic(
    message: OpenRouterMessage,
) -> Option<AnthropicMessage> {
    match message {
        OpenRouterMessage::User(user_msg) => {
            let content: Vec<AnthropicContent> = user_msg
                .content
                .0
                .into_iter()
                .filter_map(|part| match part {
                    ContentPart::Text(text) => Some(AnthropicContent::Text(
                        anthropic_ox::message::Text::new(text.text),
                    )),
                    ContentPart::ImageUrl(image) => {
                        // Try to parse data URL
                        if let Some(data_url) = image.image_url.url.strip_prefix("data:") {
                            if let Some((media_part, data_part)) = data_url.split_once(";base64,") {
                                return Some(AnthropicContent::Image {
                                    source: anthropic_ox::message::ImageSource::Base64 {
                                        media_type: media_part.to_string(),
                                        data: data_part.to_string(),
                                    },
                                });
                            }
                        }
                        None
                    }
                })
                .collect();

            if !content.is_empty() {
                Some(AnthropicMessage::new(AnthropicRole::User, content))
            } else {
                None
            }
        }
        OpenRouterMessage::Assistant(assistant_msg) => {
            let mut content: Vec<AnthropicContent> = assistant_msg
                .content
                .0
                .into_iter()
                .filter_map(|part| match part {
                    ContentPart::Text(text) => Some(AnthropicContent::Text(
                        anthropic_ox::message::Text::new(text.text),
                    )),
                    ContentPart::ImageUrl(image) => {
                        // Try to parse data URL
                        if let Some(data_url) = image.image_url.url.strip_prefix("data:") {
                            if let Some((media_part, data_part)) = data_url.split_once(";base64,") {
                                return Some(AnthropicContent::Image {
                                    source: anthropic_ox::message::ImageSource::Base64 {
                                        media_type: media_part.to_string(),
                                        data: data_part.to_string(),
                                    },
                                });
                            }
                        }
                        None
                    }
                })
                .collect();

            // Add tool calls
            if let Some(tool_calls) = assistant_msg.tool_calls {
                for tool_call in tool_calls {
                    if let (Some(id), Some(name)) = (tool_call.id, tool_call.function.name) {
                        let input: serde_json::Value =
                            serde_json::from_str(&tool_call.function.arguments)
                                .unwrap_or(serde_json::Value::Object(Default::default()));

                        content.push(AnthropicContent::ToolUse(
                            anthropic_ox::tool::ToolUse::new(id, name, input),
                        ));
                    }
                }
            }

            if !content.is_empty() {
                Some(AnthropicMessage::new(AnthropicRole::Assistant, content))
            } else {
                None
            }
        }
        OpenRouterMessage::Tool(tool_msg) => {
            // Tool messages become user messages with ToolResult content
            let content = vec![AnthropicContent::ToolResult(
                anthropic_ox::tool::ToolResult::text(tool_msg.tool_call_id, tool_msg.content),
            )];

            Some(AnthropicMessage::new(AnthropicRole::User, content))
        }
        OpenRouterMessage::System(_) => {
            // System messages are handled at the request level
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_anthropic_to_openrouter_request() {
        let anthropic_request = AnthropicRequest::builder()
            .model("claude-3-sonnet")
            .messages(AnthropicMessages::new())
            .max_tokens(1000)
            .maybe_system(Some("You are a helpful assistant".into()))
            .build();

        let openrouter_request: OpenRouterRequest = anthropic_request.into();
        assert_eq!(openrouter_request.model, "claude-3-sonnet");
        assert_eq!(openrouter_request.max_tokens, Some(1000));

        // Check that system message was converted to first SystemMessage
        if let Some(first_msg) = openrouter_request.messages.0.first() {
            assert!(matches!(first_msg, OpenRouterMessage::System(_)));
        }
    }

    #[test]
    fn test_stop_reason_conversions() {
        assert_eq!(
            OpenRouterFinishReason::from(AnthropicStopReason::EndTurn),
            OpenRouterFinishReason::Stop
        );
        assert_eq!(
            OpenRouterFinishReason::from(AnthropicStopReason::MaxTokens),
            OpenRouterFinishReason::Length
        );
        assert_eq!(
            OpenRouterFinishReason::from(AnthropicStopReason::ToolUse),
            OpenRouterFinishReason::ToolCalls
        );

        assert_eq!(
            AnthropicStopReason::from(OpenRouterFinishReason::Stop),
            AnthropicStopReason::EndTurn
        );
        assert_eq!(
            AnthropicStopReason::from(OpenRouterFinishReason::Length),
            AnthropicStopReason::MaxTokens
        );
        assert_eq!(
            AnthropicStopReason::from(OpenRouterFinishReason::ToolCalls),
            AnthropicStopReason::ToolUse
        );
    }

    #[test]
    fn test_tool_conversions() {
        let anthropic_tool = AnthropicTool::new("test_function".to_string(), "A test function".to_string())
            .with_schema(json!({"type": "object", "properties": {}}));

        let openrouter_tool: OpenRouterTool = anthropic_tool.clone().into();
        assert_eq!(openrouter_tool.function.name, "test_function");
        assert_eq!(openrouter_tool.function.description, Some("A test function".to_string()));

        let back_to_anthropic: AnthropicTool = openrouter_tool.into();
        assert_eq!(back_to_anthropic.name, anthropic_tool.name);
        assert_eq!(back_to_anthropic.description, anthropic_tool.description);
    }
}

/// Streaming conversion utilities for OpenRouter ↔ Anthropic
pub mod streaming {
    use super::*;
    use anthropic_ox::response::{StreamEvent as AnthropicStreamEvent, StreamMessage, ContentBlockDelta};
    use anthropic_ox::message::ContentBlock;
    use crate::response::{ChunkChoice, Delta};

    /// Stateful converter for OpenRouter chunks to Anthropic stream events
    pub struct StreamConverter {
        message_id: Option<String>,
        content_index: usize,
        is_first_chunk: bool,
        accumulated_content: String,
    }

    impl StreamConverter {
        pub fn new() -> Self {
            Self {
                message_id: None,
                content_index: 0,
                is_first_chunk: true,
                accumulated_content: String::new(),
            }
        }

        /// Convert an OpenRouter ChatCompletionChunk to Anthropic StreamEvent
        pub fn convert_chunk(&mut self, chunk: ChatCompletionChunk) -> Vec<AnthropicStreamEvent> {
            let mut events = Vec::new();
            
            // Set message ID if not set
            if self.message_id.is_none() {
                self.message_id = Some(chunk.id.clone());
            }

            if let Some(choice) = chunk.choices.first() {
                // Handle first chunk - send MessageStart event
                if self.is_first_chunk {
                    self.is_first_chunk = false;
                    
                    let stream_message = StreamMessage {
                        id: chunk.id.clone(),
                        r#type: "message".to_string(),
                        role: anthropic_ox::message::Role::Assistant,
                        content: Vec::new(),
                        model: chunk.model.clone(),
                        stop_reason: None,
                        stop_sequence: None,
                        usage: anthropic_ox::response::Usage {
                            input_tokens: None,
                            output_tokens: None,
                        },
                    };
                    
                    events.push(AnthropicStreamEvent::MessageStart {
                        message: stream_message,
                    });

                    // Send ContentBlockStart if there's content
                    if choice.delta.content.is_some() {
                        events.push(AnthropicStreamEvent::ContentBlockStart {
                            index: self.content_index,
                            content_block: ContentBlock::Text { 
                                text: String::new() 
                            },
                        });
                    }
                }

                // Handle content delta
                if let Some(content) = &choice.delta.content {
                    if !content.is_empty() {
                        self.accumulated_content.push_str(content);
                        
                        events.push(AnthropicStreamEvent::ContentBlockDelta {
                            index: self.content_index,
                            delta: ContentBlockDelta::TextDelta {
                                text: content.clone(),
                            },
                        });
                    }
                }

                // Handle finish reason (final chunk)
                if let Some(finish_reason) = &choice.finish_reason {
                    // Send ContentBlockStop
                    events.push(AnthropicStreamEvent::ContentBlockStop {
                        index: self.content_index,
                    });

                    // Send MessageDelta with stop reason
                    events.push(AnthropicStreamEvent::MessageDelta {
                        delta: anthropic_ox::response::MessageDelta {
                            stop_reason: Some(match finish_reason {
                                OpenRouterFinishReason::Stop => "end_turn".to_string(),
                                OpenRouterFinishReason::Length | OpenRouterFinishReason::Limit => "max_tokens".to_string(),
                                OpenRouterFinishReason::ContentFilter => "end_turn".to_string(),
                                OpenRouterFinishReason::ToolCalls => "tool_use".to_string(),
                            }),
                            stop_sequence: None,
                        },
                        usage: chunk.usage.map(|usage| anthropic_ox::response::Usage {
                            input_tokens: Some(usage.prompt_tokens),
                            output_tokens: Some(usage.completion_tokens),
                        }),
                    });

                    // Send MessageStop
                    events.push(AnthropicStreamEvent::MessageStop);
                }
            }

            events
        }
    }

    impl Default for StreamConverter {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde_json::json;

        #[test]
        fn test_stream_converter_basic_flow() {
            let mut converter = StreamConverter::new();
            
            // First chunk
            let chunk1 = ChatCompletionChunk {
                id: "msg_123".to_string(),
                provider: "openrouter".to_string(),
                model: "openai/gpt-4o".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 1234567890,
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: Some("assistant".to_string()),
                        content: Some("Hello".to_string()),
                        tool_calls: None,
                    },
                    logprobs: None,
                    finish_reason: None,
                }],
                usage: None,
            };
            
            let events1 = converter.convert_chunk(chunk1);
            assert!(events1.len() >= 2); // Should have MessageStart and ContentBlockStart
            assert!(matches!(events1[0], AnthropicStreamEvent::MessageStart { .. }));
            assert!(matches!(events1[1], AnthropicStreamEvent::ContentBlockStart { .. }));
            
            // Second chunk with content
            let chunk2 = ChatCompletionChunk {
                id: "msg_123".to_string(),
                provider: "openrouter".to_string(),
                model: "openai/gpt-4o".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 1234567890,
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: Some(" World".to_string()),
                        tool_calls: None,
                    },
                    logprobs: None,
                    finish_reason: None,
                }],
                usage: None,
            };
            
            let events2 = converter.convert_chunk(chunk2);
            assert_eq!(events2.len(), 1); // Should have ContentBlockDelta
            assert!(matches!(events2[0], AnthropicStreamEvent::ContentBlockDelta { .. }));
            
            // Final chunk with finish reason
            let chunk3 = ChatCompletionChunk {
                id: "msg_123".to_string(),
                provider: "openrouter".to_string(),
                model: "openai/gpt-4o".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 1234567890,
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: None,
                        tool_calls: None,
                    },
                    logprobs: None,
                    finish_reason: Some(OpenRouterFinishReason::Stop),
                }],
                usage: Some(crate::response::Usage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                }),
            };
            
            let events3 = converter.convert_chunk(chunk3);
            assert_eq!(events3.len(), 3); // Should have ContentBlockStop, MessageDelta, MessageStop
            assert!(matches!(events3[0], AnthropicStreamEvent::ContentBlockStop { .. }));
            assert!(matches!(events3[1], AnthropicStreamEvent::MessageDelta { .. }));
            assert!(matches!(events3[2], AnthropicStreamEvent::MessageStop));
        }
    }
}