use anthropic_ox::{
    message::{
        Content as AnthropicContent, ContentBlock, ImageSource as AnthropicImageSource,
        Message as AnthropicMessage, Messages as AnthropicMessages, Role as AnthropicRole,
        Text as AnthropicText,
    },
    request::ChatRequest,
    response::{
        ChatResponse, ContentBlockDelta, StopReason as AnthropicStopReason,
        StreamEvent as AnthropicStreamEvent,
    },
    tool::{Tool, ToolChoice},
};

use crate::{
    content::{
        delta::{FinishReason, StreamEvent, StreamStop},
        message::{Message, MessageRole},
        part::Part,
    },
    errors::GenerateContentError,
    model::ModelRequest,
    tool::Tool as AiOxTool,
    usage::Usage,
    ModelResponse,
};

/// Convert ai-ox ModelRequest to Anthropic ChatRequest
///
/// # Arguments
/// * `request` - The ai-ox model request to convert
/// * `model` - The Anthropic model name to use
/// * `system_instruction` - Optional system instruction to include
/// * `max_tokens` - Maximum tokens for the response
///
/// # Returns
/// * `Ok(ChatRequest)` - Successfully converted request
/// * `Err(GenerateContentError)` - If conversion fails due to unsupported content types
///
/// # Notes
/// - Tool schemas require the "schema" feature to be enabled
/// - Gemini-specific tools are skipped during conversion
/// - Image content is converted to Anthropic's base64 format
pub fn convert_request_to_anthropic(
    request: ModelRequest,
    model: String,
    system_instruction: Option<String>,
    max_tokens: u32,
    tools: Option<(Vec<Tool>, Option<ToolChoice>)>,
) -> Result<ChatRequest, GenerateContentError> {
    let mut anthropic_messages = AnthropicMessages::new();
    let system_message = system_instruction;

    // Convert messages, handling system messages specially
    for message in request.messages {
        let content = extract_content_from_parts(&message.content)?;
        let role = match message.role {
            MessageRole::User => AnthropicRole::User,
            MessageRole::Assistant => AnthropicRole::Assistant,
        };
        anthropic_messages.push(AnthropicMessage::new(role, content));
    }

    let (tools, tool_choice) = if let Some((tools, tool_choice)) = tools {
        (Some(tools), tool_choice)
    } else if let Some(request_tools) = request.tools {
        (Some(convert_tools_to_anthropic(request_tools)?), None)
    } else {
        (None, None)
    };

    Ok(ChatRequest::builder()
        .model(model)
        .messages(anthropic_messages)
        .max_tokens(max_tokens)
        .maybe_system(system_message.map(Into::into))
        .maybe_tools(tools)
        .maybe_tool_choice(tool_choice)
        .build())
}

impl From<anthropic_ox::response::Usage> for Usage {
    fn from(usage: anthropic_ox::response::Usage) -> Self {
        let mut new_usage = Usage::new();
        new_usage.requests = 1;
        new_usage.input_tokens_by_modality.insert(
            crate::usage::Modality::Text,
            usage.input_tokens.unwrap_or(0) as u64,
        );
        new_usage.output_tokens_by_modality.insert(
            crate::usage::Modality::Text,
            usage.output_tokens.unwrap_or(0) as u64,
        );
        new_usage
    }
}

/// Extract Anthropic content from ai-ox content parts
fn extract_content_from_parts(content: &[Part]) -> Result<Vec<AnthropicContent>, GenerateContentError> {
    let mut anthropic_content = Vec::new();
    
    for part in content {
        match part {
            Part::Text { text } => {
                anthropic_content.push(AnthropicContent::Text(AnthropicText::new(text.clone())));
            }
            Part::Image { source } => {
                // Convert ai-ox ImageSource to Anthropic ImageSource
                match source {
                    crate::content::part::ImageSource::Base64 { media_type, data } => {
                        let anthropic_source = AnthropicImageSource::Base64 {
                            media_type: media_type.clone(),
                            data: data.clone(),
                        };
                        anthropic_content.push(AnthropicContent::Image { 
                            source: anthropic_source 
                        });
                    }
                }
            }
            Part::ToolCall { id, name, args } => {
                let tool_use = anthropic_ox::tool::ToolUse::new(
                    id.clone(),
                    name.clone(),
                    args.clone(),
                );
                anthropic_content.push(AnthropicContent::ToolUse(tool_use));
            }
            Part::ToolResult { call_id, name: _, content } => {
                // Preserve JSON structure when possible
                let content_text = match content {
                    serde_json::Value::String(s) => s.clone(),
                    other => serde_json::to_string_pretty(other)
                        .unwrap_or_else(|_| other.to_string()),
                };
                
                let tool_result = anthropic_ox::tool::ToolResult::text(
                    call_id.clone(),
                    content_text,
                );
                anthropic_content.push(AnthropicContent::ToolResult(tool_result));
            }
            unsupported => {
                return Err(GenerateContentError::message_conversion(
                    &format!("Unsupported Part variant for Anthropic: {:?}", unsupported)
                ));
            }
        }
    }
    
    if anthropic_content.is_empty() {
        return Err(GenerateContentError::message_conversion(
            "No convertible content found in message - message parts must contain text, images, tool calls, or tool results compatible with Anthropic format"
        ));
    }
    
    Ok(anthropic_content)
}

/// Convert Anthropic ChatResponse to ai-ox ModelResponse
/// 
/// # Arguments
/// * `response` - The Anthropic chat response to convert
/// * `model_name` - The model name to include in the response
/// 
/// # Returns
/// * `Ok(ModelResponse)` - Successfully converted response with usage metrics
/// * `Err(GenerateContentError)` - If conversion fails
/// 
/// # Notes
/// - Tool results preserve JSON structure when possible
/// - Tool IDs are mapped from tool calls to tool results
/// - Usage metrics are converted to ai-ox format with text modality
/// - Response timestamp is set to current UTC time
pub fn convert_anthropic_response_to_ai_ox(
    response: ChatResponse,
    model_name: String,
) -> Result<ModelResponse, GenerateContentError> {
    let mut content_parts = Vec::new();
    
    // First pass: collect tool names from ToolUse for mapping to ToolResult
    let mut tool_id_to_name: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for content in &response.content {
        if let AnthropicContent::ToolUse(tool_use) = content {
            tool_id_to_name.insert(tool_use.id.clone(), tool_use.name.clone());
        }
    }
    
    // Convert content
    for content in response.content {
        match content {
            AnthropicContent::Text(text) => {
                content_parts.push(Part::Text { text: text.text });
            }
            AnthropicContent::Image { source } => {
                let source = match source {
                    AnthropicImageSource::Base64 { media_type, data } => {
                        crate::content::part::ImageSource::Base64 { media_type, data }
                    }
                };
                content_parts.push(Part::Image { source });
            }
            AnthropicContent::ToolUse(tool_use) => {
                content_parts.push(Part::ToolCall {
                    id: tool_use.id,
                    name: tool_use.name,
                    args: tool_use.input,
                });
            }
            AnthropicContent::ToolResult(tool_result) => {
                // Convert tool result content to JSON value
                let content = serde_json::json!({
                    "content": tool_result.content,
                    "is_error": tool_result.is_error
                });
                
                // Get the tool name from our mapping, fallback to extracting from ID
                let tool_name = tool_id_to_name.get(&tool_result.tool_use_id)
                    .cloned()
                    .unwrap_or_else(|| {
                        // Fallback: try to extract name from tool_use_id pattern
                        tool_result.tool_use_id.split('_')
                            .next()
                            .unwrap_or("unknown_tool")
                            .to_string()
                    });
                    
                content_parts.push(Part::ToolResult {
                    call_id: tool_result.tool_use_id,
                    name: tool_name,
                    content,
                });
            }
        }
    }
    
    let message = Message {
        role: MessageRole::Assistant,
        content: content_parts,
        timestamp: chrono::Utc::now(),
    };
    
    let usage = {
        let mut usage = Usage::new();
        usage.requests = 1;
        usage.input_tokens_by_modality.insert(crate::usage::Modality::Text, response.usage.input_tokens.unwrap_or(0) as u64);
        usage.output_tokens_by_modality.insert(crate::usage::Modality::Text, response.usage.output_tokens.unwrap_or(0) as u64);
        usage
    };
    
    Ok(ModelResponse {
        message,
        usage,
        model_name,
        vendor_name: "anthropic".to_string(),
    })
}

/// Convert streaming event to ai-ox stream events
/// 
/// **Important behavior notes:**
/// - This is a stateless converter that processes events independently
/// - Usage information is emitted from MessageDelta when available
/// - StreamStop may be emitted twice: once from MessageDelta (with stop reason) 
///   and once from MessageStop (fallback) 
/// - Consumers should handle potential duplicate StreamStop events
/// - For production use with guaranteed single StreamStop, consider a stateful converter
pub fn convert_stream_event_to_ai_ox(
    event: AnthropicStreamEvent,
) -> Vec<Result<StreamEvent, GenerateContentError>> {
    let mut events = Vec::new();
    
    match event {
        AnthropicStreamEvent::MessageStart { .. } => {
            // MessageStart contains initial message metadata - could be used for logging
            // For now, we skip as ai-ox doesn't have an equivalent event
        }
        AnthropicStreamEvent::ContentBlockStart { content_block, index } => {
            match content_block {
                ContentBlock::ToolUse { id, name, .. } => {
                    // Emit tool call start with ID and name
                    use crate::content::delta::{MessageDelta, ToolCallChunk};
                    
                    let tool_chunk = ToolCallChunk {
                        index,
                        id: Some(id),
                        name: Some(name),
                        args_delta: None,
                    };
                    
                    let message_delta = MessageDelta {
                        role: None,
                        content_delta: None,
                        tool_call_chunks: vec![tool_chunk],
                    };
                    
                    events.push(Ok(StreamEvent::MessageDelta(message_delta)));
                }
                ContentBlock::Text { .. } => {
                    // Text content blocks don't need special start handling
                }
            }
        }
        AnthropicStreamEvent::ContentBlockDelta { delta, index } => {
            match delta {
                ContentBlockDelta::TextDelta { text } => {
                    if !text.is_empty() {
                        events.push(Ok(StreamEvent::TextDelta(text)));
                    }
                }
                ContentBlockDelta::InputJsonDelta { partial_json } => {
                    // Map Anthropic's input JSON delta to ai-ox tool call chunk
                    use crate::content::delta::{MessageDelta, ToolCallChunk};
                    
                    let tool_chunk = ToolCallChunk {
                        index,
                        id: None, // ID was sent in ContentBlockStart
                        name: None, // Name was sent in ContentBlockStart
                        args_delta: Some(partial_json),
                    };
                    
                    let message_delta = MessageDelta {
                        role: None,
                        content_delta: None,
                        tool_call_chunks: vec![tool_chunk],
                    };
                    
                    events.push(Ok(StreamEvent::MessageDelta(message_delta)));
                }
            }
        }
        AnthropicStreamEvent::ContentBlockStop { .. } => {
            // ContentBlockStop signals end of content block
            // ai-ox doesn't have an equivalent event, but we could use this for validation
        }
        AnthropicStreamEvent::MessageDelta { delta, usage } => {
            // Handle usage information when available
            if let Some(usage) = usage {
                let mut ai_ox_usage = Usage::new();
                ai_ox_usage.requests = 1;
                ai_ox_usage.input_tokens_by_modality.insert(
                    crate::usage::Modality::Text, 
                    u64::from(usage.input_tokens.unwrap_or(0))
                );
                ai_ox_usage.output_tokens_by_modality.insert(
                    crate::usage::Modality::Text, 
                    u64::from(usage.output_tokens.unwrap_or(0))
                );
                
                events.push(Ok(StreamEvent::Usage(ai_ox_usage)));
            }
            
            // If we have a stop reason in delta, emit StreamStop now
            // This handles the case where MessageDelta contains the final stop reason
            if let Some(stop_reason_str) = delta.stop_reason.as_deref() {
                let stop_reason = parse_stop_reason(Some(stop_reason_str));
                let finish_reason = FinishReason::from(stop_reason);
                let usage = Usage::new(); // Basic usage since detailed usage was emitted above
                
                events.push(Ok(StreamEvent::StreamStop(StreamStop {
                    usage,
                    finish_reason,
                })));
            }
        }
        AnthropicStreamEvent::MessageStop => {
            // Emit StreamStop as a fallback
            // Note: This may result in duplicate StreamStop events if MessageDelta already emitted one
            // In a stateless converter, consumers should handle potential duplicates
            let usage = Usage::new();
            events.push(Ok(StreamEvent::StreamStop(StreamStop {
                usage,
                finish_reason: FinishReason::Stop, // Default since we don't have stop reason context
            })));
        }
        AnthropicStreamEvent::Error { error } => {
            events.push(Err(GenerateContentError::provider_error("anthropic", error.message)));
        }
        AnthropicStreamEvent::Ping => {
            // Ping events are for keep-alive, no need to emit anything
        }
    }
    
    events
}

/// Convert ai-ox Tools to Anthropic Tools
fn convert_tools_to_anthropic(tools: Vec<AiOxTool>) -> Result<Vec<anthropic_ox::tool::Tool>, GenerateContentError> {
    #[cfg(feature = "schema")]
    let mut anthropic_tools = Vec::new();
    #[cfg(not(feature = "schema"))]
    let anthropic_tools = Vec::new();
    
    for tool in tools {
        match tool {
            AiOxTool::FunctionDeclarations(functions) => {
                for func in functions {
                    let _anthropic_tool = anthropic_ox::tool::Tool::new(
                        func.name.clone(),
                        func.description.clone().unwrap_or_default(),
                    );
                    
                    // Schema support requires schema feature to be enabled in anthropic-ox
                    #[cfg(feature = "schema")]
                    let anthropic_tool = _anthropic_tool.with_schema(func.parameters.clone());
                    
                    #[cfg(not(feature = "schema"))]
                    {
                        // Return an error if schema feature is not enabled but tools are provided
                        return Err(GenerateContentError::configuration(
                            "Tool schemas require the 'schema' feature to be enabled. Please enable the 'schema' feature."
                        ));
                    }
                    
                    #[cfg(feature = "schema")]
                    anthropic_tools.push(anthropic_tool);
                }
            }
            #[cfg(feature = "gemini")]
            AiOxTool::GeminiTool(_) => {
                // Skip Gemini tools like other providers do
            }
        }
    }
    
    Ok(anthropic_tools)
}

/// Convert Anthropic stop reason string to StopReason enum
/// 
/// Maps string values from Anthropic's streaming API to the StopReason enum.
/// Used internally to parse stop reasons from MessageDelta events.
fn parse_stop_reason(reason_str: Option<&str>) -> Option<AnthropicStopReason> {
    match reason_str {
        Some("end_turn") => Some(AnthropicStopReason::EndTurn),
        Some("max_tokens") => Some(AnthropicStopReason::MaxTokens),
        Some("stop_sequence") => Some(AnthropicStopReason::StopSequence),
        Some("tool_use") => Some(AnthropicStopReason::ToolUse),
        _ => None,
    }
}

/// Convert Anthropic StopReason to ai-ox FinishReason
/// 
/// Maps Anthropic's stop reasons to ai-ox finish reasons:
/// - `EndTurn` -> `Stop` (natural completion)  
/// - `MaxTokens` -> `Length` (hit token limit)
/// - `StopSequence` -> `Stop` (hit stop sequence)
/// - `ToolUse` -> `ToolCalls` (model wants to call tools)
/// - `None` -> `Stop` (default fallback)
impl From<Option<AnthropicStopReason>> for FinishReason {
    fn from(reason: Option<AnthropicStopReason>) -> Self {
        match reason {
            Some(AnthropicStopReason::EndTurn) => FinishReason::Stop,
            Some(AnthropicStopReason::MaxTokens) => FinishReason::Length,
            Some(AnthropicStopReason::StopSequence) => FinishReason::Stop,
            Some(AnthropicStopReason::ToolUse) => FinishReason::ToolCalls,
            None => FinishReason::Stop,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{Tool as AiOxTool, FunctionMetadata};
    use serde_json::json;

    #[test]
    fn test_convert_tools_to_anthropic_multiple_functions() {
        let function1 = FunctionMetadata {
            name: "get_weather".to_string(),
            description: Some("Get weather information".to_string()),
            parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
        };
        
        let function2 = FunctionMetadata {
            name: "get_time".to_string(),
            description: Some("Get current time".to_string()),
            parameters: json!({"type": "object"}),
        };

        let tools = vec![AiOxTool::FunctionDeclarations(vec![function1, function2])];
        let result = convert_tools_to_anthropic(tools).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "get_weather");
        assert_eq!(result[1].name, "get_time");
    }

    #[test]
    fn test_convert_tools_to_anthropic_empty_tools() {
        let tools = vec![];
        let result = convert_tools_to_anthropic(tools).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    #[cfg(feature = "gemini")]
    fn test_convert_tools_to_anthropic_skips_gemini_tools() {
        use crate::tool::gemini::GeminiTool;
        
        let gemini_tool = AiOxTool::GeminiTool(GeminiTool::FunctionDeclarations(vec![]));
        
        let tools = vec![gemini_tool];
        let result = convert_tools_to_anthropic(tools).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_tool_result_json_preservation() {
        let json_content = json!({
            "temperature": 72,
            "humidity": 65,
            "condition": "sunny"
        });

        let parts = vec![
            Part::ToolResult {
                call_id: "call_123".to_string(),
                name: "get_weather".to_string(),
                content: json_content.clone(),
            }
        ];

        let result = extract_content_from_parts(&parts).unwrap();
        assert_eq!(result.len(), 1);
        
        if let AnthropicContent::ToolResult(tool_result) = &result[0] {
            assert_eq!(tool_result.tool_use_id, "call_123");
            // The JSON should be pretty-printed, not just stringified
            let content_text = match &tool_result.content[0] {
                anthropic_ox::tool::ToolResultContent::Text { text } => text,
                _ => panic!("Expected text content"),
            };
            // Should be valid JSON that can be parsed back
            let parsed: serde_json::Value = serde_json::from_str(content_text).unwrap();
            assert_eq!(parsed, json_content);
        } else {
            panic!("Expected ToolResult content");
        }
    }

    #[test]
    fn test_tool_result_string_preservation() {
        let string_content = json!("Just a simple string response");

        let parts = vec![
            Part::ToolResult {
                call_id: "call_456".to_string(),
                name: "simple_func".to_string(),
                content: string_content,
            }
        ];

        let result = extract_content_from_parts(&parts).unwrap();
        assert_eq!(result.len(), 1);
        
        if let AnthropicContent::ToolResult(tool_result) = &result[0] {
            let content_text = match &tool_result.content[0] {
                anthropic_ox::tool::ToolResultContent::Text { text } => text,
                _ => panic!("Expected text content"),
            };
            assert_eq!(content_text, "Just a simple string response");
        } else {
            panic!("Expected ToolResult content");
        }
    }

    #[test]
    fn test_stream_event_message_start_skipped() {
        use anthropic_ox::response::{StreamEvent as AnthropicStreamEvent, StreamMessage};
        
        let message_start = AnthropicStreamEvent::MessageStart {
            message: StreamMessage {
                id: "msg_123".to_string(),
                r#type: "message".to_string(),
                role: anthropic_ox::message::Role::Assistant,
                content: vec![],
                model: "claude-3-haiku".to_string(),
                stop_reason: None,
                stop_sequence: None,
                usage: anthropic_ox::response::Usage::default(),
            },
        };
        
        let result = convert_stream_event_to_ai_ox(message_start);
        assert!(result.is_empty()); // MessageStart should be skipped
    }

    #[test]
    fn test_stream_event_content_block_start_tool_use() {
        use anthropic_ox::response::StreamEvent as AnthropicStreamEvent;
        use anthropic_ox::message::message::ContentBlock;
        
        let tool_start = AnthropicStreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::ToolUse {
                id: "call_123".to_string(),
                name: "get_weather".to_string(),
                input: json!({"location": "NYC"}),
            },
        };
        
        let result = convert_stream_event_to_ai_ox(tool_start);
        assert_eq!(result.len(), 1);
        
        if let Ok(StreamEvent::MessageDelta(delta)) = &result[0] {
            assert_eq!(delta.tool_call_chunks.len(), 1);
            assert_eq!(delta.tool_call_chunks[0].index, 0);
            assert_eq!(delta.tool_call_chunks[0].id, Some("call_123".to_string()));
            assert_eq!(delta.tool_call_chunks[0].name, Some("get_weather".to_string()));
            assert_eq!(delta.tool_call_chunks[0].args_delta, None);
        } else {
            panic!("Expected MessageDelta with tool call chunk");
        }
    }

    #[test]
    fn test_stream_event_content_block_delta_text() {
        use anthropic_ox::response::{StreamEvent as AnthropicStreamEvent, ContentBlockDelta};
        
        let text_delta = AnthropicStreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::TextDelta {
                text: "Hello world".to_string(),
            },
        };
        
        let result = convert_stream_event_to_ai_ox(text_delta);
        assert_eq!(result.len(), 1);
        
        if let Ok(StreamEvent::TextDelta(text)) = &result[0] {
            assert_eq!(text, "Hello world");
        } else {
            panic!("Expected TextDelta");
        }
    }

    #[test]
    fn test_stream_event_content_block_delta_json() {
        use anthropic_ox::response::{StreamEvent as AnthropicStreamEvent, ContentBlockDelta};
        
        let json_delta = AnthropicStreamEvent::ContentBlockDelta {
            index: 1,
            delta: ContentBlockDelta::InputJsonDelta {
                partial_json: "{\"location\":".to_string(),
            },
        };
        
        let result = convert_stream_event_to_ai_ox(json_delta);
        assert_eq!(result.len(), 1);
        
        if let Ok(StreamEvent::MessageDelta(delta)) = &result[0] {
            assert_eq!(delta.tool_call_chunks.len(), 1);
            assert_eq!(delta.tool_call_chunks[0].index, 1);
            assert_eq!(delta.tool_call_chunks[0].args_delta, Some("{\"location\":".to_string()));
        } else {
            panic!("Expected MessageDelta with JSON delta");
        }
    }

    #[test]
    fn test_stream_event_message_delta_with_usage() {
        use anthropic_ox::response::{StreamEvent as AnthropicStreamEvent, MessageDelta, Usage};
        
        let message_delta = AnthropicStreamEvent::MessageDelta {
            delta: MessageDelta {
                stop_reason: None,
                stop_sequence: None,
            },
            usage: Some(Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
            }),
        };
        
        let result = convert_stream_event_to_ai_ox(message_delta);
        assert_eq!(result.len(), 1);
        
        if let Ok(StreamEvent::Usage(usage)) = &result[0] {
            assert_eq!(usage.requests, 1);
            assert_eq!(usage.input_tokens_by_modality.get(&crate::usage::Modality::Text), Some(&100));
            assert_eq!(usage.output_tokens_by_modality.get(&crate::usage::Modality::Text), Some(&50));
        } else {
            panic!("Expected Usage event");
        }
    }

    #[test]
    fn test_stream_event_message_delta_with_stop_reason() {
        use anthropic_ox::response::{StreamEvent as AnthropicStreamEvent, MessageDelta};
        
        let message_delta = AnthropicStreamEvent::MessageDelta {
            delta: MessageDelta {
                stop_reason: Some("tool_use".to_string()),
                stop_sequence: None,
            },
            usage: None,
        };
        
        let result = convert_stream_event_to_ai_ox(message_delta);
        assert_eq!(result.len(), 1);
        
        if let Ok(StreamEvent::StreamStop(stop)) = &result[0] {
            assert_eq!(stop.finish_reason, FinishReason::ToolCalls);
        } else {
            panic!("Expected StreamStop with ToolCalls reason");
        }
    }

    #[test]
    fn test_stream_event_message_stop() {
        use anthropic_ox::response::StreamEvent as AnthropicStreamEvent;
        
        let message_stop = AnthropicStreamEvent::MessageStop;
        
        let result = convert_stream_event_to_ai_ox(message_stop);
        assert_eq!(result.len(), 1);
        
        if let Ok(StreamEvent::StreamStop(stop)) = &result[0] {
            assert_eq!(stop.finish_reason, FinishReason::Stop);
        } else {
            panic!("Expected StreamStop");
        }
    }

    #[test]
    fn test_stream_event_error() {
        use anthropic_ox::response::StreamEvent as AnthropicStreamEvent;
        use anthropic_ox::error::ErrorInfo;
        
        let error_event = AnthropicStreamEvent::Error {
            error: ErrorInfo {
                r#type: "error".to_string(),
                message: "Something went wrong".to_string(),
            },
        };
        
        let result = convert_stream_event_to_ai_ox(error_event);
        assert_eq!(result.len(), 1);
        
        if let Err(err) = &result[0] {
            assert!(err.to_string().contains("Something went wrong"));
        } else {
            panic!("Expected error result");
        }
    }

    #[test]
    fn test_stream_event_ping_skipped() {
        use anthropic_ox::response::StreamEvent as AnthropicStreamEvent;
        
        let ping = AnthropicStreamEvent::Ping;
        
        let result = convert_stream_event_to_ai_ox(ping);
        assert!(result.is_empty()); // Ping should be skipped
    }

    #[test]
    fn test_stop_reason_from_conversion() {
        assert_eq!(FinishReason::from(Some(AnthropicStopReason::EndTurn)), FinishReason::Stop);
        assert_eq!(FinishReason::from(Some(AnthropicStopReason::MaxTokens)), FinishReason::Length);
        assert_eq!(FinishReason::from(Some(AnthropicStopReason::StopSequence)), FinishReason::Stop);
        assert_eq!(FinishReason::from(Some(AnthropicStopReason::ToolUse)), FinishReason::ToolCalls);
        assert_eq!(FinishReason::from(None), FinishReason::Stop);
    }

    #[test]
    fn test_parse_stop_reason() {
        assert_eq!(parse_stop_reason(Some("end_turn")), Some(AnthropicStopReason::EndTurn));
        assert_eq!(parse_stop_reason(Some("max_tokens")), Some(AnthropicStopReason::MaxTokens));
        assert_eq!(parse_stop_reason(Some("stop_sequence")), Some(AnthropicStopReason::StopSequence));
        assert_eq!(parse_stop_reason(Some("tool_use")), Some(AnthropicStopReason::ToolUse));
        assert_eq!(parse_stop_reason(Some("unknown")), None);
        assert_eq!(parse_stop_reason(None), None);
    }
}