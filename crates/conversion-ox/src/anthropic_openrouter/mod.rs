//! Direct conversions between Anthropic and OpenRouter formats
//!
//! This module provides explicit conversion functions between Anthropic and OpenRouter
//! API formats, eliminating the triple conversion chain that was causing complexity.
//!
//! ## Design Principles
//!
//! - **Explicit over implicit**: Use named functions instead of From traits for complex conversions
//! - **Direct conversions**: Single-hop Anthropic ↔ OpenRouter without intermediate formats  
//! - **Centralized logic**: All conversion logic lives in this module
//! - **Proper error handling**: Return Result types for fallible conversions
//!
//! ## Limitations
//!
//! - System messages: Anthropic has dedicated system field, OpenRouter uses message chain
//! - Tool results: Different representations (content vs separate messages)
//! - Tool names in responses: OpenRouter doesn't preserve tool names in some cases
//! - Streaming events: Completely different architectures (requires stateful conversion)
//! - Computer tools: Not supported in OpenRouter (logged and skipped)

use anthropic_ox::{
    message::{
        Content as AnthropicContent, Message as AnthropicMessage, 
        Role as AnthropicRole,
    },
    request::ChatRequest as AnthropicRequest,
    response::{ChatResponse as AnthropicResponse, StopReason as AnthropicStopReason},
    tool::Tool as AnthropicTool,
};

use openrouter_ox::{
    message::{
        AssistantMessage, ContentPart, Message as OpenRouterMessage,
        SystemMessage, ToolMessage, UserMessage,
    },
    request::ChatRequest as OpenRouterRequest,
    response::{
        ChatCompletionResponse as OpenRouterResponse, 
        FinishReason as OpenRouterFinishReason,
    },
    tool::{FunctionMetadata, Tool as OpenRouterTool},
};

use crate::ConversionError;

pub mod streaming;

/// Convert Anthropic ChatRequest directly to OpenRouter ChatRequest
/// 
/// This is an explicit, single-hop conversion that handles all edge cases
/// without going through intermediate formats.
pub fn anthropic_to_openrouter_request(
    anthropic_request: AnthropicRequest,
) -> Result<OpenRouterRequest, ConversionError> {
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
                        AnthropicContent::SearchResult(search_result) => {
                            log::warn!("SearchResult content in system message converted to text");
                            Some(format!("Search Result: {}\n{}", search_result.title, search_result.source))
                        },
                        AnthropicContent::Thinking(thinking) => {
                            log::debug!("Converting thinking content in system message to text");
                            Some(thinking.text.clone())
                        },
                        _ => {
                            log::warn!("Unsupported content type in system message, skipping");
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        };
        if !system_content.is_empty() {
            openrouter_messages.push(OpenRouterMessage::System(SystemMessage::text(system_content)));
        }
    }

    // Check if we have thinking content to enable reasoning (before moving messages)
    let has_thinking = anthropic_request.messages.0.iter().any(|msg| {
        msg.content.as_vec().iter().any(|content| {
            matches!(content, AnthropicContent::Thinking(_))
        })
    });

    // Convert messages using helper function
    let converted_messages = convert_anthropic_messages_to_openrouter(anthropic_request.messages.0)?;
    openrouter_messages.extend(converted_messages);

    // Convert tools - direct conversion without intermediate format
    let tools: Option<Vec<ai_ox_common::openai_format::Tool>> = if let Some(anthropic_tools) = anthropic_request.tools {
        let mut openrouter_tools = Vec::new();
        for tool in anthropic_tools {
            match helpers::anthropic_tool_to_openrouter(tool) {
                Ok(or_tool) => {
                    // Convert to ai_ox_common::openai_format::Tool for OpenRouter request
                    let final_tool = ai_ox_common::openai_format::Tool {
                        r#type: or_tool.tool_type,
                        function: ai_ox_common::openai_format::Function {
                            name: or_tool.function.name,
                            description: or_tool.function.description,
                            parameters: Some(or_tool.function.parameters),
                        }
                    };
                    openrouter_tools.push(final_tool);
                }
                Err(e) => {
                    log::warn!("Skipping unsupported tool: {}", e);
                    // Continue without this tool instead of failing the whole request
                }
            }
        }
        if openrouter_tools.is_empty() { None } else { Some(openrouter_tools) }
    } else {
        None
    };

    // Build OpenRouter request
    let request_builder = OpenRouterRequest::builder()
        .model(anthropic_request.model)
        .messages(openrouter_messages)
        .maybe_max_tokens(Some(anthropic_request.max_tokens))
        .maybe_temperature(anthropic_request.temperature.map(|t| t as f64))
        .maybe_top_p(anthropic_request.top_p.map(|tp| tp as f64))
        .maybe_top_k(anthropic_request.top_k.map(|tk| tk as u32))
        .maybe_tools(tools)
        .maybe_stop(anthropic_request.stop_sequences);
    
    // Enable reasoning if thinking content is present
    let final_request = if has_thinking {
        request_builder.maybe_include_reasoning(Some(true)).build()
    } else {
        request_builder.build()
    };
    
    Ok(final_request)
}

/// Convert OpenRouter ChatResponse directly to Anthropic ChatResponse
/// 
/// This is an explicit, single-hop conversion that handles all edge cases
/// without going through intermediate formats.
pub fn openrouter_to_anthropic_response(
    openrouter_response: OpenRouterResponse,
) -> Result<AnthropicResponse, ConversionError> {
    let first_choice = openrouter_response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| ConversionError::MissingData("No choices in OpenRouter response".to_string()))?;

    let mut content = Vec::new();

    // Convert reasoning to thinking content if present
    if let Some(reasoning) = &first_choice.reasoning {
        let mut thinking = anthropic_ox::message::ThinkingContent::new(reasoning.clone());
        // If we have reasoning_details, use the first one as the main thinking text
        if let Some(details) = &first_choice.reasoning_details {
            if let Some(first_detail) = details.first() {
                thinking.text = first_detail.text.clone();
            }
        }
        content.push(AnthropicContent::Thinking(thinking));
    }

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
                        .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));

                content.push(AnthropicContent::ToolUse(anthropic_ox::tool::ToolUse {
                    id,
                    name,
                    input,
                    cache_control: None,
                }));
            }
        }
    }

    // Convert stop reason
    let stop_reason = match first_choice.finish_reason {
        OpenRouterFinishReason::Stop => Some(AnthropicStopReason::EndTurn),
        OpenRouterFinishReason::Length => Some(AnthropicStopReason::MaxTokens),
        OpenRouterFinishReason::Limit => Some(AnthropicStopReason::MaxTokens),
        OpenRouterFinishReason::ToolCalls => Some(AnthropicStopReason::ToolUse),
        OpenRouterFinishReason::ContentFilter => Some(AnthropicStopReason::EndTurn), // Map to closest equivalent
    };

    Ok(AnthropicResponse {
        id: openrouter_response.id,
        r#type: "message".to_string(),
        role: AnthropicRole::Assistant,
        content,
        model: openrouter_response.model,
        stop_reason,
        stop_sequence: None, // OpenRouter doesn't provide this
        usage: anthropic_ox::response::Usage {
            input_tokens: Some(openrouter_response.usage.prompt_tokens),
            output_tokens: Some(openrouter_response.usage.completion_tokens),
            thinking_tokens: None, // OpenRouter doesn't provide thinking tokens
        },
    })
}

/// Convert Anthropic messages to OpenRouter messages
fn convert_anthropic_messages_to_openrouter(
    messages: Vec<AnthropicMessage>,
) -> Result<Vec<OpenRouterMessage>, ConversionError> {
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
                                        openrouter_ox::message::ImageContent::new(data_url),
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
                            log::warn!("ToolUse content found in user message, skipping");
                        }
                        AnthropicContent::Thinking(thinking) => {
                            log::debug!("Converting thinking content in user message to text");
                            text_parts.push(ContentPart::Text(thinking.text.into()));
                        }
                        AnthropicContent::SearchResult(search_result) => {
                            log::warn!("SearchResult content converted to text for OpenRouter");
                            let text_content = format!("Search Result: {}\n{}", search_result.title, search_result.source);
                            text_parts.push(ContentPart::Text(text_content.into()));
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
                                        openrouter_ox::message::ImageContent::new(data_url),
                                    ));
                                }
                            }
                        }
                        AnthropicContent::ToolUse(tool_use) => {
                            tool_calls.push(openrouter_ox::response::ToolCall {
                                index: None,
                                id: Some(tool_use.id),
                                type_field: "function".to_string(),
                                function: openrouter_ox::response::FunctionCall {
                                    name: Some(tool_use.name),
                                    arguments: serde_json::to_string(&tool_use.input).unwrap_or_default(),
                                },
                            });
                        }
                        AnthropicContent::ToolResult(_) => {
                            log::warn!("ToolResult content found in assistant message, skipping");
                        }
                        AnthropicContent::Thinking(thinking) => {
                            log::debug!("Converting thinking content in assistant message to text");
                            text_parts.push(ContentPart::Text(thinking.text.into()));
                        }
                        AnthropicContent::SearchResult(search_result) => {
                            log::warn!("SearchResult content converted to text for OpenRouter");
                            let text_content = format!("Search Result: {}\n{}", search_result.title, search_result.source);
                            text_parts.push(ContentPart::Text(text_content.into()));
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

    Ok(result)
}

// Helper functions for internal use
mod helpers {
    use super::*;
    
    /// Convert Anthropic Tool to OpenRouter Tool (direct conversion)
    pub fn anthropic_tool_to_openrouter(tool: AnthropicTool) -> Result<OpenRouterTool, ConversionError> {
        match tool {
            AnthropicTool::Custom(custom_tool) => {
                Ok(OpenRouterTool {
                    tool_type: "function".to_string(),
                    function: FunctionMetadata {
                        name: custom_tool.name,
                        description: Some(custom_tool.description),
                        parameters: custom_tool.input_schema,
                    }
                })
            }
            AnthropicTool::Computer(_) => {
                log::warn!("Computer tool not supported in OpenRouter, skipping");
                Err(ConversionError::UnsupportedConversion(
                    "Computer tools are not supported by OpenRouter".to_string()
                ))
            }
        }
    }

    /// Convert OpenRouter Tool to Anthropic Tool (direct conversion)
    pub fn openrouter_tool_to_anthropic(tool: OpenRouterTool) -> Result<AnthropicTool, ConversionError> {
        let custom_tool = anthropic_ox::tool::CustomTool::new(
            tool.function.name,
            tool.function.description.unwrap_or_default(),
        ).with_schema(tool.function.parameters);
        
        Ok(AnthropicTool::Custom(custom_tool))
    }
}