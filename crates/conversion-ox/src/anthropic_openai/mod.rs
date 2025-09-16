//! Direct conversions between Anthropic and OpenAI formats
//!
//! This module provides explicit conversion functions between Anthropic and OpenAI
//! API formats, enabling full roundtrip compatibility for requests and responses.
//!
//! ## Design Principles
//!
//! - **Type-safe conversions**: Use native types from both anthropic-ox and openai-ox
//! - **Full roundtrip support**: Request and response conversions in both directions
//! - **Explicit over implicit**: Named functions instead of From traits
//! - **Proper error handling**: Return Result types for fallible conversions
//!
//! ## Supported Conversions
//!
//! ### Chat Completions API
//! - `anthropic_to_openai_request()` - Convert AnthropicRequest → OpenAI ChatRequest
//! - `openai_to_anthropic_response()` - Convert OpenAI ChatResponse → AnthropicResponse
//!
//! ### Responses API (Reasoning Models)
//! - `anthropic_to_openai_responses_request()` - Convert AnthropicRequest → OpenAI ResponsesRequest
//! - `openai_responses_to_anthropic_response()` - Convert OpenAI ResponsesResponse → AnthropicResponse
//! - `anthropic_to_openai_responses_response()` - Convert AnthropicResponse → OpenAI ResponsesResponse
//! - `openai_responses_to_anthropic_request()` - Convert OpenAI ResponsesRequest → AnthropicRequest
//!
//! ## Limitations
//!
//! - System messages: Anthropic has dedicated system field, OpenAI uses message chain
//! - Tool representations: Different structures for tool definitions and results
//! - Content types: Anthropic supports richer content types (thinking, search results)
//! - OpenAI-specific parameters: Some OpenAI parameters have no Anthropic equivalent

mod constants;

use anthropic_ox::{
    message::{
        Content as AnthropicContent, Message as AnthropicMessage,
        Role as AnthropicRole, StringOrContents, Text as AnthropicText,
        ThinkingContent,
    },
    request::{ChatRequest as AnthropicRequest, ThinkingConfig},
    response::{ChatResponse as AnthropicResponse, Usage as AnthropicUsage, StopReason},
    tool::{Tool as AnthropicTool, ToolUse, ToolResult as AnthropicToolResult, ToolResultContent},
};

use openai_ox::{
    request::ChatRequest as OpenAIRequest,
    response::{ChatResponse as OpenAIResponse, Choice as OpenAIChoice},
    responses::{
        ResponsesRequest, ResponsesResponse, ResponsesInput,
        ResponseOutputItem, ResponseOutputContent, ReasoningItem, ResponseMessage,
        ReasoningConfig, ResponsesUsage, ResponsesTool,
    },
};

use ai_ox_common::openai_format::{Message as OpenAIMessage, MessageRole as OpenAIRole};

use crate::ConversionError;
use self::constants::*;
use serde_json;
use uuid;

/// Helper function to extract text from Anthropic content blocks
fn extract_text_from_contents(contents: Vec<AnthropicContent>) -> String {
    contents
        .iter()
        .filter_map(|content| match content {
            AnthropicContent::Text(text) => Some(text.text.clone()),
            AnthropicContent::Thinking(thinking) => Some(thinking.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Helper function to extract text from a single Anthropic content block
fn extract_text_from_single_content(content: AnthropicContent) -> Option<String> {
    match content {
        AnthropicContent::Text(text) => Some(text.text),
        AnthropicContent::Thinking(thinking) => Some(thinking.text),
        _ => None,
    }
}

/// Helper function to decode a tool result from encoded text format
fn decode_tool_result_from_text(text: &str) -> Option<anthropic_ox::tool::ToolResult> {
    // Check if the text starts with our encoded tool result format
    if let Some(rest) = text.strip_prefix("[TOOL_RESULT:") {
        if let Some(end_pos) = rest.find("]") {
            let tool_use_id = rest[..end_pos].to_string();
            let encoded_content = &rest[end_pos + 1..];

            let mut content_parts = Vec::new();

            for part in encoded_content.split('|') {
                if let Some(text_part) = part.strip_prefix("text:") {
                    content_parts.push(anthropic_ox::tool::ToolResultContent::Text {
                        text: text_part.to_string()
                    });
                } else if let Some(image_part) = part.strip_prefix("image:") {
                    if let Some(colon_pos) = image_part.find(':') {
                        let media_type = image_part[..colon_pos].to_string();
                        let data = image_part[colon_pos + 1..].to_string();
                        content_parts.push(anthropic_ox::tool::ToolResultContent::Image {
                            source: anthropic_ox::message::ImageSource::Base64 {
                                media_type,
                                data,
                            }
                        });
                    }
                }
            }

            if !content_parts.is_empty() {
                return Some(anthropic_ox::tool::ToolResult {
                    tool_use_id,
                    content: content_parts,
                    is_error: None,
                    cache_control: None,
                });
            }
        }
    }

    None
}


/// Validate common request parameters
fn validate_request_params(model: &str, max_tokens: Option<u32>) -> Result<(), ConversionError> {
    if model.is_empty() {
        return Err(ConversionError::MissingData("Model name cannot be empty".to_string()));
    }
    
    if let Some(tokens) = max_tokens {
        if tokens == 0 {
            return Err(ConversionError::MissingData("Max tokens must be greater than 0".to_string()));
        }
        if tokens > 1_000_000 {
            return Err(ConversionError::MissingData(
                format!("Max tokens {} exceeds reasonable limit of 1,000,000", tokens)
            ));
        }
    }
    
    Ok(())
}

/// Convert Anthropic ChatRequest to OpenAI ChatRequest
/// 
/// This converts the Anthropic request format to OpenAI format, handling:
/// - System message conversion (dedicated field → system message)
/// - Message role and content mapping
/// - Basic parameters (model, temperature, max_tokens)
pub fn anthropic_to_openai_request(
    anthropic_request: AnthropicRequest,
) -> Result<OpenAIRequest, ConversionError> {
    // Validate input parameters
    validate_request_params(&anthropic_request.model, Some(anthropic_request.max_tokens))?;
    
    let mut openai_messages = Vec::new();

    // Handle system message: convert from dedicated system field to first system message
    if let Some(system) = anthropic_request.system {
        let system_content = match system {
            StringOrContents::String(s) => s,
            StringOrContents::Contents(contents) => extract_text_from_contents(contents),
        };

        if !system_content.is_empty() {
            openai_messages.push(OpenAIMessage {
                role: OpenAIRole::System,
                content: Some(system_content),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }

    // Store message count before consuming
    let anthropic_message_count = anthropic_request.messages.len();
    
    // Convert conversation messages
    for message in anthropic_request.messages {
        let openai_role = match message.role {
            AnthropicRole::User => OpenAIRole::User,
            AnthropicRole::Assistant => OpenAIRole::Assistant,
        };

        let mut content_parts = Vec::new();
        let mut tool_calls = Vec::new();

        match message.content {
            StringOrContents::String(s) => {
                if !s.is_empty() {
                    content_parts.push(s);
                }
            }
            StringOrContents::Contents(contents) => {
                for content in contents {
                    match content {
                        AnthropicContent::Text(text) => {
                            content_parts.push(text.text);
                        }
                        AnthropicContent::ToolUse(tool_use) => {
                            tool_calls.push(ai_ox_common::openai_format::ToolCall {
                                id: tool_use.id,
                                r#type: "function".to_string(),
                                function: ai_ox_common::openai_format::FunctionCall {
                                    name: tool_use.name,
                                    arguments: serde_json::to_string(&tool_use.input).unwrap_or_default(),
                                },
                            });
                        }
                        AnthropicContent::ToolResult(tool_result) => {
                            // Tool results become separate tool messages
                            let result_content = tool_result.content.iter()
                                .filter_map(|c| match c {
                                    anthropic_ox::tool::ToolResultContent::Text { text } => Some(text.clone()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join(" ");

                            openai_messages.push(OpenAIMessage {
                                role: OpenAIRole::Tool,
                                content: Some(result_content),
                                name: None,
                                tool_calls: None,
                                tool_call_id: Some(tool_result.tool_use_id),
                            });
                        }
                        _ => {
                            // For other content types, extract text if possible
                            if let Some(text) = extract_text_from_single_content(content) {
                                content_parts.push(text);
                            }
                        }
                    }
                }
            }
        }

        // Only add the message if it has content or tool calls
        if !content_parts.is_empty() || !tool_calls.is_empty() {
            let content = if content_parts.is_empty() {
                None
            } else {
                Some(content_parts.join(" "))
            };

            let tool_calls = if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            };

            openai_messages.push(OpenAIMessage {
                role: openai_role,
                content,
                name: None,
                tool_calls,
                tool_call_id: None,
            });
        }
    }

    if openai_messages.is_empty() {
        return Err(ConversionError::MissingData(
            format!("No messages found after conversion from {} Anthropic messages", 
                    anthropic_message_count)
        ));
    }

    // Convert tools from Anthropic to OpenAI format
    let tools = anthropic_request.tools.map(|anthropic_tools| {
        anthropic_tools.into_iter().filter_map(|tool| {
            match tool {
                anthropic_ox::tool::Tool::Custom(custom_tool) => {
                    Some(ai_ox_common::openai_format::Tool {
                        r#type: "function".to_string(),
                        function: ai_ox_common::openai_format::Function {
                            name: custom_tool.name,
                            description: Some(custom_tool.description),
                            parameters: Some(custom_tool.input_schema),
                        },
                    })
                },
                // Skip other tool types for now
                _ => None,
            }
        }).collect::<Vec<ai_ox_common::openai_format::Tool>>()
    });

    // Build OpenAI request - need to handle optional parameters in a single chain due to type-state builder
    let mut request = match anthropic_request.temperature {
        Some(temp) => {
            OpenAIRequest::builder()
                .model(anthropic_request.model)
                .messages(openai_messages)
                .max_tokens(anthropic_request.max_tokens)
                .temperature(temp)
                .build()
        },
        None => {
            OpenAIRequest::builder()
                .model(anthropic_request.model)
                .messages(openai_messages)
                .max_tokens(anthropic_request.max_tokens)
                .build()
        }
    };

    // Set tools if present
    if let Some(tools_list) = tools {
        request.tools = Some(tools_list);
    }

    Ok(request)
}

/// Convert OpenAI ChatResponse to Anthropic ChatResponse
/// 
/// This converts the OpenAI response format to Anthropic format, handling:
/// - Choice extraction (first choice becomes main response)
/// - Content and role mapping
/// - Usage statistics conversion
pub fn openai_to_anthropic_response(
    openai_response: OpenAIResponse,
) -> Result<AnthropicResponse, ConversionError> {
    // Get the first choice from the OpenAI response
    let choice = openai_response.choices
        .into_iter()
        .next()
        .ok_or_else(|| ConversionError::MissingData("No choices in OpenAI response".to_string()))?;

    // Convert the message content
    let content = if let Some(text) = choice.message.content {
        vec![AnthropicContent::Text(AnthropicText::new(text))]
    } else {
        return Err(ConversionError::MissingData(
            "No content in OpenAI response message".to_string()
        ));
    };

    // Convert role
    let role = match choice.message.role {
        OpenAIRole::Assistant => AnthropicRole::Assistant,
        OpenAIRole::User => AnthropicRole::User, 
        OpenAIRole::System => {
            return Err(ConversionError::UnsupportedConversion(
                "System role not supported in Anthropic responses".to_string()
            ));
        }
        _ => {
            return Err(ConversionError::UnsupportedConversion(
                format!("Unsupported role in OpenAI response: {:?}", choice.message.role)
            ));
        }
    };

    // Create the Anthropic response using constructor
    let anthropic_response = AnthropicResponse {
        id: openai_response.id,
        r#type: MESSAGE_TYPE.to_string(),
        model: openai_response.model,
        role,
        content,
        stop_reason: None,
        stop_sequence: None,
        usage: AnthropicUsage::default(),
    };

    Ok(anthropic_response)
}

/// Convert Anthropic ChatRequest to OpenAI ResponsesRequest
/// 
/// This converts the Anthropic request format to OpenAI Responses API format, handling:
/// - System message inclusion in message array
/// - Message conversion to ResponsesInput::Messages
/// - Thinking configuration mapping to ReasoningConfig
/// - Basic parameters (model, max_tokens)
/// 
/// Note: Sets `store=false` and `stream=true` as required by OpenAI Responses API
pub fn anthropic_to_openai_responses_request(
    anthropic_request: AnthropicRequest,
) -> Result<ResponsesRequest, ConversionError> {
    // Validate input parameters
    validate_request_params(&anthropic_request.model, Some(anthropic_request.max_tokens))?;
    
    if anthropic_request.messages.is_empty() {
        return Err(ConversionError::MissingData(
            "Anthropic request has no messages".to_string()
        ));
    }

    // Extract instructions from system prompt - pass through raw content
    let instructions = if let Some(system) = anthropic_request.system {
        match system {
            StringOrContents::String(s) => s,
            StringOrContents::Contents(contents) => extract_text_from_contents(contents),
        }
    } else {
        String::new()
    };
    
    // Convert non-system messages for input
    let mut openai_messages = Vec::new();
    for message in anthropic_request.messages {
        let openai_role = match message.role {
            AnthropicRole::User => OpenAIRole::User,
            AnthropicRole::Assistant => OpenAIRole::Assistant,
        };

        let content_text = match message.content {
            StringOrContents::String(s) => s,
            StringOrContents::Contents(contents) => extract_text_from_contents(contents),
        };

        if !content_text.is_empty() {
            openai_messages.push(OpenAIMessage {
                role: openai_role,
                content: Some(content_text),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }

    // Create reasoning config if thinking is present
    let reasoning = anthropic_request.thinking.map(|_config| {
        ReasoningConfig {
            effort: Some(REASONING_EFFORT_HIGH.to_string()),
            summary: Some(REASONING_SUMMARY_AUTO.to_string()),
        }
    });

    // Determine include field - always provide empty array as per codex-openai-proxy
    let include = if reasoning.is_some() {
        Some(vec!["reasoning.encrypted_content".to_string()])
    } else {
        Some(vec![])  // Empty array instead of None
    };

    // Convert tools to Responses API format (uses "type": "custom" instead of "function")
    let tools: Option<Vec<ResponsesTool>> = anthropic_request.tools.map(|anthropic_tools| {
        anthropic_tools.into_iter().filter_map(|tool| {
            // Convert Anthropic tool to OpenAI Responses API tool format
            match tool {
                anthropic_ox::tool::Tool::Custom(custom_tool) => {
                    Some(ResponsesTool {
                        tool_type: "custom".to_string(),
                        name: custom_tool.name,
                        description: Some(custom_tool.description),
                        format: None, // No grammar format for now
                        parameters: None, // Responses API doesn't support parameters field
                    })
                },
                // Skip computer tools as they don't map to OpenAI
                anthropic_ox::tool::Tool::Computer(_) => None,
            }
        }).collect()
    });

    // Build the ResponsesRequest with all fields
    let mut request = ResponsesRequest::builder()
        .model(anthropic_request.model)
        .input(ResponsesInput::Messages(openai_messages))
        .max_output_tokens(anthropic_request.max_tokens)
        .store(false)  // Required by OpenAI Responses API
        .stream(true)  // Required by OpenAI Responses API
        .build();

    // Add optional fields after building
    // Always set instructions - even if empty, to ensure the field is present
    request.instructions = Some(instructions);
    
    if let Some(reasoning_config) = reasoning {
        request.reasoning = Some(reasoning_config);
    }
    
    // Always set include field (empty array if no special includes)
    if let Some(include_fields) = include {
        request.include = Some(include_fields);
    }
    
    // Set tools and related fields
    if let Some(tool_list) = tools {
        if !tool_list.is_empty() {
            request.tools = Some(tool_list);
            request.tool_choice = Some("auto".to_string());
            request.parallel_tool_calls = Some(false);
        }
        // If tools list is empty, don't set tools field at all
    }
    // Don't set tools fields when no tools are present

    Ok(request)
}

/// Convert OpenAI ResponsesResponse to Anthropic ChatResponse
/// 
/// This converts the OpenAI Responses API format to Anthropic format, handling:
/// - OutputItem array to content blocks conversion
/// - ReasoningItem to ThinkingContent mapping
/// - Message and text items to Text content
/// - Tool calls conversion
pub fn openai_responses_to_anthropic_response(
    openai_response: ResponsesResponse,
) -> Result<AnthropicResponse, ConversionError> {
    if openai_response.output.is_empty() {
        return Err(ConversionError::MissingData(
            "OpenAI response has no output items".to_string()
        ));
    }

    // Check if completed before consuming values
    let is_completed = openai_response.is_completed();
    let output_items_count = openai_response.output.len();
    
    let mut content_blocks = Vec::new();

    // Convert each output item to Anthropic content
    for item in openai_response.output {
        match item {
            ResponseOutputItem::Reasoning { id: _, summary, content: _ } => {
                // Convert reasoning to thinking content
                // Summary is an array of values - try to extract text
                let text = if summary.is_empty() {
                    "Reasoning in progress".to_string()
                } else {
                    summary.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                
                if !text.is_empty() {
                    content_blocks.push(AnthropicContent::Thinking(
                        ThinkingContent::new(text)
                    ));
                }
            }
            ResponseOutputItem::Message { id: _, status: _, content, role: _ } => {
                 // Extract text from message content items
                 for content_item in content {
                     match content_item {
                         ResponseOutputContent::Text { text, annotations: _ } => {
                             // Check if this is an encoded tool result
                             if let Some(tool_result) = decode_tool_result_from_text(&text) {
                                 content_blocks.push(AnthropicContent::ToolResult(tool_result));
                             } else {
                                 content_blocks.push(AnthropicContent::Text(
                                     AnthropicText::new(text)
                                 ));
                             }
                         }
                         ResponseOutputContent::Refusal { refusal } => {
                             content_blocks.push(AnthropicContent::Text(
                                 AnthropicText::new(format!("[Refusal: {}]", refusal))
                             ));
                         }
                     }
                 }
             }
            ResponseOutputItem::FunctionToolCall { id, details: _ } |
            ResponseOutputItem::FileSearchToolCall { id, details: _ } |
            ResponseOutputItem::ComputerToolCall { id, details: _ } |
            ResponseOutputItem::CodeInterpreterToolCall { id, details: _ } |
            ResponseOutputItem::CustomToolCall { id, details: _ } => {
                // Tool calls conversion not yet implemented - return error instead of silent failure
                return Err(ConversionError::UnsupportedConversion(
                    format!("Tool call conversion from Responses API not yet implemented. Tool call ID: {}", id)
                ));
            }
        }
    }

    if content_blocks.is_empty() {
        return Err(ConversionError::MissingData(
            format!("No content blocks generated from OpenAI response '{}' with {} output items", 
                    openai_response.id, output_items_count)
        ));
    }
    
    // Create Anthropic response
    let anthropic_response = AnthropicResponse {
        id: openai_response.id,
        r#type: MESSAGE_TYPE.to_string(),
        model: openai_response.model,
        role: AnthropicRole::Assistant,
        content: content_blocks,
        stop_reason: if is_completed {
            Some(StopReason::EndTurn)
        } else {
            None
        },
        stop_sequence: None,
        usage: openai_response.usage
            .map(|u| AnthropicUsage {
                input_tokens: Some(u.input_tokens),
                output_tokens: Some(u.output_tokens),
                thinking_tokens: None,
            })
            .unwrap_or_default(),
    };

    Ok(anthropic_response)
}

/// Convert Anthropic ChatResponse to OpenAI ResponsesResponse
/// 
/// This converts an Anthropic response to OpenAI Responses API format, handling:
/// - Content blocks to OutputItem array conversion
/// - ThinkingContent to ReasoningItem with encrypted_content
/// - Text content to Message items
pub fn anthropic_to_openai_responses_response(
    anthropic_response: AnthropicResponse,
) -> Result<ResponsesResponse, ConversionError> {
    if anthropic_response.content.is_empty() {
        return Err(ConversionError::MissingData(
            "Anthropic response has no content".to_string()
        ));
    }

    let mut output_items = Vec::new();
    let mut all_text = Vec::new();
    
    // Store content length before consuming
    let content_blocks_count = anthropic_response.content.len();

    // Convert each content block to output items
    for content in anthropic_response.content {
        match content {
            AnthropicContent::Thinking(thinking) => {
                // Convert thinking to reasoning item with proper structure
                output_items.push(ResponseOutputItem::Reasoning {
                    id: format!("rs_{}", uuid::Uuid::new_v4()),
                    summary: if let Some(sig) = thinking.signature {
                        vec![serde_json::Value::String(sig)]
                    } else {
                        vec![serde_json::Value::String(thinking.text.clone())]
                    },
                    content: None,
                });
            }
             AnthropicContent::Text(text) => {
                 // Collect text for a single message at the end
                 all_text.push(text.text);
             }
             AnthropicContent::ToolResult(tool_result) => {
                 // Convert tool result to a message output item with structured encoding
                 let mut result_parts = Vec::new();

                 for content in &tool_result.content {
                     match content {
                         anthropic_ox::tool::ToolResultContent::Text { text } => {
                             result_parts.push(format!("text:{}", text));
                         }
                         anthropic_ox::tool::ToolResultContent::Image { source } => {
                             match source {
                                 anthropic_ox::message::ImageSource::Base64 { media_type, data } => {
                                     result_parts.push(format!("image:{}:{}", media_type, data));
                                 }
                             }
                         }
                     }
                 }

                 if !result_parts.is_empty() {
                     let encoded_content = result_parts.join("|");
                     output_items.push(ResponseOutputItem::Message {
                         id: format!("tool_result_{}", uuid::Uuid::new_v4()),
                         status: "completed".to_string(),
                         content: vec![ResponseOutputContent::Text {
                             text: format!("[TOOL_RESULT:{}]{}", tool_result.tool_use_id, encoded_content),
                             annotations: vec![],
                         }],
                         role: ROLE_ASSISTANT.to_string(),
                     });
                 }
             }
             _ => {
                 log::debug!("Skipping unsupported content type in conversion");
             }
        }
    }

    // If we have text content, add it as a message
    if !all_text.is_empty() {
        let combined_text = all_text.join("\n");
        output_items.push(ResponseOutputItem::Message {
            id: format!("msg_{}", uuid::Uuid::new_v4()),
            status: "completed".to_string(),
            content: vec![ResponseOutputContent::Text {
                text: combined_text,
                annotations: vec![],
            }],
            role: match anthropic_response.role {
                AnthropicRole::Assistant => ROLE_ASSISTANT.to_string(),
                AnthropicRole::User => ROLE_USER.to_string(),
            },
        });
    }

    if output_items.is_empty() {
        return Err(ConversionError::MissingData(
            format!("No output items generated from Anthropic response '{}' with {} content blocks", 
                    anthropic_response.id, content_blocks_count)
        ));
    }

    // output_text will be generated by the SDK's add_output_text function
    // We don't need to generate it here since it's skip_deserializing

    // Create ResponsesResponse matching official SDK structure
    let responses_response = ResponsesResponse {
        id: anthropic_response.id,
        created_at: chrono::Utc::now().timestamp() as u64,
        output_text: String::new(), // Will be filled by add_output_text
        error: None,
        incomplete_details: None,
        instructions: None,
        metadata: None,
        model: anthropic_response.model,
        object: "response".to_string(),
        output: output_items,
        parallel_tool_calls: false,
        temperature: None,
        tool_choice: None,
        tools: vec![],
        top_p: None,
        background: None,
        conversation: None,
        max_output_tokens: None,
        previous_response_id: None,
        prompt_cache_key: None,
        max_tool_calls: None,
        service_tier: Some("default".to_string()),
        top_logprobs: None,
        reasoning: None,
        safety_identifier: None,
        status: if anthropic_response.stop_reason.is_some() {
            Some(STATUS_COMPLETED.to_string())
        } else {
            Some(STATUS_IN_PROGRESS.to_string())
        },
        text: None,
        truncation: None,
        usage: Some(ResponsesUsage {
            input_tokens: anthropic_response.usage.input_tokens.unwrap_or(0),
            output_tokens: anthropic_response.usage.output_tokens.unwrap_or(0),
            total_tokens: anthropic_response.usage.input_tokens.unwrap_or(0) + anthropic_response.usage.output_tokens.unwrap_or(0),
            input_tokens_details: None,
            output_tokens_details: None,
            reasoning_tokens: None,
            cache: None,
        }),
        user: None,
    };

    Ok(responses_response)
}

/// Convert OpenAI ResponsesRequest to Anthropic ChatRequest
/// 
/// This converts an OpenAI Responses API request to Anthropic format, handling:
/// - ResponsesInput to messages conversion
/// - ReasoningConfig to thinking_config mapping
/// - System message extraction
pub fn openai_responses_to_anthropic_request(
    openai_request: ResponsesRequest,
) -> Result<AnthropicRequest, ConversionError> {
    // Validate input parameters
    validate_request_params(&openai_request.model, openai_request.max_output_tokens)?;
    
    // Use instructions field as system content if present
    let system_content = openai_request.instructions.clone();
    
    // Extract messages from input
    let openai_messages = match openai_request.input {
        ResponsesInput::Text(text) => {
            // Convert single text to user message
            vec![OpenAIMessage {
                role: OpenAIRole::User,
                content: Some(text),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }]
        }
        ResponsesInput::Messages(messages) => messages,
        ResponsesInput::Mixed(parts) => {
            // Convert mixed input to messages
            let text_parts: Vec<String> = parts
                .into_iter()
                .filter_map(|part| part.text)
                .collect();
            
            if text_parts.is_empty() {
                return Err(ConversionError::MissingData(
                    "No text content in mixed input".to_string()
                ));
            }

            vec![OpenAIMessage {
                role: OpenAIRole::User,
                content: Some(text_parts.join("\n")),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }]
        }
    };

    // Process messages (no longer need to extract system messages)
    let mut conversation_messages = Vec::new();

    for message in openai_messages {
        match message.role {
            OpenAIRole::User => {
                conversation_messages.push(AnthropicMessage {
                    role: AnthropicRole::User,
                    content: StringOrContents::String(
                        message.content.unwrap_or_default()
                    ),
                });
            }
            OpenAIRole::Assistant => {
                conversation_messages.push(AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: StringOrContents::String(
                        message.content.unwrap_or_default()
                    ),
                });
            }
            // System messages should not be in the input messages when using instructions field
            OpenAIRole::System => {
                log::debug!("System message found in ResponsesInput messages - this should be in instructions field");
            }
            _ => {
                log::warn!("Unsupported role in ResponsesRequest, skipping");
            }
        }
    }

    if conversation_messages.is_empty() {
        return Err(ConversionError::MissingData(
            format!("No conversation messages found after conversion from ResponsesInput for model '{}'", 
                    openai_request.model)
        ));
    }

    // Convert reasoning config to thinking config
    let thinking_config = openai_request.reasoning.and_then(|_reasoning| {
        Some(ThinkingConfig::new(DEFAULT_THINKING_BUDGET))
    });

    // Build Anthropic request with base fields, then add optional ones
    let mut request = AnthropicRequest::builder()
        .model(openai_request.model)
        .messages(conversation_messages)
        .max_tokens(openai_request.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS))
        .build();
    
    // Add optional fields after building
    if let Some(system) = system_content {
        request.system = Some(StringOrContents::String(system));
    }
    
    if let Some(config) = thinking_config {
        request.thinking = Some(config);
    }
    
    Ok(request)
}

/// Convert OpenAI ChatRequest to Anthropic ChatRequest
///
/// This converts the OpenAI request format to Anthropic format, handling:
/// - System message extraction (first system message becomes dedicated system field)
/// - Message role and content mapping
/// - Basic parameters (model, temperature, max_tokens)
/// - Tool conversion (OpenAI functions to Anthropic tools)
pub fn openai_to_anthropic_request(
    openai_request: OpenAIRequest,
) -> Result<AnthropicRequest, ConversionError> {
    // Validate input parameters
    validate_request_params(&openai_request.model, openai_request.max_tokens)?;

    let mut anthropic_messages = Vec::new();
    let mut system_message = None;

    // Process messages, extracting system message
    for message in &openai_request.messages {
        match message.role {
            OpenAIRole::System => {
                // Store system message separately for Anthropic
                if system_message.is_none() {
                    system_message = message.content.clone();
                } else {
                    // Multiple system messages - concatenate
                    system_message = Some(format!(
                        "{}\n{}",
                        system_message.unwrap(),
                        message.content.clone().unwrap_or_default()
                    ));
                }
            }
            OpenAIRole::User => {
                let content = vec![AnthropicContent::Text(AnthropicText {
                    text: message.content.as_ref().unwrap_or(&String::new()).clone(),
                    cache_control: None,
                })];
                anthropic_messages.push(AnthropicMessage {
                    role: AnthropicRole::User,
                    content: content.into(),
                });
            }
            OpenAIRole::Assistant => {
                let mut content = Vec::new();

                // Add text content
                if let Some(text) = message.content.as_ref() {
                    content.push(AnthropicContent::Text(AnthropicText {
                        text: text.clone(),
                        cache_control: None,
                    }));
                }

                // Add tool calls
                if let Some(tool_calls) = &message.tool_calls {
                    for tool_call in tool_calls {
                        content.push(AnthropicContent::ToolUse(ToolUse {
                            id: tool_call.id.clone(),
                            name: tool_call.function.name.clone(),
                            input: serde_json::from_str(&tool_call.function.arguments).unwrap_or(serde_json::Value::Null),
                            cache_control: None,
                        }));
                    }
                }

                anthropic_messages.push(AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: content.into(),
                });
            }
            OpenAIRole::Tool => {
                // Tool results become user messages with tool results
                let content = vec![AnthropicContent::ToolResult(anthropic_ox::tool::ToolResult {
                    tool_use_id: message.tool_call_id.clone().unwrap_or_default(),
                    content: vec![anthropic_ox::tool::ToolResultContent::Text {
                        text: message.content.as_ref().unwrap_or(&String::new()).clone(),
                    }],
                    is_error: Some(false),
                    cache_control: None,
                })];
                anthropic_messages.push(AnthropicMessage {
                    role: AnthropicRole::User,
                    content: content.into(),
                });
            }
        }
    }

    if anthropic_messages.is_empty() {
        return Err(ConversionError::MissingData(
            format!("No messages found after conversion from {} OpenAI messages",
                    openai_request.messages.len())
        ));
    }

    // Convert tools from OpenAI format to Anthropic format
    let tools = openai_request.tools.as_ref().map(|openai_tools| {
        openai_tools.iter().filter_map(|tool| {
            Some(anthropic_ox::tool::Tool::Custom(anthropic_ox::tool::CustomTool::new(
                tool.function.name.clone(),
                tool.function.description.clone().unwrap_or_default(),
            ).with_schema(tool.function.parameters.clone().unwrap_or(serde_json::json!({})))))
        }).collect::<Vec<anthropic_ox::tool::Tool>>()
    });

    // Build Anthropic request - need to handle optional parameters in a single chain due to type-state builder
    let mut request = match openai_request.temperature {
        Some(temp) => {
            AnthropicRequest::builder()
                .model(openai_request.model)
                .messages(anthropic_messages)
                .max_tokens(openai_request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS))
                .temperature(temp as f32)
                .build()
        },
        None => {
            AnthropicRequest::builder()
                .model(openai_request.model)
                .messages(anthropic_messages)
                .max_tokens(openai_request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS))
                .build()
        }
    };

    // Set optional fields manually on the built request
    if let Some(system) = system_message {
        request.system = Some(StringOrContents::String(system));
    }
    if let Some(tools) = tools {
        request.tools = Some(tools);
    }

    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_anthropic_to_openai_request_basic() {
        let anthropic_request = AnthropicRequest::builder()
            .model("claude-3-haiku-20240307")
            .system(StringOrContents::String("You are a helpful assistant".to_string()))
            .messages(vec![
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: StringOrContents::String("Hello".to_string()),
                },
            ])
            .temperature(0.7)
            .max_tokens(1000)
            .build();

        let result = anthropic_to_openai_request(anthropic_request).unwrap();

        assert_eq!(result.model, "claude-3-haiku-20240307");
        assert_eq!(result.temperature, Some(0.7));
        assert_eq!(result.max_tokens, Some(1000));
        assert_eq!(result.messages.len(), 2); // system + user
        
        // Check system message
        assert_eq!(result.messages[0].role, OpenAIRole::System);
        assert_eq!(result.messages[0].content, Some("You are a helpful assistant".to_string()));

        // Check user message
        assert_eq!(result.messages[1].role, OpenAIRole::User);
        assert_eq!(result.messages[1].content, Some("Hello".to_string()));
    }

    #[test]
    fn test_openai_to_anthropic_response_basic() {
        use openai_ox::response::Choice;
        let openai_choice = Choice {
            index: 0,
            message: OpenAIMessage {
                role: OpenAIRole::Assistant,
                content: Some("Hello there!".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: Some("stop".to_string()),
            logprobs: None,
        };

        let openai_response = OpenAIResponse {
            id: "response-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-3.5-turbo".to_string(),
            choices: vec![openai_choice],
            usage: None,
            system_fingerprint: None,
        };

        let result = openai_to_anthropic_response(openai_response).unwrap();

        assert_eq!(result.id, "response-123");
        assert_eq!(result.model, "gpt-3.5-turbo");
        assert_eq!(result.role, AnthropicRole::Assistant);

        assert_eq!(result.content.len(), 1);
        if let AnthropicContent::Text(text) = &result.content[0] {
            assert_eq!(text.text, "Hello there!");
        } else {
            panic!("Expected text content");
        }
    }
}