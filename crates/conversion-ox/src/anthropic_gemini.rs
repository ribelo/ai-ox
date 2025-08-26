//! Conversions between Anthropic and Gemini formats
//!
//! This module provides functions to convert between Anthropic and Gemini API formats.

use anthropic_ox::{
    message::{
        Content as AnthropicContent,
        Role as AnthropicRole,
    },
    request::ChatRequest as AnthropicRequest,
    response::{ChatResponse as AnthropicResponse, StopReason as AnthropicStopReason},
    tool::Tool as AnthropicTool,
};

use gemini_ox::{
    content::{Content as GeminiContent, Part as GeminiPart, Role as GeminiRole, Text as GeminiText, PartData, Blob},
    generate_content::{
        request::GenerateContentRequest as GeminiRequest,
        response::GenerateContentResponse as GeminiResponse,
    },
    tool::{Tool as GeminiTool, FunctionMetadata},
};

/// Convert Anthropic ChatRequest to Gemini GenerateContentRequest
pub fn anthropic_to_gemini_request(anthropic_request: AnthropicRequest) -> GeminiRequest {
    let mut gemini_contents = Vec::new();
    
    // Convert messages to Gemini contents
    for message in anthropic_request.messages.0 {
        match message.role {
            AnthropicRole::User => {
                let parts = convert_anthropic_message_content_to_parts(&message.content);
                gemini_contents.push(GeminiContent {
                    role: GeminiRole::User,
                    parts,
                });
            }
            AnthropicRole::Assistant => {
                let parts = convert_anthropic_message_content_to_parts(&message.content);
                gemini_contents.push(GeminiContent {
                    role: GeminiRole::Model,
                    parts,
                });
            }
        }
    }

    // Handle system instruction
    let system_instruction = anthropic_request.system.map(|system| {
        let system_text = match system {
            anthropic_ox::message::StringOrContents::String(s) => s,
            anthropic_ox::message::StringOrContents::Contents(contents) => {
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
        
        GeminiContent {
            role: GeminiRole::User,
            parts: vec![GeminiPart::new(PartData::Text(GeminiText::from(system_text)))],
        }
    });

    // Convert tools
    let tools = anthropic_request.tools.map(|anthropic_tools| {
        anthropic_tools
            .into_iter()
            .map(|tool| {
                serde_json::to_value(anthropic_tool_to_gemini_tool(tool))
                    .unwrap_or_default()
            })
            .collect()
    });

    // Build Gemini request
    let mut request = GeminiRequest::builder()
        .model(anthropic_request.model)
        .content_list(gemini_contents)
        .build();
    
    if let Some(system) = system_instruction {
        request.system_instruction = Some(system);
    }

    if let Some(tools) = tools {
        request.tools = Some(tools);
    }

    request
}

/// Convert Gemini GenerateContentResponse to Anthropic ChatResponse  
pub fn gemini_to_anthropic_response(gemini_response: GeminiResponse) -> AnthropicResponse {
    let content = if let Some(candidate) = gemini_response.candidates.first() {
        convert_gemini_parts_to_anthropic_content(&candidate.content.parts)
    } else {
        Vec::new()
    };

    let stop_reason = gemini_response
        .candidates
        .first()
        .and_then(|c| c.finish_reason.as_ref())
        .map(|reason| match reason {
            gemini_ox::generate_content::FinishReason::Stop => AnthropicStopReason::EndTurn,
            gemini_ox::generate_content::FinishReason::MaxTokens => AnthropicStopReason::MaxTokens,
            gemini_ox::generate_content::FinishReason::Safety => AnthropicStopReason::StopSequence,
            gemini_ox::generate_content::FinishReason::Recitation => AnthropicStopReason::StopSequence,
            _ => AnthropicStopReason::EndTurn,
        });

    let usage = gemini_response.usage_metadata.map(|usage| anthropic_ox::response::Usage {
        input_tokens: Some(usage.prompt_token_count),
        output_tokens: usage.candidates_token_count,
    }).unwrap_or_default();

    AnthropicResponse {
        id: uuid::Uuid::new_v4().to_string(),
        r#type: "message".to_string(),
        role: AnthropicRole::Assistant,
        content,
        model: gemini_response.model_version.unwrap_or_default(),
        stop_reason,
        stop_sequence: None,
        usage,
    }
}

/// Convert Anthropic message content (StringOrContents) to Gemini parts
fn convert_anthropic_message_content_to_parts(content: &anthropic_ox::message::StringOrContents) -> Vec<GeminiPart> {
    match content {
        anthropic_ox::message::StringOrContents::String(text) => {
            vec![GeminiPart::new(PartData::Text(GeminiText::from(text.clone())))]
        }
        anthropic_ox::message::StringOrContents::Contents(contents) => {
            convert_anthropic_content_to_parts(contents)
        }
    }
}

/// Convert Anthropic content blocks to Gemini parts
fn convert_anthropic_content_to_parts(content: &[AnthropicContent]) -> Vec<GeminiPart> {
    content
        .iter()
        .filter_map(|content| match content {
            AnthropicContent::Text(text) => {
                Some(GeminiPart::new(PartData::Text(GeminiText::from(text.text.clone()))))
            }
            AnthropicContent::Image { source } => {
                match source {
                    anthropic_ox::message::ImageSource::Base64 { media_type, data } => {
                        Some(GeminiPart::new(PartData::InlineData(Blob::new(
                            media_type.clone(),
                            data.clone(),
                        ))))
                    }
                }
            }
            AnthropicContent::ToolUse(tool_use) => {
                Some(GeminiPart::new(PartData::FunctionCall(gemini_ox::content::FunctionCall {
                    id: Some(tool_use.id.clone()),
                    name: tool_use.name.clone(),
                    args: Some(tool_use.input.clone()),
                })))
            }
            AnthropicContent::ToolResult(tool_result) => {
                // Convert tool result content to JSON
                let text_parts: Vec<String> = tool_result.content.iter()
                    .filter_map(|content| match content {
                        anthropic_ox::tool::ToolResultContent::Text { text } => Some(text.clone()),
                        anthropic_ox::tool::ToolResultContent::Image { .. } => None, // Skip images for now
                    })
                    .collect();
                let response = serde_json::json!({"text": text_parts.join("\n")});
                
                Some(GeminiPart::new(PartData::FunctionResponse(gemini_ox::content::FunctionResponse {
                    id: Some(tool_result.tool_use_id.clone()),
                    name: "".to_string(), // Gemini doesn't require function name in response
                    response,
                    will_continue: None,
                    scheduling: None,
                })))
            }
            AnthropicContent::Thinking(_) => {
                // Thinking content is internal reasoning, not exposed in Gemini format
                None
            }
        })
        .collect()
}

/// Convert Gemini parts to Anthropic content blocks
fn convert_gemini_parts_to_anthropic_content(parts: &[GeminiPart]) -> Vec<AnthropicContent> {
    parts
        .iter()
        .filter_map(|part| match &part.data {
            PartData::Text(text) => {
                Some(AnthropicContent::Text(anthropic_ox::message::Text::new(text.to_string())))
            }
            PartData::InlineData(blob) => {
                Some(AnthropicContent::Image {
                    source: anthropic_ox::message::ImageSource::Base64 {
                        media_type: blob.mime_type.clone(),
                        data: blob.data.clone(),
                    },
                })
            }
            PartData::FunctionCall(function_call) => {
                Some(AnthropicContent::ToolUse(anthropic_ox::tool::ToolUse {
                    id: function_call.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    name: function_call.name.clone(),
                    input: function_call.args.clone().unwrap_or_default(),
                    cache_control: None,
                }))
            }
            PartData::FunctionResponse(_) => {
                // Gemini function responses don't map directly to Anthropic content
                // They are typically handled as part of the conversation flow
                None
            }
            _ => None,
        })
        .collect()
}

/// Convert Anthropic Tool to Gemini Tool  
pub fn anthropic_tool_to_gemini_tool(anthropic_tool: AnthropicTool) -> GeminiTool {
    GeminiTool::FunctionDeclarations(vec![FunctionMetadata {
        name: anthropic_tool.name,
        description: Some(anthropic_tool.description),
        parameters: draft07_to_openapi3(anthropic_tool.input_schema),
    }])
}

/// Convert JSON Schema Draft-07 format to OpenAPI 3.0 format
/// 
/// Key transformations:
/// - Remove Draft-07 meta fields ($schema, additionalProperties, etc.)  
/// - Convert nullable: ["string", "null"] → "string" + nullable: true
/// - Remove unsupported validation constraints
/// - Recursively transform nested schemas
pub fn draft07_to_openapi3(schema: serde_json::Value) -> serde_json::Value {
    match schema {
        serde_json::Value::Object(mut obj) => {
            // 1. Remove Draft-07 specific meta fields
            obj.remove("$schema");
            obj.remove("additionalProperties");
            obj.remove("default");
            obj.remove("optional");
            obj.remove("title");
            
            // 2. Remove unsupported validation constraints
            obj.remove("maximum");
            obj.remove("minimum");
            obj.remove("exclusiveMaximum");
            obj.remove("exclusiveMinimum");
            obj.remove("multipleOf");
            obj.remove("maxLength");
            obj.remove("minLength");
            obj.remove("pattern");
            obj.remove("maxItems");
            obj.remove("minItems");
            obj.remove("uniqueItems");
            obj.remove("maxProperties");
            obj.remove("minProperties");
            
            // 3. Remove complex schema composition (not supported in OpenAPI 3.0)
            obj.remove("oneOf");
            obj.remove("anyOf");
            obj.remove("allOf");
            obj.remove("not");
            obj.remove("if");
            obj.remove("then");
            obj.remove("else");
            obj.remove("patternProperties");
            obj.remove("dependencies");
            obj.remove("additionalItems");
            obj.remove("contains");
            obj.remove("const");
            
            // 4. Convert nullable type arrays to OpenAPI 3.0 format
            if let Some(type_value) = obj.get_mut("type") {
                if let serde_json::Value::Array(type_array) = type_value {
                    // Check if this is a nullable type like ["string", "null"]
                    if type_array.len() == 2 && 
                       type_array.contains(&serde_json::Value::String("null".to_string())) {
                        
                        // Extract the non-null type
                        let non_null_type = type_array.iter()
                            .find(|&t| t != &serde_json::Value::String("null".to_string()))
                            .cloned()
                            .unwrap_or_else(|| serde_json::Value::String("string".to_string()));
                        
                        // Set single type and add nullable property
                        *type_value = non_null_type;
                        obj.insert("nullable".to_string(), serde_json::Value::Bool(true));
                    } else if type_array.len() == 1 {
                        // Convert single-item array to string
                        *type_value = type_array[0].clone();
                    }
                }
            }
            
            // 5. Recursively transform nested schemas
            if let Some(properties) = obj.get_mut("properties") {
                if let serde_json::Value::Object(props) = properties {
                    for (_, prop_value) in props.iter_mut() {
                        *prop_value = draft07_to_openapi3(prop_value.clone());
                    }
                }
            }
            
            // Transform array items
            if let Some(items) = obj.get_mut("items") {
                *items = draft07_to_openapi3(items.clone());
            }
            
            // Transform additional items (though we remove additionalItems above)
            if let Some(additional_items) = obj.get_mut("additionalItems") {
                *additional_items = draft07_to_openapi3(additional_items.clone());
            }
            
            serde_json::Value::Object(obj)
        }
        // For non-object values, return as-is
        other => other,
    }
}

/// Convert Gemini Tool to Anthropic Tool
pub fn gemini_tool_to_anthropic_tool(gemini_tool: GeminiTool) -> AnthropicTool {
    match gemini_tool {
        GeminiTool::FunctionDeclarations(functions) => {
            // Take the first function if multiple are present
            if let Some(func) = functions.into_iter().next() {
                let tool = AnthropicTool::new(
                    func.name,
                    func.description.unwrap_or_else(|| "Unknown function".to_string()),
                ).with_schema(func.parameters);
                tool
            } else {
                // Fallback for empty function list
                AnthropicTool::new(
                    "unknown".to_string(),
                    "Unknown function".to_string(),
                )
            }
        }
        // Handle other Gemini tool types by converting them to basic function tools
        GeminiTool::GoogleSearchRetrieval { .. } => {
            AnthropicTool::new(
                "google_search_retrieval".to_string(),
                "Google Search Retrieval tool".to_string(),
            )
        }
        GeminiTool::CodeExecution { .. } => {
            AnthropicTool::new(
                "code_execution".to_string(),
                "Code execution tool".to_string(),
            )
        }
        GeminiTool::GoogleSearch(_) => {
            AnthropicTool::new(
                "google_search".to_string(),
                "Google Search tool".to_string(),
            )
        }
    }
}