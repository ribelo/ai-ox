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

use uuid;

use gemini_ox::{
    content::{Content as GeminiContent, Part as GeminiPart, Role as GeminiRole, Text as GeminiText, PartData, Blob},
    generate_content::{
        request::GenerateContentRequest as GeminiRequest,
        response::GenerateContentResponse as GeminiResponse,
        ThinkingConfig,
    },
    tool::{Tool as GeminiTool, FunctionMetadata},
};

/// Convert Anthropic ChatRequest to Gemini GenerateContentRequest
pub fn anthropic_to_gemini_request(anthropic_request: AnthropicRequest) -> GeminiRequest {
    let mut gemini_contents = Vec::new();
    
    // First pass: collect all tool names from all messages for ID mapping and check for thinking content
    let mut tool_id_to_name = std::collections::HashMap::new();
    let mut has_thinking_content = false;
    
    for message in &anthropic_request.messages.0 {
        if let anthropic_ox::message::StringOrContents::Contents(contents) = &message.content {
            for content in contents {
                if let AnthropicContent::ToolUse(tool_use) = content {
                    tool_id_to_name.insert(tool_use.id.clone(), tool_use.name.clone());
                } else if let AnthropicContent::Thinking(_) = content {
                    has_thinking_content = true;
                }
            }
        }
    }
    
    // Convert messages to Gemini contents
    for message in anthropic_request.messages.0 {
        match message.role {
            AnthropicRole::User => {
                let parts = convert_anthropic_message_content_to_parts(&message.content, &tool_id_to_name);
                gemini_contents.push(GeminiContent {
                    role: GeminiRole::User,
                    parts,
                });
            }
            AnthropicRole::Assistant => {
                let parts = convert_anthropic_message_content_to_parts(&message.content, &tool_id_to_name);
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

    // Enable thinking config if we detected thinking content or model name suggests thinking
    if has_thinking_content || request.model.contains("thinking") {
        // Set up generation config with thinking support
        let mut generation_config = request.generation_config.unwrap_or_default();
        generation_config.thinking_config = Some(ThinkingConfig {
            include_thoughts: true,
            thinking_budget: -1, // Dynamic thinking budget
        });
        request.generation_config = Some(generation_config);
    }

    request
}

/// Convert Gemini GenerateContentResponse to Anthropic ChatResponse
pub fn gemini_to_anthropic_response(gemini_response: GeminiResponse) -> Result<AnthropicResponse, crate::ConversionError> {
    let content = if let Some(candidate) = gemini_response.candidates.first() {
        convert_gemini_parts_to_anthropic_content(&candidate.content.parts)?
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
        thinking_tokens: usage.thoughts_token_count,
    }).unwrap_or_default();

    Ok(AnthropicResponse {
        id: uuid::Uuid::new_v4().to_string(),
        r#type: "message".to_string(),
        role: AnthropicRole::Assistant,
        content,
        model: gemini_response.model_version.unwrap_or_default(),
        stop_reason,
        stop_sequence: None,
        usage,
    })
}

/// Convert Anthropic message content (StringOrContents) to Gemini parts
fn convert_anthropic_message_content_to_parts(
    content: &anthropic_ox::message::StringOrContents,
    tool_id_to_name: &std::collections::HashMap<String, String>,
) -> Vec<GeminiPart> {
    match content {
        anthropic_ox::message::StringOrContents::String(text) => {
            vec![GeminiPart::new(PartData::Text(GeminiText::from(text.clone())))]
        }
        anthropic_ox::message::StringOrContents::Contents(contents) => {
            convert_anthropic_content_to_parts(contents, tool_id_to_name)
        }
    }
}

/// Convert Anthropic content blocks to Gemini parts
fn convert_anthropic_content_to_parts(
    content: &[AnthropicContent],
    tool_id_to_name: &std::collections::HashMap<String, String>,
) -> Vec<GeminiPart> {
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
                // Convert tool result content to JSON, preserving all content types
                let mut content_parts = Vec::new();

                for content in &tool_result.content {
                    match content {
                        anthropic_ox::tool::ToolResultContent::Text { text } => {
                            content_parts.push(serde_json::json!({"type": "text", "text": text}));
                        }
                        anthropic_ox::tool::ToolResultContent::Image { source } => {
                            match source {
                                anthropic_ox::message::ImageSource::Base64 { media_type, data } => {
                                    content_parts.push(serde_json::json!({
                                        "type": "image",
                                        "media_type": media_type,
                                        "data": data
                                    }));
                                }
                            }
                        }
                    }
                }

                let response = if content_parts.len() == 1 {
                    // Single content part - return it directly
                    content_parts.into_iter().next().unwrap()
                } else {
                    // Multiple content parts - return as array
                    serde_json::Value::Array(content_parts)
                };
                
                // Get the tool name from the mapping, with fallback to extracting from ID
                let tool_name = tool_id_to_name
                    .get(&tool_result.tool_use_id)
                    .cloned()
                    .unwrap_or_else(|| {
                        // Fallback: try to extract from tool_use_id if it contains tool name
                        // For UUIDs, use a generic name
                        if tool_result.tool_use_id.len() == 36 && tool_result.tool_use_id.matches('-').count() == 4 {
                            "function_tool".to_string()
                        } else {
                            tool_result.tool_use_id.split('_').next().unwrap_or("unknown_tool").to_string()
                        }
                    });
                
                Some(GeminiPart::new(PartData::FunctionResponse(gemini_ox::content::FunctionResponse {
                    id: Some(tool_result.tool_use_id.clone()),
                    name: tool_name,
                    response,
                    will_continue: None,
                    scheduling: None,
                })))
            }
            AnthropicContent::Thinking(thinking) => {
                // Convert Anthropic thinking content to Gemini thought part
                let mut part = GeminiPart::new_with_thought(
                    PartData::Text(GeminiText::from(thinking.text.clone())),
                    true
                );
                
                // If Anthropic thinking content has a signature, use it for Gemini's thoughtSignature
                if let Some(ref signature) = thinking.signature {
                    part.thought_signature = Some(signature.clone());
                }
                
                Some(part)
            }
            AnthropicContent::SearchResult(search_result) => {
                // Convert search result to text format for Gemini
                let text_content = format!("Search Result: {}\n{}", search_result.title, search_result.source);
                Some(GeminiPart::new(PartData::Text(GeminiText::from(text_content))))
            }
        })
        .collect()
}

/// Convert Gemini parts to Anthropic content blocks
fn convert_gemini_parts_to_anthropic_content(parts: &[GeminiPart]) -> Result<Vec<AnthropicContent>, crate::ConversionError> {
    let mut anthropic_contents = Vec::new();

    for part in parts {
        match &part.data {
            PartData::Text(text) => {
                // Check if this is a thinking part based on the thought field
                if part.thought == Some(true) {
                    // Create Anthropic thinking content with signature if available
                    let mut thinking = anthropic_ox::message::ThinkingContent::new(text.to_string());

                    // If Gemini has a thoughtSignature, use it for Anthropic's signature
                    if let Some(ref signature) = part.thought_signature {
                        thinking.signature = Some(signature.clone());
                    }

                    anthropic_contents.push(AnthropicContent::Thinking(thinking));
                } else {
                    anthropic_contents.push(AnthropicContent::Text(anthropic_ox::message::Text::new(text.to_string())));
                }
            }
            PartData::InlineData(blob) => {
                anthropic_contents.push(AnthropicContent::Image {
                    source: anthropic_ox::message::ImageSource::Base64 {
                        media_type: blob.mime_type.clone(),
                        data: blob.data.clone(),
                    },
                });
            }
            PartData::FunctionCall(function_call) => {
                anthropic_contents.push(AnthropicContent::ToolUse(anthropic_ox::tool::ToolUse {
                    id: function_call.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    name: function_call.name.clone(),
                    input: function_call.args.clone().unwrap_or_default(),
                    cache_control: None,
                }));
            }
            PartData::FunctionResponse(func_response) => {
                // Convert Gemini function response back to Anthropic ToolResult
                let tool_use_id = func_response.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

                // Parse the response JSON to extract content
                let content = match &func_response.response {
                    serde_json::Value::String(text) => {
                        vec![anthropic_ox::tool::ToolResultContent::Text { text: text.clone() }]
                    }
                    serde_json::Value::Object(obj) => {
                        // Check if it's our structured format
                        if let (Some(type_val), Some(text_val)) = (obj.get("type"), obj.get("text")) {
                            if type_val == "text" {
                                if let Some(text) = text_val.as_str() {
                                    vec![anthropic_ox::tool::ToolResultContent::Text { text: text.to_string() }]
                                } else {
                                    vec![anthropic_ox::tool::ToolResultContent::Text {
                                        text: text_val.to_string()
                                    }]
                                }
                            } else if type_val == "image" {
                                // Handle image content
                                if let (Some(media_type), Some(data)) = (obj.get("media_type"), obj.get("data")) {
                                    if let (Some(mt), Some(d)) = (media_type.as_str(), data.as_str()) {
                                        vec![anthropic_ox::tool::ToolResultContent::Image {
                                            source: anthropic_ox::message::ImageSource::Base64 {
                                                media_type: mt.to_string(),
                                                data: d.to_string(),
                                            }
                                        }]
                                    } else {
                                        vec![anthropic_ox::tool::ToolResultContent::Text {
                                            text: serde_json::to_string(&func_response.response).unwrap_or_default()
                                        }]
                                    }
                                } else {
                                    vec![anthropic_ox::tool::ToolResultContent::Text {
                                        text: serde_json::to_string(&func_response.response).unwrap_or_default()
                                    }]
                                }
                            } else {
                                vec![anthropic_ox::tool::ToolResultContent::Text {
                                    text: serde_json::to_string(&func_response.response).unwrap_or_default()
                                }]
                            }
                        } else {
                            // Legacy format or complex objects
                            if let Some(text_value) = obj.get("text") {
                                if let Some(text) = text_value.as_str() {
                                    vec![anthropic_ox::tool::ToolResultContent::Text { text: text.to_string() }]
                                } else {
                                    vec![anthropic_ox::tool::ToolResultContent::Text {
                                        text: serde_json::to_string(&func_response.response).unwrap_or_default()
                                    }]
                                }
                            } else {
                                vec![anthropic_ox::tool::ToolResultContent::Text {
                                    text: serde_json::to_string(&func_response.response).unwrap_or_default()
                                }]
                            }
                        }
                    }
                    serde_json::Value::Array(arr) => {
                        // Handle array of structured content parts
                        let mut contents = Vec::new();
                        for item in arr {
                            if let serde_json::Value::Object(obj) = &item {
                                if let Some(type_val) = obj.get("type") {
                                    if let Some(type_str) = type_val.as_str() {
                                        if type_str == "text" {
                                            if let Some(text_val) = obj.get("text") {
                                                if let Some(text) = text_val.as_str() {
                                                    contents.push(anthropic_ox::tool::ToolResultContent::Text { text: text.to_string() });
                                                }
                                            }
                                        } else if type_str == "image" {
                                            if let (Some(media_type), Some(data)) = (obj.get("media_type"), obj.get("data")) {
                                                if let (Some(mt), Some(d)) = (media_type.as_str(), data.as_str()) {
                                                    contents.push(anthropic_ox::tool::ToolResultContent::Image {
                                                        source: anthropic_ox::message::ImageSource::Base64 {
                                                            media_type: mt.to_string(),
                                                            data: d.to_string(),
                                                        }
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                // Fallback for non-structured array items
                                match &item {
                                    serde_json::Value::String(text) => {
                                        contents.push(anthropic_ox::tool::ToolResultContent::Text { text: text.to_string() });
                                    }
                                    _ => {
                                        contents.push(anthropic_ox::tool::ToolResultContent::Text {
                                            text: serde_json::to_string(&item).unwrap_or_default()
                                        });
                                    }
                                }
                            }
                        }
                        contents
                    }
                    _ => {
                        // For other types, convert to string representation
                        vec![anthropic_ox::tool::ToolResultContent::Text {
                            text: serde_json::to_string(&func_response.response).unwrap_or_default()
                        }]
                    }
                };

                anthropic_contents.push(AnthropicContent::ToolResult(anthropic_ox::tool::ToolResult {
                    tool_use_id,
                    content,
                    is_error: None,
                    cache_control: None,
                }));
            }
            PartData::FileData(file_data) => {
                return Err(crate::ConversionError::UnsupportedConversion(format!(
                    "Cannot convert Gemini FileData to Anthropic format. File attachments require conversion to a supported format. File URI: {:?}",
                    file_data.file_uri
                )));
            }
            PartData::ExecutableCode(code) => {
                return Err(crate::ConversionError::UnsupportedConversion(format!(
                    "Cannot convert Gemini ExecutableCode to Anthropic format. Code execution is not supported in Anthropic. Language: {}, code length: {} chars",
                    code.language, code.code.len()
                )));
            }
            PartData::CodeExecutionResult(result) => {
                return Err(crate::ConversionError::UnsupportedConversion(format!(
                    "Cannot convert Gemini CodeExecutionResult to Anthropic format. Code execution results are not supported in Anthropic. Outcome: {:?}",
                    result.outcome
                )));
            }

        }
    }

    Ok(anthropic_contents)
}

/// Convert Anthropic Tool to Gemini Tool  
pub fn anthropic_tool_to_gemini_tool(anthropic_tool: AnthropicTool) -> GeminiTool {
    match anthropic_tool {
        AnthropicTool::Custom(custom_tool) => {
            GeminiTool::FunctionDeclarations(vec![FunctionMetadata {
                name: custom_tool.name,
                description: Some(custom_tool.description),
                parameters: draft07_to_openapi3(custom_tool.input_schema),
            }])
        }
        AnthropicTool::Computer(_) => {
            // Computer tool doesn't convert directly to Gemini - would need special handling
            GeminiTool::FunctionDeclarations(vec![FunctionMetadata {
                name: "computer".to_string(),
                description: Some("Computer use tool (not supported in Gemini)".to_string()),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            }])
        }
    }
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
                let custom_tool = anthropic_ox::tool::CustomTool::new(
                    func.name,
                    func.description.unwrap_or_else(|| "Unknown function".to_string()),
                ).with_schema(func.parameters);
                let tool = AnthropicTool::Custom(custom_tool);
                tool
            } else {
                // Fallback for empty function list
                AnthropicTool::Custom(anthropic_ox::tool::CustomTool::new(
                    "unknown".to_string(),
                    "Unknown function".to_string(),
                ))
            }
        }
        // Handle other Gemini tool types by converting them to basic function tools
        GeminiTool::GoogleSearchRetrieval { .. } => {
            AnthropicTool::Custom(anthropic_ox::tool::CustomTool::new(
                "google_search_retrieval".to_string(),
                "Google Search Retrieval tool".to_string(),
            ))
        }
        GeminiTool::CodeExecution { .. } => {
            AnthropicTool::Custom(anthropic_ox::tool::CustomTool::new(
                "code_execution".to_string(),
                "Code execution tool".to_string(),
            ))
        }
        GeminiTool::GoogleSearch(_) => {
            AnthropicTool::Custom(anthropic_ox::tool::CustomTool::new(
                "google_search".to_string(),
                "Google Search tool".to_string(),
            ))
        }
    }
}

/// Convert Gemini GenerateContentRequest to Anthropic ChatRequest
pub fn gemini_to_anthropic_request(gemini_request: GeminiRequest) -> Result<AnthropicRequest, crate::ConversionError> {
    let mut anthropic_messages = Vec::new();
    
    // Convert Gemini contents to Anthropic messages
    for content in gemini_request.contents {
        let role = match content.role {
            GeminiRole::User => AnthropicRole::User,
            GeminiRole::Model => AnthropicRole::Assistant,
        };
        
        let mut anthropic_contents = Vec::new();
        
        // Convert parts to Anthropic content
        for part in content.parts {
            match part.data {
                PartData::Text(text) => {
                    anthropic_contents.push(AnthropicContent::Text(
                        anthropic_ox::message::Text::new(text.to_string())
                    ));
                }
                PartData::InlineData(blob) => {
                    // Convert blob to base64 image content  
                    anthropic_contents.push(AnthropicContent::Image {
                        source: anthropic_ox::message::ImageSource::Base64 {
                            media_type: blob.mime_type.clone(),
                            data: blob.data.clone(),
                        },
                    });
                }
                PartData::FunctionCall(func_call) => {
                    let input = func_call.args.unwrap_or_default();
                    let id = func_call.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                    anthropic_contents.push(AnthropicContent::ToolUse(
                        anthropic_ox::tool::ToolUse {
                            id,
                            name: func_call.name,
                            input,
                            cache_control: None,
                        }
                    ));
                }
                PartData::FunctionResponse(func_response) => {
                 let tool_use_id = func_response.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                    let text_response = match func_response.response {
                        serde_json::Value::String(s) => s,
                        other => serde_json::to_string(&other).unwrap_or_default(),
                    };
                    anthropic_contents.push(AnthropicContent::ToolResult(
                        anthropic_ox::tool::ToolResult {
                            tool_use_id,
                            content: vec![anthropic_ox::tool::ToolResultContent::Text { text: text_response }],
                            is_error: None,
                            cache_control: None,
                        }
                    ));
                }
                PartData::FileData(file_data) => {
                    return Err(crate::ConversionError::UnsupportedConversion(format!(
                        "Cannot convert Gemini FileData to Anthropic format. File attachments require conversion to a supported format. File URI: {:?}",
                        file_data.file_uri
                    )));
                }
                PartData::ExecutableCode(code) => {
                    return Err(crate::ConversionError::UnsupportedConversion(format!(
                        "Cannot convert Gemini ExecutableCode to Anthropic format. Code execution is not supported in Anthropic. Language: {}, code length: {} chars",
                        code.language, code.code.len()
                    )));
                }
                PartData::CodeExecutionResult(result) => {
                    return Err(crate::ConversionError::UnsupportedConversion(format!(
                        "Cannot convert Gemini CodeExecutionResult to Anthropic format. Code execution results are not supported in Anthropic. Outcome: {:?}",
                        result.outcome
                    )));
                }
            }
        }
        
        if !anthropic_contents.is_empty() {
            anthropic_messages.push(anthropic_ox::message::Message {
                role,
                content: anthropic_ox::message::StringOrContents::Contents(anthropic_contents),
            });
        }
    }
    
    // Prepare optional fields
    let system_instruction = if let Some(system_content) = gemini_request.system_instruction {
        if let Some(first_part) = system_content.parts.first() {
            if let PartData::Text(text) = &first_part.data {
                Some(text.to_string())
            } else { None }
        } else { None }
    } else { None };
    
    let anthropic_tools = if let Some(tools) = gemini_request.tools {
        let mut converted_tools = Vec::new();
        for tool_json in tools {
            // Try to deserialize JSON Value to Tool
            match serde_json::from_value::<GeminiTool>(tool_json) {
                Ok(tool) => converted_tools.push(gemini_tool_to_anthropic_tool(tool)),
                Err(e) => {
                    return Err(crate::ConversionError::ContentConversion(format!("Failed to deserialize tool: {:?}", e)));
                }
            }
        }
        if converted_tools.is_empty() { None } else { Some(converted_tools) }
    } else { None };
    
    // Build request using chained maybe_ methods for all optional fields
    let request = AnthropicRequest::builder()
        .model(gemini_request.model)
        .messages(anthropic_ox::message::Messages(anthropic_messages))
        .maybe_system(system_instruction.map(anthropic_ox::message::StringOrContents::String))
        .maybe_max_tokens(
            gemini_request.generation_config
                .as_ref()
                .and_then(|c| c.max_output_tokens)
                .map(|t| t as u32)
        )
        .maybe_temperature(
            gemini_request.generation_config
                .as_ref()
                .and_then(|c| c.temperature)
                .map(|t| t as f32)
        )
        .maybe_top_p(
            gemini_request.generation_config
                .as_ref()
                .and_then(|c| c.top_p)
                .map(|tp| tp as f32)
        )
        .maybe_top_k(
            gemini_request.generation_config
                .as_ref()
                .and_then(|c| c.top_k)
                .map(|tk| tk as i32)
        )
        .maybe_thinking(
            gemini_request.generation_config
                .as_ref()
                .and_then(|c| c.thinking_config.as_ref())
                .map(|tc| {
                    let budget = if tc.thinking_budget < 0 { 
                        u32::MAX // Use max for dynamic budget
                    } else { 
                        tc.thinking_budget as u32 
                    };
                    anthropic_ox::request::ThinkingConfig::new(budget)
                })
        )
        .maybe_tools(anthropic_tools)
        .build();
    
    Ok(request)
}

/// Convert Anthropic ChatResponse to Gemini GenerateContentResponse
pub fn anthropic_to_gemini_response(anthropic_response: AnthropicResponse) -> Result<GeminiResponse, crate::ConversionError> {
    use gemini_ox::generate_content::{
        ResponseCandidate as Candidate, 
        FinishReason,
        usage::UsageMetadata,
    };
    use gemini_ox::content::{Content as GeminiContent, Part as GeminiPart, Role as GeminiRole};
    
    let mut gemini_parts = Vec::new();
    
    // Convert Anthropic content to Gemini parts - content is now directly Vec<Content>
    for content in anthropic_response.content {
        match content {
            AnthropicContent::Text(text) => {
                gemini_parts.push(GeminiPart {
                    data: PartData::Text(GeminiText::from(text.text)),
                    thought: None,
                    thought_signature: None,
                    video_metadata: None,
                });
            }
            AnthropicContent::Thinking(thinking) => {
                gemini_parts.push(GeminiPart {
                    data: PartData::Text(GeminiText::from(thinking.text)),
                    thought: Some(true),
                    thought_signature: thinking.signature,
                    video_metadata: None,
                });
            }
            AnthropicContent::ToolUse(tool_use) => {
                gemini_parts.push(GeminiPart {
                    data: PartData::FunctionCall(gemini_ox::content::FunctionCall {
                        id: Some(tool_use.id),
                        name: tool_use.name,
                        args: Some(tool_use.input),
                    }),
                    thought: None,
                    thought_signature: None,
                    video_metadata: None,
                });
            }
            AnthropicContent::ToolResult(tool_result) => {
                // Convert tool result content to JSON, preserving all content types
                let mut content_parts = Vec::new();

                for content in &tool_result.content {
                    match content {
                        anthropic_ox::tool::ToolResultContent::Text { text } => {
                            content_parts.push(serde_json::json!({"type": "text", "text": text}));
                        }
                        anthropic_ox::tool::ToolResultContent::Image { source } => {
                            match source {
                                anthropic_ox::message::ImageSource::Base64 { media_type, data } => {
                                    content_parts.push(serde_json::json!({
                                        "type": "image",
                                        "media_type": media_type,
                                        "data": data
                                    }));
                                }
                            }
                        }
                    }
                }

                let response = if content_parts.len() == 1 {
                    // Single content part - return it directly
                    content_parts.into_iter().next().unwrap()
                } else {
                    // Multiple content parts - return as array
                    serde_json::Value::Array(content_parts)
                };
                gemini_parts.push(GeminiPart {
                    data: PartData::FunctionResponse(gemini_ox::content::FunctionResponse {
                        id: Some(tool_result.tool_use_id),
                        name: "function_tool".to_string(),
                        response,
                        will_continue: None,
                        scheduling: None,
                    }),
                    thought: None,
                    thought_signature: None,
                    video_metadata: None,
                });
            }
            // Handle unsupported content types explicitly
            AnthropicContent::Image { .. } => {
                return Err(crate::ConversionError::UnsupportedConversion(
                    "Cannot convert Anthropic Image content to Gemini format in response context".to_string()
                ));
            }
            AnthropicContent::SearchResult(_) => {
                return Err(crate::ConversionError::UnsupportedConversion(
                    "Cannot convert Anthropic SearchResult to Gemini format".to_string()
                ));
            }
        }
    }
    
    let candidate = Candidate {
        content: GeminiContent {
            role: GeminiRole::Model,
            parts: gemini_parts,
        },
        finish_reason: Some(match anthropic_response.stop_reason {
            Some(anthropic_ox::response::StopReason::EndTurn) => FinishReason::Stop,
            Some(anthropic_ox::response::StopReason::MaxTokens) => FinishReason::MaxTokens,
            Some(anthropic_ox::response::StopReason::StopSequence) => FinishReason::Stop,
            Some(anthropic_ox::response::StopReason::ToolUse) => FinishReason::Stop,
            None => FinishReason::Stop,
        }),
        index: Some(0),
        safety_ratings: Vec::new(), // Could be enhanced to convert safety info
        citation_metadata: None,
        token_count: None,
        grounding_attributions: None,
        avg_logprobs: None,
        logprobs_result: None,
        grounding_metadata: None,
    };
    
    Ok(GeminiResponse {
        candidates: vec![candidate],
        prompt_feedback: None,
        usage_metadata: Some(UsageMetadata {
            prompt_token_count: anthropic_response.usage.input_tokens.unwrap_or_default(),
            candidates_token_count: anthropic_response.usage.output_tokens,
            total_token_count: anthropic_response.usage.input_tokens.unwrap_or_default() + anthropic_response.usage.output_tokens.unwrap_or_default(),
            cached_content_token_count: None,
            thoughts_token_count: anthropic_response.usage.thinking_tokens,
            cache_tokens_details: None,
            candidates_tokens_details: None,
            prompt_tokens_details: None,
            tool_use_prompt_tokens_details: None,
            tool_use_prompt_token_count: None,
        }),
        model_version: Some(anthropic_response.model),
    })
}