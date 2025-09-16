use std::collections::{BTreeMap, HashMap};

use anthropic_ox::{
    message::{
        Content as AnthropicContent, ImageSource as AnthropicImageSource,
        Message as AnthropicMessage, Messages as AnthropicMessages, Role as AnthropicRole,
        StringOrContents, Text as AnthropicText,
    },
    request::ChatRequest as AnthropicRequest,
    tool::{Tool as AnthropicTool, ToolResult as AnthropicToolResult, ToolResultContent, ToolUse},
};

use crate::{
    content::{
        message::{Message, MessageRole},
        part::{DataRef, Part},
    },
    errors::GenerateContentError,
    model::request::ModelRequest,
    tool::{FunctionMetadata, Tool},
};

fn anthropic_content_to_part(
    content: &AnthropicContent,
    tool_name_lookup: &HashMap<String, String>,
) -> Result<Part, GenerateContentError> {
    Ok(match content {
        AnthropicContent::Text(text) => Part::Text {
            text: text.text.clone(),
            ext: BTreeMap::new(),
        },
        AnthropicContent::Image { source } => match source {
            AnthropicImageSource::Base64 { media_type, data } => Part::Blob {
                data_ref: DataRef::Base64 { data: data.clone() },
                mime_type: media_type.clone(),
                name: None,
                description: None,
                ext: BTreeMap::new(),
            },
        },
        AnthropicContent::ToolUse(tool_use) => Part::ToolUse {
            id: tool_use.id.clone(),
            name: tool_use.name.clone(),
            args: tool_use.input.clone(),
            ext: BTreeMap::new(),
        },
        AnthropicContent::ToolResult(tool_result) => {
            let mut ext = BTreeMap::new();
            if let Some(is_error) = tool_result.is_error {
                ext.insert(
                    "anthropic.is_error".to_string(),
                    serde_json::Value::Bool(is_error),
                );
            }
            let name = tool_name_lookup
                .get(&tool_result.tool_use_id)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            let parts = tool_result
                .content
                .iter()
                .map(|item| match item {
                    ToolResultContent::Text { text } => Ok(Part::Text {
                        text: text.clone(),
                        ext: BTreeMap::new(),
                    }),
                    ToolResultContent::Image { source } => match source {
                        AnthropicImageSource::Base64 { media_type, data } => Ok(Part::Blob {
                            data_ref: DataRef::Base64 { data: data.clone() },
                            mime_type: media_type.clone(),
                            name: None,
                            description: None,
                            ext: BTreeMap::new(),
                        }),
                    },
                })
                .collect::<Result<Vec<_>, GenerateContentError>>()?;
            Part::ToolResult {
                id: tool_result.tool_use_id.clone(),
                name,
                parts,
                ext,
            }
        }
        AnthropicContent::Thinking(_) | AnthropicContent::SearchResult(_) => {
            return Err(GenerateContentError::unsupported_feature(
                "Unsupported Anthropic content type for ai-ox request conversion",
            ));
        }
    })
}

fn convert_anthropic_tools(tools: &[AnthropicTool]) -> Vec<Tool> {
    if tools.is_empty() {
        return Vec::new();
    }

    let mut function_declarations = Vec::new();
    for tool in tools {
        if let AnthropicTool::Custom(custom) = tool {
            function_declarations.push(FunctionMetadata {
                name: custom.name.clone(),
                description: Some(custom.description.clone()),
                parameters: custom.input_schema.clone(),
            });
        }
    }

    if function_declarations.is_empty() {
        Vec::new()
    } else {
        vec![Tool::FunctionDeclarations(function_declarations)]
    }
}

fn convert_model_tools(tools: Option<&[Tool]>) -> Option<Vec<AnthropicTool>> {
    tools.map(|entries| {
        let mut anthropic_tools = Vec::new();
        for tool in entries {
            match tool {
                Tool::FunctionDeclarations(functions) => {
                    for func in functions {
                        let mut custom = anthropic_ox::tool::CustomTool::new(
                            func.name.clone(),
                            func.description.clone().unwrap_or_default(),
                        );
                        custom = custom.with_schema(func.parameters.clone());
                        anthropic_tools.push(AnthropicTool::Custom(custom));
                    }
                }
                #[cfg(feature = "gemini")]
                Tool::GeminiTool(_) => {
                    // Gemini-specific tools cannot be expressed in Anthropic
                }
            }
        }
        anthropic_tools
    })
}

fn convert_string_or_contents_to_message(
    system: &StringOrContents,
) -> Result<Message, GenerateContentError> {
    let mut parts = Vec::new();
    match system {
        StringOrContents::String(text) => {
            parts.push(Part::Text {
                text: text.clone(),
                ext: BTreeMap::new(),
            });
        }
        StringOrContents::Contents(contents) => {
            // For now only support text content
            for content in contents {
                if let AnthropicContent::Text(text) = content {
                    parts.push(Part::Text {
                        text: text.text.clone(),
                        ext: BTreeMap::new(),
                    });
                } else {
                    return Err(GenerateContentError::unsupported_feature(
                        "Non-text system content not supported in conversion",
                    ));
                }
            }
        }
    }
    Ok(Message::new(MessageRole::System, parts))
}

/// Convert an Anthropic ChatRequest into an ai-ox ModelRequest, preserving tool
/// metadata required for faithful roundtrips.
pub fn anthropic_request_to_model_request(
    request: &AnthropicRequest,
) -> Result<ModelRequest, GenerateContentError> {
    let mut tool_name_lookup = HashMap::new();
    for message in &request.messages.0 {
        if let StringOrContents::Contents(contents) = &message.content {
            for content in contents {
                if let AnthropicContent::ToolUse(tool_use) = content {
                    tool_name_lookup.insert(tool_use.id.clone(), tool_use.name.clone());
                }
            }
        }
    }

    let messages: Vec<Message> = request
        .messages
        .0
        .iter()
        .map(|message| {
            let role = match message.role {
                AnthropicRole::User => MessageRole::User,
                AnthropicRole::Assistant => MessageRole::Assistant,
            };

            let parts = match &message.content {
                StringOrContents::String(text) => vec![Part::Text {
                    text: text.clone(),
                    ext: BTreeMap::new(),
                }],
                StringOrContents::Contents(contents) => contents
                    .iter()
                    .map(|content| anthropic_content_to_part(content, &tool_name_lookup))
                    .collect::<Result<Vec<_>, _>>()?,
            };

            Ok(Message::new(role, parts))
        })
        .collect::<Result<_, GenerateContentError>>()?;

    let system_message = if let Some(system) = &request.system {
        Some(convert_string_or_contents_to_message(system)?)
    } else {
        None
    };

    let tools = request
        .tools
        .as_ref()
        .map(|tools| convert_anthropic_tools(tools))
        .filter(|tools| !tools.is_empty());

    Ok(ModelRequest {
        messages,
        tools,
        system_message,
    })
}

fn convert_part_to_anthropic_content(
    part: &Part,
) -> Result<AnthropicContent, GenerateContentError> {
    Ok(match part {
        Part::Text { text, .. } => AnthropicContent::Text(AnthropicText::new(text.clone())),
        Part::Blob {
            data_ref: DataRef::Base64 { data },
            mime_type,
            ..
        } => AnthropicContent::Image {
            source: AnthropicImageSource::Base64 {
                media_type: mime_type.clone(),
                data: data.clone(),
            },
        },
        Part::Blob {
            data_ref: DataRef::Uri { .. },
            ..
        } => {
            return Err(GenerateContentError::unsupported_feature(
                "URI blobs cannot be converted to Anthropic format",
            ));
        }
        Part::ToolUse { id, name, args, .. } => AnthropicContent::ToolUse(ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: args.clone(),
            cache_control: None,
        }),
        Part::ToolResult { id, parts, ext, .. } => {
            let is_error = ext
                .get("anthropic.is_error")
                .and_then(|value| value.as_bool());
            let content = parts
                .iter()
                .map(|part| match part {
                    Part::Text { text, .. } => Ok(ToolResultContent::Text { text: text.clone() }),
                    Part::Blob {
                        data_ref: DataRef::Base64 { data },
                        mime_type,
                        ..
                    } => Ok(ToolResultContent::Image {
                        source: AnthropicImageSource::Base64 {
                            media_type: mime_type.clone(),
                            data: data.clone(),
                        },
                    }),
                    _ => Err(GenerateContentError::unsupported_feature(
                        "Unsupported nested part in tool result conversion",
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;

            AnthropicContent::ToolResult(AnthropicToolResult {
                tool_use_id: id.clone(),
                content,
                is_error,
                cache_control: None,
            })
        }
        Part::Opaque { .. } => {
            return Err(GenerateContentError::unsupported_feature(
                "Opaque parts cannot be converted to Anthropic content",
            ));
        }
    })
}

/// Convert an ai-ox ModelRequest back into an Anthropic request, cloning the
/// provided template for metadata (model name, sampling settings, etc.).
pub fn model_request_to_anthropic_request(
    request: &ModelRequest,
    template: &AnthropicRequest,
) -> Result<AnthropicRequest, GenerateContentError> {
    let mut converted_messages: Vec<(AnthropicRole, Vec<AnthropicContent>)> = Vec::new();
    for message in &request.messages {
        if matches!(message.role, MessageRole::System) {
            continue; // handled separately
        }

        let role = match message.role {
            MessageRole::Assistant => AnthropicRole::Assistant,
            _ => AnthropicRole::User,
        };

        let mut contents = Vec::new();
        for part in &message.content {
            contents.push(convert_part_to_anthropic_content(part)?);
        }
        converted_messages.push((role, contents));
    }

    let mut anthropic_messages = AnthropicMessages::new();
    for (index, (role, contents)) in converted_messages.into_iter().enumerate() {
        let desired_variant = template
            .messages
            .0
            .get(index)
            .map(|message| &message.content);

        let content_variant = match desired_variant {
            Some(StringOrContents::String(_)) => {
                if contents.len() == 1 {
                    if let AnthropicContent::Text(text) = &contents[0] {
                        StringOrContents::String(text.text.clone())
                    } else {
                        StringOrContents::Contents(contents.clone())
                    }
                } else {
                    StringOrContents::Contents(contents.clone())
                }
            }
            _ => StringOrContents::Contents(contents.clone()),
        };

        anthropic_messages.add_message(AnthropicMessage {
            role,
            content: content_variant,
        });
    }

    let mut output = template.clone();
    output.messages = anthropic_messages;

    if let Some(system_message) = &request.system_message {
        let mut text_parts = Vec::new();
        for part in &system_message.content {
            if let Part::Text { text, .. } = part {
                text_parts.push(text.clone());
            } else {
                return Err(GenerateContentError::unsupported_feature(
                    "Non-text system parts are not supported when converting to Anthropic",
                ));
            }
        }

        if text_parts.len() == 1 {
            output.system = Some(StringOrContents::String(text_parts.remove(0)));
        } else if !text_parts.is_empty() {
            let contents = text_parts
                .into_iter()
                .map(|text| AnthropicContent::Text(AnthropicText::new(text)))
                .collect();
            output.system = Some(StringOrContents::Contents(contents));
        } else {
            output.system = None;
        }
    }

    if let Some(tools) = request.tools.as_ref() {
        let anthropic_tools = convert_model_tools(Some(tools));
        if let Some(tools) = anthropic_tools {
            output.tools = Some(tools);
        }
    }

    Ok(output)
}
