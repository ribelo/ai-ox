use std::collections::BTreeMap;

use ai_ox_common::openai_format::{
    Function as OpenAIFunction, FunctionCall as OpenAIFunctionCall, Message as OpenAIMessage,
    MessageRole as OpenAIRole, Tool as OpenAITool, ToolCall as OpenAIToolCall,
};
use openai_ox::request::ChatRequest as OpenAIChatRequest;
use serde_json::Value;

use crate::{
    content::{
        message::{Message, MessageRole},
        part::Part,
    },
    errors::GenerateContentError,
    model::request::ModelRequest,
    tool::{FunctionMetadata, Tool},
};

pub fn model_request_to_openai_chat_request(
    request: &ModelRequest,
    model: impl Into<String>,
) -> Result<OpenAIChatRequest, GenerateContentError> {
    let mut messages = Vec::new();

    if let Some(system_message) = &request.system_message {
        messages.push(OpenAIMessage {
            role: OpenAIRole::System,
            content: Some(collect_text_content(system_message)?),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    for message in &request.messages {
        let (role, base_message, mut tool_messages) = convert_message_to_openai(message)?;
        if let Some(base) = base_message {
            messages.push(base);
        } else if role == OpenAIRole::Assistant {
            messages.push(OpenAIMessage {
                role,
                content: None,
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
        messages.append(&mut tool_messages);
    }

    let mut openai_request = OpenAIChatRequest::builder()
        .model(model.into())
        .messages(messages)
        .build();

    if let Some(tools) = &request.tools {
        let converted = convert_tools_to_openai(tools);
        if !converted.is_empty() {
            openai_request.tools = Some(converted);
        }
    }

    Ok(openai_request)
}

pub fn openai_chat_request_to_model_request(
    request: &OpenAIChatRequest,
) -> Result<ModelRequest, GenerateContentError> {
    let mut messages = Vec::new();
    let mut system_message = None;
    let mut tool_call_names = std::collections::HashMap::new();

    for message in &request.messages {
        match message.role {
            OpenAIRole::System => {
                if let Some(content) = &message.content {
                    system_message = Some(Message::new(
                        MessageRole::System,
                        vec![Part::Text {
                            text: content.clone(),
                            ext: BTreeMap::new(),
                        }],
                    ));
                }
            }
            OpenAIRole::User => {
                messages.push(Message::new(
                    MessageRole::User,
                    vec![Part::Text {
                        text: message.content.clone().unwrap_or_default(),
                        ext: BTreeMap::new(),
                    }],
                ));
            }
            OpenAIRole::Assistant => {
                let mut parts = Vec::new();
                if let Some(content) = &message.content {
                    if !content.is_empty() {
                        parts.push(Part::Text {
                            text: content.clone(),
                            ext: BTreeMap::new(),
                        });
                    }
                }
                if let Some(tool_calls) = &message.tool_calls {
                    for call in tool_calls {
                        let args =
                            serde_json::from_str(&call.function.arguments).map_err(|err| {
                                GenerateContentError::message_conversion(&format!(
                                    "Failed to parse tool call arguments: {}",
                                    err
                                ))
                            })?;
                        tool_call_names.insert(call.id.clone(), call.function.name.clone());
                        parts.push(Part::ToolUse {
                            id: call.id.clone(),
                            name: call.function.name.clone(),
                            args,
                            ext: BTreeMap::new(),
                        });
                    }
                }
                messages.push(Message::new(MessageRole::Assistant, parts));
            }
            OpenAIRole::Tool => {
                let (name, parts, ext) =
                    decode_tool_result_content(message.content.as_deref().unwrap_or(
                        "{\"ai_ox_tool_result\": {\"name\": \"unknown\", \"content\": []}}",
                    ))?;
                if let Some(tool_call_id) = &message.tool_call_id {
                    let resolved_name = tool_call_names.get(tool_call_id).cloned().unwrap_or(name);
                    messages.push(Message::new(
                        MessageRole::Assistant,
                        vec![Part::ToolResult {
                            id: tool_call_id.clone(),
                            name: resolved_name,
                            parts,
                            ext,
                        }],
                    ));
                }
            }
        }
    }

    let tool_definitions = request
        .tools
        .as_ref()
        .map(|tools| convert_openai_tools(tools))
        .filter(|tools| !tools.is_empty());

    Ok(ModelRequest {
        messages,
        tools: tool_definitions,
        system_message,
    })
}

fn collect_text_content(message: &Message) -> Result<String, GenerateContentError> {
    let mut texts = Vec::new();
    for part in &message.content {
        if let Part::Text { text, .. } = part {
            texts.push(text.clone());
        } else {
            return Err(GenerateContentError::unsupported_feature(
                "Non-text system parts cannot be converted to OpenAI format",
            ));
        }
    }
    Ok(texts.join(" \n"))
}

fn convert_message_to_openai(
    message: &Message,
) -> Result<(OpenAIRole, Option<OpenAIMessage>, Vec<OpenAIMessage>), GenerateContentError> {
    let role = match message.role {
        MessageRole::User | MessageRole::Unknown(_) => OpenAIRole::User,
        MessageRole::Assistant => OpenAIRole::Assistant,
        MessageRole::System => OpenAIRole::System,
    };

    let mut text_segments = Vec::new();
    let mut tool_calls = Vec::new();
    let mut tool_messages = Vec::new();

    for part in &message.content {
        match part {
            Part::Text { text, .. } => text_segments.push(text.clone()),
            Part::ToolUse { id, name, args, .. } => {
                let arguments = serde_json::to_string(args).map_err(|err| {
                    GenerateContentError::message_conversion(&format!(
                        "Failed to serialize tool call arguments: {}",
                        err
                    ))
                })?;
                tool_calls.push(OpenAIToolCall {
                    id: id.clone(),
                    r#type: "function".to_string(),
                    function: OpenAIFunctionCall {
                        name: name.clone(),
                        arguments,
                    },
                });
            }
            Part::ToolResult {
                id,
                name,
                parts,
                ext,
            } => {
                let content = encode_tool_result_payload(name, parts, ext)?;
                tool_messages.push(OpenAIMessage {
                    role: OpenAIRole::Tool,
                    content: Some(content),
                    name: Some(name.clone()),
                    tool_calls: None,
                    tool_call_id: Some(id.clone()),
                });
            }
            Part::Blob { .. } | Part::Opaque { .. } => {
                return Err(GenerateContentError::unsupported_feature(
                    "Unsupported part type for OpenAI request conversion",
                ));
            }
        }
    }

    let base_message = if text_segments.is_empty() && tool_calls.is_empty() {
        None
    } else {
        Some(OpenAIMessage {
            role: role.clone(),
            content: if text_segments.is_empty() {
                None
            } else {
                Some(text_segments.join(" \n"))
            },
            name: None,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
        })
    };

    Ok((role, base_message, tool_messages))
}

fn encode_tool_result_payload(
    name: &str,
    parts: &[Part],
    ext: &BTreeMap<String, Value>,
) -> Result<String, GenerateContentError> {
    let mut payload = serde_json::json!({
        "ai_ox_tool_result": {
            "name": name,
            "content": parts,
        }
    });

    if !ext.is_empty() {
        if let Some(obj) = payload
            .get_mut("ai_ox_tool_result")
            .and_then(|value| value.as_object_mut())
        {
            obj.insert("ext".to_string(), serde_json::to_value(ext).unwrap());
        }
    }

    Ok(payload.to_string())
}

fn decode_tool_result_content(
    content: &str,
) -> Result<(String, Vec<Part>, BTreeMap<String, Value>), GenerateContentError> {
    let payload: serde_json::Value = serde_json::from_str(content).map_err(|err| {
        GenerateContentError::message_conversion(&format!(
            "Failed to decode OpenAI tool result content: {}",
            err
        ))
    })?;

    let object = payload
        .get("ai_ox_tool_result")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            GenerateContentError::message_conversion("Missing ai_ox_tool_result in tool content")
        })?;

    let name = object
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| GenerateContentError::message_conversion("Tool result missing name"))?
        .to_string();

    let content_value = object
        .get("content")
        .ok_or_else(|| GenerateContentError::message_conversion("Tool result missing content"))?;
    let parts: Vec<Part> = serde_json::from_value(content_value.clone()).map_err(|err| {
        GenerateContentError::message_conversion(&format!(
            "Failed to deserialize tool result parts: {}",
            err
        ))
    })?;

    let ext = object
        .get("ext")
        .cloned()
        .map(|value| serde_json::from_value(value).unwrap_or_default())
        .unwrap_or_default();

    Ok((name, parts, ext))
}

fn convert_tools_to_openai(tools: &[Tool]) -> Vec<OpenAITool> {
    let mut converted = Vec::new();
    for tool in tools {
        if let Tool::FunctionDeclarations(functions) = tool {
            for func in functions {
                converted.push(OpenAITool {
                    r#type: "function".to_string(),
                    function: OpenAIFunction {
                        name: func.name.clone(),
                        description: func.description.clone(),
                        parameters: Some(func.parameters.clone()),
                    },
                });
            }
        }
    }
    converted
}

fn convert_openai_tools(tools: &[OpenAITool]) -> Vec<Tool> {
    if tools.is_empty() {
        return Vec::new();
    }

    let functions = tools
        .iter()
        .map(|tool| FunctionMetadata {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            parameters: tool
                .function
                .parameters
                .clone()
                .unwrap_or_else(|| serde_json::json!({})),
        })
        .collect();

    vec![Tool::FunctionDeclarations(functions)]
}
