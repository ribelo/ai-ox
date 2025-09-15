use std::convert::TryFrom;

use crate::{
    content::{
        delta::{FinishReason, StreamEvent, StreamStop},
        message::{Message, MessageRole},
        part::Part,
    },
    errors::GenerateContentError,
    model::{ModelRequest, response::ModelResponse},
    tool::{Tool, encode_tool_result_parts, decode_tool_result_parts},
};
use gemini_ox::{
    content::{Content as GeminiContent, Part as GeminiPart, Role as GeminiRole},
    generate_content::{
        request::GenerateContentRequest as GeminiGenerateContentRequest, GenerationConfig,
        response::GenerateContentResponse,
        SafetySettings,
    },
    tool::{config::ToolConfig},
};



impl From<MessageRole> for GeminiRole {
    fn from(role: MessageRole) -> Self {
        match role {
            MessageRole::User => Self::User,
            MessageRole::Assistant => Self::Model,
            MessageRole::System => Self::User, // Map System to User since Gemini doesn't have System role
            MessageRole::Unknown(_) => {
                // Map unknown roles to User as default
                Self::User
            }
        }
    }
}

impl TryFrom<Message> for GeminiContent {
    type Error = GenerateContentError;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        let role = message.role.into();
        let parts = message
            .content
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<GeminiPart>, _>>()?;
        Ok(Self { role, parts })
    }
}

impl TryFrom<Part> for GeminiPart {
    type Error = GenerateContentError;

    fn try_from(part: Part) -> Result<Self, Self::Error> {
        match part {
            Part::Text { text, .. } => Ok(Self::new(gemini_ox::content::PartData::Text(text.into()))),
            Part::ToolUse { id, name, args, .. } => Ok(Self::new(
                gemini_ox::content::PartData::FunctionCall(gemini_ox::content::FunctionCall {
                    id: Some(id),
                    name,
                    args: Some(args),
                }),
            )),
            Part::ToolResult { id, name, parts, .. } => {
                let encoded_content = encode_tool_result_parts(&name, &parts)?;
                Ok(Self::new(
                    gemini_ox::content::PartData::FunctionResponse(gemini_ox::content::FunctionResponse {
                        id: Some(id),
                        name,
                        response: serde_json::from_str(&encoded_content).unwrap_or(serde_json::Value::Null),
                        will_continue: None,
                        scheduling: None,
                    }),
                ))
            }
            Part::Blob { data_ref, mime_type, .. } => match data_ref {
                crate::content::part::DataRef::Uri { uri } => Ok(Self::new(
                    gemini_ox::content::PartData::FileData(gemini_ox::content::FileData {
                        mime_type,
                        file_uri: uri,
                        display_name: None,
                    }),
                )),
                crate::content::part::DataRef::Base64 { data } => Ok(Self::new(
                    gemini_ox::content::PartData::InlineData(gemini_ox::content::Blob {
                        mime_type,
                        data,
                        display_name: None,
                    }),
                )),
            },
            Part::Opaque { provider, .. } => Err(GenerateContentError::unsupported_feature(
                &format!("Opaque parts not supported by Gemini provider. Provider: {}", provider),
            )),
            _ => Err(GenerateContentError::unsupported_feature(
                "Unsupported part type for Gemini models.",
            )),
        }
    }
}

impl TryFrom<GeminiContent> for Message {
    type Error = GenerateContentError;

    fn try_from(content: GeminiContent) -> Result<Self, Self::Error> {
        let role = match content.role {
            GeminiRole::User => MessageRole::User,
            GeminiRole::Model => MessageRole::Assistant,
        };
        let content = content
            .parts
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<Part>, _>>()?;
        Ok(Self {
            role,
            content,
            timestamp: None,
            ext: Some(std::collections::BTreeMap::new()),
        })
    }
}

impl TryFrom<GeminiPart> for Part {
    type Error = GenerateContentError;

    fn try_from(part: GeminiPart) -> Result<Self, Self::Error> {
        match part.data {
            gemini_ox::content::PartData::Text(text) => Ok(Part::Text { text: text.to_string(), ext: std::collections::BTreeMap::new() }),
            gemini_ox::content::PartData::FunctionCall(function_call) => Ok(Part::ToolUse {
                id: function_call.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                name: function_call.name,
                args: function_call.args.unwrap_or_default(),
                ext: std::collections::BTreeMap::new(),
            }),
            gemini_ox::content::PartData::FunctionResponse(function_response) => {
                let encoded_response = serde_json::to_string(&function_response.response)?;
                let (decoded_name, parts) = decode_tool_result_parts(&encoded_response)?;
                // Verify the decoded name matches the expected function name
                if decoded_name != function_response.name {
                    return Err(GenerateContentError::message_conversion(
                        &format!("Function name mismatch: expected '{}', got '{}'", function_response.name, decoded_name)
                    ));
                }
                Ok(Part::ToolResult {
                    id: function_response.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    name: function_response.name,
                    parts,
                    ext: std::collections::BTreeMap::new(),
                })
            }
            gemini_ox::content::PartData::InlineData(blob) => Ok(Part::Blob {
                data_ref: crate::content::part::DataRef::Base64 { data: blob.data },
                mime_type: blob.mime_type,
                name: blob.display_name,
                description: None,
                ext: std::collections::BTreeMap::new(),
            }),
            gemini_ox::content::PartData::FileData(file_data) => Ok(Part::Blob {
                data_ref: crate::content::part::DataRef::Uri { uri: file_data.file_uri },
                mime_type: file_data.mime_type,
                name: file_data.display_name,
                description: None,
                ext: std::collections::BTreeMap::new(),
            }),
            _ => Err(GenerateContentError::unsupported_feature(
                "Unsupported Gemini part type.",
            )),
        }
    }
}



impl From<Tool> for serde_json::Value {
    fn from(tool: Tool) -> Self {
        serde_json::to_value(tool).unwrap()
    }
}

impl From<crate::tool::FunctionMetadata> for gemini_ox::tool::FunctionMetadata {
    fn from(metadata: crate::tool::FunctionMetadata) -> Self {
        Self {
            name: metadata.name,
            description: metadata.description,
            parameters: metadata.parameters,
        }
    }
}

pub(super) fn convert_response_to_stream_events(
    response: GenerateContentResponse,
) -> Vec<Result<StreamEvent, GenerateContentError>> {
    let mut events = Vec::new();

    if let Some(candidate) = response.candidates.first() {
        for part in &candidate.content.parts {
            if let Ok(event) = StreamEvent::try_from(part.clone()) {
                events.push(Ok(event));
            }
        }
    }

    if let Some(usage_metadata) = &response.usage_metadata {
        events.push(Ok(StreamEvent::Usage(usage_metadata.clone().into())));

        let finish_reason = response
            .candidates
            .first()
            .and_then(|c| c.finish_reason.as_ref())
            .map(|fr| fr.into())
            .unwrap_or(FinishReason::Stop);

        events.push(Ok(StreamEvent::StreamStop(StreamStop {
            finish_reason,
            usage: usage_metadata.clone().into(),
        })));
    }

    events
}

impl TryFrom<GeminiPart> for StreamEvent {
    type Error = GenerateContentError;

    fn try_from(part: GeminiPart) -> Result<Self, Self::Error> {
        match part.data {
            gemini_ox::content::PartData::Text(text) => Ok(StreamEvent::TextDelta(text.to_string())),
            gemini_ox::content::PartData::FunctionCall(function_call) => {
                Ok(StreamEvent::ToolCall(function_call.into()))
            }
            gemini_ox::content::PartData::FunctionResponse(function_response) => {
                let encoded_response = serde_json::to_string(&function_response.response)?;
                let (decoded_name, parts) = decode_tool_result_parts(&encoded_response)?;
                // Verify the decoded name matches the expected function name
                if decoded_name != function_response.name {
                    return Err(GenerateContentError::message_conversion(
                        &format!("Function name mismatch: expected '{}', got '{}'", function_response.name, decoded_name)
                    ));
                }
                Ok(StreamEvent::ToolResult(Part::ToolResult {
                    id: function_response.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    name: function_response.name,
                    parts,
                    ext: std::collections::BTreeMap::new(),
                }))
            }
            gemini_ox::content::PartData::InlineData(blob) => {
                Ok(StreamEvent::ToolResult(Part::Blob {
                    data_ref: crate::content::part::DataRef::Base64 { data: blob.data },
                    mime_type: blob.mime_type,
                    name: blob.display_name,
                    description: None,
                    ext: std::collections::BTreeMap::new(),
                }))
            }
            gemini_ox::content::PartData::FileData(file_data) => {
                Ok(StreamEvent::ToolResult(Part::Blob {
                    data_ref: crate::content::part::DataRef::Uri { uri: file_data.file_uri },
                    mime_type: file_data.mime_type,
                    name: file_data.display_name,
                    description: None,
                    ext: std::collections::BTreeMap::new(),
                }))
            }
            _ => Err(GenerateContentError::unsupported_feature(
                "Unsupported Gemini part type for streaming.",
            )),
        }
    }
}

impl From<&gemini_ox::generate_content::FinishReason> for FinishReason {
    fn from(reason: &gemini_ox::generate_content::FinishReason) -> Self {
        match reason {
            gemini_ox::generate_content::FinishReason::Stop => Self::Stop,
            gemini_ox::generate_content::FinishReason::MaxTokens => Self::Length,
            gemini_ox::generate_content::FinishReason::Safety => Self::ContentFilter,
            gemini_ox::generate_content::FinishReason::Recitation => Self::ContentFilter,
            _ => Self::Other,
        }
    }
}

pub(super) fn convert_request_to_gemini(
    request: ModelRequest,
    model: String,
    system_instruction: Option<GeminiContent>,
    tool_config: Option<ToolConfig>,
    safety_settings: Option<SafetySettings>,
    generation_config: Option<GenerationConfig>,
    cached_content: Option<String>,
) -> Result<GeminiGenerateContentRequest, GenerateContentError> {
    let contents = request
        .messages
        .into_iter()
        .map(TryInto::try_into)
        .collect::<Result<Vec<GeminiContent>, _>>()?;

    let tools = request
        .tools
        .map(|tools| tools.into_iter().map(|tool| tool.into()).collect());

    Ok(GeminiGenerateContentRequest {
        model,
        contents,
        system_instruction,
        tools,
        tool_config,
        safety_settings,
        generation_config,
        cached_content,
    })
}

pub(super) fn convert_gemini_response_to_ai_ox(
    response: GenerateContentResponse,
    model_name: String,
) -> Result<ModelResponse, GenerateContentError> {
    let message = response
        .candidates
        .first()
        .map(|candidate| candidate.content.clone().try_into())
        .transpose()?        .unwrap_or(Message::new(MessageRole::Assistant, vec![]));

    let usage = response
        .usage_metadata
        .map(|usage_metadata| usage_metadata.into())
        .unwrap_or_default();

    Ok(ModelResponse {
        message,
        model_name,
        vendor_name: "google".to_string(),
        usage,
    })
}
