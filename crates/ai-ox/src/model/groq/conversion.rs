use ai_ox_common::openai_format::Message as GroqMessage;
use groq_ox::{
    request::ChatRequest,
    response::{ChatCompletionChunk, ChatResponse},
};

use crate::{
    ModelResponse,
    content::{
        delta::{FinishReason, StreamEvent, StreamStop},
        message::{Message, MessageRole},
        part::Part,
    },
    errors::GenerateContentError,
    model::ModelRequest,
    usage::Usage,
};

use super::GroqError;

/// Convert from ai-ox ModelRequest to Groq ChatRequest
pub fn convert_request_to_groq(
    request: ModelRequest,
    model: String,
    system_instruction: Option<String>,
    _tool_choice: Option<ai_ox_common::openai_format::ToolChoice>,
) -> Result<ChatRequest, GenerateContentError> {
    let mut groq_messages = Vec::new();

    // Add system instruction if provided
    if let Some(system_msg) = system_instruction {
        groq_messages.push(GroqMessage::system(system_msg));
    }

    // Convert messages - simplified version
    for message in request.messages {
        match message.role {
            MessageRole::User => {
                let content = extract_text_content(&message.content)?;
                groq_messages.push(GroqMessage::user(content));
            }
            MessageRole::Assistant => {
                let content = extract_text_content(&message.content)?;
                groq_messages.push(GroqMessage::assistant(content));
            }
            MessageRole::System => {
                let content = extract_text_content(&message.content)?;
                groq_messages.push(GroqMessage::system(content));
            }
            MessageRole::Unknown(role) => {
                return Err(GenerateContentError::message_conversion(&format!(
                    "Unknown role: {}",
                    role
                )));
            }
        }
    }

    // TODO: Add tool support later
    Ok(ChatRequest::builder()
        .model(model)
        .messages(groq_messages)
        .build())
}

/// Extract text content from content parts
fn extract_text_content(content: &[Part]) -> Result<String, GenerateContentError> {
    let mut text = String::new();
    for part in content {
        if let Part::Text {
            text: part_text, ..
        } = part
        {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(part_text);
        }
    }
    if text.is_empty() {
        return Err(GenerateContentError::message_conversion(
            "No text content found in message",
        ));
    }
    Ok(text)
}

/// Convert Groq response to ai-ox ModelResponse
pub fn convert_groq_response_to_ai_ox(
    response: ChatResponse,
    model_name: String,
) -> Result<ModelResponse, GenerateContentError> {
    let choice = response
        .choices
        .first()
        .ok_or_else(|| GroqError::ResponseParsing("No choices in response".to_string()))?;

    let mut content_parts = Vec::new();

    // Add text content if present
    if let Some(text) = &choice.message.content {
        if !text.is_empty() {
            content_parts.push(Part::Text {
                text: text.clone(),
                ext: std::collections::BTreeMap::new(),
            });
        }
    }

    let message = Message {
        role: MessageRole::Assistant,
        content: content_parts,
        timestamp: Some(chrono::Utc::now()),
        ext: None,
    };

    let usage = response
        .usage
        .map(|u| {
            let mut usage = Usage::new();
            usage.requests = 1;
            usage
                .input_tokens_by_modality
                .insert(crate::usage::Modality::Text, u.prompt_tokens() as u64);
            usage
                .output_tokens_by_modality
                .insert(crate::usage::Modality::Text, u.completion_tokens() as u64);
            usage
        })
        .unwrap_or_else(Usage::new);

    Ok(ModelResponse {
        message,
        usage,
        model_name,
        vendor_name: "groq".to_string(),
    })
}

/// Convert streaming response to stream events
pub fn convert_response_to_stream_events(
    chunk: ChatCompletionChunk,
) -> Vec<Result<StreamEvent, GenerateContentError>> {
    let mut events = Vec::new();

    if let Some(choice) = chunk.choices.first() {
        let delta = &choice.delta;

        // Handle content delta
        if let Some(content) = &delta.content {
            if !content.is_empty() {
                events.push(Ok(StreamEvent::TextDelta(content.clone())));
            }
        }

        // Handle finish reason
        if let Some(finish_reason) = &choice.finish_reason {
            let reason = match finish_reason.as_str() {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                _ => FinishReason::Other,
            };

            let usage = chunk
                .usage
                .map(|u| {
                    let mut usage = Usage::new();
                    usage.requests = 1;
                    usage
                        .input_tokens_by_modality
                        .insert(crate::usage::Modality::Text, u.prompt_tokens() as u64);
                    usage
                        .output_tokens_by_modality
                        .insert(crate::usage::Modality::Text, u.completion_tokens() as u64);
                    usage
                })
                .unwrap_or_else(Usage::new);

            events.push(Ok(StreamEvent::StreamStop(StreamStop {
                usage,
                finish_reason: reason,
            })));
        }
    }

    events
}
