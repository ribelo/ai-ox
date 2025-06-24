use crate::{
    content::{
        delta::{FinishReason, StreamEvent, StreamStop},
        message::{Message, MessageRole},
        part::{FileData, ImageSource, Part},
    },
    tool::ToolCall,
    usage::Usage,
};
use gemini_ox::{
    content::{
        Blob as GeminiBlob, Content as GeminiContent, FileData as GeminiFileData,
        FunctionCall as GeminiFunctionCall, FunctionResponse as GeminiFunctionResponse,
        Part as GeminiPart, PartData as GeminiPartData, Role as GeminiRole, Text as GeminiText,
    },
    generate_content::response::GenerateContentResponse as GeminiResponse,
};

/// Converts an `ai-ox` `Message` to a `gemini-ox` `Content`.
impl From<Message> for GeminiContent {
    fn from(message: Message) -> Self {
        let role = match message.role {
            MessageRole::User => GeminiRole::User,
            MessageRole::Assistant => GeminiRole::Model,
        };

        let parts: Vec<GeminiPart> = message.content.into_iter().map(Into::into).collect();

        GeminiContent { role, parts }
    }
}

/// Converts an `ai-ox` `Part` to a `gemini-ox` `Part`.
impl From<Part> for GeminiPart {
    fn from(part: Part) -> Self {
        let data = match part {
            Part::Text { text } => GeminiPartData::Text(GeminiText::new(text)),
            Part::Image { source } => match source {
                ImageSource::Base64 { media_type, data } => {
                    GeminiPartData::InlineData(GeminiBlob::new(media_type, data))
                }
            },
            Part::File(file_data) => {
                let gemini_file_data = if let Some(display_name) = file_data.display_name {
                    GeminiFileData::new_with_display_name(
                        file_data.file_uri,
                        file_data.mime_type,
                        display_name,
                    )
                } else {
                    GeminiFileData::new(file_data.file_uri, file_data.mime_type)
                };
                GeminiPartData::FileData(gemini_file_data)
            }
            Part::ToolCall { id, name, args } => {
                // Convert tool call to proper Gemini FunctionCall
                let function_call = GeminiFunctionCall {
                    id: Some(id),
                    name,
                    args: Some(args),
                };
                GeminiPartData::FunctionCall(function_call)
            }
            Part::ToolResult {
                call_id,
                name,
                content,
            } => {
                // Convert tool result to proper Gemini FunctionResponse
                let function_response = GeminiFunctionResponse {
                    id: Some(call_id),
                    name,
                    response: content,
                    will_continue: None,
                    scheduling: None,
                };
                GeminiPartData::FunctionResponse(function_response)
            }
        };

        GeminiPart {
            thought: None,
            video_metadata: None,
            data,
        }
    }
}

/// Converts a `gemini-ox` `Content` to an `ai-ox` `Message`.
impl From<GeminiContent> for Message {
    fn from(content: GeminiContent) -> Self {
        let role = match content.role {
            GeminiRole::User => MessageRole::User,
            GeminiRole::Model => MessageRole::Assistant,
        };

        let parts = content.parts.into_iter().map(Into::into).collect();

        Message::new(role, parts)
    }
}

/// Converts a `gemini-ox` `Part` to an `ai-ox` `Part`.
impl From<GeminiPart> for Part {
    fn from(part: GeminiPart) -> Self {
        match part.data {
            GeminiPartData::Text(text) => Part::Text {
                text: text.to_string(),
            },
            GeminiPartData::InlineData(blob) => Part::Image {
                source: ImageSource::Base64 {
                    media_type: blob.mime_type,
                    data: blob.data,
                },
            },
            GeminiPartData::FileData(file_data) => {
                let ai_file_data = FileData {
                    file_uri: file_data.file_uri,
                    mime_type: file_data.mime_type,
                    display_name: file_data.display_name,
                };
                Part::File(ai_file_data)
            }
            GeminiPartData::FunctionCall(function_call) => Part::ToolCall {
                id: function_call.id.unwrap_or_default(),
                name: function_call.name,
                args: function_call.args.unwrap_or_default(),
            },
            GeminiPartData::FunctionResponse(function_response) => Part::ToolResult {
                call_id: function_response.id.unwrap_or_default(),
                name: function_response.name,
                content: function_response.response,
            },
            GeminiPartData::ExecutableCode(executable_code) => Part::ToolCall {
                // Gemini provides no ID for code execution requests. An empty string is used
                // as a placeholder, making the `name` field essential for identification.
                id: String::new(),
                name: "code_interpreter".to_string(),
                args: serde_json::json!({
                    "language": executable_code.language.to_string(),
                    "code": executable_code.code,
                }),
            },
            GeminiPartData::CodeExecutionResult(code_execution_result) => Part::ToolResult {
                // Gemini provides no ID for code execution results. An empty string is used
                // as a placeholder, making the `name` field essential for identification.
                call_id: String::new(),
                name: "code_interpreter".to_string(),
                content: serde_json::json!({
                    "outcome": code_execution_result.outcome.to_string(),
                    "output": code_execution_result.output,
                }),
            },
        }
    }
}

/// Converts a `gemini-ox` `FunctionCall` to a `ToolCall`.
impl From<GeminiFunctionCall> for ToolCall {
    fn from(function_call: GeminiFunctionCall) -> Self {
        ToolCall {
            id: function_call.id.unwrap_or_default(),
            name: function_call.name,
            args: function_call.args.unwrap_or_default(),
        }
    }
}

/// Converts a streaming Gemini response to ai-ox StreamEvents
pub fn convert_streaming_response(response: GeminiResponse) -> Vec<StreamEvent> {
    to_ai_ox_stream_events(response)
}

/// Converts a raw Gemini chunk to our internal StreamEvent format
fn to_ai_ox_stream_events(gemini_chunk: GeminiResponse) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    // Process candidates
    if let Some(candidate) = gemini_chunk.candidates.first() {
        for part in &candidate.content.parts {
            match &part.data {
                GeminiPartData::Text(text) => {
                    events.push(StreamEvent::TextDelta(text.to_string()));
                }
                GeminiPartData::FunctionCall(function_call) => {
                    events.push(StreamEvent::ToolCall(ToolCall::from(function_call.clone())));
                }
                GeminiPartData::ExecutableCode(executable_code) => {
                    let tool_call = crate::tool::ToolCall {
                        id: String::new(), // Gemini doesn't provide IDs for code execution
                        name: "code_interpreter".to_string(),
                        args: serde_json::json!({
                            "language": executable_code.language.to_string(),
                            "code": executable_code.code,
                        }),
                    };
                    events.push(StreamEvent::ToolCall(tool_call));
                }
                GeminiPartData::CodeExecutionResult(code_execution_result) => {
                    use crate::content::{message::{Message, MessageRole}, part::Part};
                    
                    let content_json = serde_json::json!({
                        "outcome": code_execution_result.outcome.to_string(),
                        "output": code_execution_result.output,
                    });
                    
                    let message = Message::new(
                        MessageRole::Assistant,
                        vec![Part::Text { text: content_json.to_string() }]
                    );
                    
                    let tool_result = crate::tool::ToolResult {
                        id: String::new(), // Gemini doesn't provide IDs for code execution results
                        name: "code_interpreter".to_string(),
                        response: vec![message],
                    };
                    events.push(StreamEvent::ToolResult(tool_result));
                }
                GeminiPartData::InlineData(_) | 
                GeminiPartData::FileData(_) | 
                GeminiPartData::FunctionResponse(_) => {
                    // These part types are handled but don't generate stream events
                }
            }
        }
    }

    // Handle usage metadata
    if let Some(usage_metadata) = &gemini_chunk.usage_metadata {
        let usage = Usage::from(usage_metadata.clone());
        events.push(StreamEvent::Usage(usage.clone()));
        
        // Determine finish reason
        let finish_reason = gemini_chunk.candidates
            .first()
            .and_then(|candidate| candidate.finish_reason.as_ref())
            .map(|reason| match reason {
                gemini_ox::generate_content::FinishReason::Stop => FinishReason::Stop,
                gemini_ox::generate_content::FinishReason::MaxTokens => FinishReason::Length,
                gemini_ox::generate_content::FinishReason::Safety => FinishReason::ContentFilter,
                gemini_ox::generate_content::FinishReason::Recitation => FinishReason::ContentFilter,
                _ => FinishReason::Other,
            })
            .unwrap_or(FinishReason::Stop);

        events.push(StreamEvent::StreamStop(StreamStop {
            finish_reason,
            usage,
        }));
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::part::FileData;

    #[test]
    fn test_message_to_content_user_role() {
        let message = Message {
            role: MessageRole::User,
            content: vec![Part::Text {
                text: "Hello, world!".to_string(),
            }],
            timestamp: chrono::Utc::now(),
        };

        let content: GeminiContent = message.into();
        assert_eq!(content.role, GeminiRole::User);
        assert_eq!(content.parts.len(), 1);
    }

    #[test]
    fn test_message_to_content_assistant_role() {
        let message = Message {
            role: MessageRole::Assistant,
            content: vec![Part::Text {
                text: "Hi there!".to_string(),
            }],
            timestamp: chrono::Utc::now(),
        };

        let content: GeminiContent = message.into();
        assert_eq!(content.role, GeminiRole::Model);
        assert_eq!(content.parts.len(), 1);
    }

    #[test]
    fn test_text_part_conversion() {
        let part = Part::Text {
            text: "Test text".to_string(),
        };

        let gemini_part: GeminiPart = part.into();
        let text_data = gemini_part.data.as_text().unwrap();
        assert_eq!(text_data.to_string(), "Test text");
    }

    #[test]
    fn test_image_part_conversion() {
        let part = Part::Image {
            source: ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: "base64data".to_string(),
            },
        };

        let gemini_part: GeminiPart = part.into();
        let blob_data = gemini_part.data.as_inline_data().unwrap();
        assert_eq!(blob_data.mime_type, "image/png");
        assert_eq!(blob_data.data, "base64data");
    }

    #[test]
    fn test_file_part_conversion() {
        let file_data = FileData::new_with_display_name(
            "gs://bucket/file.pdf",
            "application/pdf",
            "document.pdf",
        );
        let part = Part::File(file_data);

        let gemini_part: GeminiPart = part.into();
        let file_data_result = gemini_part.data.as_file_data().unwrap();
        assert_eq!(file_data_result.file_uri, "gs://bucket/file.pdf");
        assert_eq!(file_data_result.mime_type, "application/pdf");
        assert_eq!(
            file_data_result.display_name,
            Some("document.pdf".to_string())
        );
    }

    #[test]
    fn test_reverse_text_part_conversion() {
        let gemini_part = GeminiPart::new(GeminiText::new("Hello, world!"));
        let ai_part: Part = gemini_part.into();

        match ai_part {
            Part::Text { text } => assert_eq!(text, "Hello, world!"),
            _ => panic!("Expected text part"),
        }
    }

    #[test]
    fn test_tool_call_conversion() {
        use serde_json::json;

        let part = Part::ToolCall {
            id: "call_456".to_string(),
            name: "search_web".to_string(),
            args: json!({"query": "rust programming"}),
        };

        let gemini_part: GeminiPart = part.into();
        let function_call = gemini_part.data.as_function_call().unwrap();
        assert_eq!(function_call.id, Some("call_456".to_string()));
        assert_eq!(function_call.name, "search_web");
        assert_eq!(
            function_call.args,
            Some(json!({"query": "rust programming"}))
        );
    }

    #[test]
    fn test_reverse_function_call_conversion() {
        use serde_json::json;

        let function_call = GeminiFunctionCall {
            id: Some("call_789".to_string()),
            name: "calculate".to_string(),
            args: Some(json!({"expression": "2 + 2"})),
        };
        let gemini_part = GeminiPart::new(function_call);
        let ai_part: Part = gemini_part.into();

        match ai_part {
            Part::ToolCall { id, name, args } => {
                assert_eq!(id, "call_789");
                assert_eq!(name, "calculate");
                assert_eq!(args, json!({"expression": "2 + 2"}));
            }
            _ => panic!("Expected tool call part"),
        }
    }

    #[test]
    fn test_reverse_function_response_conversion() {
        use serde_json::json;

        let function_response = GeminiFunctionResponse::new_with_id(
            "call_123",
            "test_function",
            json!({"result": "success"}),
        );
        let gemini_part = GeminiPart::new(function_response);
        let ai_part: Part = gemini_part.into();

        match ai_part {
            Part::ToolResult {
                call_id,
                name,
                content,
            } => {
                assert_eq!(call_id, "call_123");
                assert_eq!(name, "test_function");
                assert_eq!(content, json!({"result": "success"}));
            }
            _ => panic!("Expected tool result part"),
        }
    }

    #[test]
    fn test_reverse_message_conversion() {
        let gemini_content = GeminiContent {
            role: GeminiRole::Model,
            parts: vec![GeminiPart::new(GeminiText::new("AI response"))],
        };

        let ai_message: Message = gemini_content.into();
        assert_eq!(ai_message.role, MessageRole::Assistant);
        assert_eq!(ai_message.content.len(), 1);

        match &ai_message.content[0] {
            Part::Text { text } => assert_eq!(text, "AI response"),
            _ => panic!("Expected text part"),
        }
    }
}