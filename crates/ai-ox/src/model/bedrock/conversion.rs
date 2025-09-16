use super::error::BedrockError;
use crate::{
    content::{
        delta::FinishReason,
        message::Message,
        message::MessageRole,
        part::{DataRef, Part},
    },
    model::response::ModelResponse,
    tool::{Tool, decode_tool_result_parts, encode_tool_result_parts},
    usage::Usage,
};
use aws_sdk_bedrockruntime::types::{
    ContentBlock, Message as BedrockMessage, StopReason, ToolResultBlock, ToolSpecification,
    ToolUseBlock,
};
use base64::prelude::*;
use serde_json::Value;
use std::collections::BTreeMap;

/// Helper function to convert serde_json::Value to aws_smithy_types::Document
fn json_to_document(value: Value) -> aws_smithy_types::Document {
    match value {
        Value::Null => aws_smithy_types::Document::Null,
        Value::Bool(b) => aws_smithy_types::Document::Bool(b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                aws_smithy_types::Document::Number(aws_smithy_types::Number::PosInt(i as u64))
            } else if let Some(f) = n.as_f64() {
                aws_smithy_types::Document::Number(aws_smithy_types::Number::Float(f))
            } else {
                aws_smithy_types::Document::Null
            }
        }
        Value::String(s) => aws_smithy_types::Document::String(s),
        Value::Array(arr) => {
            aws_smithy_types::Document::Array(arr.into_iter().map(json_to_document).collect())
        }
        Value::Object(obj) => aws_smithy_types::Document::Object(
            obj.into_iter()
                .map(|(k, v)| (k, json_to_document(v)))
                .collect(),
        ),
    }
}

/// Helper function to convert aws_smithy_types::Document to serde_json::Value
pub(super) fn document_to_json(doc: &aws_smithy_types::Document) -> Value {
    match doc {
        aws_smithy_types::Document::Null => Value::Null,
        aws_smithy_types::Document::Bool(b) => Value::Bool(*b),
        aws_smithy_types::Document::Number(n) => match n {
            aws_smithy_types::Number::PosInt(i) => Value::Number((*i).into()),
            aws_smithy_types::Number::NegInt(i) => Value::Number((*i).into()),
            aws_smithy_types::Number::Float(f) => {
                Value::Number(serde_json::Number::from_f64(*f).unwrap_or_else(|| 0.into()))
            }
        },
        aws_smithy_types::Document::String(s) => Value::String(s.clone()),
        aws_smithy_types::Document::Array(arr) => {
            Value::Array(arr.iter().map(document_to_json).collect())
        }
        aws_smithy_types::Document::Object(obj) => Value::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), document_to_json(v)))
                .collect(),
        ),
    }
}

/// Converts ai-ox messages to Bedrock format
pub(super) fn convert_ai_ox_messages_to_bedrock(
    messages: Vec<Message>,
) -> Result<Vec<BedrockMessage>, BedrockError> {
    let mut bedrock_messages = Vec::new();

    for message in messages {
        let role = match message.role {
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::System => "system",
            MessageRole::Unknown(_) => "user", // Map unknown to user
        };

        let content_blocks = convert_parts_to_content_blocks(message.content)?;

        let bedrock_message = BedrockMessage::builder()
            .role(role.into())
            .set_content(Some(content_blocks))
            .build()
            .map_err(|e| {
                BedrockError::MessageConversion(format!("Failed to build message: {}", e))
            })?;

        bedrock_messages.push(bedrock_message);
    }

    Ok(bedrock_messages)
}

/// Converts ai-ox Parts to Bedrock ContentBlocks
fn convert_parts_to_content_blocks(parts: Vec<Part>) -> Result<Vec<ContentBlock>, BedrockError> {
    let mut content_blocks = Vec::new();

    for part in parts {
        let content_block = match part {
            Part::Text { text, .. } => ContentBlock::Text(text),
            Part::Blob {
                mime_type,
                data_ref,
                ..
            } => {
                if mime_type.starts_with("image/") {
                    match data_ref {
                        DataRef::Base64 { data } => {
                            let image_format = determine_image_format(&mime_type)?;
                            let decoded_data = BASE64_STANDARD.decode(&data).map_err(|e| {
                                BedrockError::MessageConversion(format!(
                                    "Invalid base64 image data: {}",
                                    e
                                ))
                            })?;

                            let image_block = aws_sdk_bedrockruntime::types::ImageBlock::builder()
                                .format(image_format)
                                .source(aws_sdk_bedrockruntime::types::ImageSource::Bytes(
                                    aws_smithy_types::Blob::new(decoded_data),
                                ))
                                .build()
                                .map_err(|e| {
                                    BedrockError::MessageConversion(format!(
                                        "Failed to build image block: {}",
                                        e
                                    ))
                                })?;
                            ContentBlock::Image(image_block)
                        }
                        _ => {
                            return Err(BedrockError::MessageConversion(
                                "Unsupported data_ref for image".to_string(),
                            ));
                        }
                    }
                } else if mime_type.starts_with("audio/") {
                    return Err(BedrockError::MessageConversion(
                        "Audio content is not supported by Bedrock".to_string(),
                    ));
                } else {
                    // treat as document
                    match data_ref {
                        DataRef::Uri { uri } => {
                            if uri.starts_with("file://") {
                                let path = uri.strip_prefix("file://").unwrap_or(&uri);
                                let document_format = determine_document_format(&mime_type)?;
                                let file_contents = std::fs::read(&path).map_err(|e| {
                                    BedrockError::MessageConversion(format!(
                                        "Failed to read file {}: {}",
                                        path, e
                                    ))
                                })?;

                                let document_block =
                                    aws_sdk_bedrockruntime::types::DocumentBlock::builder()
                                        .format(document_format)
                                        .name("document".to_string())
                                        .source(
                                            aws_sdk_bedrockruntime::types::DocumentSource::Bytes(
                                                aws_smithy_types::Blob::new(file_contents),
                                            ),
                                        )
                                        .build()
                                        .map_err(|e| {
                                            BedrockError::MessageConversion(format!(
                                                "Failed to build document block: {}",
                                                e
                                            ))
                                        })?;
                                ContentBlock::Document(document_block)
                            } else {
                                return Err(BedrockError::MessageConversion(format!(
                                    "Unsupported URI scheme for document: {}",
                                    uri
                                )));
                            }
                        }
                        _ => {
                            return Err(BedrockError::MessageConversion(
                                "Unsupported data_ref for document".to_string(),
                            ));
                        }
                    }
                }
            }
            Part::ToolUse { id, name, args, .. } => {
                let input_doc = aws_smithy_types::Document::Object(
                    args.as_object()
                        .ok_or_else(|| {
                            BedrockError::MessageConversion(
                                "Tool args must be an object".to_string(),
                            )
                        })?
                        .iter()
                        .map(|(k, v)| (k.clone(), json_to_document(v.clone())))
                        .collect(),
                );

                let tool_use_block = ToolUseBlock::builder()
                    .tool_use_id(id)
                    .name(name)
                    .input(input_doc)
                    .build()
                    .map_err(|e| {
                        BedrockError::MessageConversion(format!(
                            "Failed to build tool use block: {}",
                            e
                        ))
                    })?;
                ContentBlock::ToolUse(tool_use_block)
            }
            Part::ToolResult {
                id, name, parts, ..
            } => {
                let encoded = encode_tool_result_parts(&name, &parts).map_err(|e| {
                    BedrockError::MessageConversion(format!("Failed to encode tool result: {}", e))
                })?;
                let tool_result_block = ToolResultBlock::builder()
                    .tool_use_id(id)
                    .content(aws_sdk_bedrockruntime::types::ToolResultContentBlock::Text(
                        encoded,
                    ))
                    .build()
                    .map_err(|e| {
                        BedrockError::MessageConversion(format!(
                            "Failed to build tool result block: {}",
                            e
                        ))
                    })?;
                ContentBlock::ToolResult(tool_result_block)
            }
            Part::Opaque { .. } => {
                return Err(BedrockError::MessageConversion(
                    "Opaque parts are not supported by Bedrock".to_string(),
                ));
            }
        };
        content_blocks.push(content_block);
    }

    Ok(content_blocks)
}

/// Helper to determine image format from MIME type
fn determine_image_format(
    mime_type: &str,
) -> Result<aws_sdk_bedrockruntime::types::ImageFormat, BedrockError> {
    match mime_type {
        "image/png" => Ok(aws_sdk_bedrockruntime::types::ImageFormat::Png),
        "image/jpeg" | "image/jpg" => Ok(aws_sdk_bedrockruntime::types::ImageFormat::Jpeg),
        "image/gif" => Ok(aws_sdk_bedrockruntime::types::ImageFormat::Gif),
        "image/webp" => Ok(aws_sdk_bedrockruntime::types::ImageFormat::Webp),
        _ => Err(BedrockError::MessageConversion(format!(
            "Unsupported image format: {}",
            mime_type
        ))),
    }
}

/// Helper to determine document format from MIME type
fn determine_document_format(
    mime_type: &str,
) -> Result<aws_sdk_bedrockruntime::types::DocumentFormat, BedrockError> {
    match mime_type {
        "application/pdf" => Ok(aws_sdk_bedrockruntime::types::DocumentFormat::Pdf),
        "text/csv" => Ok(aws_sdk_bedrockruntime::types::DocumentFormat::Csv),
        "application/msword"
        | "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
            Ok(aws_sdk_bedrockruntime::types::DocumentFormat::Doc)
        }
        "application/vnd.ms-excel"
        | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => {
            Ok(aws_sdk_bedrockruntime::types::DocumentFormat::Xls)
        }
        "text/html" => Ok(aws_sdk_bedrockruntime::types::DocumentFormat::Html),
        "text/plain" => Ok(aws_sdk_bedrockruntime::types::DocumentFormat::Txt),
        "text/markdown" => Ok(aws_sdk_bedrockruntime::types::DocumentFormat::Md),
        _ => Err(BedrockError::MessageConversion(format!(
            "Unsupported document format: {}",
            mime_type
        ))),
    }
}

/// Converts Bedrock response to ai-ox ModelResponse
pub(super) fn convert_bedrock_response_to_ai_ox(
    output: aws_sdk_bedrockruntime::types::ConverseOutput,
    model_name: String,
    usage: aws_sdk_bedrockruntime::types::TokenUsage,
) -> Result<ModelResponse, BedrockError> {
    let message = match output {
        aws_sdk_bedrockruntime::types::ConverseOutput::Message(msg) => msg,
        _ => return Err(BedrockError::NoResponse),
    };

    let content_blocks = message.content();
    let parts = convert_content_blocks_to_parts(content_blocks)?;

    let ai_ox_message = Message {
        role: MessageRole::Assistant,
        content: parts,
        timestamp: Some(chrono::Utc::now()),
        ext: None,
    };

    let ai_ox_usage = convert_token_usage_to_ai_ox(usage);

    Ok(ModelResponse {
        message: ai_ox_message,
        model_name,
        usage: ai_ox_usage,
        vendor_name: "bedrock".to_string(),
    })
}

/// Converts Bedrock ContentBlocks to ai-ox Parts
fn convert_content_blocks_to_parts(
    content_blocks: &[ContentBlock],
) -> Result<Vec<Part>, BedrockError> {
    let mut parts = Vec::new();

    for block in content_blocks {
        let part = match block {
            ContentBlock::Text(text) => Part::Text {
                text: text.clone(),
                ext: Default::default(),
            },
            ContentBlock::Image(image_block) => {
                let format = image_block.format();
                let mime_type = match format {
                    aws_sdk_bedrockruntime::types::ImageFormat::Png => "image/png",
                    aws_sdk_bedrockruntime::types::ImageFormat::Jpeg => "image/jpeg",
                    aws_sdk_bedrockruntime::types::ImageFormat::Gif => "image/gif",
                    aws_sdk_bedrockruntime::types::ImageFormat::Webp => "image/webp",
                    _ => {
                        return Err(BedrockError::MessageConversion(
                            "Unknown image format".to_string(),
                        ));
                    }
                };

                let data = match image_block.source() {
                    Some(aws_sdk_bedrockruntime::types::ImageSource::Bytes(blob)) => {
                        BASE64_STANDARD.encode(blob.as_ref())
                    }
                    _ => {
                        return Err(BedrockError::MessageConversion(
                            "Unsupported image source".to_string(),
                        ));
                    }
                };

                Part::Blob {
                    mime_type: mime_type.to_string(),
                    data_ref: DataRef::Base64 { data },
                    name: None,
                    description: None,
                    ext: Default::default(),
                }
            }
            ContentBlock::ToolUse(tool_use) => Part::ToolUse {
                id: tool_use.tool_use_id().to_string(),
                name: tool_use.name().to_string(),
                args: document_to_json(tool_use.input()),
                ext: Default::default(),
            },
            ContentBlock::ToolResult(tool_result) => {
                let content_text = match tool_result.content() {
                    [aws_sdk_bedrockruntime::types::ToolResultContentBlock::Text(text)] => {
                        text.clone()
                    }
                    content_blocks => content_blocks
                        .iter()
                        .filter_map(|block| match block {
                            aws_sdk_bedrockruntime::types::ToolResultContentBlock::Text(text) => {
                                Some(text.as_str())
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                };

                let (decoded_name, parts) = decode_tool_result_parts(&content_text)?;
                Part::ToolResult {
                    id: tool_result.tool_use_id().to_string(),
                    name: decoded_name,
                    parts,
                    ext: Default::default(),
                }
            }
            _ => {
                return Err(BedrockError::MessageConversion(
                    "Unsupported content block type".to_string(),
                ));
            }
        };
        parts.push(part);
    }

    Ok(parts)
}

/// Converts Bedrock TokenUsage to ai-ox Usage
pub(super) fn convert_token_usage_to_ai_ox(
    token_usage: aws_sdk_bedrockruntime::types::TokenUsage,
) -> Usage {
    let mut usage = Usage::default();

    usage.requests = 1;

    let input_tokens = token_usage.input_tokens();
    if input_tokens > 0 {
        usage
            .input_tokens_by_modality
            .insert(crate::usage::Modality::Text, input_tokens as u64);
    }

    let output_tokens = token_usage.output_tokens();
    if output_tokens > 0 {
        usage
            .output_tokens_by_modality
            .insert(crate::usage::Modality::Text, output_tokens as u64);
    }

    usage
}

/// Converts ai-ox Tool to Bedrock Tools
pub(super) fn convert_ai_ox_tools_to_bedrock(
    tools: Vec<Tool>,
) -> Result<Vec<aws_sdk_bedrockruntime::types::Tool>, BedrockError> {
    let mut tool_specs = Vec::new();

    for tool in tools {
        match tool {
            Tool::FunctionDeclarations(functions) => {
                for func in functions {
                    let input_schema = aws_sdk_bedrockruntime::types::ToolInputSchema::Json(
                        json_to_document(func.parameters.clone()),
                    );

                    let tool_spec = ToolSpecification::builder()
                        .name(func.name)
                        .set_description(func.description)
                        .input_schema(input_schema)
                        .build()
                        .map_err(|e| {
                            BedrockError::MessageConversion(format!(
                                "Failed to build tool specification: {}",
                                e
                            ))
                        })?;
                    tool_specs.push(aws_sdk_bedrockruntime::types::Tool::ToolSpec(tool_spec));
                }
            }
            #[cfg(feature = "gemini")]
            Tool::GeminiTool(_) => {
                return Err(BedrockError::MessageConversion(
                    "Cannot convert Gemini-specific tool to Bedrock format".to_string(),
                ));
            }
        }
    }

    Ok(tool_specs)
}

/// Converts Bedrock finish reason to ai-ox FinishReason
pub(super) fn convert_bedrock_finish_reason(stop_reason: StopReason) -> FinishReason {
    match stop_reason {
        StopReason::EndTurn => FinishReason::Stop,
        StopReason::ToolUse => FinishReason::ToolCalls,
        StopReason::MaxTokens => FinishReason::Length,
        StopReason::StopSequence => FinishReason::Stop,
        StopReason::ContentFiltered => FinishReason::ContentFilter,
        StopReason::GuardrailIntervened => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        content::{message::Message, message::MessageRole, part::Part},
        tool::{FunctionMetadata, Tool},
    };
    use aws_sdk_bedrockruntime::types::*;
    use chrono::Utc;
    use serde_json::json;

    fn create_test_message(role: MessageRole, content: Vec<Part>) -> Message {
        Message {
            role,
            content,
            timestamp: Some(Utc::now()),
            ext: None,
        }
    }

    #[test]
    fn test_convert_text_message() {
        let ai_ox_message = create_test_message(
            MessageRole::User,
            vec![Part::Text {
                text: "Hello, world!".to_string(),
                ext: Default::default(),
            }],
        );

        let result = convert_ai_ox_messages_to_bedrock(vec![ai_ox_message]).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role().as_str(), "user");

        let content = result[0].content();
        assert_eq!(content.len(), 1);
        match &content[0] {
            ContentBlock::Text(text) => assert_eq!(text, "Hello, world!"),
            _ => panic!("Expected text content block"),
        }
    }

    #[test]
    fn test_convert_assistant_message() {
        let ai_ox_message = create_test_message(
            MessageRole::Assistant,
            vec![Part::Text {
                text: "Assistant response".to_string(),
                ext: Default::default(),
            }],
        );

        let result = convert_ai_ox_messages_to_bedrock(vec![ai_ox_message]).unwrap();

        assert_eq!(result[0].role().as_str(), "assistant");
    }

    #[test]
    fn test_convert_multiple_text_parts() {
        let ai_ox_message = create_test_message(
            MessageRole::User,
            vec![
                Part::Text {
                    text: "First part".to_string(),
                    ext: Default::default(),
                },
                Part::Text {
                    text: "Second part".to_string(),
                    ext: Default::default(),
                },
            ],
        );

        let result = convert_ai_ox_messages_to_bedrock(vec![ai_ox_message]).unwrap();

        let content = result[0].content();
        assert_eq!(content.len(), 2);

        match (&content[0], &content[1]) {
            (ContentBlock::Text(text1), ContentBlock::Text(text2)) => {
                assert_eq!(text1, "First part");
                assert_eq!(text2, "Second part");
            }
            _ => panic!("Expected text content blocks"),
        }
    }

    #[test]
    fn test_determine_image_formats() {
        assert!(matches!(
            determine_image_format("image/png").unwrap(),
            ImageFormat::Png
        ));
        assert!(matches!(
            determine_image_format("image/jpeg").unwrap(),
            ImageFormat::Jpeg
        ));
        assert!(matches!(
            determine_image_format("image/jpg").unwrap(),
            ImageFormat::Jpeg
        ));
        assert!(matches!(
            determine_image_format("image/gif").unwrap(),
            ImageFormat::Gif
        ));
        assert!(matches!(
            determine_image_format("image/webp").unwrap(),
            ImageFormat::Webp
        ));

        assert!(determine_image_format("image/invalid").is_err());
    }

    #[test]
    fn test_determine_document_formats() {
        assert!(matches!(
            determine_document_format("application/pdf").unwrap(),
            DocumentFormat::Pdf
        ));
        assert!(matches!(
            determine_document_format("text/csv").unwrap(),
            DocumentFormat::Csv
        ));
        assert!(matches!(
            determine_document_format("text/html").unwrap(),
            DocumentFormat::Html
        ));
        assert!(matches!(
            determine_document_format("text/plain").unwrap(),
            DocumentFormat::Txt
        ));
        assert!(matches!(
            determine_document_format("text/markdown").unwrap(),
            DocumentFormat::Md
        ));

        assert!(determine_document_format("application/invalid").is_err());
    }

    #[test]
    fn test_convert_ai_ox_tools_to_bedrock() {
        let function_metadata = vec![FunctionMetadata {
            name: "get_weather".to_string(),
            description: Some("Get weather information".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }),
        }];

        let ai_ox_tools = vec![Tool::FunctionDeclarations(function_metadata)];
        let result = convert_ai_ox_tools_to_bedrock(ai_ox_tools).unwrap();

        assert_eq!(result.len(), 1);
        match &result[0] {
            aws_sdk_bedrockruntime::types::Tool::ToolSpec(tool_spec) => {
                assert_eq!(tool_spec.name(), "get_weather");
                assert_eq!(
                    tool_spec.description(),
                    Some("Get weather information").as_deref()
                );
            }
            _ => panic!("Expected ToolSpec variant"),
        }
    }

    #[test]
    #[cfg(feature = "gemini")]
    fn test_convert_gemini_tool_error() {
        let gemini_tool = Tool::GeminiTool(gemini_ox::tool::Tool::FunctionDeclarations(vec![]));
        let result = convert_ai_ox_tools_to_bedrock(vec![gemini_tool]);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BedrockError::MessageConversion(_)
        ));
    }

    #[test]
    fn test_convert_bedrock_finish_reasons() {
        assert!(matches!(
            convert_bedrock_finish_reason(StopReason::EndTurn),
            FinishReason::Stop
        ));
        assert!(matches!(
            convert_bedrock_finish_reason(StopReason::ToolUse),
            FinishReason::ToolCalls
        ));
        assert!(matches!(
            convert_bedrock_finish_reason(StopReason::MaxTokens),
            FinishReason::Length
        ));
        assert!(matches!(
            convert_bedrock_finish_reason(StopReason::StopSequence),
            FinishReason::Stop
        ));
        assert!(matches!(
            convert_bedrock_finish_reason(StopReason::ContentFiltered),
            FinishReason::ContentFilter
        ));
        assert!(matches!(
            convert_bedrock_finish_reason(StopReason::GuardrailIntervened),
            FinishReason::ContentFilter
        ));
    }

    #[test]
    fn test_convert_token_usage() {
        let token_usage = TokenUsage::builder()
            .input_tokens(100)
            .output_tokens(50)
            .total_tokens(150)
            .build()
            .unwrap();

        let result = convert_token_usage_to_ai_ox(token_usage);

        assert_eq!(result.requests, 1);
        assert_eq!(
            result
                .input_tokens_by_modality
                .get(&crate::usage::Modality::Text),
            Some(&100)
        );
        assert_eq!(
            result
                .output_tokens_by_modality
                .get(&crate::usage::Modality::Text),
            Some(&50)
        );
    }

    #[test]
    fn test_convert_empty_messages() {
        let result = convert_ai_ox_messages_to_bedrock(vec![]).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_convert_message_with_empty_content() {
        let ai_ox_message = create_test_message(MessageRole::User, vec![]);
        let result = convert_ai_ox_messages_to_bedrock(vec![ai_ox_message]).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content().len(), 0);
    }
}
