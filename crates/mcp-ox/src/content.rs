use crate::{FromMcp, McpConversionError, ToMcp};
use ai_ox::content::{DataRef, Message, Part};
use mcp_sdk::types::ToolResponseContent;
use url::Url;

impl ToMcp<Vec<ToolResponseContent>> for Vec<Message> {
    fn to_mcp(&self) -> Result<Vec<ToolResponseContent>, McpConversionError> {
        let mut contents = Vec::new();
        for message in self {
            for part in &message.content {
                let content = part_to_tool_response_content(part)?;
                contents.push(content);
            }
        }
        Ok(contents)
    }
}

impl FromMcp<Vec<ToolResponseContent>> for Vec<Message> {
    fn from_mcp(value: Vec<ToolResponseContent>) -> Result<Vec<Message>, McpConversionError> {
        let mut messages = Vec::new();
        let mut parts = Vec::new();
        for content in value {
            let part = tool_response_content_to_part(&content)?;
            parts.push(part);
        }
        // Assume all parts are in one message, role Assistant
        if !parts.is_empty() {
            messages.push(Message::new(ai_ox::content::MessageRole::Assistant, parts));
        }
        Ok(messages)
    }
}

fn part_to_tool_response_content(part: &Part) -> Result<ToolResponseContent, McpConversionError> {
    match part {
        Part::Text { text, .. } => Ok(ToolResponseContent::Text { text: text.clone() }),
        Part::Blob {
            data_ref,
            mime_type,
            ..
        } => match data_ref {
            DataRef::Base64 { data } => {
                if mime_type.starts_with("image/") {
                    Ok(ToolResponseContent::Image {
                        data: data.clone(),
                        mime_type: mime_type.clone(),
                    })
                } else {
                    Err(McpConversionError::InvalidFormat(
                        "Base64 blobs not supported for ToolResponseContent unless image"
                            .to_string(),
                    ))
                }
            }
            DataRef::Uri { uri } => Ok(ToolResponseContent::Resource {
                resource: mcp_sdk::types::ResourceContents {
                    uri: Url::parse(uri).map_err(|_| {
                        McpConversionError::InvalidFormat("Invalid URI".to_string())
                    })?,
                    mime_type: Some(mime_type.clone()),
                },
            }),
        },
        _ => Err(McpConversionError::InvalidFormat(
            "Unsupported Part type for ToolResponseContent".to_string(),
        )),
    }
}

fn tool_response_content_to_part(
    content: &ToolResponseContent,
) -> Result<Part, McpConversionError> {
    match content {
        ToolResponseContent::Text { text } => Ok(Part::Text {
            text: text.clone(),
            ext: std::collections::BTreeMap::new(),
        }),
        ToolResponseContent::Image { data, mime_type } => Ok(Part::Blob {
            data_ref: DataRef::Base64 { data: data.clone() },
            mime_type: mime_type.clone(),
            name: None,
            description: None,
            ext: std::collections::BTreeMap::new(),
        }),
        ToolResponseContent::Resource { resource } => Ok(Part::Blob {
            data_ref: DataRef::Uri {
                uri: resource.uri.to_string(),
            },
            mime_type: resource
                .mime_type
                .clone()
                .unwrap_or_else(|| "application/octet-stream".to_string()),
            name: Some("resource".to_string()),
            description: None,
            ext: std::collections::BTreeMap::new(),
        }),
    }
}
