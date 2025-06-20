use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Represents file data with URI and metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileData {
    /// URI identifying the file.
    pub file_uri: String,
    /// The IANA standard MIME type of the source data.
    pub mime_type: String,
    /// Optional display name for the file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
}

impl FileData {
    /// Creates a new FileData with the given URI and MIME type.
    pub fn new(file_uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            file_uri: file_uri.into(),
            mime_type: mime_type.into(),
            display_name: None,
        }
    }

    /// Creates a new FileData with the given URI, MIME type, and display name.
    pub fn new_with_display_name(
        file_uri: impl Into<String>,
        mime_type: impl Into<String>,
        display_name: impl Into<String>,
    ) -> Self {
        Self {
            file_uri: file_uri.into(),
            mime_type: mime_type.into(),
            display_name: Some(display_name.into()),
        }
    }
}

/// Source of image data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ImageSource {
    /// Base64 encoded image data.
    Base64 {
        /// MIME type of the image (e.g., "image/png", "image/jpeg").
        media_type: String,
        /// Base64 encoded image data.
        data: String,
    },
}

/// Represents a single piece of content that can be part of a message.
/// Supports text, images, files, and tool results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Part {
    /// Plain text content.
    Text {
        /// The text content.
        text: String,
    },
    /// Image content with source information.
    Image {
        /// The source of the image data.
        source: ImageSource,
    },
    /// File content referenced by URI.
    File(FileData),
    /// Result from a tool execution.
    ToolResult {
        /// ID of the tool call this is responding to.
        call_id: String,
        /// Name of the tool that was called.
        name: String,
        /// The result data from the tool execution.
        content: Value,
    },
}
