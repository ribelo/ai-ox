use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Reference to binary data - either a URI or inline base64
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "dataType", rename_all = "camelCase")]
pub enum DataRef {
    /// External URI reference
    Uri { uri: String },
    /// Inline base64-encoded data
    Base64 { data: String },
}

impl DataRef {
    /// Create a URI reference
    pub fn uri(uri: impl Into<String>) -> Self {
        Self::Uri { uri: uri.into() }
    }

    /// Create a base64 reference
    pub fn base64(data: impl Into<String>) -> Self {
        Self::Base64 { data: data.into() }
    }

    /// Get the size if this is base64 data
    pub fn base64_size(&self) -> Option<usize> {
        match self {
            Self::Base64 { data } => {
                // Rough estimate: base64 is ~4/3 of original
                Some(data.len() * 3 / 4)
            }
            Self::Uri { .. } => None,
        }
    }
}

/// Simplified Part enum following Feynman's design
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Part {
    /// Plain text content
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        ext: BTreeMap<String, Value>,
    },

    /// Generic binary content with MIME type (replaces Image/Audio/Resource/File)
    Blob {
        /// Reference to the data (URI or base64)
        #[serde(flatten)]
        data_ref: DataRef,
        /// MIME type (e.g., "image/png", "audio/wav", "application/pdf")
        mime_type: String,
        /// Optional human-friendly name
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Optional description
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        ext: BTreeMap<String, Value>,
    },

    /// Request to call a tool
    ToolUse {
        id: String,
        name: String,
        args: Value,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        ext: BTreeMap<String, Value>,
    },

    /// Result from tool execution (contains nested parts for rich content)
    ToolResult {
        id: String,
        name: String,
        /// Can contain multiple parts (text, blobs, even nested tool results)
        parts: Vec<Part>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        ext: BTreeMap<String, Value>,
    },

    /// Provider-specific content we don't understand
    Opaque {
        provider: String,
        kind: String,
        payload: Value,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        ext: BTreeMap<String, Value>,
    },
}

impl Part {
    /// Create a text part
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            ext: BTreeMap::new(),
        }
    }

    /// Create a blob from URI
    pub fn blob_uri(uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Blob {
            data_ref: DataRef::uri(uri),
            mime_type: mime_type.into(),
            name: None,
            description: None,
            ext: BTreeMap::new(),
        }
    }

    /// Create a blob from base64 data
    pub fn blob_base64(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Blob {
            data_ref: DataRef::base64(data),
            mime_type: mime_type.into(),
            name: None,
            description: None,
            ext: BTreeMap::new(),
        }
    }

    /// Create an image blob (convenience method)
    pub fn image_uri(uri: impl Into<String>) -> Self {
        Self::blob_uri(uri, "image/jpeg")
    }

    /// Create an audio blob (convenience method)
    pub fn audio_uri(uri: impl Into<String>) -> Self {
        Self::blob_uri(uri, "audio/wav")
    }

    /// Create a tool use request
    pub fn tool_use(id: impl Into<String>, name: impl Into<String>, args: Value) -> Self {
        Self::ToolUse {
            id: id.into(),
            name: name.into(),
            args,
            ext: BTreeMap::new(),
        }
    }

    /// Create a tool result with parts
    pub fn tool_result(id: impl Into<String>, name: impl Into<String>, parts: Vec<Part>) -> Self {
        Self::ToolResult {
            id: id.into(),
            name: name.into(),
            parts,
            ext: BTreeMap::new(),
        }
    }

    /// Get the MIME type if this is a blob
    pub fn mime_type(&self) -> Option<&str> {
        match self {
            Self::Blob { mime_type, .. } => Some(mime_type),
            _ => None,
        }
    }

    /// Check if this part is a specific MIME type category
    pub fn is_image(&self) -> bool {
        self.mime_type().map_or(false, |m| m.starts_with("image/"))
    }

    pub fn is_audio(&self) -> bool {
        self.mime_type().map_or(false, |m| m.starts_with("audio/"))
    }

    pub fn is_video(&self) -> bool {
        self.mime_type().map_or(false, |m| m.starts_with("video/"))
    }

    /// Add a namespaced extension value
    pub fn with_ext(mut self, namespace: &str, key: &str, value: Value) -> Self {
        let full_key = format!("{}.{}", namespace, key);
        match &mut self {
            Self::Text { ext, .. }
            | Self::Blob { ext, .. }
            | Self::ToolUse { ext, .. }
            | Self::ToolResult { ext, .. }
            | Self::Opaque { ext, .. } => {
                ext.insert(full_key, value);
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_creation() {
        let blob = Part::blob_uri("https://example.com/image.jpg", "image/jpeg");
        assert!(blob.is_image());
        assert!(!blob.is_audio());
        assert_eq!(blob.mime_type(), Some("image/jpeg"));
    }

    #[test]
    fn test_tool_result_with_parts() {
        let result = Part::tool_result(
            "call_123",
            "search",
            vec![
                Part::text("Found 3 results:"),
                Part::blob_uri("https://example.com/result1.png", "image/png"),
            ],
        );

        if let Part::ToolResult { parts, .. } = result {
            assert_eq!(parts.len(), 2);
        } else {
            panic!("Expected ToolResult");
        }
    }

    #[test]
    fn test_ext_namespacing() {
        let part = Part::text("Hello")
            .with_ext("anthropic", "thinking", Value::Bool(true))
            .with_ext("openai", "model", Value::String("gpt-4".into()));

        if let Part::Text { ext, .. } = part {
            assert_eq!(ext.get("anthropic.thinking"), Some(&Value::Bool(true)));
            assert_eq!(
                ext.get("openai.model"),
                Some(&Value::String("gpt-4".into()))
            );
        }
    }

    #[test]
    fn test_data_ref_size_estimation() {
        let base64_data = "SGVsbG8gV29ybGQ="; // "Hello World" in base64
        let data_ref = DataRef::base64(base64_data);
        let size = data_ref.base64_size().unwrap();
        assert!(size > 0);
        assert!(size < 100); // Reasonable size for small string
    }
}
