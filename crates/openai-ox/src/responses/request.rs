use bon::Builder;
use serde::{Deserialize, Serialize};
use ai_ox_common::openai_format::Message;
use serde_json::Value;

/// Tool definition for OpenAI Responses API - supports custom types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesTool {
    /// Type of tool (e.g., "custom", "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    
    /// Name of the tool
    pub name: String,
    
    /// Optional description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    
    /// Tool format configuration (for grammar-based tools)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<ToolFormat>,
    
    /// Parameters schema (for simple tools without grammar)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

/// Tool format configuration for grammar-based tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFormat {
    /// Format type (e.g., "grammar")
    #[serde(rename = "type")]
    pub format_type: String,
    
    /// Grammar syntax (e.g., "lark")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub syntax: Option<String>,
    
    /// Grammar definition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub definition: Option<String>,
}

/// Request for OpenAI Responses API - supports reasoning models
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(builder_type(vis = "pub"), state_mod(vis = "pub"))]
pub struct ResponsesRequest {
    /// Model to use (e.g., "o3", "o4-mini", "gpt-5")
    #[builder(into)]
    pub model: String,

    /// Input content - can be text, messages, or mixed content
    pub input: ResponsesInput,

    /// Instructions for the model (system prompt)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Reasoning configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    /// Text output configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,

    /// Tools/functions available to the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,
    
    /// Tool choice configuration (e.g., "auto", "none", or specific tool)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    
    /// Whether to allow parallel tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Items to include in response (e.g., ["reasoning.encrypted_content"])
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,

    /// Whether to store the response for future use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    /// Enable streaming response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Maximum number of output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// Previous response ID for conversation chaining
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// User identifier for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Reasoning configuration for the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Reasoning effort level: "minimal", "low", "medium", "high"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,

    /// Summary configuration: "auto" for automatic summaries
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Text output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    /// Verbosity level: "low", "medium", "high"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
}

/// Input for the Responses API - flexible input types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    /// Simple text input
    Text(String),
    /// Array of messages (chat format)
    Messages(Vec<Message>),
    /// Mixed content with files, images, etc.
    Mixed(Vec<InputPart>),
}

/// Input part for mixed content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputPart {
    /// Type of input part
    #[serde(rename = "type")]
    pub part_type: String,
    
    /// Text content (for text parts)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    
    /// Image data (for image parts)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageData>,
    
    /// File reference (for file parts)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<FileReference>,
}

/// Image data for input parts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    /// Image URL or base64 data
    pub url: String,
    /// Optional detail level
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// File reference for input parts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileReference {
    /// File ID from uploaded files
    pub id: String,
}

// Helper methods for ResponsesInput
impl ResponsesInput {
    /// Create text input
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(content.into())
    }

    /// Create messages input
    pub fn messages(messages: Vec<Message>) -> Self {
        Self::Messages(messages)
    }

    /// Create mixed input
    pub fn mixed(parts: Vec<InputPart>) -> Self {
        Self::Mixed(parts)
    }
}

// Helper methods for InputPart
impl InputPart {
    /// Create text input part
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            part_type: "text".to_string(),
            text: Some(content.into()),
            image: None,
            file: None,
        }
    }

    /// Create image input part
    pub fn image(url: impl Into<String>, detail: Option<String>) -> Self {
        Self {
            part_type: "image".to_string(),
            text: None,
            image: Some(ImageData {
                url: url.into(),
                detail,
            }),
            file: None,
        }
    }

    /// Create file input part
    pub fn file(file_id: impl Into<String>) -> Self {
        Self {
            part_type: "file".to_string(),
            text: None,
            image: None,
            file: Some(FileReference {
                id: file_id.into(),
            }),
        }
    }
}

// Helper methods for ReasoningConfig
impl ReasoningConfig {
    /// Create reasoning config with effort level
    pub fn with_effort(effort: impl Into<String>) -> Self {
        Self {
            effort: Some(effort.into()),
            summary: None,
        }
    }

    /// Create reasoning config with automatic summary
    pub fn with_auto_summary() -> Self {
        Self {
            effort: None,
            summary: Some("auto".to_string()),
        }
    }

    /// Create reasoning config with both effort and summary
    pub fn with_effort_and_summary(effort: impl Into<String>) -> Self {
        Self {
            effort: Some(effort.into()),
            summary: Some("auto".to_string()),
        }
    }
}

// Helper methods for TextConfig
impl TextConfig {
    /// Create text config with verbosity level
    pub fn with_verbosity(verbosity: impl Into<String>) -> Self {
        Self {
            verbosity: Some(verbosity.into()),
        }
    }
}

