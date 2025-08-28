// Base OpenAI format types shared across compatible providers
// Extracted from the 80% duplication across openai-ox, groq-ox, mistral-ox, openrouter-ox

use serde::{Deserialize, Serialize};
use serde_json::Value;
use bon::Builder;

/// Core message roles used across all OpenAI-format providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User, 
    Assistant,
    Tool,
}

/// Base message structure used across all OpenAI-format providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Tool call structure (identical across providers)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String, // Usually "function"
    pub function: FunctionCall,
}

/// Function call details (identical across providers)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String, // JSON string
}

/// Tool definition structure (identical across providers)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub r#type: String, // Usually "function"
    pub function: Function,
}

/// Function definition (identical across providers) 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>, // JSON schema
}

/// Tool choice options (identical across providers)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Specific { r#type: String, function: Function },
}

/// Core chat request fields present in ALL OpenAI-format providers
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(builder_type(vis = "pub"))]
pub struct ChatRequest {
    /// Messages in the conversation
    #[builder(field)]
    pub messages: Vec<Message>,
    
    /// Model identifier
    #[builder(into)]
    pub model: String,
    
    /// Temperature for randomness (0.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    
    /// Maximum tokens in response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    
    /// Top-p sampling (0.0 to 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    
    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    
    /// Available tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    
    /// Tool choice strategy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

/// Usage statistics (identical across providers)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Base chat response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String, // "chat.completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

/// Response choice structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
}

/// Streaming response delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<MessageRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Streaming choice structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

/// Streaming response chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamResponse {
    pub id: String,
    pub object: String, // "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

impl ChatRequest {
    /// Create a new chat request with minimal required fields
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            messages,
            model: model.into(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            stop: None,
            stream: None,
            tools: None,
            tool_choice: None,
        }
    }
}

impl Message {
    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
    
    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
    
    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}