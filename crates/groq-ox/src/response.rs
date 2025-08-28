use serde::{Deserialize, Serialize};
use ai_ox_common::openai_format::{
    ChatCompletionResponse, CompletionChoice, TokenUsage, 
    ToolCall
};
use crate::error::GroqRequestError;

// Use shared response types from ai-ox-common
pub type ChatResponse = ChatCompletionResponse;
pub type Choice = CompletionChoice;
pub type Usage = TokenUsage;

// Groq uses custom streaming types due to complex tool call deltas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

impl ChatCompletionChunk {
    /// Parse streaming data and return a vector of parsed chunks
    pub fn from_streaming_data(data: &str) -> Vec<Result<Self, GroqRequestError>> {
        let mut chunks = Vec::new();
        
        for line in data.lines() {
            let line = line.trim();
            
            // Skip empty lines and comments
            if line.is_empty() || line.starts_with(':') {
                continue;
            }
            
            // Handle data lines
            if let Some(data_content) = line.strip_prefix("data: ") {
                // Check for end of stream
                if data_content.trim() == "[DONE]" {
                    break;
                }
                
                // Try to parse the JSON
                match serde_json::from_str::<ChatCompletionChunk>(data_content) {
                    Ok(chunk) => chunks.push(Ok(chunk)),
                    Err(e) => chunks.push(Err(GroqRequestError::JsonDeserializationError(e))),
                }
            }
        }
        
        chunks
    }
}

/// Extension trait for ChatResponse convenience methods
pub trait ChatResponseExt {
    /// Get the text content from the first choice, if available
    fn text(&self) -> Option<&str>;
    /// Get tool calls from the first choice, if available
    fn tool_calls(&self) -> Option<&[ToolCall]>;
}

impl ChatResponseExt for ChatResponse {
    fn text(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|choice| choice.message.content.as_deref())
    }

    fn tool_calls(&self) -> Option<&[ToolCall]> {
        self.choices
            .first()
            .and_then(|choice| choice.message.tool_calls.as_deref())
    }
}