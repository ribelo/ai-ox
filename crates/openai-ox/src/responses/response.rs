use serde::{Deserialize, Serialize};
use ai_ox_common::openai_format::ToolCall;
use crate::Usage;

/// Response from OpenAI Responses API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesResponse {
    /// Unique identifier for the response
    pub id: String,

    /// Unix timestamp of creation
    pub created_at: u64,

    /// Model used for the response
    pub model: String,

    /// Array of output items
    pub output: Vec<OutputItem>,

    /// Response status (e.g., "completed", "in_progress", "failed")
    pub status: String,

    /// Usage statistics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponsesUsage>,

    /// System fingerprint
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// Individual output item in the response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    /// Reasoning item with potential summary and encrypted content
    #[serde(rename = "reasoning")]
    ReasoningItem(ReasoningItem),
    
    /// Message response (text content)
    #[serde(rename = "message")]
    Message(ResponseMessage),
    
    /// Tool/function call
    #[serde(rename = "tool_call")]
    ToolCall(ToolCallItem),
    
    /// Plain text response
    #[serde(rename = "text")]
    Text(TextItem),
}

/// Reasoning item containing the model's internal reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningItem {
    /// Unique identifier for this reasoning item
    pub id: String,

    /// Human-readable summary of the reasoning (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,

    /// Encrypted reasoning content for ZDR compliance
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,

    /// Token usage for this reasoning
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ReasoningUsage>,
}

/// Message response item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMessage {
    /// Message role (typically "assistant")
    pub role: String,

    /// Message content
    pub content: String,

    /// Tool calls in this message (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Tool call item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallItem {
    /// Tool call details
    #[serde(flatten)]
    pub tool_call: ToolCall,

    /// Result of the tool call (if completed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,

    /// Status of the tool call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

/// Text item for simple text responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextItem {
    /// Text content
    pub text: String,
}

/// Usage statistics for Responses API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesUsage {
    /// Total input tokens
    pub input_tokens: u32,

    /// Total output tokens
    pub output_tokens: u32,

    /// Total tokens used
    pub total_tokens: u32,

    /// Reasoning tokens used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,

    /// Cache-related token details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheUsage>,
}

/// Cache usage details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheUsage {
    /// Cached input tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u32>,

    /// Cache hit rate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hit_rate: Option<f64>,
}

/// Usage statistics for individual reasoning items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningUsage {
    /// Tokens used in this reasoning step
    pub reasoning_tokens: u32,
}

/// Streaming response for Responses API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesStreamChunk {
    /// Chunk identifier
    pub id: String,

    /// Model used
    pub model: String,

    /// Output delta for this chunk
    pub output: Vec<OutputDelta>,

    /// Current status
    pub status: String,

    /// Final usage (only in last chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponsesUsage>,
}

/// Delta for streaming output items
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputDelta {
    /// Reasoning delta
    #[serde(rename = "reasoning")]
    ReasoningDelta(ReasoningDelta),
    
    /// Message delta
    #[serde(rename = "message")]
    MessageDelta(MessageDelta),
    
    /// Tool call delta
    #[serde(rename = "tool_call")]
    ToolCallDelta(ToolCallDelta),
    
    /// Text delta
    #[serde(rename = "text")]
    TextDelta(TextDelta),
}

/// Reasoning delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningDelta {
    /// Reasoning item ID
    pub id: String,

    /// Partial summary update
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,

    /// Partial encrypted content update
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
}

/// Message delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDelta {
    /// Role (if starting new message)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// Partial content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Tool call deltas
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// Tool call delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Tool call index
    pub index: u32,

    /// Tool call ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Function name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<String>,

    /// Partial arguments
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Text delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDelta {
    /// Partial text content
    pub text: String,
}

// Helper methods for ResponsesResponse
impl ResponsesResponse {
    /// Get the main text content from the response
    pub fn content(&self) -> Option<String> {
        let mut content_parts = Vec::new();

        for item in &self.output {
            match item {
                OutputItem::Message(msg) => content_parts.push(&msg.content),
                OutputItem::Text(text) => content_parts.push(&text.text),
                OutputItem::ReasoningItem(reasoning) => {
                    if let Some(ref summary) = reasoning.summary {
                        content_parts.push(summary);
                    }
                }
                _ => {}
            }
        }

        if content_parts.is_empty() {
            None
        } else {
            Some(content_parts.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n"))
        }
    }

    /// Get all reasoning items from the response
    pub fn reasoning_items(&self) -> Vec<&ReasoningItem> {
        self.output
            .iter()
            .filter_map(|item| match item {
                OutputItem::ReasoningItem(reasoning) => Some(reasoning),
                _ => None,
            })
            .collect()
    }

    /// Get all messages from the response
    pub fn messages(&self) -> Vec<&ResponseMessage> {
        self.output
            .iter()
            .filter_map(|item| match item {
                OutputItem::Message(message) => Some(message),
                _ => None,
            })
            .collect()
    }

    /// Get all tool calls from the response
    pub fn tool_calls(&self) -> Vec<&ToolCallItem> {
        self.output
            .iter()
            .filter_map(|item| match item {
                OutputItem::ToolCall(tool_call) => Some(tool_call),
                _ => None,
            })
            .collect()
    }

    /// Check if the response is completed
    pub fn is_completed(&self) -> bool {
        self.status == "completed"
    }

    /// Check if the response is still in progress
    pub fn is_in_progress(&self) -> bool {
        self.status == "in_progress"
    }

    /// Check if the response failed
    pub fn is_failed(&self) -> bool {
        self.status == "failed"
    }

    /// Get total reasoning tokens used
    pub fn reasoning_tokens(&self) -> u32 {
        self.usage
            .as_ref()
            .and_then(|u| u.reasoning_tokens)
            .unwrap_or(0)
    }

    /// Check if response contains encrypted reasoning
    pub fn has_encrypted_reasoning(&self) -> bool {
        self.reasoning_items()
            .iter()
            .any(|item| item.encrypted_content.is_some())
    }
}

// Conversion from ResponsesUsage to standard Usage
impl From<ResponsesUsage> for Usage {
    fn from(responses_usage: ResponsesUsage) -> Self {
        Self {
            prompt_tokens: responses_usage.input_tokens,
            completion_tokens: responses_usage.output_tokens,
            total_tokens: responses_usage.total_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: responses_usage.reasoning_tokens.map(|reasoning_tokens| {
                crate::usage::CompletionTokensDetails {
                    reasoning_tokens: Some(reasoning_tokens),
                    audio_tokens: None,
                }
            }),
        }
    }
}

// Conversion from standard Usage to ResponsesUsage
impl From<Usage> for ResponsesUsage {
    fn from(usage: Usage) -> Self {
        Self {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            reasoning_tokens: None,
            cache: None,
        }
    }
}