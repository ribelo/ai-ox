use serde::{Deserialize, Serialize};

use crate::{Message, Usage};

/// Response from chat completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Unique identifier for the response
    pub id: String,
    
    /// Object type (usually "chat.completion")
    pub object: String,
    
    /// Unix timestamp of creation
    pub created: u64,
    
    /// Model used for the completion
    pub model: String,
    
    /// List of completion choices
    pub choices: Vec<Choice>,
    
    /// Usage statistics
    pub usage: Option<Usage>,
    
    /// System fingerprint
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// A completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    /// Index of this choice
    pub index: u32,
    
    /// The completion message
    pub message: Message,
    
    /// Reason for stopping
    pub finish_reason: Option<String>,
    
    /// Log probabilities (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

/// Streaming choice delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceDelta {
    /// Index of this choice
    pub index: u32,
    
    /// The partial message delta
    pub delta: MessageDelta,
    
    /// Reason for stopping
    pub finish_reason: Option<String>,
    
    /// Log probabilities (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

/// Partial message for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDelta {
    /// Message role
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    
    /// Partial content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    
    /// Tool calls (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

impl ChatResponse {
    /// Get the content of the first choice, if available
    pub fn content(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|choice| choice.message.content.as_deref())
    }
    
    /// Get the first choice, if available
    pub fn first_choice(&self) -> Option<&Choice> {
        self.choices.first()
    }
    
    /// Check if the response is finished
    pub fn is_finished(&self) -> bool {
        self.choices
            .first()
            .map(|choice| choice.finish_reason.is_some())
            .unwrap_or(false)
    }
    
    /// Get the finish reason of the first choice
    pub fn finish_reason(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|choice| choice.finish_reason.as_deref())
    }
}