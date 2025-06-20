use crate::content::message::{Message, MessageRole};
use crate::content::part::Part;
use chrono::Utc;
use serde::Serialize;
use serde_json::Value;

/// Represents a request to generate content from a model.
#[derive(Debug, Clone, Serialize)]
pub struct GenerateContentRequest {
    pub model: String,
    /// A list of messages forming the conversation history.
    pub messages: Vec<Message>,

    /// A list of tools the model may call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,

    /// System-level instructions for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Message>,
}

impl From<&str> for Message {
    fn from(text: &str) -> Self {
        Message {
            role: MessageRole::User,
            content: vec![Part::Text {
                text: text.to_string(),
            }],
            timestamp: Utc::now(),
        }
    }
}

impl From<String> for Message {
    fn from(text: String) -> Self {
        Message {
            role: MessageRole::User,
            content: vec![Part::Text { text }],
            timestamp: Utc::now(),
        }
    }
}
