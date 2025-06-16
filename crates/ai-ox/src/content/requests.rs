use crate::content::message::Message;
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
