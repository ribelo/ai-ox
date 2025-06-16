use serde::Serialize;
use serde_json::Value;

use crate::content::message::Message;
// Re-export the content structures for convenience
pub use crate::content::requests::GenerateContentRequest;
pub use crate::content::response::GenerateContentResponse;

#[derive(Debug, Clone, Serialize)]
pub struct ModelRequest {
    /// A list of messages forming the conversation history.
    pub messages: Vec<Message>,

    /// A list of tools the model may call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,

    /// System-level instructions for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Message>,
}

impl From<Vec<Message>> for ModelRequest {
    fn from(messages: Vec<Message>) -> Self {
        ModelRequest {
            messages,
            tools: None,
            instructions: None,
        }
    }
}
