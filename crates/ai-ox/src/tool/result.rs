use crate::content::message::Message;
use serde::{Deserialize, Serialize};

/// Represents the result of a successful tool invocation.
///
/// Contains the response data and metadata about the tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// The original tool call ID
    pub id: String,

    /// Name of the function that was called
    pub name: String,

    /// Response messages from the tool execution
    /// This allows tools to return conversational responses
    pub response: Vec<Message>,
}

impl ToolResult {
    /// Creates a new ToolResult with the given parameters.
    pub fn new(id: impl Into<String>, name: impl Into<String>, response: Vec<Message>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            response,
        }
    }

    /// Creates a ToolResult with a single message response.
    pub fn single_message(
        id: impl Into<String>,
        name: impl Into<String>,
        message: Message,
    ) -> Self {
        Self::new(id, name, vec![message])
    }
}
