use serde::{Deserialize, Serialize};

/// Represents partial updates to a tool function call during streaming.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolFunctionDelta {
    /// The function name (sent once at the beginning).
    pub name: Option<String>,
    /// Incremental arguments as a JSON string fragment.
    pub arguments: Option<String>,
}

/// Represents a partial update to a content part during streaming.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum PartDelta {
    /// Incremental text content.
    Text {
        /// The text fragment to append.
        text: String,
    },
    /// Incremental tool call updates.
    ToolCall {
        /// Index of the tool call being updated in the message's tool call list.
        index: usize,
        /// Tool call ID (sent once at the beginning).
        id: Option<String>,
        /// Function details being updated.
        function: Option<ToolFunctionDelta>,
    },
    // Note: Image and ToolResult parts are not typically streamed as deltas,
    // they appear fully formed in the final message.
}

/// Represents a partial update to a message during streaming.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageDelta {
    /// The message role (typically sent once at the start of the message).
    pub role: Option<String>,
    /// Content part deltas for this message.
    pub content: Option<Vec<PartDelta>>,
}

/// Represents events in a message stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MessageStreamEvent {
    /// A partial message delta update.
    Delta(MessageDelta),
    /// Indicates the end of the message stream.
    End,
}
