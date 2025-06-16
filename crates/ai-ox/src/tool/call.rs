use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Represents a call to a tool function.
///
/// This struct contains all the information needed to invoke a specific tool,
/// including the function name and its arguments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call
    pub id: String,

    /// Name of the function to call
    pub name: String,

    /// Arguments to pass to the function, serialized as JSON
    pub args: Value,
}

impl ToolCall {
    /// Creates a new ToolCall with the given parameters.
    pub fn new(id: impl Into<String>, name: impl Into<String>, args: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            args,
        }
    }
}
