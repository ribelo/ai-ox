use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

use ai_ox_common;

/// Result from a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// ID of the tool call this is responding to.
    pub id: String,
    /// Name of the tool that was called.
    pub name: String,
    /// The result data from the tool execution.
    pub content: Vec<crate::content::Part>,
    /// Extension metadata for MCP compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ext: Option<BTreeMap<String, Value>>,
}

impl ToolResult {
    /// Creates a new ToolResult.
    pub fn new(id: impl Into<String>, name: impl Into<String>, content: Vec<crate::content::Part>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            content,
            ext: None,
        }
    }
}


/// Represents a request to call a tool function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolUse {
    /// Unique identifier for this tool call.
    pub id: String,
    /// Name of the function to call.
    pub name: String,
    /// Arguments to pass to the function.
    pub args: Value,
    /// Extension metadata for MCP compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ext: Option<BTreeMap<String, Value>>,
}

impl ToolUse {
    /// Creates a new ToolUse with the given id, name, and arguments.
    pub fn new(id: impl Into<String>, name: impl Into<String>, args: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            args,
            ext: None,
        }
    }
}

impl From<ai_ox_common::ToolCall> for ToolUse {
    fn from(call: ai_ox_common::ToolCall) -> Self {
        let args = serde_json::from_str(&call.function.arguments).unwrap_or(Value::Null);
        Self {
            id: call.id,
            name: call.function.name,
            args,
            ext: None,
        }
    }
}
