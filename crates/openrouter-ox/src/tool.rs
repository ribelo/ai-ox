use serde::{Deserialize, Serialize};
use serde_json::Value;

// Note: ToolCall is defined in response module but not used in this simplified version

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDescription {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub name: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "auto")]
    Auto,
    Function {
        #[serde(rename = "type")]
        choice_type: String,
        function: FunctionName,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionName {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionMetadata {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Debug, thiserror::Error, Serialize)]
pub enum ToolError {
    #[error("Failed to execute tool: {0}")]
    ExecutionFailed(String),
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Failed to deserialize input: {0}")]
    InputDeserializationFailed(String),
    #[error("Failed to serialize output: {0}")]
    OutputSerializationFailed(String),
    #[error("Missing arguments: {0}")]
    MissingArguments(String),
}