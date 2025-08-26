use serde::{Deserialize, Serialize};

use crate::{
    message::{Messages, StringOrContents},
    request::ThinkingConfig,
    tool::{Tool, ToolChoice},
};

/// A request to count the number of tokens in a message.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenCountRequest {
    /// The model to use for token counting.
    pub model: String,
    /// The messages to count the tokens of.
    pub messages: Messages,
    /// An optional system prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<StringOrContents>,
    /// An optional list of tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// An optional choice of tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// An optional thinking configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

/// The response from a token count request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenCountResponse {
    /// The number of input tokens.
    pub input_tokens: u32,
}
