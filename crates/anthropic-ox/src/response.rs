use serde::{Deserialize, Serialize};
use crate::{
    message::{Role, Content, ContentBlock},
    error::ErrorInfo,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatResponse {
    pub id: String,
    pub r#type: String,
    pub role: Role,
    pub content: Vec<Content>,
    pub model: String,
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

impl ChatResponse {
    pub fn text_content(&self) -> Vec<&str> {
        self.content
            .iter()
            .filter_map(|content| {
                if let Content::Text(text) = content {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn tool_uses(&self) -> impl Iterator<Item = &crate::tool::ToolUse> {
        self.content.iter().filter_map(|content| {
            if let Content::ToolUse(tool_use) = content {
                Some(tool_use)
            } else {
                None
            }
        })
    }

    pub fn has_tool_use(&self) -> bool {
        self.content
            .iter()
            .any(|content| matches!(content, Content::ToolUse(_)))
    }
}

impl std::fmt::Display for ChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ChatResponse {{ id: {}, type: {}, role: {:?}, model: {}, content: [{}] }}",
            self.id,
            self.r#type,
            self.role,
            self.model,
            self.text_content().join(", ")
        )
    }
}

// Streaming types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamMessage {
    pub id: String,
    pub r#type: String,
    pub role: Role,
    pub content: Vec<Content>,
    pub model: String,
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart {
        message: StreamMessage,
    },
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: ContentBlockDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDelta,
        usage: Option<Usage>,
    },
    MessageStop,
    Ping,
    Error {
        error: ErrorInfo,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}