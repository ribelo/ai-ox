use bon::Builder;
use serde::{Deserialize, Serialize};

pub type Messages = Vec<Message>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    #[serde(rename = "system")]
    System(SystemMessage),
    #[serde(rename = "user")]
    User(UserMessage),
    #[serde(rename = "assistant")]
    Assistant(AssistantMessage),
    #[serde(rename = "tool")]
    Tool(ToolMessage),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Builder)]
pub struct SystemMessage {
    pub content: String,
}

impl SystemMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Builder)]
pub struct UserMessage {
    pub content: String,
}

impl UserMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Builder)]
pub struct AssistantMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::tool::ToolCall>>,
}

impl AssistantMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            tool_calls: None,
        }
    }

    pub fn with_tool_calls(tool_calls: Vec<crate::tool::ToolCall>) -> Self {
        Self {
            content: None,
            tool_calls: Some(tool_calls),
        }
    }

    pub fn with_content_and_tools(
        content: impl Into<String>,
        tool_calls: Vec<crate::tool::ToolCall>,
    ) -> Self {
        Self {
            content: Some(content.into()),
            tool_calls: Some(tool_calls),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Builder)]
pub struct ToolMessage {
    pub tool_call_id: String,
    pub content: String,
}

impl ToolMessage {
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }
}

impl From<SystemMessage> for Message {
    fn from(msg: SystemMessage) -> Self {
        Message::System(msg)
    }
}

impl From<UserMessage> for Message {
    fn from(msg: UserMessage) -> Self {
        Message::User(msg)
    }
}

impl From<AssistantMessage> for Message {
    fn from(msg: AssistantMessage) -> Self {
        Message::Assistant(msg)
    }
}

impl From<ToolMessage> for Message {
    fn from(msg: ToolMessage) -> Self {
        Message::Tool(msg)
    }
}