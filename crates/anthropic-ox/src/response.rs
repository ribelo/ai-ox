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

    pub fn thinking_content(&self) -> Vec<&str> {
        self.content
            .iter()
            .filter_map(|content| {
                if let Content::Thinking(thinking) = content {
                    Some(thinking.text.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn thinking_blocks(&self) -> impl Iterator<Item = &crate::message::ThinkingContent> {
        self.content.iter().filter_map(|content| {
            if let Content::Thinking(thinking) = content {
                Some(thinking)
            } else {
                None
            }
        })
    }

    pub fn has_thinking(&self) -> bool {
        self.content
            .iter()
            .any(|content| matches!(content, Content::Thinking(_)))
    }
}

impl std::fmt::Display for ChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut content_summary = Vec::new();
        
        let text_parts = self.text_content();
        if !text_parts.is_empty() {
            content_summary.push(format!("text: [{}]", text_parts.join(", ")));
        }
        
        let thinking_parts = self.thinking_content();
        if !thinking_parts.is_empty() {
            content_summary.push(format!("thinking: [{}]", thinking_parts.len()));
        }
        
        if self.has_tool_use() {
            content_summary.push("tools".to_string());
        }
        
        write!(
            f,
            "ChatResponse {{ id: {}, type: {}, role: {:?}, model: {}, content: {} }}",
            self.id,
            self.r#type,
            self.role,
            self.model,
            content_summary.join(", ")
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
    ThinkingDelta { text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Role, Content, ThinkingContent, Text};
    
    fn create_test_response_with_thinking() -> ChatResponse {
        ChatResponse {
            id: "test_id".to_string(),
            r#type: "message".to_string(),
            role: Role::Assistant,
            content: vec![
                Content::Thinking(ThinkingContent::new("Let me think about this...".to_string())),
                Content::Text(Text::new("The answer is 42.".to_string())),
                Content::Thinking(ThinkingContent::with_signature(
                    "Additional reasoning...".to_string(),
                    "sig123".to_string()
                )),
            ],
            model: "claude-3-sonnet".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            stop_sequence: None,
            usage: Usage { input_tokens: Some(10), output_tokens: Some(20) },
        }
    }
    
    #[test]
    fn test_thinking_content_extraction() {
        let response = create_test_response_with_thinking();
        
        let thinking_texts = response.thinking_content();
        assert_eq!(thinking_texts.len(), 2);
        assert_eq!(thinking_texts[0], "Let me think about this...");
        assert_eq!(thinking_texts[1], "Additional reasoning...");
    }
    
    #[test]
    fn test_thinking_blocks_iterator() {
        let response = create_test_response_with_thinking();
        
        let thinking_blocks: Vec<_> = response.thinking_blocks().collect();
        assert_eq!(thinking_blocks.len(), 2);
        
        assert_eq!(thinking_blocks[0].text, "Let me think about this...");
        assert_eq!(thinking_blocks[0].signature, None);
        
        assert_eq!(thinking_blocks[1].text, "Additional reasoning...");
        assert_eq!(thinking_blocks[1].signature, Some("sig123".to_string()));
    }
    
    #[test]
    fn test_has_thinking() {
        let response = create_test_response_with_thinking();
        assert!(response.has_thinking());
        
        let response_without_thinking = ChatResponse {
            id: "test_id".to_string(),
            r#type: "message".to_string(),
            role: Role::Assistant,
            content: vec![Content::Text(Text::new("Just text".to_string()))],
            model: "claude-3-sonnet".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            stop_sequence: None,
            usage: Usage::default(),
        };
        
        assert!(!response_without_thinking.has_thinking());
    }
    
    #[test]
    fn test_content_block_delta_thinking() {
        let delta = ContentBlockDelta::ThinkingDelta { 
            text: "More reasoning...".to_string() 
        };
        
        let json = serde_json::to_string(&delta).unwrap();
        let expected = r#"{"type":"thinking_delta","text":"More reasoning..."}"#;
        assert_eq!(json, expected);
        
        let deserialized: ContentBlockDelta = serde_json::from_str(&json).unwrap();
        assert_eq!(delta, deserialized);
    }
    
    #[test]
    fn test_response_display_with_thinking() {
        let response = create_test_response_with_thinking();
        let display = format!("{}", response);
        
        assert!(display.contains("thinking: [2]"));
        assert!(display.contains("text:"));
        assert!(display.contains("ChatResponse"));
    }
    
    #[test]
    fn test_response_display_without_thinking() {
        let response = ChatResponse {
            id: "test_id".to_string(),
            r#type: "message".to_string(),
            role: Role::Assistant,
            content: vec![Content::Text(Text::new("Just text".to_string()))],
            model: "claude-3-sonnet".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            stop_sequence: None,
            usage: Usage::default(),
        };
        
        let display = format!("{}", response);
        assert!(!display.contains("thinking"));
        assert!(display.contains("text:"));
    }
    
    #[test]
    fn test_stream_event_content_block_start_thinking() {
        let thinking_block = crate::message::ContentBlock::Thinking {
            text: "Initial thinking...".to_string(),
            signature: Some("start_sig".to_string()),
        };
        
        let event = StreamEvent::ContentBlockStart {
            index: 0,
            content_block: thinking_block,
        };
        
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("content_block_start"));
        assert!(json.contains("thinking"));
        assert!(json.contains("Initial thinking..."));
        assert!(json.contains("start_sig"));
        
        let deserialized: StreamEvent = serde_json::from_str(&json).unwrap();
        match deserialized {
            StreamEvent::ContentBlockStart { index, content_block } => {
                assert_eq!(index, 0);
                match content_block {
                    crate::message::ContentBlock::Thinking { text, signature } => {
                        assert_eq!(text, "Initial thinking...");
                        assert_eq!(signature, Some("start_sig".to_string()));
                    },
                    _ => panic!("Expected thinking content block"),
                }
            },
            _ => panic!("Expected ContentBlockStart event"),
        }
    }
    
    #[test]
    fn test_thinking_content_empty_response() {
        let response = ChatResponse {
            id: "empty_test".to_string(),
            r#type: "message".to_string(),
            role: Role::Assistant,
            content: vec![],
            model: "claude-3-sonnet".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            stop_sequence: None,
            usage: Usage::default(),
        };
        
        assert!(!response.has_thinking());
        assert!(response.thinking_content().is_empty());
        assert_eq!(response.thinking_blocks().count(), 0);
    }
}