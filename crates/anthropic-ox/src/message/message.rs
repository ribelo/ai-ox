use std::{fmt, path::Path};

use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::tool::{ToolResult, ToolUse};

use strum::{Display, EnumString};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Display, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum ImageSource {
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
}

impl ImageSource {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let path = path.as_ref();
        let data = std::fs::read(path)?;
        let base64_data = base64::engine::general_purpose::STANDARD.encode(data);
        let media_type = mime_guess::from_path(path)
            .first_or_octet_stream()
            .to_string();

        Ok(ImageSource::Base64 {
            media_type,
            data: base64_data,
        })
    }
}

impl fmt::Display for ImageSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageSource::Base64 { media_type, data } => {
                let truncated_data = if data.len() > 20 {
                    format!("{}...", &data[..20])
                } else {
                    data.clone()
                };
                write!(f, "Base64 ({}, {})", media_type, truncated_data)
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum Content {
    #[serde(rename = "text")]
    Text(Text),
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "tool_use")]
    ToolUse(ToolUse),
    #[serde(rename = "tool_result")]
    ToolResult(ToolResult),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Text {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl Text {
    pub fn new(text: String) -> Self {
        Self { text, cache_control: None }
    }
    
    pub fn as_str(&self) -> &str {
        &self.text
    }
}

impl fmt::Display for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    ToolUse { 
        id: String, 
        name: String, 
        input: serde_json::Value 
    },
}

impl From<String> for Content {
    fn from(text: String) -> Self {
        Content::Text(Text { text, cache_control: None })
    }
}

impl From<&str> for Content {
    fn from(text: &str) -> Self {
        Content::Text(Text {
            text: text.to_string(),
            cache_control: None,
        })
    }
}

impl From<Text> for Content {
    fn from(text: Text) -> Self {
        Content::Text(text)
    }
}

impl fmt::Display for Content {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Content::Text(text) => write!(f, "{}", text.text),
            Content::Image { source } => write!(f, "[Image: {}]", source),
            Content::ToolUse(tool_use) => write!(f, "[Tool Use: {}]", tool_use.name),
            Content::ToolResult(tool_result) => write!(f, "[Tool Result: {}]", tool_result.tool_use_id),
        }
    }
}

// Follow Zed's pattern: String first, then Contents
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum StringOrContents {
    String(String),
    Contents(Vec<Content>),
}


impl StringOrContents {
    pub fn as_vec(&self) -> Vec<Content> {
        match self {
            StringOrContents::String(text) => vec![Content::Text(Text::new(text.clone()))],
            StringOrContents::Contents(contents) => contents.clone(),
        }
    }

    pub fn into_vec(self) -> Vec<Content> {
        match self {
            StringOrContents::String(text) => vec![Content::Text(Text::new(text))],
            StringOrContents::Contents(contents) => contents,
        }
    }

    pub fn as_string(&self) -> String {
        match self {
            StringOrContents::String(text) => text.clone(),
            StringOrContents::Contents(contents) => {
                contents.iter()
                    .filter_map(|content| match content {
                        Content::Text(text) => Some(text.text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("")
            }
        }
    }

    pub fn into_string(self) -> String {
        match self {
            StringOrContents::String(text) => text,
            StringOrContents::Contents(contents) => {
                contents.into_iter()
                    .filter_map(|content| match content {
                        Content::Text(text) => Some(text.text),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("")
            }
        }
    }
}

impl From<String> for StringOrContents {
    fn from(text: String) -> Self {
        StringOrContents::String(text)
    }
}

impl From<&str> for StringOrContents {
    fn from(text: &str) -> Self {
        StringOrContents::String(text.to_string())
    }
}

impl From<Vec<Content>> for StringOrContents {
    fn from(contents: Vec<Content>) -> Self {
        StringOrContents::Contents(contents)
    }
}

impl From<Content> for StringOrContents {
    fn from(content: Content) -> Self {
        StringOrContents::Contents(vec![content])
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: StringOrContents,
}

impl Message {
    pub fn new(role: Role, content: Vec<Content>) -> Self {
        Self { 
            role, 
            content: StringOrContents::Contents(content) 
        }
    }

    pub fn user<T: Into<Content>>(content: Vec<T>) -> Self {
        Self {
            role: Role::User,
            content: StringOrContents::Contents(content.into_iter().map(Into::into).collect()),
        }
    }

    pub fn assistant<T: Into<Content>>(content: Vec<T>) -> Self {
        Self {
            role: Role::Assistant,
            content: StringOrContents::Contents(content.into_iter().map(Into::into).collect()),
        }
    }

    pub fn add_content<T: Into<Content>>(&mut self, content: T) {
        match &mut self.content {
            StringOrContents::String(text) => {
                let mut contents = vec![Content::Text(Text::new(text.clone()))];
                contents.push(content.into());
                self.content = StringOrContents::Contents(contents);
            }
            StringOrContents::Contents(contents) => {
                contents.push(content.into());
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.content {
            StringOrContents::String(text) => text.is_empty(),
            StringOrContents::Contents(contents) => contents.is_empty(),
        }
    }

    pub fn len(&self) -> usize {
        match &self.content {
            StringOrContents::String(_) => 1,
            StringOrContents::Contents(contents) => contents.len(),
        }
    }
}

impl<T: Into<Content>> From<T> for Message {
    fn from(content: T) -> Self {
        Message::user(vec![content])
    }
}

impl From<Vec<Content>> for Message {
    fn from(content: Vec<Content>) -> Self {
        Message::user(content)
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: ", self.role)?;
        match &self.content {
            StringOrContents::String(text) => write!(f, "{}", text),
            StringOrContents::Contents(contents) => {
                for (i, content) in contents.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", content)?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Messages(pub Vec<Message>);

impl FromIterator<Message> for Messages {
    fn from_iter<T: IntoIterator<Item = Message>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl Messages {
    pub fn new() -> Self {
        Self(vec![])
    }

    pub fn add_message(&mut self, message: Message) {
        self.0.push(message);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::ops::Deref for Messages {
    type Target = Vec<Message>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Messages {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Vec<Message>> for Messages {
    fn from(messages: Vec<Message>) -> Self {
        Self(messages)
    }
}

impl From<Message> for Messages {
    fn from(message: Message) -> Self {
        Self(vec![message])
    }
}

impl IntoIterator for Messages {
    type Item = Message;
    type IntoIter = std::vec::IntoIter<Message>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Messages {
    type Item = &'a Message;
    type IntoIter = std::slice::Iter<'a, Message>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_string_or_contents_string_deserialization() {
        // Test deserializing a simple string
        let json = r#""Hello world""#;
        let result: Result<StringOrContents, _> = serde_json::from_str(json);
        
        assert!(result.is_ok());
        let content = result.unwrap();
        
        match content {
            StringOrContents::String(s) => assert_eq!(s, "Hello world"),
            StringOrContents::Contents(_) => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_string_or_contents_array_deserialization() {
        // Test deserializing a content array
        let json = r#"[{"type": "text", "text": "Hello world"}]"#;
        let result: Result<StringOrContents, _> = serde_json::from_str(json);
        
        assert!(result.is_ok());
        let content = result.unwrap();
        
        match content {
            StringOrContents::String(_) => panic!("Expected Contents variant"),
            StringOrContents::Contents(contents) => {
                assert_eq!(contents.len(), 1);
                match &contents[0] {
                    Content::Text(text) => assert_eq!(text.text, "Hello world"),
                    _ => panic!("Expected Text content"),
                }
            }
        }
    }

    #[test]
    fn test_message_with_string_content() {
        let json = r#"{
            "role": "user",
            "content": "Hello world"
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let message = result.unwrap();
        assert_eq!(message.role, Role::User);
        
        match message.content {
            StringOrContents::String(s) => assert_eq!(s, "Hello world"),
            StringOrContents::Contents(_) => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_message_with_array_content() {
        let json = r#"{
            "role": "user", 
            "content": [{"type": "text", "text": "Hello world"}]
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let message = result.unwrap();
        assert_eq!(message.role, Role::User);
        
        match message.content {
            StringOrContents::String(_) => panic!("Expected Contents variant"),
            StringOrContents::Contents(contents) => {
                assert_eq!(contents.len(), 1);
                match &contents[0] {
                    Content::Text(text) => assert_eq!(text.text, "Hello world"),
                    _ => panic!("Expected Text content"),
                }
            }
        }
    }

    #[test] 
    fn test_serialization_roundtrip_string() {
        let original = StringOrContents::String("Hello world".to_string());
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: StringOrContents = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_serialization_roundtrip_contents() {
        let original = StringOrContents::Contents(vec![
            Content::Text(Text::new("Hello world".to_string()))
        ]);
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: StringOrContents = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_complex_claude_code_request() {
        // Test a typical Claude Code request with multiple content items
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "Please help me with this code:"},
                {"type": "text", "text": "fn main() { println!(\"Hello\"); }"}
            ]
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let message = result.unwrap();
        match message.content {
            StringOrContents::Contents(contents) => {
                assert_eq!(contents.len(), 2);
                match &contents[0] {
                    Content::Text(text) => assert_eq!(text.text, "Please help me with this code:"),
                    _ => panic!("Expected Text content"),
                }
                match &contents[1] {
                    Content::Text(text) => assert_eq!(text.text, "fn main() { println!(\"Hello\"); }"),
                    _ => panic!("Expected Text content"),
                }
            },
            StringOrContents::String(_) => panic!("Expected Contents variant"),
        }
    }

    // Tests based on Anthropic API documentation examples

    #[test]
    fn test_text_with_cache_control() {
        // RED: Test for cache_control support in text content
        let json = r#"{
            "type": "text",
            "text": "Summarize this coding conversation in under 50 characters.\nCapture the main task, key files, problems addressed, and current status.",
            "cache_control": {"type": "ephemeral"}
        }"#;
        
        let result: Result<Content, _> = serde_json::from_str(json);
        assert!(result.is_ok(), "Failed to deserialize text with cache_control: {:?}", result.err());
        
        let content = result.unwrap();
        match content {
            Content::Text(text) => {
                assert_eq!(text.text, "Summarize this coding conversation in under 50 characters.\nCapture the main task, key files, problems addressed, and current status.");
                assert!(text.cache_control.is_some());
                let cache_control = text.cache_control.unwrap();
                assert_eq!(cache_control.cache_type, "ephemeral");
            },
            _ => panic!("Expected Text content"),
        }
    }

    #[test]
    fn test_text_without_cache_control() {
        // Ensure text without cache_control still works
        let json = r#"{
            "type": "text",
            "text": "Hello world"
        }"#;
        
        let result: Result<Content, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let content = result.unwrap();
        match content {
            Content::Text(text) => {
                assert_eq!(text.text, "Hello world");
                assert!(text.cache_control.is_none());
            },
            _ => panic!("Expected Text content"),
        }
    }

    #[test]
    fn test_system_content_with_cache_control() {
        // RED: Test for system content with cache_control (from failed logs)
        let json = r#"[{
            "type": "text",
            "text": "Summarize this coding conversation in under 50 characters.\nCapture the main task, key files, problems addressed, and current status.",
            "cache_control": {"type": "ephemeral"}
        }]"#;
        
        let result: Result<StringOrContents, _> = serde_json::from_str(json);
        assert!(result.is_ok(), "Failed to deserialize system content with cache_control: {:?}", result.err());
        
        let content = result.unwrap();
        match content {
            StringOrContents::Contents(contents) => {
                assert_eq!(contents.len(), 1);
                match &contents[0] {
                    Content::Text(text) => {
                        assert_eq!(text.text, "Summarize this coding conversation in under 50 characters.\nCapture the main task, key files, problems addressed, and current status.");
                        assert!(text.cache_control.is_some());
                        let cache_control = text.cache_control.as_ref().unwrap();
                        assert_eq!(cache_control.cache_type, "ephemeral");
                    },
                    _ => panic!("Expected Text content"),
                }
            },
            StringOrContents::String(_) => panic!("Expected Contents variant"),
        }
    }

    #[test]
    fn test_cache_control_serialization_roundtrip() {
        // Ensure cache_control is preserved during serialization
        let original = Content::Text(Text {
            text: "Test text".to_string(),
            cache_control: Some(CacheControl {
                cache_type: "ephemeral".to_string(),
            }),
        });
        
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Content = serde_json::from_str(&json).unwrap();
        
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_real_claude_code_tool_error_scenario() {
        // RED: Test for the tool schema error we saw in logs
        // This validates we handle complex content with multiple text blocks
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "Please help me with this code:"},
                {"type": "text", "text": "fn main() {\n    println!(\"Hello, world!\");\n}"},
                {"type": "text", "text": "And analyze the following error:"},
                {"type": "text", "text": "tools.0.custom.input_schema: Field required"}
            ]
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok(), "Failed to parse multi-text content message: {:?}", result.err());
        
        let message = result.unwrap();
        match message.content {
            StringOrContents::Contents(contents) => {
                assert_eq!(contents.len(), 4);
                // Verify all text blocks are preserved
                let texts: Vec<String> = contents.iter()
                    .filter_map(|c| match c {
                        Content::Text(t) => Some(t.text.clone()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(texts.len(), 4);
                assert!(texts[3].contains("tools.0.custom.input_schema"));
            },
            StringOrContents::String(_) => panic!("Expected Contents variant for multi-text"),
        }
    }

    #[test]
    fn test_anthropic_beta_header_scenarios() {
        // Test content that might come from requests with different beta headers
        let json = r#"{
            "role": "user", 
            "content": [
                {"type": "text", "text": "Testing claude-code-20250219 features"},
                {"type": "text", "text": "With interleaved-thinking-2025-05-14"},
                {"type": "text", "text": "And fine-grained-tool-streaming-2025-05-14"}
            ]
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let message = result.unwrap();
        match message.content {
            StringOrContents::Contents(contents) => {
                assert_eq!(contents.len(), 3);
                // Verify beta feature references are preserved
                let all_text = contents.iter()
                    .filter_map(|c| match c {
                        Content::Text(t) => Some(t.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                assert!(all_text.contains("claude-code-20250219"));
                assert!(all_text.contains("interleaved-thinking-2025-05-14"));
                assert!(all_text.contains("fine-grained-tool-streaming-2025-05-14"));
            },
            StringOrContents::String(_) => panic!("Expected Contents for beta features test"),
        }
    }

    #[test]
    fn test_large_content_handling() {
        // Test handling of large content like we saw in logs (112KB+ requests)
        let large_text = "x".repeat(100000); // 100KB of text
        let json = format!(r#"{{
            "role": "user",
            "content": [
                {{"type": "text", "text": "Processing large content:"}},
                {{"type": "text", "text": "{}"}}
            ]
        }}"#, large_text);
        
        let result: Result<Message, _> = serde_json::from_str(&json);
        assert!(result.is_ok(), "Failed to parse large content message");
        
        let message = result.unwrap();
        match message.content {
            StringOrContents::Contents(contents) => {
                assert_eq!(contents.len(), 2);
                match &contents[1] {
                    Content::Text(text) => {
                        assert_eq!(text.text.len(), 100000);
                        assert_eq!(text.text, large_text);
                    },
                    _ => panic!("Expected Text content"),
                }
            },
            StringOrContents::String(_) => panic!("Expected Contents for large content"),
        }
    }

    #[test]
    fn test_mixed_string_and_content_scenarios() {
        // Test edge case: ensure we handle ordering correctly (String first in untagged)
        let string_json = r#"{"role": "user", "content": "Simple string"}"#;
        let array_json = r#"{"role": "user", "content": [{"type": "text", "text": "Array content"}]}"#;
        
        // Test string parsing
        let string_result: Result<Message, _> = serde_json::from_str(string_json);
        assert!(string_result.is_ok());
        match string_result.unwrap().content {
            StringOrContents::String(s) => assert_eq!(s, "Simple string"),
            StringOrContents::Contents(_) => panic!("Should parse as string"),
        }
        
        // Test array parsing  
        let array_result: Result<Message, _> = serde_json::from_str(array_json);
        assert!(array_result.is_ok());
        match array_result.unwrap().content {
            StringOrContents::Contents(contents) => {
                assert_eq!(contents.len(), 1);
                match &contents[0] {
                    Content::Text(text) => assert_eq!(text.text, "Array content"),
                    _ => panic!("Expected Text content"),
                }
            },
            StringOrContents::String(_) => panic!("Should parse as array"),
        }
    }

    #[test]
    fn test_basic_user_message() {
        // Basic user message with simple string content
        let json = r#"{
            "role": "user", 
            "content": "Hello, Claude"
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let message = result.unwrap();
        assert_eq!(message.role, Role::User);
        match message.content {
            StringOrContents::String(s) => assert_eq!(s, "Hello, Claude"),
            StringOrContents::Contents(_) => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_assistant_response_array_format() {
        // Assistant response in array format (typical API response)
        let json = r#"{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello! How can I help you today?"}
            ]
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let message = result.unwrap();
        assert_eq!(message.role, Role::Assistant);
        match message.content {
            StringOrContents::Contents(contents) => {
                assert_eq!(contents.len(), 1);
                match &contents[0] {
                    Content::Text(text) => assert_eq!(text.text, "Hello! How can I help you today?"),
                    _ => panic!("Expected Text content"),
                }
            },
            StringOrContents::String(_) => panic!("Expected Contents variant"),
        }
    }

    #[test]
    fn test_message_with_image() {
        // Message with image content (base64)
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                    }
                }
            ]
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let message = result.unwrap();
        match message.content {
            StringOrContents::Contents(contents) => {
                assert_eq!(contents.len(), 2);
                
                // First content: text
                match &contents[0] {
                    Content::Text(text) => assert_eq!(text.text, "What's in this image?"),
                    _ => panic!("Expected Text content"),
                }
                
                // Second content: image
                match &contents[1] {
                    Content::Image { source } => {
                        match source {
                            ImageSource::Base64 { media_type, data } => {
                                assert_eq!(media_type, "image/png");
                                assert_eq!(data, "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==");
                            }
                        }
                    },
                    _ => panic!("Expected Image content"),
                }
            },
            StringOrContents::String(_) => panic!("Expected Contents variant"),
        }
    }

    #[test]
    fn test_message_with_tool_use() {
        // Assistant message with tool use
        let json = r#"{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll help you get the weather."},
                {
                    "type": "tool_use",
                    "id": "call_1234567890",
                    "name": "get_weather",
                    "input": {
                        "location": "New York, NY",
                        "unit": "celsius"
                    }
                }
            ]
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let message = result.unwrap();
        assert_eq!(message.role, Role::Assistant);
        match message.content {
            StringOrContents::Contents(contents) => {
                assert_eq!(contents.len(), 2);
                
                // First content: text
                match &contents[0] {
                    Content::Text(text) => assert_eq!(text.text, "I'll help you get the weather."),
                    _ => panic!("Expected Text content"),
                }
                
                // Second content: tool use
                match &contents[1] {
                    Content::ToolUse(tool_use) => {
                        assert_eq!(tool_use.id, "call_1234567890");
                        assert_eq!(tool_use.name, "get_weather");
                        let expected_input = serde_json::json!({
                            "location": "New York, NY",
                            "unit": "celsius"
                        });
                        assert_eq!(tool_use.input, expected_input);
                    },
                    _ => panic!("Expected ToolUse content"),
                }
            },
            StringOrContents::String(_) => panic!("Expected Contents variant"),
        }
    }

    #[test]
    fn test_message_with_tool_result() {
        // User message with tool result
        let json = r#"{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1234567890",
                    "content": [
                        {"type": "text", "text": "The weather in New York, NY is 22°C and sunny."}
                    ]
                }
            ]
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let message = result.unwrap();
        assert_eq!(message.role, Role::User);
        match message.content {
            StringOrContents::Contents(contents) => {
                assert_eq!(contents.len(), 1);
                
                match &contents[0] {
                    Content::ToolResult(tool_result) => {
                        assert_eq!(tool_result.tool_use_id, "call_1234567890");
                        assert_eq!(tool_result.is_error, None); // Default when not specified
                        assert_eq!(tool_result.content.len(), 1);
                    },
                    _ => panic!("Expected ToolResult content"),
                }
            },
            StringOrContents::String(_) => panic!("Expected Contents variant"),
        }
    }

    #[test]
    fn test_multiple_text_blocks() {
        // Message with multiple separate text blocks
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "First paragraph."},
                {"type": "text", "text": "Second paragraph."},
                {"type": "text", "text": "Third paragraph."}
            ]
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let message = result.unwrap();
        match message.content {
            StringOrContents::Contents(contents) => {
                assert_eq!(contents.len(), 3);
                
                let expected_texts = ["First paragraph.", "Second paragraph.", "Third paragraph."];
                for (i, expected) in expected_texts.iter().enumerate() {
                    match &contents[i] {
                        Content::Text(text) => assert_eq!(text.text, *expected),
                        _ => panic!("Expected Text content at index {}", i),
                    }
                }
            },
            StringOrContents::String(_) => panic!("Expected Contents variant"),
        }
    }

    #[test]
    fn test_empty_string_content() {
        // Message with empty string content
        let json = r#"{
            "role": "user",
            "content": ""
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let message = result.unwrap();
        match message.content {
            StringOrContents::String(s) => assert_eq!(s, ""),
            StringOrContents::Contents(_) => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_empty_content_array() {
        // Message with empty content array
        let json = r#"{
            "role": "user",
            "content": []
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let message = result.unwrap();
        match message.content {
            StringOrContents::Contents(contents) => assert_eq!(contents.len(), 0),
            StringOrContents::String(_) => panic!("Expected Contents variant"),
        }
    }

    #[test]
    fn test_partial_assistant_response() {
        // Partial assistant response (like in completion scenarios)
        let json = r#"{
            "role": "assistant",
            "content": "The answer is ("
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let message = result.unwrap();
        assert_eq!(message.role, Role::Assistant);
        match message.content {
            StringOrContents::String(s) => assert_eq!(s, "The answer is ("),
            StringOrContents::Contents(_) => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_mixed_content_types() {
        // Message with text, image, and tool use together
        let json = r#"{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I can see the image you shared."},
                {
                    "type": "image", 
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg", 
                        "data": "/9j/4AAQSkZJRgABAQEAYABgAAD//2Q="
                    }
                },
                {"type": "text", "text": "Let me analyze it for you."},
                {
                    "type": "tool_use",
                    "id": "analyze_123",
                    "name": "analyze_image",
                    "input": {"mode": "detailed"}
                }
            ]
        }"#;
        
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        
        let message = result.unwrap();
        match message.content {
            StringOrContents::Contents(contents) => {
                assert_eq!(contents.len(), 4);
                
                // Verify each content type in order
                match &contents[0] {
                    Content::Text(text) => assert_eq!(text.text, "I can see the image you shared."),
                    _ => panic!("Expected Text at index 0"),
                }
                
                match &contents[1] {
                    Content::Image { source } => {
                        match source {
                            ImageSource::Base64 { media_type, data } => {
                                assert_eq!(media_type, "image/jpeg");
                                assert_eq!(data, "/9j/4AAQSkZJRgABAQEAYABgAAD//2Q=");
                            }
                        }
                    },
                    _ => panic!("Expected Image at index 1"),
                }
                
                match &contents[2] {
                    Content::Text(text) => assert_eq!(text.text, "Let me analyze it for you."),
                    _ => panic!("Expected Text at index 2"),
                }
                
                match &contents[3] {
                    Content::ToolUse(tool_use) => {
                        assert_eq!(tool_use.id, "analyze_123");
                        assert_eq!(tool_use.name, "analyze_image");
                        let expected = serde_json::json!({"mode": "detailed"});
                        assert_eq!(tool_use.input, expected);
                    },
                    _ => panic!("Expected ToolUse at index 3"),
                }
            },
            StringOrContents::String(_) => panic!("Expected Contents variant"),
        }
    }

    #[test]
    fn test_serialization_preserves_format() {
        // Test that serialization preserves the original format
        let original_string = Message {
            role: Role::User,
            content: StringOrContents::String("Hello".to_string()),
        };
        
        let json_string = serde_json::to_string(&original_string).unwrap();
        let deserialized_string: Message = serde_json::from_str(&json_string).unwrap();
        
        assert_eq!(original_string, deserialized_string);
        
        let original_array = Message {
            role: Role::Assistant,
            content: StringOrContents::Contents(vec![
                Content::Text(Text::new("Hello".to_string()))
            ]),
        };
        
        let json_array = serde_json::to_string(&original_array).unwrap();
        let deserialized_array: Message = serde_json::from_str(&json_array).unwrap();
        
        assert_eq!(original_array, deserialized_array);
    }
}