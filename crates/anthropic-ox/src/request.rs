use bon::Builder;
use serde::{Deserialize, Serialize};
use crate::{
    message::{Message, Messages, StringOrContents},
    tool::{Tool, ToolChoice},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub config_type: String,
    pub budget_tokens: u32,
}

impl ThinkingConfig {
    /// Create a new thinking configuration with the specified token budget
    /// Minimum budget is 1024 tokens as per Anthropic API requirements
    pub fn new(budget_tokens: u32) -> Self {
        Self {
            config_type: "enabled".to_string(),
            budget_tokens: budget_tokens.max(1024), // Ensure minimum budget
        }
    }
    
    /// Create thinking config with default budget (1024 tokens)
    pub fn enabled() -> Self {
        Self::new(1024)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(builder_type(vis = "pub"), state_mod(vis = "pub"))]
pub struct ChatRequest {
    #[builder(field)]
    pub messages: Messages,
    #[builder(into)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<StringOrContents>,
    #[builder(default = 4096)]
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

impl<S: chat_request_builder::State> ChatRequestBuilder<S> {
    pub fn messages(mut self, messages: impl IntoIterator<Item = impl Into<Message>>) -> Self {
        self.messages = messages.into_iter().map(Into::into).collect();
        self
    }
    
    pub fn message(mut self, message: impl Into<Message>) -> Self {
        self.messages.push(message.into());
        self
    }
    
    // Note: Tool choice helpers would need complex generic bounds with bon
    // For now, users can call .tool_choice(Some(ToolChoice::Auto)) directly
}

impl ChatRequest {
    pub fn push_message(&mut self, message: impl Into<Message>) {
        self.messages.push(message.into());
    }
    
    /// Enable streaming for this request
    pub fn streaming(mut self) -> Self {
        self.stream = Some(true);
        self
    }
    
    /// Set temperature for response randomness (0.0 to 1.0)
    pub fn temp(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
    
    /// Set top_p for nucleus sampling (0.0 to 1.0)
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }
    
    /// Set top_k for top-k sampling
    pub fn top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }
    
    /// Add stop sequences to halt generation
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }
    
    /// Add a single stop sequence
    pub fn stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.stop_sequences.get_or_insert_with(Vec::new).push(sequence.into());
        self
    }
    
    /// Enable thinking with the specified token budget
    pub fn with_thinking(mut self, budget_tokens: u32) -> Self {
        self.thinking = Some(ThinkingConfig::new(budget_tokens));
        self
    }
    
    /// Enable thinking with default budget (1024 tokens)
    pub fn enable_thinking(mut self) -> Self {
        self.thinking = Some(ThinkingConfig::enabled());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    
    #[test]
    fn test_thinking_config_creation() {
        let config = ThinkingConfig::new(2048);
        assert_eq!(config.config_type, "enabled");
        assert_eq!(config.budget_tokens, 2048);
        
        let default_config = ThinkingConfig::enabled();
        assert_eq!(default_config.config_type, "enabled");
        assert_eq!(default_config.budget_tokens, 1024);
    }
    
    #[test]
    fn test_thinking_config_minimum_budget() {
        // Test that budget is enforced to minimum 1024
        let config = ThinkingConfig::new(512);
        assert_eq!(config.budget_tokens, 1024);
        
        let config = ThinkingConfig::new(0);
        assert_eq!(config.budget_tokens, 1024);
        
        let config = ThinkingConfig::new(2048);
        assert_eq!(config.budget_tokens, 2048);
    }
    
    #[test]
    fn test_thinking_config_serialization() {
        let config = ThinkingConfig::new(4096);
        let json = serde_json::to_string(&config).unwrap();
        
        let expected = r#"{"type":"enabled","budget_tokens":4096}"#;
        assert_eq!(json, expected);
        
        let deserialized: ThinkingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }
    
    #[test]
    fn test_chat_request_with_thinking() {
        let request = ChatRequest::builder()
            .model("claude-3-sonnet")
            .messages(vec![Message::from("Hello")])
            .thinking(ThinkingConfig::new(2048))
            .build();
            
        assert!(request.thinking.is_some());
        let thinking = request.thinking.unwrap();
        assert_eq!(thinking.config_type, "enabled");
        assert_eq!(thinking.budget_tokens, 2048);
    }
    
    #[test]
    fn test_chat_request_enable_thinking() {
        let mut request = ChatRequest::builder()
            .model("claude-3-sonnet")
            .messages(vec![Message::from("Hello")])
            .build();
            
        request = request.enable_thinking();
            
        assert!(request.thinking.is_some());
        let thinking = request.thinking.unwrap();
        assert_eq!(thinking.config_type, "enabled");
        assert_eq!(thinking.budget_tokens, 1024);
    }
    
    #[test]
    fn test_chat_request_thinking_serialization() {
        let mut request = ChatRequest::builder()
            .model("claude-3-sonnet")
            .messages(vec![Message::from("Test message")])
            .build();
            
        request = request.with_thinking(3072);
            
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"thinking\":"));
        assert!(json.contains("\"budget_tokens\":3072"));
        
        let deserialized: ChatRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(request.thinking, deserialized.thinking);
    }
    
    #[test]
    fn test_chat_request_without_thinking() {
        let request = ChatRequest::builder()
            .model("claude-3-sonnet")
            .messages(vec![Message::from("Hello")])
            .build();
            
        assert!(request.thinking.is_none());
        
        let json = serde_json::to_string(&request).unwrap();
        assert!(!json.contains("thinking"), "Thinking should be omitted when None");
    }
    
    #[test]
    fn test_thinking_config_equality() {
        let config1 = ThinkingConfig::new(2048);
        let config2 = ThinkingConfig::new(2048);
        let config3 = ThinkingConfig::new(1024);
        
        assert_eq!(config1, config2);
        assert_ne!(config1, config3);
    }
}