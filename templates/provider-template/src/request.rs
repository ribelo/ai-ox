use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::{Message, Model, Tool, Usage};

/// Request for chat completion
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct ChatRequest {
    /// The model to use for completion
    #[builder(into)]
    pub model: String,
    
    /// List of messages in the conversation
    #[builder(default)]
    pub messages: Vec<Message>,
    
    /// Maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    
    /// Sampling temperature (0.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    
    /// Top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    
    /// Number of completions to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    
    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    
    /// Presence penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    
    /// Frequency penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    
    /// Tools available to the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    
    /// Tool choice preference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    
    /// Response format (for structured output)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    
    /// Random seed for deterministic output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    
    /// User identifier for abuse monitoring
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl ChatRequest {
    /// Create a new chat request with the given model
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            seed: None,
            user: None,
        }
    }
}

// Builder extensions for convenience methods
impl<S: ChatRequestBuilderState> ChatRequestBuilder<S> {
    /// Add a user message
    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.get_or_insert_default().push(Message::user(content));
        self
    }
    
    /// Add an assistant message
    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.messages.get_or_insert_default().push(Message::assistant(content));
        self
    }
    
    /// Add a system message
    pub fn system(mut self, content: impl Into<String>) -> Self {
        self.messages.get_or_insert_default().push(Message::system(content));
        self
    }
    
    /// Add a message
    pub fn message(mut self, message: Message) -> Self {
        self.messages.get_or_insert_default().push(message);
        self
    }
    
    /// Add multiple messages
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.messages.get_or_insert_default().extend(messages);
        self
    }
    
    /// Add a tool
    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.get_or_insert_default().push(tool);
        self
    }
    
    /// Add multiple tools
    pub fn tools_list(mut self, tools: impl IntoIterator<Item = Tool>) -> Self {
        self.tools.get_or_insert_default().extend(tools);
        self
    }
}