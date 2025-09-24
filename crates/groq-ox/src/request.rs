// Groq request types using base OpenAI format from ai-ox-common
// This demonstrates Grug's approach: inherit base types, add provider extensions

use bon::Builder;
use serde::Serialize;

// Import base OpenAI-format types and shared response format
use ai_ox_common::{
    openai_format::{Message, Tool, ToolChoice},
    response_format::ResponseFormat,
};

/// Groq chat request - uses base OpenAI format with Groq-specific extensions
///
/// This replaces the old 113-line ChatRequest with one that reuses base types.
/// Demonstrates ~50% reduction in code by sharing common OpenAI structures.
#[derive(Debug, Clone, Serialize, Builder)]
#[builder(builder_type(vis = "pub"), state_mod(vis = "pub"))]
pub struct ChatRequest {
    // Core OpenAI-format fields (using shared base types from ai-ox-common)
    #[builder(field)]
    pub messages: Vec<Message>,
    #[builder(into)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    // Groq-specific extensions beyond base OpenAI format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    // Groq uses max_completion_tokens instead of max_tokens (provider quirk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
}

impl ChatRequest {
    /// Create a simple chat request with model and messages
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            tools: None,
            tool_choice: None,
            frequency_penalty: None,
            presence_penalty: None,
            response_format: None,
            seed: None,
            user: None,
            max_completion_tokens: None,
        }
    }

    /// Create a chat request with JSON response format
    pub fn with_json_response(model: impl Into<String>, messages: Vec<Message>) -> Self {
        let mut request = Self::new(model, messages);
        request.response_format = Some(ResponseFormat::JsonObject);
        request
    }
}

// Builder extension methods
impl<S: chat_request_builder::State> ChatRequestBuilder<S> {
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.messages = messages.into_iter().collect();
        self
    }

    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    pub fn user_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::user(content));
        self
    }

    pub fn system_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::system(content));
        self
    }
}
