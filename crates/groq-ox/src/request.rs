use bon::Builder;
use serde::{Serialize, Deserialize};
use serde_json::Value;

use crate::{
    message::{Message, Messages},
    tool::{Tool, ToolChoice},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    #[serde(untagged)]
    JsonSchema { 
        r#type: String,
        json_schema: Value 
    },
}

#[derive(Debug, Clone, Serialize, Builder)]
#[builder(builder_type(vis = "pub"), state_mod(vis = "pub"))]
pub struct ChatRequest {
    #[builder(field)]
    pub messages: Messages,
    #[builder(into)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
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
}

impl ChatRequest {
    pub fn push_message(&mut self, message: impl Into<Message>) {
        self.messages.push(message.into());
    }

    /// Create a new ChatRequest with JSON object response format
    pub fn new_with_json_object(model: impl Into<String>, messages: Messages) -> Self {
        Self {
            model: model.into(),
            messages,
            response_format: Some(ResponseFormat::JsonObject),
            temperature: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: None,
            stop: None,
            tools: None,
            tool_choice: None,
            seed: None,
            user: None,
        }
    }

    /// Create a new ChatRequest with JSON schema response format
    pub fn new_with_json_schema(model: impl Into<String>, messages: Messages, schema: Value) -> Self {
        Self {
            model: model.into(),
            messages,
            response_format: Some(ResponseFormat::JsonSchema {
                r#type: "json_schema".to_string(),
                json_schema: schema,
            }),
            temperature: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: None,
            stop: None,
            tools: None,
            tool_choice: None,
            seed: None,
            user: None,
        }
    }
}