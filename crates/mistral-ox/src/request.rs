use bon::Builder;
#[cfg(feature = "schema")]
use schemars::{generate::SchemaSettings, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::Messages;

// Import base OpenAI format types
use ai_ox_common::openai_format::{Message, Tool, ToolChoice};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
}

#[derive(Debug, Clone, Serialize, Builder)]
#[builder(builder_type(vis = "pub"), state_mod(vis = "pub"))]
pub struct ChatRequest {
    #[builder(field)]
    pub messages: Vec<Message>,
    #[builder(field)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[builder(into)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_prompt: Option<bool>,
}

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
    
    #[cfg(feature = "schema")]
    pub fn response_format<T: JsonSchema + DeserializeOwned>(mut self) -> Self {
        let _type_name = std::any::type_name::<T>().split("::").last().unwrap();
        let mut schema_settings = SchemaSettings::draft2020_12();
        schema_settings.inline_subschemas = true;
        let schema_generator = schema_settings.into_generator();
        let _json_schema = schema_generator.into_root_schema_for::<T>();
        let response_format = json!({
            "type": "json_object"
        });
        self.response_format = Some(response_format);
        self
    }
}

impl ChatRequest {
    pub fn push_message(&mut self, message: Message) {
        self.messages.push(message);
    }
}

/// Request for embeddings endpoint
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct EmbeddingsRequest {
    /// Model to use for embeddings
    #[builder(into)]
    pub model: String,
    
    /// Text or array of texts to embed
    #[serde(rename = "input")]
    pub input: EmbeddingInput,
    
    /// Encoding format for embeddings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
}

/// Input for embeddings (can be string or array of strings)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

/// Request for moderation endpoint
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct ModerationRequest {
    /// Text or array of texts to moderate
    #[serde(rename = "input")]
    pub input: ModerationInput,
    
    /// Model to use for moderation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Input for moderation (can be string or array of strings)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModerationInput {
    Single(String),
    Multiple(Vec<String>),
}

/// Request for chat moderation endpoint
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct ChatModerationRequest {
    /// Model to use for chat moderation
    #[builder(into)]
    pub model: String,
    
    /// Messages to moderate
    pub messages: Messages,
}

/// Request for fine-tuning job creation
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct FineTuningRequest {
    /// Model to fine-tune
    #[builder(into)]
    pub model: String,
    
    /// Training data files
    pub training_files: Vec<TrainingFile>,
    
    /// Validation files
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_files: Option<Vec<String>>,
    
    /// Hyperparameters for fine-tuning
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperparameters: Option<FineTuningHyperparameters>,
    
    /// Suffix for fine-tuned model name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    
    /// Integrations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub integrations: Option<Vec<serde_json::Value>>,
    
    /// Whether to auto-start the job
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_start: Option<bool>,
}

/// Training file reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingFile {
    pub file_id: String,
    pub weight: Option<f64>,
}

/// Fine-tuning hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct FineTuningHyperparameters {
    pub training_steps: u32,
    pub learning_rate: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_decay: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup_fraction: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epochs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fim_ratio: Option<f64>,
}

/// Request for batch job creation
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct BatchJobRequest {
    /// Input file ID containing requests
    pub input_file_id: String,
    
    /// API endpoint to use for batch processing
    #[builder(into)]
    pub endpoint: String,
    
    /// Completion window (24h)
    #[builder(into)]
    pub completion_window: String,
    
    /// Optional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Request for fill-in-the-middle completion
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct FimRequest {
    /// Model to use for completion
    #[builder(into)]
    pub model: String,
    
    /// Text before the completion
    pub prompt: String,
    
    /// Text after the completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    
    /// Sampling temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    
    /// Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    
    /// Random seed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u32>,
}

/// Request for agents completion
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct AgentsRequest {
    /// Messages for the agent
    pub messages: Messages,
    
    /// Agent ID to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    
    /// Sampling temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    
    /// Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    
    /// Random seed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u32>,
}