// OpenRouter request types using base OpenAI format from ai-ox-common
// Most complex provider - extensive OpenAI base + many OpenRouter-specific extensions

use bon::Builder;
use schemars::{JsonSchema, generate::SchemaSettings};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

// Import base OpenAI format types for tools only
use ai_ox_common::openai_format::{Tool, ToolChoice};

use crate::{message::Message, provider_preference::ProviderPreferences};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub r#type: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>, // "high", "medium", "low"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

/// OpenRouter chat request - uses base OpenAI format with extensive OpenRouter extensions
///
/// This is the most complex provider - demonstrates the pattern can handle extensive extensions
/// while still sharing the core OpenAI format types.
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(builder_type(vis = "pub"), state_mod(vis = "pub"))]
pub struct ChatRequest {
    // Core OpenAI-format fields (using shared base types from ai-ox-common)
    #[builder(field)]
    pub messages: Vec<Message>,
    #[builder(field)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[builder(into)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>, // OpenRouter uses f64 for temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>, // OpenRouter uses f64 for top_p
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    // OpenRouter-specific extensions (LOTS of them!)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_a: Option<f64>,

    // OpenRouter routing and provider selection features
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<Prediction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transforms: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub models: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub preset: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub provider: Option<ProviderPreferences>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
}

// Builder extension methods (same pattern as Groq/Mistral)
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

    pub fn response_format<T: JsonSchema + DeserializeOwned>(mut self) -> Self {
        let type_name = std::any::type_name::<T>().split("::").last().unwrap();
        let mut schema_settings = SchemaSettings::draft2020_12();
        schema_settings.inline_subschemas = true;
        let schema_generator = schema_settings.into_generator();
        let json_schema = schema_generator.into_root_schema_for::<T>();
        let response_format = json!({
            "type": "json_schema",
            "json_schema": {"name": type_name, "schema": json_schema, "strict": true},
        });
        self.response_format = Some(response_format);
        self
    }
}

impl ChatRequest {
    /// Create a simple chat request with model and messages
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            stream: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            seed: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            logit_bias: None,
            top_logprobs: None,
            min_p: None,
            top_a: None,
            prediction: None,
            transforms: None,
            models: None,
            route: None,
            preset: None,
            provider: None,
            reasoning: None,
        }
    }

    /// Enable OpenRouter's reasoning inclusion feature
    pub fn with_reasoning(mut self) -> Self {
        self.reasoning = Some(ReasoningConfig {
            enabled: Some(true),
            effort: None,
            max_tokens: None,
            exclude: None,
        });
        self
    }

    /// Enable OpenRouter's reasoning with specific effort level
    pub fn with_reasoning_effort(mut self, effort: &str) -> Self {
        self.reasoning = Some(ReasoningConfig {
            enabled: Some(true),
            effort: Some(effort.to_string()),
            max_tokens: None,
            exclude: None,
        });
        self
    }

    /// Enable OpenRouter's reasoning with specific token limit
    pub fn with_reasoning_tokens(mut self, max_tokens: u32) -> Self {
        self.reasoning = Some(ReasoningConfig {
            enabled: Some(true),
            effort: None,
            max_tokens: Some(max_tokens),
            exclude: None,
        });
        self
    }

    /// Set specific provider preferences for OpenRouter
    pub fn with_provider(mut self, provider: ProviderPreferences) -> Self {
        self.provider = Some(provider);
        self
    }
}
