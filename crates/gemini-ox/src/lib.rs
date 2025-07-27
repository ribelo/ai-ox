// pub mod agent;
pub mod cache;
pub mod content;
// pub mod files;
pub mod embedding;
pub mod generate_content;
pub mod live;
pub mod model;
pub mod tokens;
pub mod tool;

// Re-export types from modules
pub use crate::model::response::{ListModelsResponse, Model as ApiModel};

// Re-export the procedural macro from gemini-ox-macros if the 'macros' feature is enabled.
// #[cfg(feature = "macros")]
// pub use gemini_ox_macros::toolbox;
// Re-export async_trait if the 'macros' feature is enabled, as the toolbox macro uses it.
// #[cfg(feature = "macros")]
// pub use async_trait::async_trait;

use core::fmt;
// use std::sync::Arc; // Unused import removed - ensure this line is gone

use bon::Builder;
#[cfg(feature = "leaky-bucket")] // Add cfg attribute here
use leaky_bucket::RateLimiter;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, Serializer, ser::SerializeStruct};
use serde_json::Value;
#[cfg(feature = "leaky-bucket")] // Add cfg attribute here
use std::sync::Arc;
use thiserror::Error;

const BASE_URL: &str = "https://generativelanguage.googleapis.com";

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    PartialOrd,
    strum::EnumString,
    strum::Display,
    strum::IntoStaticStr,
)]
pub enum Model {
    // --- Gemini 1.5 Series ---

    // Gemini 1.5 Flash 8B
    #[strum(to_string = "gemini-1.5-flash-8b-001")]
    Gemini15Flash8b001, // Stable
    #[strum(to_string = "gemini-1.5-flash-8b")]
    Gemini15Flash8b, // Stable Alias
    #[strum(to_string = "gemini-1.5-flash-8b-latest")]
    Gemini15Flash8bLatest, // Stable Latest Alias

    // Gemini 1.5 Flash (Standard)
    #[strum(to_string = "gemini-1.5-flash-001")]
    Gemini15Flash001, // Stable
    #[strum(to_string = "gemini-1.5-flash-002")]
    Gemini15Flash002, // Stable
    #[strum(to_string = "gemini-1.5-flash")]
    Gemini15Flash, // Stable Alias
    #[strum(to_string = "gemini-1.5-flash-latest")]
    Gemini15FlashLatest, // Stable Latest Alias
    #[strum(to_string = "gemini-1.5-flash-001-tuning")]
    Gemini15Flash001Tuning, // Tuning Version

    // Gemini 1.5 Pro
    #[strum(to_string = "gemini-1.5-pro-001")]
    Gemini15Pro001, // Stable
    #[strum(to_string = "gemini-1.5-pro-002")]
    Gemini15Pro002, // Stable
    #[strum(to_string = "gemini-1.5-pro")]
    Gemini15Pro, // Stable Alias
    #[strum(to_string = "gemini-1.5-pro-latest")]
    Gemini15ProLatest, // Stable Latest Alias

    // --- Gemini 2.0 / 2.5 Series ---

    // Gemini 2.0 Flash Lite
    #[strum(to_string = "gemini-2.0-flash-lite-001")]
    Gemini20FlashLite001, // Stable
    #[strum(to_string = "gemini-2.0-flash-lite")]
    Gemini20FlashLite, // Stable Alias
    #[strum(to_string = "gemini-2.0-flash-lite-preview-02-05")]
    Gemini20FlashLitePreview0205, // Preview
    #[strum(to_string = "gemini-2.0-flash-lite-preview")]
    Gemini20FlashLitePreview, // Preview Alias

    // Gemini 2.0 Flash (Standard)
    #[strum(to_string = "gemini-2.0-flash-001")]
    Gemini20Flash001, // Stable
    #[strum(to_string = "gemini-2.0-flash")]
    Gemini20Flash, // Stable Alias
    #[strum(to_string = "gemini-2.0-flash-live-001")]
    Gemini20FlashLive001, // For bidiGenerateContent?
    #[strum(to_string = "gemini-2.0-flash-exp")]
    Gemini20FlashExp, // Experimental

    // Gemini 2.0 Flash Thinking (Experimental)
    #[strum(to_string = "gemini-2.0-flash-thinking-exp-01-21")]
    Gemini20FlashThinkingExp0121,
    #[strum(to_string = "gemini-2.0-flash-thinking-exp")] // Alias for exp-01-21
    Gemini20FlashThinkingExp,
    #[strum(to_string = "gemini-2.0-flash-thinking-exp-1219")]
    Gemini20FlashThinkingExp1219, // Older experimental?

    // Gemini 2.0 / 2.5 Pro (Experimental/Preview)
    // Note: Several names point to the same underlying experimental release (2.5-exp-03-25)
    #[strum(to_string = "gemini-2.5-pro-exp-03-25")]
    Gemini25ProExp0325,
    #[strum(to_string = "gemini-2.5-pro-preview-03-25")]
    Gemini25ProPreview0325,
    #[strum(to_string = "gemini-2.0-pro-exp")] // Alias for 2.5-exp-03-25
    Gemini20ProExp,
    #[strum(to_string = "gemini-2.0-pro-exp-02-05")] // Alias for 2.5-exp-03-25
    Gemini20ProExp0205,
    #[strum(to_string = "gemini-exp-1206")] // Alias for 2.5-exp-03-25
    GeminiExp1206,

    // Gemini 2.5 Flash (Stable and Preview)
    #[strum(to_string = "gemini-2.5-flash")]
    Gemini25Flash, // Stable
    #[strum(to_string = "gemini-2.5-flash-lite")]
    Gemini25FlashLite, // Stable Lite
    #[strum(to_string = "gemini-2.5-flash-lite-preview-06-17")]
    Gemini25FlashLitePreview0617, // Preview
    #[strum(to_string = "gemini-2.5-flash-preview-05-20")]
    Gemini25FlashPreview0520,
    #[strum(to_string = "gemini-2.5-flash-preview-tts")]
    Gemini25FlashPreviewTts, // TTS Preview
    #[strum(to_string = "gemini-2.5-flash-preview-native-audio-dialog")]
    Gemini25FlashPreviewNativeAudioDialog,
    
    // Gemini 2.5 Pro (Stable and Preview)
    #[strum(to_string = "gemini-2.5-pro")]
    Gemini25Pro, // Stable
    #[strum(to_string = "gemini-2.5-pro-preview-05-06")]
    Gemini25ProPreview0506, // Preview
    #[strum(to_string = "gemini-2.5-pro-preview-06-05")]
    Gemini25ProPreview0605, // Preview
    #[strum(to_string = "gemini-2.5-pro-preview-tts")]
    Gemini25ProPreviewTts, // TTS Preview

    // --- Gemma 3 Series ---
    #[strum(to_string = "gemma-3-1b-it")]
    Gemma3_1bIt,
    #[strum(to_string = "gemma-3-4b-it")]
    Gemma3_4bIt,
    #[strum(to_string = "gemma-3-12b-it")]
    Gemma3_12bIt,
    #[strum(to_string = "gemma-3-27b-it")]
    Gemma3_27bIt,
    #[strum(to_string = "gemma-3n-e2b-it")]
    Gemma3nE2bIt,
    #[strum(to_string = "gemma-3n-e4b-it")]
    Gemma3nE4bIt,

    // --- Embedding Models ---
    #[strum(to_string = "embedding-001")]
    Embedding001,
    #[strum(to_string = "text-embedding-004")]
    TextEmbedding004,
    #[strum(to_string = "gemini-embedding-001")]
    GeminiEmbedding001,
    #[strum(to_string = "gemini-embedding-exp-03-07")] // Experimental
    GeminiEmbeddingExp0307,
    #[strum(to_string = "gemini-embedding-exp")] // Alias for exp-03-07
    GeminiEmbeddingExp,

    // --- Specialized Models ---

    // LearnLM Models (Experimental)
    #[strum(to_string = "learnlm-1.5-pro-experimental")]
    LearnLm15ProExperimental,
    #[strum(to_string = "learnlm-2.0-flash-experimental")]
    LearnLm20FlashExperimental,

    // Attributed Question Answering Model
    #[strum(to_string = "aqa")]
    Aqa,
    // --- Legacy / Deprecated Models ---
    // (Removed from active list, kept for reference if needed)
    // #[strum(to_string = "embedding-gecko-001")]
    // EmbeddingGecko001,
    // #[strum(to_string = "gemini-1.0-pro-vision-latest")]
    // Gemini10ProVisionLatest, // Deprecated
    // #[strum(to_string = "gemini-pro-vision")]
    // GeminiProVision, // Deprecated alias
    // #[strum(to_string = "chat-bison-001")]
    // ChatBison001, // Legacy PaLM 2
    // #[strum(to_string = "text-bison-001")]
    // TextBison001, // Legacy PaLM 2
}

impl From<Model> for String {
    fn from(model: Model) -> Self {
        model.to_string()
    }
}

#[derive(Clone, Default, Builder)]
pub struct Gemini {
    #[builder(into)]
    pub(crate) api_key: String,
    #[builder(default)]
    pub(crate) client: reqwest::Client,
    #[cfg(feature = "leaky-bucket")]
    pub(crate) leaky_bucket: Option<Arc<RateLimiter>>,
    #[builder(default = "v1beta".to_string(), into)]
    pub(crate) api_version: String,
}

impl Gemini {
    /// Create a new Gemini client with the provided API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
            #[cfg(feature = "leaky-bucket")]
            leaky_bucket: None,
            api_version: "v1beta".to_string(),
        }
    }

    pub fn load_from_env() -> Result<Self, std::env::VarError> {
        let api_key =
            std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_AI_API_KEY"))?;
        Ok(Self::builder().api_key(api_key).build())
    }

    /// Create a Live API session builder
    ///
    /// Returns a `LiveOperationBuilder` that can be configured and then used to
    /// establish a WebSocket connection to the Gemini Live API.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use gemini_ox::{Gemini, Model};
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let gemini = Gemini::new("your-api-key");
    /// let session = gemini.live_session()
    ///     .model(Model::Gemini20FlashLive001)
    ///     .build()
    ///     .connect()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn live_session(
        &self,
    ) -> crate::live::live_operation::LiveOperationBuilder<
        crate::live::live_operation::live_operation_builder::SetGemini,
    > {
        crate::live::LiveOperation::builder().gemini(self.clone())
    }

    /// Returns a handler for caching operations.
    pub fn caches(&self) -> crate::cache::Caches {
        crate::cache::Caches::new(self.clone())
    }
}

impl fmt::Debug for Gemini {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gemini")
            .field("api_key", &"[REDACTED]")
            .field("client", &self.client)
            .field("api_version", &self.api_version)
            .finish_non_exhaustive()
    }
}

pub struct ResponseSchema;

impl ResponseSchema {
    #[must_use]
    pub fn from<T: JsonSchema>() -> Value {
        let settings = schemars::generate::SchemaSettings::openapi3().with(|s| {
            s.inline_subschemas = true;
            s.meta_schema = None;
        });
        let r#gen = schemars::generate::SchemaGenerator::new(settings);
        let root_schema = r#gen.into_root_schema_for::<T>();
        let mut json_schema = serde_json::to_value(root_schema).unwrap();

        json_schema
            .as_object_mut()
            .unwrap()
            .remove("title")
            .unwrap();

        json_schema
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleApiErrorPayload {
    error: GoogleApiErrorDetails,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleApiErrorDetails {
    code: Option<i32>,
    message: String,
    status: Option<String>,
    details: Option<Value>,
}

#[derive(Debug, Error)]
pub enum GeminiRequestError {
    /// Errors from the HTTP client
    #[error(transparent)]
    ReqwestError(#[from] reqwest::Error),

    /// JSON serialization/deserialization errors
    #[error(transparent)]
    SerdeError(#[from] serde_json::Error),

    /// JSON deserialization errors with context
    #[error("Failed to deserialize JSON: {0}")]
    JsonDeserializationError(serde_json::Error),

    /// Invalid request errors from the API
    #[error("Invalid request error: {message}")]
    InvalidRequestError {
        code: Option<String>,
        details: serde_json::Value,
        message: String,
        status: Option<String>,
    },

    /// Unexpected response from the API
    #[error("Unexpected response from API: {0}")]
    UnexpectedResponse(String),

    /// Invalid event data in stream
    #[error("Invalid event data: {0}")]
    InvalidEventData(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimit,

    /// URL building error
    #[error("URL build failed: {0}")]
    UrlBuildError(String),

    /// I/O errors
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

impl Serialize for GeminiRequestError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            GeminiRequestError::ReqwestError(e) => {
                let mut state = serializer.serialize_struct("GeminiRequestError", 2)?;
                state.serialize_field("type", "ReqwestError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            GeminiRequestError::SerdeError(e) => {
                let mut state = serializer.serialize_struct("GeminiRequestError", 2)?;
                state.serialize_field("type", "SerdeError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            GeminiRequestError::JsonDeserializationError(e) => {
                let mut state = serializer.serialize_struct("GeminiRequestError", 2)?;
                state.serialize_field("type", "JsonDeserializationError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            GeminiRequestError::InvalidRequestError {
                code,
                details,
                message,
                status,
            } => {
                let field_count = 1 // type
                    + if code.is_some() { 1 } else { 0 }
                    + 1 // details
                    + 1 // message
                    + if status.is_some() { 1 } else { 0 };
                let mut state = serializer.serialize_struct("GeminiRequestError", field_count)?;
                state.serialize_field("type", "InvalidRequestError")?;
                if let Some(c) = code {
                    state.serialize_field("code", c)?;
                }
                state.serialize_field("details", details)?;
                state.serialize_field("message", message)?;
                if let Some(s) = status {
                    state.serialize_field("status", s)?;
                }
                state.end()
            }
            GeminiRequestError::UnexpectedResponse(response) => {
                let mut state = serializer.serialize_struct("GeminiRequestError", 2)?;
                state.serialize_field("type", "UnexpectedResponse")?;
                state.serialize_field("response", response)?;
                state.end()
            }
            GeminiRequestError::InvalidEventData(message) => {
                let mut state = serializer.serialize_struct("GeminiRequestError", 2)?;
                state.serialize_field("type", "InvalidEventData")?;
                state.serialize_field("message", message)?;
                state.end()
            }
            GeminiRequestError::RateLimit => {
                let mut state = serializer.serialize_struct("GeminiRequestError", 1)?;
                state.serialize_field("type", "RateLimit")?;
                state.end()
            }
            GeminiRequestError::UrlBuildError(message) => {
                let mut state = serializer.serialize_struct("GeminiRequestError", 2)?;
                state.serialize_field("type", "UrlBuildError")?;
                state.serialize_field("message", message)?;
                state.end()
            }
            GeminiRequestError::IoError(e) => {
                let mut state = serializer.serialize_struct("GeminiRequestError", 2)?;
                state.serialize_field("type", "IoError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
        }
    }
}

/// Parse an error response from the Google API.
/// This function handles both JSON format errors and plain text errors.
fn parse_error_response(status: reqwest::StatusCode, bytes: bytes::Bytes) -> GeminiRequestError {
    // Try to parse as a structured Google API error first
    if let Ok(payload) = serde_json::from_slice::<GoogleApiErrorPayload>(&bytes) {
        GeminiRequestError::InvalidRequestError {
            code: payload.error.code.map(|c| c.to_string()),
            message: payload.error.message,
            status: payload.error.status,
            details: payload.error.details.unwrap_or(Value::Null),
        }
    } else {
        // Fall back to text
        let error_text = String::from_utf8_lossy(&bytes).to_string();
        GeminiRequestError::UnexpectedResponse(format!(
            "HTTP status {}: {}",
            status.as_u16(),
            error_text
        ))
    }
}
