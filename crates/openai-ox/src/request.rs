// OpenAI request types using base OpenAI format from ai-ox-common
// This is the "canonical" OpenAI provider - should be closest to base format

use bon::Builder;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// Import base OpenAI format types  
use ai_ox_common::openai_format::{BaseMessage, BaseTool, BaseToolChoice};

/// Request for chat completion - uses base OpenAI format with OpenAI-specific extensions
/// 
/// This demonstrates the base format working for the "canonical" OpenAI provider.
/// Most fields come from base format, with OpenAI-specific extensions like `n`, `user`, etc.
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(builder_type(vis = "pub"), state_mod(vis = "pub"))]
pub struct ChatRequest {
    // Core OpenAI-format fields (using shared base types from ai-ox-common)
    #[builder(field)]
    pub messages: Vec<BaseMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(field)]
    pub tools: Option<Vec<BaseTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(field)]
    pub user: Option<String>,
    #[builder(into)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    
    // OpenAI-specific extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,  // Number of completions to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<BaseToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

/// Request for embeddings
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(builder_type(vis = "pub"), state_mod(vis = "pub"))]
pub struct EmbeddingsRequest {
    /// Input text or array of texts
    #[builder(into)]
    pub input: EmbeddingInput,
    
    /// Model to use for embeddings
    #[builder(into)]
    pub model: String,
    
    /// Format for embeddings (float, base64)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    
    /// Number of dimensions (for text-embedding-3 models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    
    /// User identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Input for embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
    Tokens(Vec<u32>),
    TokenArrays(Vec<Vec<u32>>),
}

/// Request for content moderation
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(builder_type(vis = "pub"), state_mod(vis = "pub"))]
pub struct ModerationRequest {
    /// Input text to moderate
    #[builder(into)]
    pub input: ModerationInput,
    
    /// Moderation model to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Input for moderation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModerationInput {
    Single(String),
    Multiple(Vec<String>),
}

/// Request for image generation
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct ImageRequest {
    /// Text description of desired image
    #[builder(into)]
    pub prompt: String,
    
    /// Model to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    
    /// Number of images to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    
    /// Quality of images
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    
    /// Response format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    
    /// Size of images
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    
    /// Style of images
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    
    /// User identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Request for audio processing
#[derive(Debug, Clone, Builder)]
pub struct AudioRequest {
    /// Audio file to transcribe/translate
    pub file: Vec<u8>,
    
    /// Filename of the audio file
    pub filename: String,
    
    /// Model to use
    #[builder(into)]
    pub model: String,
    
    /// Language (for transcription)
    pub language: Option<String>,
    
    /// Prompt to guide model
    pub prompt: Option<String>,
    
    /// Response format
    pub response_format: Option<String>,
    
    /// Temperature for transcription
    pub temperature: Option<f32>,
}

/// Request for fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct FineTuningRequest {
    /// Training file ID
    pub training_file: String,
    
    /// Model to fine-tune
    #[builder(into)]
    pub model: String,
    
    /// Validation file ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_file: Option<String>,
    
    /// Hyperparameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperparameters: Option<Value>,
    
    /// Model suffix
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
}

/// Request for Assistant API
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct AssistantRequest {
    /// Model to use for assistant  
    #[builder(into)]
    pub model: String,
    
    /// Name of assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    
    /// Description of assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    
    /// Instructions for assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    
    /// Tools available to assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<BaseTool>>,
    
    /// File IDs accessible to assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_ids: Option<Vec<String>>,
    
    /// Metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

// Builder extension methods (same pattern as other providers)
impl<S: chat_request_builder::State> ChatRequestBuilder<S> {
    pub fn messages(mut self, messages: impl IntoIterator<Item = BaseMessage>) -> Self {
        self.messages = messages.into_iter().collect();
        self
    }
    
    pub fn message(mut self, message: BaseMessage) -> Self {
        self.messages.push(message);
        self
    }
    
    pub fn user_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(BaseMessage::user(content));
        self
    }
    
    pub fn system_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(BaseMessage::system(content));
        self
    }
}