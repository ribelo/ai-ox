use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::{Message, Tool};

/// Request for chat completion
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(builder_type(vis = "pub"), state_mod(vis = "pub"))]
pub struct ChatRequest {
    /// List of messages in the conversation
    #[builder(field)]
    pub messages: Vec<Message>,

    /// Tools available to the model
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(field)]
    pub tools: Option<Vec<Tool>>,

    /// User identifier for abuse monitoring
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(field)]
    pub user: Option<String>,

    /// The model to use for completion
    #[builder(into)]
    pub model: String,

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

    /// Tool choice preference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,

    /// Response format (for structured output)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,

    /// Random seed for deterministic output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
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
impl<S: chat_request_builder::State> ChatRequestBuilder<S> {
    /// Add a user message
    pub fn user_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::user(content));
        self
    }

    /// Add an assistant message
    pub fn assistant_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::assistant(content));
        self
    }

    /// Add a system message
    pub fn system_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::system(content));
        self
    }

    /// Add a message
    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Add a tool
    pub fn tool(mut self, tool: Tool) -> Self {
        if self.tools.is_none() {
            self.tools = Some(Vec::new());
        }
        self.tools.as_mut().unwrap().push(tool);
        self
    }
}

/// Request for embeddings endpoint
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct EmbeddingsRequest {
    /// Text or array of texts to embed
    #[serde(rename = "input")]
    pub input: EmbeddingInput,
    
    /// Model to use for embeddings
    pub model: String,
    
    /// Encoding format for embeddings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    
    /// User identifier for abuse monitoring
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
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
    
    /// Model to use for moderation (defaults to text-moderation-latest)
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

/// Request for image generation
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct ImageRequest {
    /// Text description of the desired image
    pub prompt: String,
    
    /// Model to use for image generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    
    /// Number of images to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    
    /// Image quality (standard or hd)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    
    /// Response format (url or b64_json)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    
    /// Image size
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    
    /// Image style (vivid or natural)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    
    /// User identifier for abuse monitoring
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Request for audio transcription/translation
#[derive(Debug, Clone, Builder)]
pub struct AudioRequest {
    /// Audio file to transcribe/translate
    pub file: Vec<u8>,
    
    /// Filename of the audio file
    pub filename: String,
    
    /// Model to use (whisper-1)
    pub model: String,
    
    /// Language of the input audio (for transcription)
    pub language: Option<String>,
    
    /// Optional text prompt to guide the model's style
    pub prompt: Option<String>,
    
    /// Response format (json, text, srt, verbose_json, vtt)
    pub response_format: Option<String>,
    
    /// Sampling temperature
    pub temperature: Option<f32>,
}

/// Request for fine-tuning job creation
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct FineTuningRequest {
    /// Model to fine-tune
    pub model: String,
    
    /// Training data file ID
    pub training_file: String,
    
    /// Validation data file ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_file: Option<String>,
    
    /// Hyperparameters for fine-tuning
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperparameters: Option<serde_json::Value>,
    
    /// Suffix for fine-tuned model name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
}

/// Request for assistant creation
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct AssistantRequest {
    /// Model to use
    pub model: String,
    
    /// Name of the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    
    /// Description of the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    
    /// System instructions for the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    
    /// Tools available to the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    
    /// File IDs available to the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_ids: Option<Vec<String>>,
    
    /// Metadata for the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}