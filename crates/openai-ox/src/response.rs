use crate::{Message, Usage};
use serde::{Deserialize, Serialize};

/// Response from chat completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Unique identifier for the response
    pub id: String,

    /// Object type (usually "chat.completion")
    pub object: String,

    /// Unix timestamp of creation
    pub created: u64,

    /// Model used for the completion
    pub model: String,

    /// List of completion choices
    pub choices: Vec<Choice>,

    /// Usage statistics
    pub usage: Option<Usage>,

    /// System fingerprint
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// A completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    /// Index of this choice
    pub index: u32,

    /// The completion message
    pub message: Message,

    /// Reason for stopping
    pub finish_reason: Option<String>,

    /// Log probabilities (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

/// Streaming choice delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceDelta {
    /// Index of this choice
    pub index: u32,

    /// The partial message delta
    pub delta: MessageDelta,

    /// Reason for stopping
    pub finish_reason: Option<String>,

    /// Log probabilities (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

/// Partial message for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDelta {
    /// Message role
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// Partial content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Tool calls (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

impl ChatResponse {
    /// Get the content of the first choice, if available
    pub fn content(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|choice| choice.message.content.as_deref())
    }

    /// Get the first choice, if available
    pub fn first_choice(&self) -> Option<&Choice> {
        self.choices.first()
    }

    /// Check if the response is finished
    pub fn is_finished(&self) -> bool {
        self.choices
            .first()
            .map(|choice| choice.finish_reason.is_some())
            .unwrap_or(false)
    }

    /// Get the finish reason of the first choice
    pub fn finish_reason(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|choice| choice.finish_reason.as_deref())
    }
}

/// Response from models list endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// Response from embeddings endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

/// Embedding data item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: u32,
}

/// Usage information for embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

impl EmbeddingsUsage {
    pub fn prompt_tokens(&self) -> u32 {
        self.prompt_tokens
    }

    pub fn total_tokens(&self) -> u32 {
        self.total_tokens
    }
}

/// Response from moderation endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationResponse {
    pub id: String,
    pub model: String,
    pub results: Vec<ModerationResult>,
}

/// Moderation result for a single input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationResult {
    pub flagged: bool,
    pub categories: ModerationCategories,
    pub category_scores: ModerationCategoryScores,
}

/// Moderation categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationCategories {
    pub sexual: bool,
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: bool,
    pub harassment: bool,
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: bool,
    pub hate: bool,
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: bool,
    #[serde(rename = "self-harm")]
    pub self_harm: bool,
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: bool,
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: bool,
    pub violence: bool,
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: bool,
}

/// Moderation category scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationCategoryScores {
    pub sexual: f64,
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: f64,
    pub harassment: f64,
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: f64,
    pub hate: f64,
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: f64,
    #[serde(rename = "self-harm")]
    pub self_harm: f64,
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: f64,
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: f64,
    pub violence: f64,
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: f64,
}

/// Response from image generation endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
}

/// Image data item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

/// Response from audio transcription/translation endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioResponse {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<AudioSegment>>,
}

/// Audio segment (for detailed transcription)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSegment {
    pub id: u32,
    pub seek: u32,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub tokens: Vec<u32>,
    pub temperature: f64,
    pub avg_logprob: f64,
    pub compression_ratio: f64,
    pub no_speech_prob: f64,
}

/// Response from files list endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesResponse {
    pub object: String,
    pub data: Vec<FileInfo>,
}

/// File information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub id: String,
    pub object: String,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_details: Option<String>,
}

/// Response from file upload endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileUploadResponse {
    pub id: String,
    pub object: String,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String,
}

/// Response from fine-tuning jobs list endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningJobsResponse {
    pub object: String,
    pub data: Vec<FineTuningJob>,
    pub has_more: bool,
}

/// Fine-tuning job information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningJob {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub finished_at: Option<u64>,
    pub model: String,
    pub fine_tuned_model: Option<String>,
    pub organization_id: String,
    pub status: String,
    pub hyperparameters: FineTuningHyperparameters,
    pub training_file: String,
    pub validation_file: Option<String>,
    pub result_files: Vec<String>,
    pub trained_tokens: Option<u64>,
    pub error: Option<serde_json::Value>,
}

/// Fine-tuning hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningHyperparameters {
    pub n_epochs: u32,
    pub batch_size: Option<u32>,
    pub learning_rate_multiplier: Option<f64>,
}

/// Response from assistants list endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantsResponse {
    pub object: String,
    pub data: Vec<AssistantInfo>,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
    pub has_more: bool,
}

/// Assistant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantInfo {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub name: Option<String>,
    pub description: Option<String>,
    pub model: String,
    pub instructions: Option<String>,
    pub tools: Vec<serde_json::Value>,
    pub file_ids: Vec<String>,
    pub metadata: serde_json::Value,
}
