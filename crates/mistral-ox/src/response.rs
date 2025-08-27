use serde::{Deserialize, Serialize};

use crate::message::AssistantMessage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: AssistantMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

impl ChatCompletionChunk {
    /// Parse streaming data and extract ChatCompletionChunk objects
    pub fn from_streaming_data(data: &str) -> Vec<Result<Self, crate::error::MistralRequestError>> {
        let mut chunks = Vec::new();
        
        for line in data.lines() {
            let line = line.trim();
            
            // Skip empty lines
            if line.is_empty() {
                continue;
            }
            
            // Handle SSE format (data: prefix)
            if let Some(json_str) = line.strip_prefix("data: ") {
                let json_str = json_str.trim();
                
                // Skip [DONE] marker
                if json_str == "[DONE]" {
                    continue;
                }
                
                // Try to parse the JSON
                match serde_json::from_str::<ChatCompletionChunk>(json_str) {
                    Ok(chunk) => chunks.push(Ok(chunk)),
                    Err(e) => chunks.push(Err(crate::error::MistralRequestError::JsonDeserializationError(e))),
                }
            }
        }
        
        chunks
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aliases: Option<Vec<String>>,
}

/// Response from embeddings endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub id: String,
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
    pub categories: ModerationCategories,
    pub category_scores: ModerationCategoryScores,
}

/// Moderation categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationCategories {
    pub sexual: bool,
    pub hate_and_discrimination: bool,
    pub violence_and_threats: bool,
    pub dangerous_or_criminal_content: bool,
    pub selfharm: bool,
    pub health: bool,
    pub financial: bool,
    pub law: bool,
    pub pii: bool,
}

/// Moderation category scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationCategoryScores {
    pub sexual: f64,
    pub hate_and_discrimination: f64,
    pub violence_and_threats: f64,
    pub dangerous_or_criminal_content: f64,
    pub selfharm: f64,
    pub health: f64,
    pub financial: f64,
    pub law: f64,
    pub pii: f64,
}

/// Response from fine-tuning jobs list endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningJobsResponse {
    pub data: Vec<FineTuningJob>,
    pub object: String,
    pub total: u32,
}

/// Fine-tuning job information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningJob {
    pub id: String,
    pub hyperparameters: FineTuningHyperparameters,
    pub model: String,
    pub status: String,
    pub job_type: String,
    pub created_at: u64,
    pub modified_at: u64,
    pub training_files: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_files: Option<Vec<String>>,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fine_tuned_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    pub integrations: Vec<serde_json::Value>,
    pub trained_tokens: Option<u64>,
    pub repositories: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_start: Option<bool>,
}

/// Fine-tuning hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Response from batch jobs list endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJobsResponse {
    pub data: Vec<BatchJob>,
    pub object: String,
    pub total: u32,
}

/// Batch job information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJob {
    pub id: String,
    pub object: String,
    pub endpoint: String,
    pub input_file_id: String,
    pub completion_window: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_file_id: Option<String>,
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_progress_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finalizing_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expired_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancelling_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancelled_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_counts: Option<BatchRequestCounts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Batch request counts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequestCounts {
    pub total: u32,
    pub completed: u32,
    pub failed: u32,
}

/// Response from files list endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesResponse {
    pub data: Vec<FileInfo>,
    pub object: String,
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
    pub sample_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_lines: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
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

/// Response from file delete endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDeleteResponse {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}