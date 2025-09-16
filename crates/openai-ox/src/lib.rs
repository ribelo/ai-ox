//! OpenAI AI API client for Rust
//!
//! This crate provides a Rust client for the OpenAI AI API, with support for:
//! - Chat completions and streaming
//! - Model listing
//! - Text embeddings
//! - Content moderation  
//! - Image generation (DALL-E)
//! - Audio transcription/translation (Whisper)
//! - File management
//! - Fine-tuning jobs
//! - Assistants API
//! - Tool/function calling
//! - Error handling and rate limiting
//!
//! # Example
//!
//! ```rust,no_run
//! use openai_ox::OpenAI;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = OpenAI::new("your-api-key");
//!
//!     let request = client
//!         .chat()
//!         .model("gpt-3.5-turbo")
//!         .user_message("Hello, world!")
//!         .build();
//!
//!     let response = client.send(&request).await?;
//!     println!("{}", response.content().unwrap_or("No content"));
//!
//!     Ok(())
//! }
//! ```

pub mod client;
pub mod error;
mod internal;
pub mod model;
pub mod request;
pub mod response;
pub mod responses;
pub mod usage;

// Re-export main types
pub use client::OpenAI;
pub use error::OpenAIRequestError;
pub use model::Model;
pub use usage::Usage;

// Re-export shared types from ai-ox-common
pub use ai_ox_common::openai_format::{Message, Tool, ToolCall, ToolChoice};

// Re-export request types
pub use request::{
    AssistantRequest, AudioRequest, ChatRequest, EmbeddingInput, EmbeddingsRequest,
    FineTuningRequest, ImageRequest, ModerationInput, ModerationRequest,
};

// Re-export response types
pub use response::{
    AssistantInfo, AssistantsResponse, AudioResponse, AudioSegment, ChatResponse, EmbeddingData,
    EmbeddingsResponse, FileInfo, FileUploadResponse, FilesResponse, FineTuningJob,
    FineTuningJobsResponse, ImageData, ImageResponse, ModelInfo, ModelsResponse,
    ModerationResponse, ModerationResult,
};

// Re-export Responses API types
pub use responses::{
    Conversation, IncompleteDetails, OutputDelta, ReasoningConfig, ReasoningItem, ResponseError,
    ResponseMessage, ResponseOutputContent, ResponseOutputItem, ResponsesInput, ResponsesRequest,
    ResponsesRequestBuilder, ResponsesResponse, ResponsesStreamChunk, ResponsesUsage, TextConfig,
    ToolCallItem,
};
