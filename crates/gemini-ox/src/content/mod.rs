//! Core content types for the Gemini API
//!
//! This module contains the fundamental content types that are used across
//! all parts of the Gemini API including content generation, live API,
//! embeddings, and function calling.

pub mod mime_types;
pub mod part;
pub mod types;

// Re-export the main types for convenient access
pub use part::{
    Blob, CodeExecutionResult, ExecutableCode, FileData, FunctionCall, FunctionResponse,
    FunctionResponseScheduling, Language, Outcome, Part, PartData, Text, VideoMetadata,
};
pub use types::{Content, ContentError, Role};

// Re-export MIME types
pub use mime_types::*;
