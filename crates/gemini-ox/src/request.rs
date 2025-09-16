//! Central module for all request types across Gemini-ox
//!
//! This module provides a unified interface to all request types used by the
//! Gemini API, allowing users to import from a single location:
//!
//! ```no_run
//! use gemini_ox::request::{GenerateContentRequest, EmbedContentRequest};
//! ```

// Re-export all request types from their respective modules
pub use crate::cache::CreateCachedContentRequest;
pub use crate::embedding::request::{EmbedContentRequest, TaskType};
pub use crate::generate_content::request::GenerateContentRequest;

// Re-export related configuration types that are commonly used with requests
pub use crate::generate_content::{
    GenerationConfig, HarmBlockThreshold, HarmCategory, SafetySetting, SafetySettings,
};
