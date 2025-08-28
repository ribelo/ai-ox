#![cfg_attr(not(test), deny(unsafe_code))]
#![warn(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::missing_docs_in_private_items
)]

//! Shared HTTP client abstractions for AI provider clients
//!
//! This crate provides common patterns and utilities used across all ai-ox provider clients
//! to eliminate code duplication and ensure consistent behavior.

pub mod error;
pub mod openai_format;
pub mod request_builder;
pub mod streaming;

pub use error::CommonRequestError;
pub use openai_format::*;
pub use request_builder::{Endpoint, HttpMethod, RequestBuilder, MultipartForm};
pub use streaming::SseParser;

/// Re-export common types for convenience
pub use async_trait::async_trait;
pub use futures_util::stream::BoxStream;
pub use serde::{Deserialize, Serialize};