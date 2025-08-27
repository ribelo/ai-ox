#![cfg_attr(not(test), deny(unsafe_code))]
#![warn(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::missing_docs_in_private_items
)]

pub mod client;
pub mod error;
#[doc(hidden)]
pub mod internal;
pub mod message;
pub mod model;
#[cfg(feature = "models")]
pub mod models;
#[cfg(feature = "batches")]
pub mod batches;
#[cfg(feature = "files")]
pub mod files;
#[cfg(feature = "admin")]
pub mod admin;
pub mod prelude;
pub mod request;
pub mod response;
pub mod tool;
#[cfg(feature = "tokens")]
pub mod tokens;
pub mod usage;

// Re-export main types
pub use client::Anthropic;
pub use error::AnthropicRequestError;
pub use model::Model;
pub use request::ChatRequest;
pub use response::{ChatResponse, StreamEvent};
