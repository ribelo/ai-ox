#![cfg_attr(not(test), deny(unsafe_code))]
#![warn(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::missing_docs_in_private_items
)]

pub mod audio;
pub mod client;
pub mod content;
pub mod error;
#[doc(hidden)]
pub mod internal;
pub mod message;
pub mod model;
pub mod request;
pub mod response;
pub mod tool;
pub mod usage;

// Re-export main types
pub use client::Mistral;
pub use error::MistralRequestError;
pub use model::Model;
pub use request::ChatRequest;
pub use response::ChatResponse;
