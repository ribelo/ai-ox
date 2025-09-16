//! Common imports for working with the Anthropic API.
//!
//! This module re-exports the most commonly used types and traits.
//!
//! ```rust,no_run
//! use anthropic_ox::prelude::*;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = Anthropic::new("your-api-key");
//! let request = ChatRequest::builder()
//!     .model(Model::Claude35Haiku20241022)
//!     .messages(vec![Message::from("Hello!")])
//!     .temperature(0.7)
//!     .build();
//!
//! let response = client.send(&request).await?;
//! # Ok(())
//! # }
//! ```

pub use crate::{
    Anthropic, AnthropicRequestError, ChatRequest, ChatResponse, Model,
    message::{Content, ImageSource, Message, Messages, Role, Text},
    tool::{Tool, ToolChoice, ToolResult, ToolResultContent, ToolUse},
    usage::Usage,
};
