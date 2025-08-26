//! OpenAI AI API client for Rust
//!
//! This crate provides a Rust client for the OpenAI AI API, with support for:
//! - Chat completions
//! - Streaming responses
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
//!         .user("Hello, world!")
//!         .build();
//!
//!     let response = client.send(&request).await?;
//!     println!("{}", response.content());
//!
//!     Ok(())
//! }
//! ```

pub mod client;
pub mod error;
pub mod message;
pub mod model;
pub mod request;
pub mod response;
pub mod tool;
pub mod usage;

// Re-export main types
pub use client::OpenAI;
pub use error::OpenAIRequestError;
pub use message::Message;
pub use model::Model;
pub use request::ChatRequest;
pub use response::ChatResponse;
pub use tool::Tool;
pub use usage::Usage;