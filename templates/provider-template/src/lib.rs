//! {{Provider}} AI API client for Rust
//!
//! This crate provides a Rust client for the {{Provider}} AI API, with support for:
//! - Chat completions
//! - Streaming responses
//! - Tool/function calling
//! - Error handling and rate limiting
//!
//! # Example
//!
//! ```rust,no_run
//! use {{provider}}_ox::{{Provider}};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = {{Provider}}::new("your-api-key");
//!     
//!     let request = client
//!         .chat()
//!         .model("{{default_model}}")
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
pub use client::{{Provider}};
pub use error::{{Provider}}RequestError;
pub use message::Message;
pub use model::Model;
pub use request::ChatRequest;
pub use response::ChatResponse;
pub use tool::Tool;
pub use usage::Usage;