#![cfg_attr(not(test), deny(unsafe_code))]
#![warn(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::missing_docs_in_private_items
)]

pub mod audio;
pub mod error;
mod internal;
pub mod model;
pub mod models;
pub mod request;
pub mod response;
pub mod usage;

// Re-export main types
pub use error::GroqRequestError;
pub use model::Model;
pub use models::response::{ListModelsResponse, ModelInfo};
pub use request::ChatRequest;
pub use response::{ChatResponse, ChatCompletionChunk};
pub use usage::Usage;

// Re-export types from ai-ox-common for convenience
pub use ai_ox_common::openai_format::{Message, Tool, ToolChoice, ToolCall};

// Create a tool module with helper functions for backward compatibility
pub mod tool {
    use ai_ox_common::openai_format::{Tool, Function};
    
    pub struct ToolFunction;
    
    impl ToolFunction {
        pub fn with_parameters(name: &str, description: &str, parameters: serde_json::Value) -> Tool {
            Tool {
                r#type: "function".to_string(),
                function: Function {
                    name: name.to_string(),
                    description: Some(description.to_string()),
                    parameters: Some(parameters),
                },
            }
        }
    }
}

use bon::Builder;
use core::fmt;
use futures_util::stream::BoxStream;
#[cfg(feature = "leaky-bucket")]
use leaky_bucket::RateLimiter;
#[cfg(feature = "leaky-bucket")]
use std::sync::Arc;

use crate::internal::GroqRequestHelper;

const BASE_URL: &str = "https://api.groq.com";

#[derive(Clone, Default, Builder)]
pub struct Groq {
    #[builder(into)]
    pub(crate) api_key: String,
    #[builder(default)]
    pub(crate) client: reqwest::Client,
    #[cfg(feature = "leaky-bucket")]
    pub(crate) leaky_bucket: Option<Arc<RateLimiter>>,
    #[builder(default = BASE_URL.to_string(), into)]
    pub(crate) base_url: String,
}

impl Groq {
    /// Create a new Groq client with the provided API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
            #[cfg(feature = "leaky-bucket")]
            leaky_bucket: None,
            base_url: BASE_URL.to_string(),
        }
    }

    pub fn load_from_env() -> Result<Self, std::env::VarError> {
        let api_key = std::env::var("GROQ_API_KEY")?;
        Ok(Self::builder().api_key(api_key).build())
    }

    /// Create request helper for internal use
    fn request_helper(&self) -> GroqRequestHelper {
        GroqRequestHelper::new(self.client.clone(), &self.base_url, &self.api_key)
    }
}

impl Groq {
    pub async fn send(
        &self,
        request: &request::ChatRequest,
    ) -> Result<response::ChatResponse, GroqRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().send_chat_request(request).await
    }

    pub fn stream(
        &self,
        request: &request::ChatRequest,
    ) -> BoxStream<'static, Result<response::ChatCompletionChunk, GroqRequestError>> {
        use async_stream::try_stream;
        
        let helper = self.request_helper();
        let mut request_data = request.clone();
        request_data.stream = Some(true);

        #[cfg(feature = "leaky-bucket")]
        let rate_limiter = self.leaky_bucket.clone();

        Box::pin(try_stream! {
            #[cfg(feature = "leaky-bucket")]
            if let Some(ref limiter) = rate_limiter {
                limiter.acquire_one().await;
            }

            let mut stream = helper.stream_chat_request(&request_data);
            use futures_util::StreamExt;
            
            while let Some(result) = stream.next().await {
                yield result?;
            }
        })
    }
}

impl fmt::Debug for Groq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Groq")
            .field("api_key", &"[REDACTED]")
            .field("client", &self.client)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}