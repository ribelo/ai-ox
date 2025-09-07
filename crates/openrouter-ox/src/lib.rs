#![cfg_attr(not(test), deny(unsafe_code))]
#![warn(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::missing_docs_in_private_items
)]

use async_stream::try_stream;
use bon::Builder;
use futures_util::stream::BoxStream;

pub mod conversion;
pub mod error;
mod internal;
pub mod message;
pub mod models;
pub mod provider_preference;
pub mod request;
pub mod response;
pub mod router;
pub mod tool;

use crate::internal::OpenRouterRequestHelper;

const BASE_URL: &str = "https://openrouter.ai";

#[cfg(feature = "leaky-bucket")]
pub use leaky_bucket::RateLimiter;
#[cfg(feature = "leaky-bucket")]
use std::sync::Arc;
use std::{collections::HashMap, fmt};

#[derive(Clone, Default, Builder)]
pub struct OpenRouter {
    #[builder(into)]
    api_key: String,
    #[builder(default = BASE_URL.to_string(), into)]
    base_url: String,
    #[builder(default)]
    #[allow(dead_code)]
    headers: HashMap<String, String>,
    #[builder(default)]
    client: reqwest::Client,
    #[cfg(feature = "leaky-bucket")]
    #[allow(dead_code)]
    leaky_bucket: Option<Arc<RateLimiter>>,
}

impl OpenRouter {
    /// Create a new OpenRouter client with the provided API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: BASE_URL.to_string(),
            headers: HashMap::new(),
            client: reqwest::Client::new(),
            #[cfg(feature = "leaky-bucket")]
            leaky_bucket: None,
        }
    }

    pub fn load_from_env() -> Result<Self, std::env::VarError> {
        let api_key = std::env::var("OPENROUTER_API_KEY")?;
        Ok(Self::builder().api_key(api_key).build())
    }

    /// Create request helper for internal use
    fn request_helper(&self) -> OpenRouterRequestHelper {
        OpenRouterRequestHelper::new(self.client.clone(), &self.base_url, &self.api_key)
    }

    pub async fn send(
        &self,
        request: &request::ChatRequest,
    ) -> Result<response::ChatCompletionResponse, OpenRouterRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().send_chat_request(request).await
    }

    pub fn stream(
        &self,
        request: &request::ChatRequest,
    ) -> BoxStream<'static, Result<response::ChatCompletionChunk, OpenRouterRequestError>> {
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

    /// List all available models from OpenRouter
    pub async fn list_models(&self) -> Result<response::ModelsResponse, OpenRouterRequestError> {
        self.request_helper().list_models().await
    }

    /// Get detailed information about a specific generation by its ID
    pub async fn get_generation(&self, generation_id: &str) -> Result<response::GenerationInfo, OpenRouterRequestError> {
        self.request_helper().get_generation(generation_id).await
    }

    /// Check the status of the API key, including usage, limits, and credits
    pub async fn get_key_status(&self) -> Result<response::KeyStatus, OpenRouterRequestError> {
        self.request_helper().get_key_status().await
    }
}

impl fmt::Debug for OpenRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAi")
            .field("api_key", &"[REDACTED]")
            .field("client", &self.client)
            .finish()
    }
}

pub use conversion::ConversionError;
pub use error::OpenRouterRequestError;
pub use request::{ChatRequest, ReasoningConfig};
pub use response::{
    ChatCompletionResponse, ChatCompletionChunk, GenerationInfo, KeyStatus, KeyStatusData, 
    KeyRateLimit, ModelInfo, ModelsResponse, ModelPricing, ModelArchitecture, ModelProvider
};
