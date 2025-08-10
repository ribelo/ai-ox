#![cfg_attr(not(test), deny(unsafe_code))]
#![warn(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::missing_docs_in_private_items
)]

pub mod error;
pub mod message;
pub mod model;
pub mod prelude;
pub mod request;
pub mod response;
pub mod tool;
pub mod usage;

// Re-export main types
pub use error::AnthropicRequestError;
pub use model::Model;
pub use request::ChatRequest;
pub use response::{ChatResponse, StreamEvent};

use bon::Builder;
use core::fmt;
use futures_util::stream::BoxStream;
#[cfg(feature = "leaky-bucket")]
use leaky_bucket::RateLimiter;
#[cfg(feature = "leaky-bucket")]
use std::sync::Arc;

const BASE_URL: &str = "https://api.anthropic.com";
const CHAT_URL: &str = "v1/messages";
const API_VERSION: &str = "2023-06-01";

#[derive(Clone, Default, Builder)]
pub struct Anthropic {
    #[builder(into)]
    pub(crate) api_key: String,
    #[builder(default)]
    pub(crate) client: reqwest::Client,
    #[cfg(feature = "leaky-bucket")]
    pub(crate) leaky_bucket: Option<Arc<RateLimiter>>,
    #[builder(default = BASE_URL.to_string(), into)]
    pub(crate) base_url: String,
    #[builder(default = API_VERSION.to_string(), into)]
    pub(crate) api_version: String,
}

impl Anthropic {
    /// Create a new Anthropic client with the provided API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
            #[cfg(feature = "leaky-bucket")]
            leaky_bucket: None,
            base_url: BASE_URL.to_string(),
            api_version: API_VERSION.to_string(),
        }
    }

    pub fn load_from_env() -> Result<Self, std::env::VarError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")?;
        Ok(Self::builder().api_key(api_key).build())
    }
}

impl Anthropic {
    pub async fn send(
        &self,
        request: &request::ChatRequest,
    ) -> Result<response::ChatResponse, AnthropicRequestError> {
        let url = format!("{}/{}", self.base_url, CHAT_URL);

        let res = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header("content-type", "application/json")
            .json(request)
            .send()
            .await?;

        if res.status().is_success() {
            Ok(res.json::<response::ChatResponse>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    pub fn stream(
        &self,
        request: &request::ChatRequest,
    ) -> BoxStream<'static, Result<response::StreamEvent, AnthropicRequestError>> {
        use async_stream::try_stream;
        use futures_util::StreamExt;
        
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let api_version = self.api_version.clone();
        let url = format!("{}/{}", self.base_url, CHAT_URL);
        let mut request_data = request.clone();
        request_data.stream = Some(true);

        Box::pin(try_stream! {
            let response = client
                .post(&url)
                .header("x-api-key", &api_key)
                .header("anthropic-version", &api_version)
                .header("content-type", "application/json")
                .json(&request_data)
                .send()
                .await?;

            let status = response.status();

            if !response.status().is_success() {
                let bytes = response.bytes().await?;
                Err(error::parse_error_response(status, bytes))?
            } else {
                let mut byte_stream = response.bytes_stream();

                while let Some(chunk_result) = byte_stream.next().await {
                    let chunk = chunk_result?;
                    let chunk_str = String::from_utf8(chunk.to_vec())
                        .map_err(|e| AnthropicRequestError::InvalidEventData(format!("UTF-8 decode error: {e}")))?;

                    for line in chunk_str.lines() {
                        if line.starts_with("data: ") {
                            let json_data = line.trim_start_matches("data: ");
                            if json_data != "[DONE]" {
                                let event: response::StreamEvent = serde_json::from_str(json_data)?;
                                yield event;
                            }
                        }
                    }
                }
            }
        })
    }
}

impl fmt::Debug for Anthropic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Anthropic")
            .field("api_key", &"[REDACTED]")
            .field("client", &self.client)
            .field("base_url", &self.base_url)
            .field("api_version", &self.api_version)
            .finish_non_exhaustive()
    }
}