pub mod audio;
pub mod content;
pub mod error;
pub mod message;
pub mod model;
pub mod request;
pub mod response;
pub mod tool;
pub mod usage;

// Re-export main types
pub use audio::{TranscriptionRequest, TranscriptionResponse};
pub use error::MistralRequestError;
pub use model::Model;
pub use request::ChatRequest;
pub use response::{ChatResponse, ChatCompletionChunk};

use bon::Builder;
use core::fmt;
use futures_util::stream::BoxStream;
#[cfg(feature = "leaky-bucket")]
use leaky_bucket::RateLimiter;
#[cfg(feature = "leaky-bucket")]
use std::sync::Arc;

const BASE_URL: &str = "https://api.mistral.ai";
const API_URL: &str = "v1/chat/completions";

#[derive(Clone, Default, Builder)]
pub struct Mistral {
    #[builder(into)]
    pub(crate) api_key: String,
    #[builder(default)]
    pub(crate) client: reqwest::Client,
    #[cfg(feature = "leaky-bucket")]
    pub(crate) leaky_bucket: Option<Arc<RateLimiter>>,
    #[builder(default = BASE_URL.to_string(), into)]
    pub(crate) base_url: String,
}

impl Mistral {
    /// Create a new Mistral client with the provided API key.
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
        let api_key = std::env::var("MISTRAL_API_KEY")?;
        Ok(Self::builder().api_key(api_key).build())
    }
}

impl Mistral {
    pub async fn send(
        &self,
        request: &request::ChatRequest,
    ) -> Result<response::ChatResponse, MistralRequestError> {
        let url = format!("{}/{}", self.base_url, API_URL);

        let res = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
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
    ) -> BoxStream<'static, Result<response::ChatCompletionChunk, MistralRequestError>> {
        use async_stream::try_stream;
        use futures_util::StreamExt;
        
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let url = format!("{}/{}", self.base_url, API_URL);
        let mut request_data = request.clone();
        request_data.stream = Some(true);

        Box::pin(try_stream! {
            let response = client
                .post(&url)
                .bearer_auth(&api_key)
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
                        .map_err(|e| MistralRequestError::InvalidEventData(format!("UTF-8 decode error: {e}")))?;

                    for parse_result in response::ChatCompletionChunk::from_streaming_data(&chunk_str) {
                        yield parse_result?;
                    }
                }
            }
        })
    }
}

impl fmt::Debug for Mistral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mistral")
            .field("api_key", &"[REDACTED]")
            .field("client", &self.client)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}