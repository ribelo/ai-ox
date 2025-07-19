use async_stream::try_stream;
use bon::Builder;
use futures_util::stream::{BoxStream, StreamExt};

pub mod error;
pub mod message;
pub mod models;
pub mod provider_preference;
pub mod request;
pub mod response;
pub mod router;
pub mod tool;
const BASE_URL: &str = "https://openrouter.ai";
const API_URL: &str = "api/v1/chat/completions";

#[cfg(feature = "leaky-bucket")]
pub use leaky_bucket::RateLimiter;
#[cfg(feature = "leaky-bucket")]
use std::sync::Arc;
use std::{collections::HashMap, fmt};

#[derive(Clone, Default, Builder)]
pub struct OpenRouter {
    #[builder(into)]
    api_key: String,
    #[builder(default)]
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

    pub async fn send(
        &self,
        request: &request::Request,
    ) -> Result<response::ChatCompletionResponse, ApiRequestError> {
        let url = format!("{BASE_URL}/{API_URL}");

        let res = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(request)
            .send()
            .await?;

        if res.status().is_success() {
            Ok(res.json::<response::ChatCompletionResponse>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?.to_vec();
            Err(error::parse_error_response(status, bytes))
        }
    }

    pub fn stream(
        &self,
        request: &request::Request,
    ) -> BoxStream<'static, Result<response::ChatCompletionChunk, ApiRequestError>> {
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let url = format!("{BASE_URL}/{API_URL}");
        let request_data = request.clone();

        Box::pin(try_stream! {
            let mut body = serde_json::to_value(&request_data)?;
            body.as_object_mut()
                .expect("Request body must be a JSON object")
                .insert("stream".to_string(), serde_json::Value::Bool(true));

            let response = client
                .post(&url)
                .bearer_auth(&api_key)
                .json(&body)
                .send()
                .await?;

            let status = response.status();

            if !response.status().is_success() {
                let bytes = response.bytes().await?.to_vec();
                Err(error::parse_error_response(status, bytes))?
            } else {
                let mut byte_stream = response.bytes_stream();

                while let Some(chunk_result) = byte_stream.next().await {
                    let chunk = chunk_result?;
                    let chunk_str = String::from_utf8(chunk.to_vec())
                        .map_err(|e| ApiRequestError::Stream(format!("UTF-8 decode error: {e}")))?;

                    for parse_result in response::ChatCompletionChunk::from_streaming_data(&chunk_str) {
                        yield parse_result?;
                    }
                }
            }
        })
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

pub use error::ApiRequestError;
