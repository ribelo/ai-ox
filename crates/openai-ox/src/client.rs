use bon::Builder;
use std::time::Duration;

use crate::{OpenAIRequestError, ChatRequest, ChatResponse};

/// OpenAI AI API client
#[derive(Debug, Clone, Builder)]
pub struct OpenAI {
    /// API key for authentication
    api_key: String,

    /// Base URL for the API (allows for custom endpoints)
    #[builder(default = "https://api.openai.com/v1".to_string(), into)]
    pub base_url: String,

    /// HTTP client for making requests
    #[builder(skip)]
    client: reqwest::Client,

    /// Rate limiter (optional)
    #[cfg(feature = "leaky-bucket")]
    #[builder(skip)]
    rate_limiter: Option<leaky_bucket::RateLimiter>,
}

impl OpenAI {
    /// Create a new OpenAI client with the given API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            client,
            #[cfg(feature = "leaky-bucket")]
            rate_limiter: None,
        }
    }

    /// Create a new OpenAI client from environment variable
    pub fn from_env() -> Result<Self, OpenAIRequestError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| OpenAIRequestError::MissingApiKey)?;
        Ok(Self::new(api_key))
    }

    /// Create a chat request builder
    pub fn chat(&self) -> crate::request::ChatRequestBuilder {
        ChatRequest::builder()
    }

    /// Send a chat request and get a response
    pub async fn send(&self, request: &ChatRequest) -> Result<ChatResponse, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(request)
            .send()
            .await?;

        if response.status().is_success() {
            Ok(response.json::<ChatResponse>().await?)
        } else {
            let status = response.status();
            let bytes = response.bytes().await?;
            Err(crate::error::parse_error_response(status, bytes))
        }
    }

    /// Send a chat request and get a streaming response
    pub fn stream(
        &self,
        request: &ChatRequest,
    ) -> futures_util::stream::BoxStream<'static, Result<ChatResponse, OpenAIRequestError>> {
        use async_stream::try_stream;
        use futures_util::StreamExt;
        
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let url = format!("{}/chat/completions", self.base_url);
        let mut request_data = request.clone();
        request_data.stream = Some(true);

        #[cfg(feature = "leaky-bucket")]
        let rate_limiter = self.rate_limiter.clone();

        Box::pin(try_stream! {
            #[cfg(feature = "leaky-bucket")]
            if let Some(ref limiter) = rate_limiter {
                limiter.acquire_one().await;
            }

            let response = client
                .post(&url)
                .bearer_auth(&api_key)
                .json(&request_data)
                .send()
                .await?;

            let status = response.status();

            if !response.status().is_success() {
                let bytes = response.bytes().await?;
                Err(crate::error::parse_error_response(status, bytes))?
            } else {
                let mut byte_stream = response.bytes_stream();

                while let Some(chunk_result) = byte_stream.next().await {
                    let chunk = chunk_result?;
                    let chunk_str = String::from_utf8(chunk.to_vec())
                        .map_err(|e| OpenAIRequestError::InvalidEventData(format!("UTF-8 decode error: {e}")))?;

                    for line in chunk_str.lines() {
                        if line.starts_with("data: ") {
                            let data = &line[6..];
                            if data == "[DONE]" {
                                return;
                            }

                            match serde_json::from_str::<ChatResponse>(data) {
                                Ok(response) => yield response,
                                Err(e) => Err(OpenAIRequestError::SerdeError(e))?,
                            }
                        }
                    }
                }
            }
        })
    }
}

#[cfg(feature = "leaky-bucket")]
impl OpenAI {
    /// Set rate limiter
    pub fn with_rate_limiter(mut self, rate_limiter: leaky_bucket::RateLimiter) -> Self {
        self.rate_limiter = Some(rate_limiter);
        self
    }
}