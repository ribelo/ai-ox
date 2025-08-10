use bon::Builder;
use std::time::Duration;

use crate::{{{Provider}}RequestError, ChatRequest, ChatResponse};

/// {{Provider}} AI API client
#[derive(Debug, Clone, Builder)]
pub struct {{Provider}} {
    /// API key for authentication
    api_key: String,
    
    /// Base URL for the API (allows for custom endpoints)
    #[builder(default = "{{base_url}}".to_string(), into)]
    pub base_url: String,
    
    /// HTTP client for making requests
    #[builder(skip)]
    client: reqwest::Client,
    
    /// Rate limiter (optional)
    #[cfg(feature = "leaky-bucket")]
    #[builder(skip)]
    rate_limiter: Option<leaky_bucket::RateLimiter>,
}

impl {{Provider}} {
    /// Create a new {{Provider}} client with the given API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            api_key: api_key.into(),
            base_url: "{{base_url}}".to_string(),
            client,
            #[cfg(feature = "leaky-bucket")]
            rate_limiter: None,
        }
    }

    /// Create a new {{Provider}} client from environment variable
    pub fn from_env() -> Result<Self, {{Provider}}RequestError> {
        let api_key = std::env::var("{{ENV_VAR}}")
            .map_err(|_| {{Provider}}RequestError::MissingApiKey)?;
        Ok(Self::new(api_key))
    }

    /// Create a chat request builder
    pub fn chat(&self) -> ChatRequestBuilder {
        ChatRequest::builder()
    }

    /// Send a chat request and get a response
    pub async fn send(&self, request: &ChatRequest) -> Result<ChatResponse, {{Provider}}RequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        let url = format!("{}/{{chat_endpoint}}", self.base_url);
        
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
    pub async fn stream(
        &self,
        request: &ChatRequest,
    ) -> Result<impl futures_util::Stream<Item = Result<ChatResponse, {{Provider}}RequestError>>, {{Provider}}RequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        let url = format!("{}/{{chat_endpoint}}", self.base_url);
        
        let mut streaming_request = request.clone();
        streaming_request.stream = Some(true);

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&streaming_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let bytes = response.bytes().await?;
            return Err(crate::error::parse_error_response(status, bytes));
        }

        Ok(async_stream::stream! {
            use futures_util::StreamExt;
            
            let mut lines = response.bytes_stream().lines();
            
            while let Some(line_result) = lines.next().await {
                match line_result {
                    Ok(line) => {
                        if line.starts_with("data: ") {
                            let data = &line[6..];
                            if data == "[DONE]" {
                                break;
                            }
                            
                            match serde_json::from_str::<ChatResponse>(data) {
                                Ok(response) => yield Ok(response),
                                Err(e) => yield Err({{Provider}}RequestError::SerdeError(e)),
                            }
                        }
                    }
                    Err(e) => yield Err({{Provider}}RequestError::InvalidEventData(e.to_string())),
                }
            }
        })
    }
}

#[cfg(feature = "leaky-bucket")]
impl {{Provider}} {
    /// Set rate limiter
    pub fn with_rate_limiter(mut self, rate_limiter: leaky_bucket::RateLimiter) -> Self {
        self.rate_limiter = Some(rate_limiter);
        self
    }
}