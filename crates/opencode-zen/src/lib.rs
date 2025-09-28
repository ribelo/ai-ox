#![cfg_attr(not(test), deny(unsafe_code))]
#![warn(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::missing_docs_in_private_items
)]

use ai_ox_common::{
    BoxStream,
    openai_format::{ChatCompletionChunk, ChatCompletionResponse, ChatRequest, ChatRequestBuilder},
};
#[cfg(feature = "leaky-bucket")]
use futures_util::StreamExt;

pub mod error;
mod internal;

use crate::{error::OpencodeZenError, internal::OpencodeRequestHelper};

const DEFAULT_BASE_URL: &str = "https://opencode.ai";

#[derive(Clone)]
pub struct OpencodeZen {
    api_key: Option<String>,
    base_url: String,
    client: reqwest::Client,
    #[cfg(feature = "leaky-bucket")]
    rate_limiter: Option<std::sync::Arc<leaky_bucket::RateLimiter>>,
}

impl Default for OpencodeZen {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::new(),
            #[cfg(feature = "leaky-bucket")]
            rate_limiter: None,
        }
    }
}

impl OpencodeZen {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn from_env() -> Result<Self, OpencodeZenError> {
        match std::env::var("OPENCODE_API_KEY") {
            Ok(value) => Ok(Self::default().with_api_key(value)),
            Err(std::env::VarError::NotPresent) => Ok(Self::default()),
            Err(_) => Err(OpencodeZenError::AuthenticationMissing),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    #[cfg(feature = "leaky-bucket")]
    pub fn with_rate_limiter(
        mut self,
        rate_limiter: std::sync::Arc<leaky_bucket::RateLimiter>,
    ) -> Self {
        self.rate_limiter = Some(rate_limiter);
        self
    }

    fn request_helper(&self) -> OpencodeRequestHelper {
        OpencodeRequestHelper::new(self.client.clone(), &self.base_url, self.api_key.as_deref())
    }

    pub fn chat(&self) -> ChatRequestBuilder {
        ChatRequest::builder()
    }

    pub async fn send(
        &self,
        request: &ChatRequest,
    ) -> Result<ChatCompletionResponse, OpencodeZenError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(limiter) = &self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().send_chat_request(request).await
    }

    pub fn stream(
        &self,
        request: &ChatRequest,
    ) -> BoxStream<'static, Result<ChatCompletionChunk, OpencodeZenError>> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(limiter) = &self.rate_limiter {
            let limiter = limiter.clone();
            let helper = self.request_helper();
            let payload = request.clone();
            return Box::pin(async_stream::try_stream! {
                limiter.acquire_one().await;
                let mut stream = helper.stream_chat_request(&payload);
                while let Some(chunk) = stream.next().await {
                    yield chunk?;
                }
            });
        }

        self.request_helper().stream_chat_request(request)
    }
}

pub use ai_ox_common::openai_format::{
    ChatCompletionChunk as StreamChunk, ChatCompletionResponse as ChatResponse, Message,
    MessageRole, Tool, ToolCall, ToolChoice,
};

pub use error::ProviderError;

pub type ChatResult = Result<ChatCompletionResponse, OpencodeZenError>;
