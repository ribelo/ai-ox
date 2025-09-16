use std::time::Duration;

use crate::{
    ChatRequest, ChatResponse, MistralRequestError, audio::TranscriptionRequest,
    audio::TranscriptionResponse, internal::MistralRequestHelper, response::ChatCompletionChunk,
};
use futures_util::stream::BoxStream;

/// Mistral AI API client
#[derive(Debug)]
pub struct Mistral {
    /// API key for authentication
    pub api_key: String,

    /// Base URL for the API (allows for custom endpoints)
    pub base_url: String,

    /// HTTP client for making requests
    pub client: reqwest::Client,

    /// Request helper
    pub helper: MistralRequestHelper,
}

impl Mistral {
    /// Create a new Mistral client with the given API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let base_url = "https://api.mistral.ai".to_string();

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        let helper = MistralRequestHelper::new(client.clone(), &base_url, &api_key);

        Self {
            api_key,
            base_url,
            client,
            helper,
        }
    }

    /// Create a new Mistral client with custom base URL
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let base_url = base_url.into();

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        let helper = MistralRequestHelper::new(client.clone(), &base_url, &api_key);

        Self {
            api_key,
            base_url,
            client,
            helper,
        }
    }

    /// Get API key (for testing)
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Send a chat completion request
    pub async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, MistralRequestError> {
        self.helper.send_chat_request(request).await
    }

    /// Stream a chat completion request
    pub fn stream_chat(
        &self,
        request: &ChatRequest,
    ) -> BoxStream<'static, Result<ChatCompletionChunk, MistralRequestError>> {
        self.helper.stream_chat_request(request)
    }

    /// Send a transcription request
    pub async fn transcribe(
        &self,
        request: &TranscriptionRequest,
    ) -> Result<TranscriptionResponse, MistralRequestError> {
        self.helper.send_transcription_request(request).await
    }

    /// Send a chat request (alias for chat method for compatibility)
    pub async fn send(&self, request: &ChatRequest) -> Result<ChatResponse, MistralRequestError> {
        self.chat(request).await
    }
}

impl Clone for Mistral {
    fn clone(&self) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        let helper = MistralRequestHelper::new(client.clone(), &self.base_url, &self.api_key);

        Self {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            client,
            helper,
        }
    }
}

impl Default for Mistral {
    fn default() -> Self {
        Self::new("")
    }
}

impl Mistral {
    /// Stream a chat request (alias for stream_chat method for compatibility)
    pub fn stream(
        &self,
        request: &ChatRequest,
    ) -> BoxStream<'static, Result<ChatCompletionChunk, MistralRequestError>> {
        self.stream_chat(request)
    }
}
