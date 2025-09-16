use crate::{ChatRequest, ChatResponse, GroqRequestError, response::ChatCompletionChunk};
use ai_ox_common::{
    BoxStream,
    error::ProviderError,
    request_builder::{AuthMethod, Endpoint, HttpMethod, RequestBuilder, RequestConfig},
};
use futures_util::stream::BoxStream as FuturesBoxStream;

/// Groq client helper methods using the common RequestBuilder
pub struct GroqRequestHelper {
    request_builder: RequestBuilder,
}

impl GroqRequestHelper {
    pub fn new(client: reqwest::Client, base_url: &str, api_key: &str) -> Self {
        let config = RequestConfig::new(base_url)
            .with_auth(AuthMethod::Bearer(api_key.to_string()))
            .with_header("content-type", "application/json");

        let request_builder = RequestBuilder::new(client, config);

        Self { request_builder }
    }

    /// Send a chat completion request
    pub async fn send_chat_request(
        &self,
        request: &ChatRequest,
    ) -> Result<ChatResponse, GroqRequestError> {
        let endpoint = Endpoint::new("openai/v1/chat/completions", HttpMethod::Post);

        Ok(self
            .request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }

    /// Stream a chat completion request
    pub fn stream_chat_request(
        &self,
        request: &ChatRequest,
    ) -> FuturesBoxStream<'static, Result<ChatCompletionChunk, GroqRequestError>> {
        let endpoint = Endpoint::new("openai/v1/chat/completions", HttpMethod::Post);

        // Use the common streaming implementation (no conversion needed - same type)
        let stream: BoxStream<'static, Result<ChatCompletionChunk, ProviderError>> =
            self.request_builder.stream(&endpoint, Some(request));

        // Direct cast since GroqRequestError = ProviderError
        stream
    }
}
