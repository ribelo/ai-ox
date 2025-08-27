use ai_ox_common::{
    request_builder::{RequestBuilder, RequestConfig, Endpoint, HttpMethod, AuthMethod},
    CommonRequestError, BoxStream
};
use futures_util::stream::BoxStream as FuturesBoxStream;
use crate::{
    OpenRouterRequestError, request::ChatRequest, 
    response::{ChatCompletionResponse, ChatCompletionChunk, ModelsResponse, GenerationInfo, KeyStatus}
};

/// Convert CommonRequestError to OpenRouterRequestError
impl From<CommonRequestError> for OpenRouterRequestError {
    fn from(err: CommonRequestError) -> Self {
        match err {
            CommonRequestError::Http(e) => OpenRouterRequestError::ReqwestError(e),
            CommonRequestError::Json(e) => OpenRouterRequestError::SerdeError(e),
            CommonRequestError::InvalidEventData(msg) => OpenRouterRequestError::InvalidEventData(msg),
            CommonRequestError::AuthenticationMissing => OpenRouterRequestError::MissingApiKey,
            CommonRequestError::InvalidMimeType(msg) => OpenRouterRequestError::InvalidEventData(msg),
            CommonRequestError::Utf8Error(e) => OpenRouterRequestError::InvalidEventData(e.to_string()),
        }
    }
}

/// OpenRouter client helper methods using the common RequestBuilder
pub struct OpenRouterRequestHelper {
    request_builder: RequestBuilder,
}

impl OpenRouterRequestHelper {
    pub fn new(client: reqwest::Client, base_url: &str, api_key: &str) -> Self {
        let config = RequestConfig::new(base_url)
            .with_auth(AuthMethod::Bearer(api_key.to_string()))
            .with_header("content-type", "application/json");

        let request_builder = RequestBuilder::new(client, config);

        Self { request_builder }
    }

    /// Send a chat completion request
    pub async fn send_chat_request(&self, request: &ChatRequest) -> Result<ChatCompletionResponse, OpenRouterRequestError> {
        let endpoint = Endpoint::new("api/v1/chat/completions", HttpMethod::Post);
        
        Ok(self.request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }

    /// Stream a chat completion request
    pub fn stream_chat_request(
        &self, 
        request: &ChatRequest
    ) -> FuturesBoxStream<'static, Result<ChatCompletionChunk, OpenRouterRequestError>> {
        let endpoint = Endpoint::new("api/v1/chat/completions", HttpMethod::Post);
        
        // Use the common streaming implementation and convert errors
        let common_stream: BoxStream<'static, Result<ChatCompletionChunk, CommonRequestError>> = 
            self.request_builder.stream(&endpoint, Some(request));
        
        Box::pin(async_stream::try_stream! {
            use futures_util::StreamExt;
            
            let mut stream = common_stream;
            while let Some(result) = stream.next().await {
                match result {
                    Ok(response) => yield response,
                    Err(e) => yield Err(OpenRouterRequestError::from(e))?,
                }
            }
        })
    }

    /// List available models
    pub async fn list_models(&self) -> Result<ModelsResponse, OpenRouterRequestError> {
        let endpoint = Endpoint::new("api/v1/models", HttpMethod::Get);
        Ok(self.request_builder
            .request_json(&endpoint, None::<&()>)
            .await?)
    }

    /// Get generation details by ID
    pub async fn get_generation(&self, generation_id: &str) -> Result<GenerationInfo, OpenRouterRequestError> {
        let endpoint = Endpoint::new(format!("api/v1/generation/{}", generation_id), HttpMethod::Get);
        Ok(self.request_builder
            .request_json(&endpoint, None::<&()>)
            .await?)
    }

    /// Check API key status and credits
    pub async fn get_key_status(&self) -> Result<KeyStatus, OpenRouterRequestError> {
        let endpoint = Endpoint::new("api/v1/auth/key", HttpMethod::Get);
        Ok(self.request_builder
            .request_json(&endpoint, None::<&()>)
            .await?)
    }
}