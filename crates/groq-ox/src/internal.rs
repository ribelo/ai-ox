use ai_ox_common::{
    request_builder::{RequestBuilder, RequestConfig, Endpoint, HttpMethod, AuthMethod},
    CommonRequestError, BoxStream
};
use futures_util::stream::BoxStream as FuturesBoxStream;
use crate::{GroqRequestError, ChatRequest, ChatResponse, response::ChatCompletionChunk};

/// Convert CommonRequestError to GroqRequestError
impl From<CommonRequestError> for GroqRequestError {
    fn from(err: CommonRequestError) -> Self {
        match err {
            CommonRequestError::Http(e) => GroqRequestError::ReqwestError(e),
            CommonRequestError::Json(e) => GroqRequestError::SerdeError(e),
            CommonRequestError::InvalidEventData(msg) => GroqRequestError::InvalidEventData(msg),
            CommonRequestError::AuthenticationMissing => GroqRequestError::MissingApiKey,
            CommonRequestError::InvalidMimeType(msg) => GroqRequestError::InvalidEventData(msg),
            CommonRequestError::Utf8Error(e) => GroqRequestError::InvalidEventData(e.to_string()),
        }
    }
}

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
    pub async fn send_chat_request(&self, request: &ChatRequest) -> Result<ChatResponse, GroqRequestError> {
        let endpoint = Endpoint::new("openai/v1/chat/completions", HttpMethod::Post);
        
        Ok(self.request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }

    /// Stream a chat completion request
    pub fn stream_chat_request(
        &self, 
        request: &ChatRequest
    ) -> FuturesBoxStream<'static, Result<ChatCompletionChunk, GroqRequestError>> {
        let endpoint = Endpoint::new("openai/v1/chat/completions", HttpMethod::Post);
        
        // Use the common streaming implementation and convert errors
        let common_stream: BoxStream<'static, Result<ChatCompletionChunk, CommonRequestError>> = 
            self.request_builder.stream(&endpoint, Some(request));
        
        Box::pin(async_stream::try_stream! {
            use futures_util::StreamExt;
            
            let mut stream = common_stream;
            while let Some(result) = stream.next().await {
                match result {
                    Ok(response) => yield response,
                    Err(e) => yield Err(GroqRequestError::from(e))?,
                }
            }
        })
    }
}