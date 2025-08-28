pub mod prelude;
pub mod request;
pub mod response;
pub mod message;

use ai_ox_common::{RequestBuilder, request_builder::RequestConfig, Endpoint, HttpMethod, CommonRequestError};
use futures_util::stream::BoxStream;
use request::ChatRequest;
use response::{ChatResponse, StreamEvent};

pub struct Bedrock {
    builder: RequestBuilder,
}

impl Bedrock {
    pub fn new() -> Self {
        let config = RequestConfig::new("https://bedrock-runtime.us-east-1.amazonaws.com");
        let client = reqwest::Client::new();
        let builder = RequestBuilder::new(client, config);
        Self { builder }
    }

    pub async fn send(&self, req: &ChatRequest) -> Result<ChatResponse, CommonRequestError> {
        let endpoint = Endpoint::new(format!("/model/{}/invoke", req.model), HttpMethod::Post);
        self.builder.request_json(&endpoint, Some(req)).await
    }

    pub fn stream(&self, req: &ChatRequest) -> BoxStream<'static, Result<StreamEvent, CommonRequestError>> {
        let endpoint = Endpoint::new(format!("/model/{}/invoke-with-response-stream", req.model), HttpMethod::Post);
        self.builder.stream(&endpoint, Some(req))
    }
}
