use ai_ox_common::{
    BoxStream,
    error::ProviderError,
    openai_format::{ChatCompletionChunk, ChatCompletionResponse, ChatRequest},
    request_builder::{AuthMethod, Endpoint, HttpMethod, RequestBuilder, RequestConfig},
};

pub(crate) struct OpencodeRequestHelper {
    request_builder: RequestBuilder,
}

impl OpencodeRequestHelper {
    pub(crate) fn new(client: reqwest::Client, base_url: &str, api_key: Option<&str>) -> Self {
        let mut config =
            RequestConfig::new(base_url).with_header("content-type", "application/json");

        if let Some(token) = api_key {
            config = config.with_auth(AuthMethod::Bearer(token.to_string()));
        }

        let request_builder = RequestBuilder::new(client, config);

        Self { request_builder }
    }

    pub(crate) async fn send_chat_request(
        &self,
        request: &ChatRequest,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        let endpoint = Endpoint::new("zen/v1/chat/completions", HttpMethod::Post);
        Ok(self
            .request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }

    pub(crate) fn stream_chat_request(
        &self,
        request: &ChatRequest,
    ) -> BoxStream<'static, Result<ChatCompletionChunk, ProviderError>> {
        let endpoint = Endpoint::new("zen/v1/chat/completions", HttpMethod::Post);
        self.request_builder.stream(&endpoint, Some(request))
    }
}
