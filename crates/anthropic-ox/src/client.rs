use bon::Builder;
use core::fmt;
use futures_util::stream::BoxStream;
#[cfg(feature = "leaky-bucket")]
use leaky_bucket::RateLimiter;
#[cfg(feature = "leaky-bucket")]
use std::sync::Arc;

use crate::{
    error::{self, AnthropicRequestError},
    request,
    response::{self, StreamEvent},
};

#[cfg(feature = "models")]
use crate::models::{ModelInfo, ModelsListResponse};
#[cfg(feature = "tokens")]
use crate::tokens::{TokenCountRequest, TokenCountResponse};

const BASE_URL: &str = "https://api.anthropic.com";
const CHAT_URL: &str = "v1/messages";
const TOKENS_URL: &str = "v1/messages/count-tokens";
const MODELS_URL: &str = "v1/models";
const API_VERSION: &str = "2023-06-01";

/// A struct to configure beta features for the Anthropic API.
#[derive(Clone, Default, Debug)]
pub struct BetaFeatures {
    /// Enables fine-grained tool streaming.
    pub fine_grained_tool_streaming: bool,
    /// Enables interleaved thinking.
    pub interleaved_thinking: bool,
}

#[derive(Clone, Default, Builder)]
pub struct Anthropic {
    #[builder(into)]
    pub(crate) api_key: Option<String>,
    #[builder(into)]
    pub(crate) oauth_token: Option<String>,
    #[builder(default)]
    pub(crate) client: reqwest::Client,
    #[cfg(feature = "leaky-bucket")]
    pub(crate) leaky_bucket: Option<Arc<RateLimiter>>,
    #[builder(default = BASE_URL.to_string(), into)]
    pub(crate) base_url: String,
    #[builder(default = API_VERSION.to_string(), into)]
    pub(crate) api_version: String,
    #[builder(default)]
    pub(crate) headers: std::collections::HashMap<String, String>,
}

impl Anthropic {
    /// Create a new Anthropic client with the provided API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: Some(api_key.into()),
            oauth_token: None,
            client: reqwest::Client::new(),
            #[cfg(feature = "leaky-bucket")]
            leaky_bucket: None,
            base_url: BASE_URL.to_string(),
            api_version: API_VERSION.to_string(),
            headers: std::collections::HashMap::new(),
        }
    }

    pub fn load_from_env() -> Result<Self, std::env::VarError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")?;
        Ok(Anthropic::builder().api_key(api_key).build())
    }

    /// Add a custom header to the client
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Enables beta features for the client.
    pub fn with_beta_features(mut self, features: BetaFeatures) -> Self {
        let mut beta_headers = Vec::new();
        if features.fine_grained_tool_streaming {
            beta_headers.push("fine-grained-tool-streaming-2025-05-14");
        }
        if features.interleaved_thinking {
            beta_headers.push("interleaved-thinking-2025-05-14");
        }

        if !beta_headers.is_empty() {
            self.headers
                .insert("anthropic-beta".to_string(), beta_headers.join(","));
        }
        self
    }
}

impl Anthropic {
    /// Lists the available models.
    #[cfg(feature = "models")]
    pub async fn list_models(
        &self,
        limit: Option<u32>,
        before_id: Option<&str>,
        after_id: Option<&str>,
    ) -> Result<ModelsListResponse, AnthropicRequestError> {
        let url = format!("{}/{}", self.base_url, MODELS_URL);
        let mut query_params = Vec::new();
        if let Some(limit) = limit {
            query_params.push(("limit", limit.to_string()));
        }
        if let Some(before_id) = before_id {
            query_params.push(("before_id", before_id.to_string()));
        }
        if let Some(after_id) = after_id {
            query_params.push(("after_id", after_id.to_string()));
        }

        let mut req = self.client.get(&url).query(&query_params);

        if let Some(oauth_token) = &self.oauth_token {
            req = req.header("authorization", format!("Bearer {}", oauth_token));
        } else if let Some(api_key) = &self.api_key {
            req = req.header("x-api-key", api_key);
        } else {
            return Err(AnthropicRequestError::AuthenticationMissing);
        }

        let mut req_with_headers = req
            .header("anthropic-version", &self.api_version)
            .header("content-type", "application/json");

        for (key, value) in &self.headers {
            req_with_headers = req_with_headers.header(key, value);
        }

        let res = req_with_headers.send().await?;

        if res.status().is_success() {
            Ok(res.json::<ModelsListResponse>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    /// Retrieves a specific model by its ID.
    #[cfg(feature = "models")]
    pub async fn get_model(&self, model_id: &str) -> Result<ModelInfo, AnthropicRequestError> {
        let url = format!("{}/{}/{}", self.base_url, MODELS_URL, model_id);

        let mut req = self.client.get(&url);

        if let Some(oauth_token) = &self.oauth_token {
            req = req.header("authorization", format!("Bearer {}", oauth_token));
        } else if let Some(api_key) = &self.api_key {
            req = req.header("x-api-key", api_key);
        } else {
            return Err(AnthropicRequestError::AuthenticationMissing);
        }

        let mut req_with_headers = req
            .header("anthropic-version", &self.api_version)
            .header("content-type", "application/json");

        for (key, value) in &self.headers {
            req_with_headers = req_with_headers.header(key, value);
        }

        let res = req_with_headers.send().await?;

        if res.status().is_success() {
            Ok(res.json::<ModelInfo>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    /// Counts the number of tokens in a message.
    #[cfg(feature = "tokens")]
    pub async fn count_tokens(
        &self,
        request: &TokenCountRequest,
    ) -> Result<TokenCountResponse, AnthropicRequestError> {
        let url = format!("{}/{}", self.base_url, TOKENS_URL);

        let mut req = self.client.post(&url);

        // Use OAuth token if available, otherwise fall back to API key
        if let Some(oauth_token) = &self.oauth_token {
            req = req.header("authorization", format!("Bearer {}", oauth_token));
        } else if let Some(api_key) = &self.api_key {
            req = req.header("x-api-key", api_key);
        } else {
            return Err(AnthropicRequestError::AuthenticationMissing);
        }

        let mut req_with_headers = req
            .header("anthropic-version", &self.api_version)
            .header("content-type", "application/json");

        // Apply custom headers
        for (key, value) in &self.headers {
            req_with_headers = req_with_headers.header(key, value);
        }

        let res = req_with_headers.json(request).send().await?;

        if res.status().is_success() {
            Ok(res.json::<TokenCountResponse>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    pub async fn send(
        &self,
        request: &request::ChatRequest,
    ) -> Result<response::ChatResponse, AnthropicRequestError> {
        let url = format!("{}/{}", self.base_url, CHAT_URL);

        let mut req = self.client.post(&url);

        // Use OAuth token if available, otherwise fall back to API key
        if let Some(oauth_token) = &self.oauth_token {
            req = req.header("authorization", format!("Bearer {}", oauth_token));
        } else if let Some(api_key) = &self.api_key {
            req = req.header("x-api-key", api_key);
        } else {
            return Err(AnthropicRequestError::AuthenticationMissing);
        }

        let mut req_with_headers = req
            .header("anthropic-version", &self.api_version)
            .header("content-type", "application/json");

        // Apply custom headers
        for (key, value) in &self.headers {
            req_with_headers = req_with_headers.header(key, value);
        }

        let res = req_with_headers.json(request).send().await?;

        if res.status().is_success() {
            Ok(res.json::<response::ChatResponse>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    pub fn stream(
        &self,
        request: &request::ChatRequest,
    ) -> BoxStream<'static, Result<StreamEvent, AnthropicRequestError>> {
        use async_stream::try_stream;
        use futures_util::StreamExt;

        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let oauth_token = self.oauth_token.clone();
        let api_version = self.api_version.clone();
        let headers = self.headers.clone();
        let url = format!("{}/{}", self.base_url, CHAT_URL);
        let mut request_data = request.clone();
        request_data.stream = Some(true);

        Box::pin(try_stream! {
            let mut req = client.post(&url);

            // Use OAuth token if available, otherwise fall back to API key
            if let Some(token) = &oauth_token {
                req = req.header("authorization", format!("Bearer {}", token));
            } else if let Some(key) = &api_key {
                req = req.header("x-api-key", key);
            } else {
                Err(AnthropicRequestError::AuthenticationMissing)?;
            }

            let mut req_with_headers = req
                .header("anthropic-version", &api_version)
                .header("content-type", "application/json");

            // Apply custom headers
            for (key, value) in &headers {
                req_with_headers = req_with_headers.header(key, value);
            }

            let response = req_with_headers
                .json(&request_data)
                .send()
                .await?;

            let status = response.status();

            if !response.status().is_success() {
                let bytes = response.bytes().await?;
                Err(error::parse_error_response(status, bytes))?
            } else {
                let mut byte_stream = response.bytes_stream();

                while let Some(chunk_result) = byte_stream.next().await {
                    let chunk = chunk_result?;
                    let chunk_str = String::from_utf8(chunk.to_vec())
                        .map_err(|e| AnthropicRequestError::InvalidEventData(format!("UTF-8 decode error: {e}")))?;

                    for line in chunk_str.lines() {
                        if line.starts_with("data: ") {
                            let json_data = line.trim_start_matches("data: ");
                            if json_data != "[DONE]" {
                                let event: response::StreamEvent = serde_json::from_str(json_data)?;
                                yield event;
                            }
                        }
                    }
                }
            }
        })
    }
}

impl fmt::Debug for Anthropic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Anthropic")
            .field("api_key", &"[REDACTED]")
            .field("oauth_token", &self.oauth_token.as_ref().map(|_| "[REDACTED]"))
            .field("client", &self.client)
            .field("base_url", &self.base_url)
            .field("api_version", &self.api_version)
            .field("headers", &self.headers)
            .finish_non_exhaustive()
    }
}
