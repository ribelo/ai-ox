use crate::error::{self, AnthropicRequestError};
use ai_ox_common::{
    request_builder::{
        AuthMethod, Endpoint as CommonEndpoint, RequestBuilder, RequestConfig,
    },
    CommonRequestError,
};
use async_stream::try_stream;
use futures_util::{stream::BoxStream, StreamExt};
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;

pub use ai_ox_common::request_builder::HttpMethod;

/// Represents an API endpoint with Anthropic-specific metadata.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub path: String,
    pub method: HttpMethod,
    pub requires_beta: Option<String>,
    pub query_params: Option<Vec<(String, String)>>,
}

impl Endpoint {
    pub fn new(path: impl Into<String>, method: HttpMethod) -> Self {
        Self {
            path: path.into(),
            method,
            requires_beta: None,
            query_params: None,
        }
    }

    pub fn with_beta(mut self, beta_header: impl Into<String>) -> Self {
        self.requires_beta = Some(beta_header.into());
        self
    }

    pub fn with_query_params(mut self, params: Vec<(String, String)>) -> Self {
        self.query_params = Some(params);
        self
    }

    fn to_common(&self) -> CommonEndpoint {
        let mut endpoint = CommonEndpoint::new(self.path.clone(), self.method.clone());
        if let Some(params) = &self.query_params {
            endpoint = endpoint.with_query_params(params.clone());
        }
        if let Some(beta) = &self.requires_beta {
            endpoint = endpoint.with_header("anthropic-beta", beta.clone());
        }
        endpoint
    }
}

#[derive(Clone)]
pub struct AnthropicRequestHelper {
    client: reqwest::Client,
    config: RequestConfig,
}

impl AnthropicRequestHelper {
    pub fn new(
        client: reqwest::Client,
        base_url: &str,
        api_key: &Option<String>,
        oauth_token: &Option<String>,
        api_version: &str,
        headers: &HashMap<String, String>,
    ) -> Result<Self, AnthropicRequestError> {
        let auth_method = if let Some(token) = oauth_token {
            AuthMethod::OAuth {
                header_name: "authorization".to_string(),
                token: token.clone(),
            }
        } else if let Some(key) = api_key {
            AuthMethod::ApiKey {
                header_name: "x-api-key".to_string(),
                key: key.clone(),
            }
        } else {
            return Err(AnthropicRequestError::AuthenticationMissing);
        };

        let mut config = RequestConfig::new(base_url.to_string())
            .with_auth(auth_method)
            .with_header("anthropic-version", api_version.to_string());

        for (key, value) in headers {
            config = config.with_header(key.clone(), value.clone());
        }

        Ok(Self { client, config })
    }

    fn builder(&self) -> RequestBuilder {
        RequestBuilder::new(self.client.clone(), self.config.clone())
    }

    fn endpoint(&self, endpoint: &Endpoint) -> CommonEndpoint {
        endpoint.to_common()
    }

    pub async fn request_json<T, B>(
        &self,
        endpoint: &Endpoint,
        body: Option<&B>,
    ) -> Result<T, AnthropicRequestError>
    where
        T: DeserializeOwned,
        B: Serialize,
    {
        self.builder()
            .request_json::<T, B>(&self.endpoint(endpoint), body)
            .await
            .map_err(AnthropicRequestError::from)
    }

    pub async fn request<T>(&self, endpoint: &Endpoint) -> Result<T, AnthropicRequestError>
    where
        T: DeserializeOwned,
    {
        self.builder()
            .request::<T>(&self.endpoint(endpoint))
            .await
            .map_err(AnthropicRequestError::from)
    }

    pub async fn request_unit(&self, endpoint: &Endpoint) -> Result<(), AnthropicRequestError> {
        self.builder()
            .request_unit(&self.endpoint(endpoint))
            .await
            .map_err(AnthropicRequestError::from)
    }

    pub async fn request_bytes(
        &self,
        endpoint: &Endpoint,
    ) -> Result<bytes::Bytes, AnthropicRequestError> {
        self.builder()
            .request_bytes(&self.endpoint(endpoint))
            .await
            .map_err(AnthropicRequestError::from)
    }

    pub fn stream<T, B>(
        &self,
        endpoint: &Endpoint,
        body: Option<&B>,
    ) -> BoxStream<'static, Result<T, AnthropicRequestError>>
    where
        T: DeserializeOwned + Send + 'static,
        B: Serialize,
    {
        let common_endpoint = self.endpoint(endpoint);
        let stream = self.builder().stream(&common_endpoint, body);

        Box::pin(stream.map(|result| result.map_err(AnthropicRequestError::from)))
    }

    #[cfg(feature = "batches")]
    pub fn stream_jsonl<T>(
        &self,
        endpoint: &Endpoint,
    ) -> BoxStream<'static, Result<T, AnthropicRequestError>>
    where
        T: DeserializeOwned + Send + 'static,
    {
        let client = self.client.clone();
        let config = self.config.clone();
        let common_endpoint = self.endpoint(endpoint);

        Box::pin(try_stream! {
            let req = RequestBuilder::new(client.clone(), config.clone()).build_request(&common_endpoint)?;
            let response = req.send().await?;
            let status = response.status();

            if !status.is_success() {
                let bytes = response.bytes().await?;
                Err(error::parse_error_response(status, bytes))?;
            } else {
                let mut byte_stream = response.bytes_stream();
                let mut buffer = Vec::new();

                while let Some(chunk_result) = byte_stream.next().await {
                    let chunk = chunk_result?;
                    buffer.extend_from_slice(&chunk);

                    while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                        let line_bytes = buffer.drain(..=pos).collect::<Vec<u8>>();
                        let line = String::from_utf8(line_bytes)
                            .map_err(|e| AnthropicRequestError::InvalidEventData(e.to_string()))?;
                        if !line.trim().is_empty() {
                            let result: T = serde_json::from_str(line.trim())
                                .map_err(AnthropicRequestError::SerdeError)?;
                            yield result;
                        }
                    }
                }

                if !buffer.is_empty() {
                    let line = String::from_utf8(buffer)
                        .map_err(|e| AnthropicRequestError::InvalidEventData(e.to_string()))?;
                    if !line.trim().is_empty() {
                        let result: T = serde_json::from_str(line.trim())
                            .map_err(AnthropicRequestError::SerdeError)?;
                        yield result;
                    }
                }
            }
        })
    }
}

impl From<CommonRequestError> for AnthropicRequestError {
    fn from(err: CommonRequestError) -> Self {
        match err {
            CommonRequestError::Http(message) => AnthropicRequestError::UnexpectedResponse(
                format!("HTTP request failed: {}", message),
            ),
            CommonRequestError::Json(message) => AnthropicRequestError::InvalidEventData(message),
            CommonRequestError::Io(message) => AnthropicRequestError::Stream(message),
            CommonRequestError::InvalidRequest {
                code,
                message,
                details,
            } => {
                let param = details
                    .as_ref()
                    .and_then(|value| value.get("param"))
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string);

                AnthropicRequestError::InvalidRequestError {
                    message,
                    param,
                    code,
                }
            }
            CommonRequestError::RateLimit => AnthropicRequestError::RateLimit,
            CommonRequestError::AuthenticationMissing => {
                AnthropicRequestError::AuthenticationMissing
            }
            CommonRequestError::InvalidModel(model) => AnthropicRequestError::InvalidRequestError {
                message: format!("Invalid model: {}", model),
                param: None,
                code: None,
            },
            CommonRequestError::UnexpectedResponse(message) => {
                AnthropicRequestError::UnexpectedResponse(message)
            }
            CommonRequestError::InvalidEventData(message) => {
                AnthropicRequestError::InvalidEventData(message)
            }
            CommonRequestError::UrlBuildError(message) => {
                AnthropicRequestError::InvalidRequestError {
                    message,
                    param: None,
                    code: None,
                }
            }
            CommonRequestError::Stream(message) => AnthropicRequestError::Stream(message),
            CommonRequestError::InvalidMimeType(message) => {
                AnthropicRequestError::InvalidRequestError {
                    message,
                    param: None,
                    code: None,
                }
            }
            CommonRequestError::Utf8Error(message) => AnthropicRequestError::InvalidUtf8(message),
            CommonRequestError::JsonDeserializationError(message) => {
                AnthropicRequestError::Deserialization(message)
            }
        }
    }
}
