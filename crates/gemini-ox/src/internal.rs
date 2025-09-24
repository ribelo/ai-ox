use crate::{
    Gemini, GeminiRequestError,
    generate_content::{request::GenerateContentRequest, response::GenerateContentResponse},
};
use ai_ox_common::{
    BoxStream, CommonRequestError,
    request_builder::{
        AuthMethod, Endpoint, HttpMethod, RequestBuilder, RequestConfig, StreamOptions,
    },
};
use futures_util::stream::BoxStream as FuturesBoxStream;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::io::{Error as IoError, ErrorKind};

/// Convert CommonRequestError to GeminiRequestError
impl From<CommonRequestError> for GeminiRequestError {
    fn from(err: CommonRequestError) -> Self {
        match err {
            CommonRequestError::Http(message) => {
                GeminiRequestError::UnexpectedResponse(format!("HTTP request failed: {}", message))
            }
            CommonRequestError::Json(message) => GeminiRequestError::SerdeError(
                serde_json::Error::io(IoError::new(ErrorKind::Other, message)),
            ),
            CommonRequestError::Io(message) => {
                GeminiRequestError::IoError(IoError::new(ErrorKind::Other, message))
            }
            CommonRequestError::InvalidRequest {
                code,
                message,
                details,
            } => GeminiRequestError::InvalidRequestError {
                code,
                message,
                status: None,
                details: details.unwrap_or(Value::Null),
            },
            CommonRequestError::RateLimit => GeminiRequestError::RateLimit,
            CommonRequestError::AuthenticationMissing => GeminiRequestError::AuthenticationMissing,
            CommonRequestError::InvalidModel(model) => GeminiRequestError::InvalidRequestError {
                code: Some("INVALID_MODEL".to_string()),
                message: format!("Invalid model: {}", model),
                status: None,
                details: Value::Null,
            },
            CommonRequestError::UnexpectedResponse(message) => {
                GeminiRequestError::UnexpectedResponse(message)
            }
            CommonRequestError::InvalidEventData(message) => {
                GeminiRequestError::InvalidEventData(message)
            }
            CommonRequestError::UrlBuildError(message) => {
                GeminiRequestError::UrlBuildError(message)
            }
            CommonRequestError::Stream(message) => GeminiRequestError::InvalidEventData(message),
            CommonRequestError::InvalidMimeType(message) => {
                GeminiRequestError::InvalidEventData(message)
            }
            CommonRequestError::Utf8Error(message) => GeminiRequestError::InvalidEventData(message),
            CommonRequestError::JsonDeserializationError(message) => {
                GeminiRequestError::JsonDeserializationError(serde_json::Error::io(IoError::new(
                    ErrorKind::Other,
                    message,
                )))
            }
        }
    }
}

/// Gemini client helper methods using the common RequestBuilder
#[derive(Clone)]
pub struct GeminiRequestHelper {
    client: reqwest::Client,
    config: RequestConfig,
    is_oauth: bool,
}

impl GeminiRequestHelper {
    const STANDARD_BASE: &'static str = "https://generativelanguage.googleapis.com";
    const CLOUD_BASE: &'static str = "https://cloudcode-pa.googleapis.com";

    fn select_auth(
        gemini: &Gemini,
        allow_oauth: bool,
    ) -> Result<(AuthMethod, bool), GeminiRequestError> {
        if allow_oauth {
            if let Some(oauth_token) = &gemini.oauth_token {
                return Ok((AuthMethod::Bearer(oauth_token.clone()), true));
            }
        }

        if let Some(api_key) = &gemini.api_key {
            Ok((
                AuthMethod::QueryParam("key".to_string(), api_key.clone()),
                false,
            ))
        } else if allow_oauth {
            Err(GeminiRequestError::AuthenticationMissing)
        } else {
            Err(GeminiRequestError::AuthenticationMissing)
        }
    }

    fn new_with_base_url(
        gemini: &Gemini,
        base_url: &str,
        allow_oauth: bool,
    ) -> Result<Self, GeminiRequestError> {
        let (auth_method, is_oauth) = Self::select_auth(gemini, allow_oauth)?;
        let config = RequestConfig::new(base_url.to_string())
            .with_auth(auth_method)
            .with_header("content-type", "application/json");

        Ok(Self {
            client: gemini.client.clone(),
            config,
            is_oauth,
        })
    }

    pub fn for_standard(gemini: &Gemini) -> Result<Self, GeminiRequestError> {
        Self::new_with_base_url(gemini, Self::STANDARD_BASE, false)
    }

    pub fn for_generate(gemini: &Gemini) -> Result<Self, GeminiRequestError> {
        if gemini.oauth_token.is_some() {
            Self::new_with_base_url(gemini, Self::CLOUD_BASE, true)
        } else {
            Self::new_with_base_url(gemini, Self::STANDARD_BASE, false)
        }
    }

    pub fn new_for_api_key(gemini: &Gemini) -> Result<Self, GeminiRequestError> {
        Self::new_with_base_url(gemini, Self::STANDARD_BASE, false)
    }

    fn builder(&self) -> RequestBuilder {
        RequestBuilder::new(self.client.clone(), self.config.clone())
    }

    pub fn is_oauth(&self) -> bool {
        self.is_oauth
    }

    pub async fn request_json<T, B>(
        &self,
        endpoint: Endpoint,
        body: Option<&B>,
    ) -> Result<T, GeminiRequestError>
    where
        T: DeserializeOwned,
        B: Serialize,
    {
        self.builder()
            .request_json::<T, B>(&endpoint, body)
            .await
            .map_err(GeminiRequestError::from)
    }

    pub async fn request<T>(&self, endpoint: Endpoint) -> Result<T, GeminiRequestError>
    where
        T: DeserializeOwned,
    {
        self.builder()
            .request::<T>(&endpoint)
            .await
            .map_err(GeminiRequestError::from)
    }

    pub async fn request_unit(&self, endpoint: Endpoint) -> Result<(), GeminiRequestError> {
        self.builder()
            .request_unit(&endpoint)
            .await
            .map_err(GeminiRequestError::from)
    }

    fn build_generate_content_body(
        &self,
        request: &GenerateContentRequest,
        gemini: &Gemini,
    ) -> Result<Value, GeminiRequestError> {
        if self.is_oauth {
            let minimal_request = serde_json::json!({
                "contents": request.contents,
                "generationConfig": request.generation_config,
                "systemInstruction": request.system_instruction
            });
            Ok(serde_json::json!({
                "model": request.model.to_string(),
                "project": gemini.project_id,
                "request": minimal_request
            }))
        } else {
            Ok(serde_json::to_value(request).map_err(GeminiRequestError::SerdeError)?)
        }
    }

    /// Send a generate content request
    pub async fn send_generate_content_request(
        &self,
        request: &GenerateContentRequest,
        gemini: &Gemini,
    ) -> Result<GenerateContentResponse, GeminiRequestError> {
        // Build endpoint based on authentication method
        let endpoint_path = if self.is_oauth {
            "v1internal:generateContent".to_string()
        } else {
            format!(
                "{}/models/{}:generateContent",
                gemini.api_version, request.model
            )
        };

        let endpoint = Endpoint::new(endpoint_path, HttpMethod::Post);

        let request_body = self.build_generate_content_body(request, gemini)?;

        if self.is_oauth {
            let response: Value = self
                .builder()
                .request_json(&endpoint, Some(&request_body))
                .await
                .map_err(GeminiRequestError::from)?;

            if let Some(inner_response) = response.get("response") {
                Ok(serde_json::from_value(inner_response.clone())?)
            } else {
                Err(GeminiRequestError::UnexpectedResponse(
                    "Missing 'response' field in Cloud Code Assist API response".to_string(),
                ))
            }
        } else {
            self.builder()
                .request_json(&endpoint, Some(&request_body))
                .await
                .map_err(GeminiRequestError::from)
        }
    }

    /// Stream a generate content request
    pub fn stream_generate_content_request(
        &self,
        request: GenerateContentRequest,
        gemini: Gemini,
    ) -> FuturesBoxStream<'static, Result<GenerateContentResponse, GeminiRequestError>> {
        // Build endpoint based on authentication method
        let endpoint_path = if self.is_oauth {
            "v1internal:streamGenerateContent".to_string()
        } else {
            format!(
                "{}/models/{}:streamGenerateContent",
                gemini.api_version, request.model
            )
        };

        let endpoint = Endpoint::new(endpoint_path, HttpMethod::Post)
            .with_query_params(vec![("alt".to_string(), "sse".to_string())]);

        let request_body = match self.build_generate_content_body(&request, &gemini) {
            Ok(body) => body,
            Err(err) => {
                return Box::pin(futures_util::stream::once(async move { Err(err) }));
            }
        };

        let common_stream: BoxStream<'static, Result<Value, CommonRequestError>> =
            self.builder().stream_with_options(
                &endpoint,
                Some(request_body),
                StreamOptions {
                    set_stream_field: false,
                },
            );

        let is_oauth = self.is_oauth;

        Box::pin(async_stream::try_stream! {
            use futures_util::StreamExt;

            let mut stream = common_stream;
            while let Some(result) = stream.next().await {
                match result {
                    Ok(value) => {
                        if is_oauth {
                            if let Some(inner_response) = value.get("response") {
                                yield serde_json::from_value::<GenerateContentResponse>(inner_response.clone())?;
                            }
                        } else {
                            yield serde_json::from_value::<GenerateContentResponse>(value)?;
                        }
                    },
                    Err(e) => Err(GeminiRequestError::from(e))?,
                }
            }
        })
    }
}
