use crate::{
    error::{self, CommonRequestError},
    streaming::SseParser,
};
use async_stream::try_stream;
use futures_util::stream::{self, BoxStream};
use reqwest::{Method, RequestBuilder as ReqwestRequestBuilder, Response};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// HTTP method for API endpoints
#[derive(Debug, Clone)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
}

impl From<HttpMethod> for Method {
    fn from(method: HttpMethod) -> Self {
        match method {
            HttpMethod::Get => Method::GET,
            HttpMethod::Post => Method::POST,
            HttpMethod::Put => Method::PUT,
            HttpMethod::Delete => Method::DELETE,
            HttpMethod::Patch => Method::PATCH,
        }
    }
}

/// Authentication method for API requests
#[derive(Debug, Clone)]
pub enum AuthMethod {
    /// Bearer token authentication (Authorization: Bearer <token>)
    Bearer(String),
    /// API key header (e.g., x-api-key: <key>)
    ApiKey { header_name: String, key: String },
    /// OAuth token with custom header
    OAuth { header_name: String, token: String },
    /// Query parameter authentication (e.g., ?key=<key>)
    QueryParam(String, String),
}

/// Represents an API endpoint with its configuration
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub path: String,
    pub method: HttpMethod,
    pub extra_headers: Option<HashMap<String, String>>,
    pub query_params: Option<Vec<(String, String)>>,
}

impl Endpoint {
    pub fn new(path: impl Into<String>, method: HttpMethod) -> Self {
        Self {
            path: path.into(),
            method,
            extra_headers: None,
            query_params: None,
        }
    }

    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let mut headers = self.extra_headers.unwrap_or_default();
        headers.insert(key.into(), value.into());
        self.extra_headers = Some(headers);
        self
    }

    pub fn with_query_params(mut self, params: Vec<(String, String)>) -> Self {
        self.query_params = Some(params);
        self
    }
}

/// Configuration for request building
#[derive(Debug, Clone)]
pub struct RequestConfig {
    pub base_url: String,
    pub auth: Option<AuthMethod>,
    pub default_headers: HashMap<String, String>,
    pub user_agent: Option<String>,
}

impl RequestConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            auth: None,
            default_headers: HashMap::new(),
            user_agent: None,
        }
    }

    pub fn with_auth(mut self, auth: AuthMethod) -> Self {
        self.auth = Some(auth);
        self
    }

    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.default_headers.insert(key.into(), value.into());
        self
    }

    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }
}

/// Generic request builder that handles common HTTP patterns
pub struct RequestBuilder {
    client: reqwest::Client,
    config: RequestConfig,
}

/// Options that control how streaming requests are constructed.
#[derive(Debug, Clone, Copy)]
pub struct StreamOptions {
    /// Whether to set `"stream": true` in the JSON body before sending the request.
    pub set_stream_field: bool,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            set_stream_field: true,
        }
    }
}

impl RequestBuilder {
    pub fn new(client: reqwest::Client, config: RequestConfig) -> Self {
        Self { client, config }
    }

    /// Build a reqwest RequestBuilder for the given endpoint
    pub fn build_request(
        &self,
        endpoint: &Endpoint,
    ) -> Result<ReqwestRequestBuilder, CommonRequestError> {
        self.build_request_with_options(endpoint, true)
    }

    /// Build a reqwest RequestBuilder with options for content-type handling
    pub fn build_request_with_options(
        &self,
        endpoint: &Endpoint,
        add_json_content_type: bool,
    ) -> Result<ReqwestRequestBuilder, CommonRequestError> {
        let url = format!(
            "{}/{}",
            self.config.base_url.trim_end_matches('/'),
            endpoint.path.trim_start_matches('/')
        );
        let method: Method = endpoint.method.clone().into();

        let mut req = self.client.request(method, &url);

        // Add query parameters if provided
        if let Some(ref params) = endpoint.query_params {
            req = req.query(&params);
        }

        // Add authentication
        if let Some(ref auth) = self.config.auth {
            req = match auth {
                AuthMethod::Bearer(token) => req.bearer_auth(token),
                AuthMethod::ApiKey { header_name, key } => req.header(header_name, key),
                AuthMethod::OAuth { header_name, token } => {
                    req.header(header_name, format!("Bearer {}", token))
                }
                AuthMethod::QueryParam(param_name, value) => req.query(&[(param_name, value)]),
            };
        }

        // Add default headers
        for (key, value) in &self.config.default_headers {
            req = req.header(key, value);
        }

        // Add endpoint-specific headers
        if let Some(ref headers) = endpoint.extra_headers {
            for (key, value) in headers {
                req = req.header(key, value);
            }
        }

        // Add user agent
        if let Some(ref user_agent) = self.config.user_agent {
            req = req.header("user-agent", user_agent);
        }

        // Add content-type for POST/PUT/PATCH requests (only for JSON requests)
        if add_json_content_type
            && matches!(
                endpoint.method,
                HttpMethod::Post | HttpMethod::Put | HttpMethod::Patch
            )
        {
            req = req.header("content-type", "application/json");
        }

        Ok(req)
    }

    /// Execute a request with JSON body and return deserialized response
    pub async fn request_json<T: for<'de> Deserialize<'de>, B: Serialize>(
        &self,
        endpoint: &Endpoint,
        body: Option<&B>,
    ) -> Result<T, CommonRequestError> {
        let mut req = self.build_request(endpoint)?;

        if let Some(body) = body {
            // Normalize body to serde_json::Value to avoid any accidental double-encoding
            let val =
                serde_json::to_value(body).map_err(|e| CommonRequestError::Json(e.to_string()))?;

            if std::env::var("AOX_HTTP_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
            {
                let kind = match &val {
                    serde_json::Value::Null => "null",
                    serde_json::Value::Bool(_) => "bool",
                    serde_json::Value::Number(_) => "number",
                    serde_json::Value::String(_) => "string",
                    serde_json::Value::Array(_) => "array",
                    serde_json::Value::Object(_) => "object",
                };
                eprintln!(
                    "[ai-ox-common::request_builder] POST {} body kind: {} payload: {}",
                    endpoint.path, kind, val
                );
            }
            req = req.json(&val);
        }

        let res = req.send().await?;
        self.handle_response(res).await
    }

    /// Execute a request without body and return deserialized response
    pub async fn request<T: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &Endpoint,
    ) -> Result<T, CommonRequestError> {
        let req = self.build_request(endpoint)?;
        let res = req.send().await?;
        self.handle_response(res).await
    }

    /// Execute a request and return unit type (for delete operations)
    pub async fn request_unit(&self, endpoint: &Endpoint) -> Result<(), CommonRequestError> {
        let req = self.build_request(endpoint)?;
        let res = req.send().await?;

        if res.status().is_success() {
            Ok(())
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    /// Execute a request and return raw bytes (for file downloads)
    pub async fn request_bytes(
        &self,
        endpoint: &Endpoint,
    ) -> Result<bytes::Bytes, CommonRequestError> {
        let req = self.build_request(endpoint)?;
        let res = req.send().await?;

        if res.status().is_success() {
            Ok(res.bytes().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    /// Execute a streaming request
    pub fn stream<T, B>(
        &self,
        endpoint: &Endpoint,
        body: Option<&B>,
    ) -> BoxStream<'static, Result<T, CommonRequestError>>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
        B: Serialize,
    {
        let body_value = match body {
            Some(b) => match serde_json::to_value(b) {
                Ok(value) => Some(value),
                Err(e) => {
                    return Box::pin(stream::once(async move {
                        Err(CommonRequestError::Json(e.to_string()))
                    }));
                }
            },
            None => None,
        };

        self.stream_with_options(endpoint, body_value, StreamOptions::default())
    }

    /// Execute a streaming request with fine-grained configuration.
    pub fn stream_with_options<T>(
        &self,
        endpoint: &Endpoint,
        body: Option<Value>,
        options: StreamOptions,
    ) -> BoxStream<'static, Result<T, CommonRequestError>>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
    {
        let client = self.client.clone();
        let config = self.config.clone();
        let endpoint = endpoint.clone();

        Box::pin(try_stream! {
            let mut req = RequestBuilder::new(client.clone(), config.clone())
                .build_request(&endpoint)?;

            if let Some(body_value) = body {
                let mut obj = match body_value {
                    Value::Object(map) => map,
                    other => {
                        Err(CommonRequestError::Json(format!(
                            "Streaming body must be a JSON object, got {}",
                            other
                        )))?
                    }
                };

                if options.set_stream_field {
                    obj.insert("stream".to_string(), Value::Bool(true));
                }

                let payload = Value::Object(obj.clone());

                if std::env::var("AOX_HTTP_DEBUG").map(|v| v == "1").unwrap_or(false) {
                    let kind = match &payload {
                        Value::Null => "null",
                        Value::Bool(_) => "bool",
                        Value::Number(_) => "number",
                        Value::String(_) => "string",
                        Value::Array(_) => "array",
                        Value::Object(_) => "object",
                    };
                    eprintln!(
                        "[ai-ox-common::request_builder] STREAM {} body kind: {} payload: {}",
                        endpoint.path,
                        kind,
                        payload
                    );
                }

                req = req.json(&payload);
            }

            let response = req.send().await?;
            let status = response.status();

            if !status.is_success() {
                let bytes = response.bytes().await?;
                Err(error::parse_error_response(status, bytes))?;
            } else {
                let mut parser = SseParser::new(response);

                while let Some(event) = parser.next_event().await? {
                    yield event;
                }
            }
        })
    }

    /// Handle response and parse errors
    async fn handle_response<T: for<'de> Deserialize<'de>>(
        &self,
        res: Response,
    ) -> Result<T, CommonRequestError> {
        let status = res.status();
        let bytes = res.bytes().await?;

        if status.is_success() {
            match serde_json::from_slice::<T>(&bytes) {
                Ok(val) => Ok(val),
                Err(e) => {
                    let body_str = String::from_utf8_lossy(&bytes);
                    Err(CommonRequestError::UnexpectedResponse(format!(
                        "HTTP {} but failed to decode JSON: {}; body: {}",
                        status.as_u16(),
                        e,
                        body_str
                    )))
                }
            }
        } else {
            Err(error::parse_error_response(status, bytes))
        }
    }

    /// Execute a multipart form request (for file uploads)
    pub async fn request_multipart<T: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &Endpoint,
        form: reqwest::multipart::Form,
    ) -> Result<T, CommonRequestError> {
        let req = self.build_request_with_options(endpoint, false)?; // Don't add JSON content-type for multipart
        let req = req.multipart(form);

        let res = req.send().await?;
        self.handle_response(res).await
    }
}

/// Helper struct for building multipart forms
pub struct MultipartForm {
    form: reqwest::multipart::Form,
}

impl MultipartForm {
    /// Create a new multipart form
    pub fn new() -> Self {
        Self {
            form: reqwest::multipart::Form::new(),
        }
    }

    /// Add a text field
    pub fn text(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.form = self.form.text(name.into(), value.into());
        self
    }

    /// Add a file from bytes
    pub fn file_from_bytes(
        mut self,
        name: impl Into<String>,
        filename: impl Into<String>,
        data: Vec<u8>,
    ) -> Self {
        let part = reqwest::multipart::Part::bytes(data).file_name(filename.into());
        self.form = self.form.part(name.into(), part);
        self
    }

    /// Add a file from bytes with custom mime type
    pub fn file_from_bytes_with_mime(
        mut self,
        name: impl Into<String>,
        filename: impl Into<String>,
        data: Vec<u8>,
        mime_type: impl Into<String>,
    ) -> Self {
        let mime_str = mime_type.into();
        let part = reqwest::multipart::Part::bytes(data.clone())
            .file_name(filename.into())
            .mime_str(&mime_str)
            .unwrap_or_else(|_| reqwest::multipart::Part::bytes(data));
        self.form = self.form.part(name.into(), part);
        self
    }

    /// Build the final form
    pub fn build(self) -> reqwest::multipart::Form {
        self.form
    }
}

impl Default for MultipartForm {
    fn default() -> Self {
        Self::new()
    }
}
