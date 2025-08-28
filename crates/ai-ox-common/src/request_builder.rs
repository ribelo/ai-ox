use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use reqwest::{Method, RequestBuilder as ReqwestRequestBuilder, Response};
use futures_util::stream::BoxStream;
use async_stream::try_stream;
use crate::{error::{self, CommonRequestError}, streaming::SseParser};

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
    #[must_use]
    pub fn new(path: impl Into<String>, method: HttpMethod) -> Self {
        Self {
            path: path.into(),
            method,
            extra_headers: None,
            query_params: None,
        }
    }

    #[must_use]
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let mut headers = self.extra_headers.unwrap_or_default();
        headers.insert(key.into(), value.into());
        self.extra_headers = Some(headers);
        self
    }

    #[must_use]
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
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            auth: None,
            default_headers: HashMap::new(),
            user_agent: None,
        }
    }

    #[must_use]
    pub fn with_auth(mut self, auth: AuthMethod) -> Self {
        self.auth = Some(auth);
        self
    }

    #[must_use]
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.default_headers.insert(key.into(), value.into());
        self
    }

    #[must_use]
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }
}

/// Generic request builder that handles common HTTP patterns
pub struct RequestBuilder {
    /// The underlying `reqwest::Client` used to send requests.
    client: reqwest::Client,
    /// The configuration for building requests.
    config: RequestConfig,
}

impl RequestBuilder {
    #[must_use]
    pub fn new(client: reqwest::Client, config: RequestConfig) -> Self {
        Self { client, config }
    }

    /// Build a reqwest `RequestBuilder` for the given endpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the URL cannot be parsed.
    pub fn build_request(&self, endpoint: &Endpoint) -> Result<ReqwestRequestBuilder, CommonRequestError> {
        self.build_request_with_options(endpoint, true)
    }

    /// Build a reqwest `RequestBuilder` with options for content-type handling.
    ///
    /// # Errors
    ///
    /// Returns an error if the URL cannot be parsed.
    pub fn build_request_with_options(&self, endpoint: &Endpoint, add_json_content_type: bool) -> Result<ReqwestRequestBuilder, CommonRequestError> {
        let url = format!("{}/{}", self.config.base_url.trim_end_matches('/'), endpoint.path.trim_start_matches('/'));
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
                AuthMethod::OAuth { header_name, token } => req.header(header_name, format!("Bearer {token}")),
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
        if add_json_content_type && matches!(endpoint.method, HttpMethod::Post | HttpMethod::Put | HttpMethod::Patch) {
            req = req.header("content-type", "application/json");
        }

        Ok(req)
    }

    /// Execute a request with JSON body and return deserialized response.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails, or if the response cannot be deserialized.
    pub async fn request_json<T: for<'de> Deserialize<'de>, B: Serialize>(
        &self,
        endpoint: &Endpoint,
        body: Option<&B>,
    ) -> Result<T, CommonRequestError> {
        let mut req = self.build_request(endpoint)?;

        if let Some(body) = body {
            req = req.json(body);
        }

        let response = req.send().await?;
        self.handle_response(response).await
    }

    /// Execute a request without body and return deserialized response.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails, or if the response cannot be deserialized.
    pub async fn request<T: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &Endpoint,
    ) -> Result<T, CommonRequestError> {
        let req = self.build_request(endpoint)?;
        let response = req.send().await?;
        self.handle_response(response).await
    }

    /// Execute a request and return unit type (for delete operations).
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn request_unit(&self, endpoint: &Endpoint) -> Result<(), CommonRequestError> {
        let req = self.build_request(endpoint)?;
        let response = req.send().await?;
        
        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let bytes = response.bytes().await?;
            Err(error::parse_error_response(status, &bytes))
        }
    }

    /// Execute a request and return raw bytes (for file downloads).
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn request_bytes(&self, endpoint: &Endpoint) -> Result<bytes::Bytes, CommonRequestError> {
        let req = self.build_request(endpoint)?;
        let response = req.send().await?;
        
        if response.status().is_success() {
            Ok(response.bytes().await?)
        } else {
            let status = response.status();
            let bytes = response.bytes().await?;
            Err(error::parse_error_response(status, &bytes))
        }
    }

    /// Execute a streaming request.
    ///
    /// # Errors
    ///
    /// Returns a stream that can yield errors if the request fails, or if an event cannot be parsed.
    ///
    /// # Panics
    ///
    /// Panics if the body cannot be serialized to a JSON value.
    #[must_use]
    pub fn stream<T, B>(
        &self,
        endpoint: &Endpoint,
        body: Option<&B>,
    ) -> BoxStream<'static, Result<T, CommonRequestError>>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
        B: Serialize,
    {
        let client = self.client.clone();
        let config = self.config.clone();
        let endpoint = endpoint.clone();
        let body_data = body.and_then(|b| serde_json::to_value(b).ok());

        Box::pin(try_stream! {
            let mut req = RequestBuilder::new(client.clone(), config.clone())
                .build_request(&endpoint)?;

            // Add body if provided and set stream=true
            if let Some(body_value) = body_data {
                if let Some(mut body_obj) = body_value.as_object().cloned() {
                    body_obj.insert("stream".to_string(), serde_json::Value::Bool(true));
                    req = req.json(&serde_json::Value::Object(body_obj));
                } else {
                    Err(CommonRequestError::RequestBuilder("Stream body must be a JSON object".to_string()))?;
                }
            }

            let response = req.send().await?;
            let status = response.status();

            if status.is_success() {
                let mut parser = SseParser::new(response);

                while let Some(event) = parser.next_event().await? {
                    yield event;
                }
            } else {
                let bytes = response.bytes().await?;
                Err(error::parse_error_response(status, &bytes))?;
            }
        })
    }

    /// Handle response and parse errors
    async fn handle_response<T: for<'de> Deserialize<'de>>(
        &self,
        response: Response,
    ) -> Result<T, CommonRequestError> {
        if response.status().is_success() {
            Ok(response.json::<T>().await?)
        } else {
            let status = response.status();
            let bytes = response.bytes().await?;
            Err(error::parse_error_response(status, &bytes))
        }
    }

    /// Execute a multipart form request (for file uploads).
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails, or if the response cannot be deserialized.
    pub async fn request_multipart<T: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &Endpoint,
        form: reqwest::multipart::Form,
    ) -> Result<T, CommonRequestError> {
        let req = self.build_request_with_options(endpoint, false)?; // Don't add JSON content-type for multipart
        let req = req.multipart(form);

        let response = req.send().await?;
        self.handle_response(response).await
    }
}

/// Helper struct for building multipart forms
pub struct MultipartForm {
    /// The underlying `reqwest::multipart::Form`.
    form: reqwest::multipart::Form,
}

impl MultipartForm {
    /// Create a new multipart form
    #[must_use]
    pub fn new() -> Self {
        Self {
            form: reqwest::multipart::Form::new(),
        }
    }

    /// Add a text field
    #[must_use]
    pub fn text(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.form = self.form.text(name.into(), value.into());
        self
    }

    /// Add a file from bytes
    #[must_use]
    pub fn file_from_bytes(
        mut self, 
        name: impl Into<String>, 
        filename: impl Into<String>,
        data: Vec<u8>
    ) -> Self {
        let part = reqwest::multipart::Part::bytes(data)
            .file_name(filename.into());
        self.form = self.form.part(name.into(), part);
        self
    }

    /// Add a file from bytes with custom mime type
    #[must_use]
    pub fn file_from_bytes_with_mime(
        mut self,
        name: impl Into<String>,
        filename: impl Into<String>, 
        data: Vec<u8>,
        mime_type: impl Into<String>
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
    #[must_use]
    pub fn build(self) -> reqwest::multipart::Form {
        self.form
    }
}

impl Default for MultipartForm {
    fn default() -> Self {
        Self::new()
    }
}