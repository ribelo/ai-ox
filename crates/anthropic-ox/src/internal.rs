use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use reqwest::{Method, RequestBuilder as ReqwestRequestBuilder, Response};
use futures_util::stream::{BoxStream, StreamExt};
use async_stream::try_stream;
use crate::error::{self, AnthropicRequestError};

/// HTTP method for API endpoints
#[derive(Debug, Clone)]
pub enum HttpMethod {
    Get,
    Post,
    Delete,
}

impl From<HttpMethod> for Method {
    fn from(method: HttpMethod) -> Self {
        match method {
            HttpMethod::Get => Method::GET,
            HttpMethod::Post => Method::POST,
            HttpMethod::Delete => Method::DELETE,
        }
    }
}

/// Represents an API endpoint with its configuration
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
}

/// Centralized request builder that handles all the duplicated HTTP logic
pub struct RequestBuilder<'a> {
    client: &'a reqwest::Client,
    base_url: &'a str,
    api_key: &'a Option<String>,
    oauth_token: &'a Option<String>,
    api_version: &'a str,
    headers: &'a HashMap<String, String>,
}

impl<'a> RequestBuilder<'a> {
    pub fn new(
        client: &'a reqwest::Client,
        base_url: &'a str,
        api_key: &'a Option<String>,
        oauth_token: &'a Option<String>,
        api_version: &'a str,
        headers: &'a HashMap<String, String>,
    ) -> Self {
        Self {
            client,
            base_url,
            api_key,
            oauth_token,
            api_version,
            headers,
        }
    }

    /// Build a request for the given endpoint
    pub fn build_request(&self, endpoint: &Endpoint) -> Result<ReqwestRequestBuilder, AnthropicRequestError> {
        let url = format!("{}/{}", self.base_url, endpoint.path);
        let method: Method = endpoint.method.clone().into();

        let mut req = self.client.request(method, &url);

        // Add query parameters if provided
        if let Some(ref params) = endpoint.query_params {
            req = req.query(&params);
        }

        // Add authentication
        if let Some(oauth_token) = self.oauth_token {
            req = req.header("authorization", format!("Bearer {}", oauth_token));
        } else if let Some(api_key) = self.api_key {
            req = req.header("x-api-key", api_key);
        } else {
            return Err(AnthropicRequestError::AuthenticationMissing);
        }

        // Add standard headers
        req = req.header("anthropic-version", self.api_version);
        
        // Only add content-type for POST requests
        if matches!(endpoint.method, HttpMethod::Post) {
            req = req.header("content-type", "application/json");
        }

        // Add beta header if required
        if let Some(ref beta) = endpoint.requires_beta {
            req = req.header("anthropic-beta", beta);
        }

        // Add custom headers
        for (key, value) in self.headers {
            req = req.header(key, value);
        }

        Ok(req)
    }

    /// Execute a request with JSON body and return deserialized response
    pub async fn request_json<T: for<'de> Deserialize<'de>, B: Serialize>(
        &self,
        endpoint: &Endpoint,
        body: Option<&B>,
    ) -> Result<T, AnthropicRequestError> {
        let mut req = self.build_request(endpoint)?;

        if let Some(body) = body {
            req = req.json(body);
        }

        let res = req.send().await?;
        self.handle_response(res).await
    }

    /// Execute a request without body and return deserialized response
    pub async fn request<T: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &Endpoint,
    ) -> Result<T, AnthropicRequestError> {
        let req = self.build_request(endpoint)?;
        let res = req.send().await?;
        self.handle_response(res).await
    }

    /// Handle response and parse errors
    pub async fn handle_response<T: for<'de> Deserialize<'de>>(
        &self,
        res: Response,
    ) -> Result<T, AnthropicRequestError> {
        if res.status().is_success() {
            Ok(res.json::<T>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    /// Execute a request and return unit type (for delete operations)
    pub async fn request_unit(&self, endpoint: &Endpoint) -> Result<(), AnthropicRequestError> {
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
    pub async fn request_bytes(&self, endpoint: &Endpoint) -> Result<bytes::Bytes, AnthropicRequestError> {
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

    /// Generic streaming method for SSE endpoints
    pub fn stream<T, B>(
        &self,
        endpoint: &Endpoint,
        body: Option<&B>,
    ) -> BoxStream<'static, Result<T, AnthropicRequestError>>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
        B: Serialize,
    {
        let client = self.client.clone();
        let base_url = self.base_url.to_string();
        let api_key = self.api_key.clone();
        let oauth_token = self.oauth_token.clone();
        let api_version = self.api_version.to_string();
        let headers = self.headers.clone();
        let endpoint = endpoint.clone();
        let body_data = body.map(|b| serde_json::to_value(b).ok()).flatten();

        Box::pin(try_stream! {
            // Rebuild the request builder for the stream context
            let stream_builder = RequestBuilder::new(
                &client,
                &base_url,
                &api_key,
                &oauth_token,
                &api_version,
                &headers,
            );
            let mut req = stream_builder.build_request(&endpoint)?;

            // Add body if provided and set stream=true
            if let Some(body_value) = body_data {
                let mut body_obj = body_value.as_object().unwrap().clone();
                body_obj.insert("stream".to_string(), serde_json::Value::Bool(true));
                req = req.json(&serde_json::Value::Object(body_obj));
            }

            let response = req.send().await?;
            let status = response.status();

            if !status.is_success() {
                let bytes = response.bytes().await?;
                Err(error::parse_error_response(status, bytes))?;
            } else {
                let mut byte_stream = response.bytes_stream();

                while let Some(chunk_result) = byte_stream.next().await {
                    let chunk = chunk_result?;
                    let chunk_str = String::from_utf8(chunk.to_vec())
                        .map_err(|e| AnthropicRequestError::InvalidEventData(format!("UTF-8 decode error: {e}")))?;

                    for event in Self::parse_sse_events(&chunk_str)? {
                        yield event;
                    }
                }
            }
        })
    }

    /// Parse Server-Sent Events from a chunk of data
    fn parse_sse_events<T: for<'de> Deserialize<'de>>(
        chunk: &str,
    ) -> Result<Vec<T>, AnthropicRequestError> {
        let mut events = Vec::new();
        
        for line in chunk.lines() {
            if line.starts_with("data: ") {
                let json_data = line.trim_start_matches("data: ");
                if json_data != "[DONE]" && !json_data.is_empty() {
                    let event: T = serde_json::from_str(json_data)
                        .map_err(|e| AnthropicRequestError::InvalidEventData(e.to_string()))?;
                    events.push(event);
                }
            }
        }
        
        Ok(events)
    }

    /// Stream JSONL data (for batch results)
    #[cfg(feature = "batches")]
    pub fn stream_jsonl<T>(
        &self,
        endpoint: &Endpoint,
    ) -> BoxStream<'static, Result<T, AnthropicRequestError>>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
    {
        let client = self.client.clone();
        let base_url = self.base_url.to_string();
        let api_key = self.api_key.clone();
        let oauth_token = self.oauth_token.clone();
        let api_version = self.api_version.to_string();
        let headers = self.headers.clone();
        let endpoint = endpoint.clone();

        Box::pin(try_stream! {
            // Rebuild the request builder for the stream context
            let stream_builder = RequestBuilder::new(
                &client,
                &base_url,
                &api_key,
                &oauth_token,
                &api_version,
                &headers,
            );
            let req = stream_builder.build_request(&endpoint)?;
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

                    // Process lines from buffer
                    while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                        let line_bytes = buffer.drain(..=pos).collect::<Vec<u8>>();
                        let line = String::from_utf8(line_bytes)
                            .map_err(|e| AnthropicRequestError::InvalidEventData(e.to_string()))?;
                        if !line.trim().is_empty() {
                            let result: T = serde_json::from_str(&line)
                                .map_err(AnthropicRequestError::SerdeError)?;
                            yield result;
                        }
                    }
                }
                
                // Process any remaining data in the buffer
                if !buffer.is_empty() {
                    let line = String::from_utf8(buffer)
                        .map_err(|e| AnthropicRequestError::InvalidEventData(e.to_string()))?;
                    if !line.trim().is_empty() {
                        let result: T = serde_json::from_str(&line)
                            .map_err(AnthropicRequestError::SerdeError)?;
                        yield result;
                    }
                }
            }
        })
    }
}