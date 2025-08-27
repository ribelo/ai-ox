use bon::Builder;
use core::fmt;
use futures_util::stream::BoxStream;
use serde::Serialize;
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
#[cfg(feature = "batches")]
use crate::batches::{BatchListResponse, MessageBatch, MessageBatchRequest};
#[cfg(feature = "files")]
use crate::files::{FileInfo, FileListResponse, FileUploadRequest};
#[cfg(feature = "admin")]
use crate::admin::{
    api_keys::{ApiKey, ApiKeyListResponse, UpdateApiKeyRequest},
    invites::{CreateInviteRequest, Invite, InviteListResponse},
    usage::{CostReportResponse, UsageReportResponse},
    users::{User, UserListResponse},
    workspaces::{CreateWorkspaceRequest, UpdateWorkspaceRequest, Workspace, WorkspaceListResponse},
};

const BASE_URL: &str = "https://api.anthropic.com";
const CHAT_URL: &str = "v1/messages";
const TOKENS_URL: &str = "v1/messages/count-tokens";
const MODELS_URL: &str = "v1/models";
const BATCHES_URL: &str = "v1/message_batches";
const FILES_URL: &str = "v1/files";
const ADMIN_ORGANIZATIONS_URL: &str = "v1/organizations";
const API_VERSION: &str = "2023-06-01";

/// A struct to configure beta features for the Anthropic API.
#[derive(Clone, Default, Debug)]
pub struct BetaFeatures {
    /// Enables fine-grained tool streaming.
    pub fine_grained_tool_streaming: bool,
    /// Enables interleaved thinking.
    pub interleaved_thinking: bool,
    /// Enables the computer use tool.
    pub computer_use: bool,
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
        if features.computer_use {
            beta_headers.push("computer-use-2025-01-24");
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

    /// Creates a new message batch.
    #[cfg(feature = "batches")]
    pub async fn create_message_batch(
        &self,
        request: &MessageBatchRequest,
    ) -> Result<MessageBatch, AnthropicRequestError> {
        let url = format!("{}/{}", self.base_url, BATCHES_URL);
        let mut req = self.client.post(&url);

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

        let res = req_with_headers.json(request).send().await?;

        if res.status().is_success() {
            Ok(res.json::<MessageBatch>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    /// Retrieves a message batch.
    #[cfg(feature = "batches")]
    pub async fn get_message_batch(
        &self,
        batch_id: &str,
    ) -> Result<MessageBatch, AnthropicRequestError> {
        let url = format!("{}/{}/{}", self.base_url, BATCHES_URL, batch_id);
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
            Ok(res.json::<MessageBatch>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    /// Lists message batches.
    #[cfg(feature = "batches")]
    pub async fn list_message_batches(
        &self,
        limit: Option<u32>,
        after: Option<&str>,
    ) -> Result<BatchListResponse, AnthropicRequestError> {
        let url = format!("{}/{}", self.base_url, BATCHES_URL);
        let mut query_params = Vec::new();
        if let Some(limit) = limit {
            query_params.push(("limit", limit.to_string()));
        }
        if let Some(after) = after {
            query_params.push(("after", after.to_string()));
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
            Ok(res.json::<BatchListResponse>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    /// Cancels a message batch.
    #[cfg(feature = "batches")]
    pub async fn cancel_message_batch(
        &self,
        batch_id: &str,
    ) -> Result<MessageBatch, AnthropicRequestError> {
        let url = format!("{}/{}/{}/cancel", self.base_url, BATCHES_URL, batch_id);
        let mut req = self.client.post(&url);

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
            Ok(res.json::<MessageBatch>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    /// Retrieves the results of a message batch.
    #[cfg(feature = "batches")]
    pub fn get_message_batch_results(
        &self,
        batch_id: &str,
    ) -> BoxStream<'static, Result<crate::batches::BatchResult, AnthropicRequestError>> {
        use async_stream::try_stream;
        use futures_util::StreamExt;

        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let oauth_token = self.oauth_token.clone();
        let api_version = self.api_version.clone();
        let headers = self.headers.clone();
        let url = format!("{}/{}/{}/results", self.base_url, BATCHES_URL, batch_id);

        Box::pin(try_stream! {
            let mut req = client.get(&url);

            if let Some(token) = &oauth_token {
                req = req.header("authorization", format!("Bearer {}", token));
            } else if let Some(key) = &api_key {
                req = req.header("x-api-key", key);
            } else {
                Err(AnthropicRequestError::AuthenticationMissing)?;
            }

            let mut req_with_headers = req
                .header("anthropic-version", &api_version);

            for (key, value) in &headers {
                req_with_headers = req_with_headers.header(key, value);
            }

            let response = req_with_headers.send().await?;
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
                        let line = String::from_utf8(line_bytes)?;
                        if !line.trim().is_empty() {
                            let result: crate::batches::BatchResult = serde_json::from_str(&line)?;
                            yield result;
                        }
                    }
                }
                // Process any remaining data in the buffer
                if !buffer.is_empty() {
                    let line = String::from_utf8(buffer)?;
                     if !line.trim().is_empty() {
                        let result: crate::batches::BatchResult = serde_json::from_str(&line)?;
                        yield result;
                    }
                }
            }
        })
    }

    /// Uploads a file to the server.
    #[cfg(feature = "files")]
    pub async fn upload_file(
        &self,
        request: &FileUploadRequest,
    ) -> Result<FileInfo, AnthropicRequestError> {
        let url = format!("{}/{}", self.base_url, FILES_URL);

        let part = reqwest::multipart::Part::bytes(request.content.clone())
            .file_name(request.filename.clone())
            .mime_str(&request.mime_type)?;

        let form = reqwest::multipart::Form::new().part("file", part);

        let mut req = self.client.post(&url).multipart(form);

        if let Some(oauth_token) = &self.oauth_token {
            req = req.header("authorization", format!("Bearer {}", oauth_token));
        } else if let Some(api_key) = &self.api_key {
            req = req.header("x-api-key", api_key);
        } else {
            return Err(AnthropicRequestError::AuthenticationMissing);
        }

        let mut req_with_headers = req
            .header("anthropic-version", &self.api_version)
            .header("anthropic-beta", "files-api-2025-04-14");

        for (key, value) in &self.headers {
            req_with_headers = req_with_headers.header(key, value);
        }

        let res = req_with_headers.send().await?;

        if res.status().is_success() {
            Ok(res.json::<FileInfo>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    /// Lists the files in the workspace.
    #[cfg(feature = "files")]
    pub async fn list_files(
        &self,
        limit: Option<u32>,
        before_id: Option<&str>,
        after_id: Option<&str>,
    ) -> Result<FileListResponse, AnthropicRequestError> {
        let url = format!("{}/{}", self.base_url, FILES_URL);
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
            .header("anthropic-beta", "files-api-2025-04-14");

        for (key, value) in &self.headers {
            req_with_headers = req_with_headers.header(key, value);
        }

        let res = req_with_headers.send().await?;

        if res.status().is_success() {
            Ok(res.json::<FileListResponse>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    /// Retrieves metadata for a specific file.
    #[cfg(feature = "files")]
    pub async fn get_file(&self, file_id: &str) -> Result<FileInfo, AnthropicRequestError> {
        let url = format!("{}/{}/{}", self.base_url, FILES_URL, file_id);
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
            .header("anthropic-beta", "files-api-2025-04-14");

        for (key, value) in &self.headers {
            req_with_headers = req_with_headers.header(key, value);
        }

        let res = req_with_headers.send().await?;

        if res.status().is_success() {
            Ok(res.json::<FileInfo>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    /// Deletes a file from the server.
    #[cfg(feature = "files")]
    pub async fn delete_file(&self, file_id: &str) -> Result<(), AnthropicRequestError> {
        let url = format!("{}/{}/{}", self.base_url, FILES_URL, file_id);
        let mut req = self.client.delete(&url);

        if let Some(oauth_token) = &self.oauth_token {
            req = req.header("authorization", format!("Bearer {}", oauth_token));
        } else if let Some(api_key) = &self.api_key {
            req = req.header("x-api-key", api_key);
        } else {
            return Err(AnthropicRequestError::AuthenticationMissing);
        }

        let mut req_with_headers = req
            .header("anthropic-version", &self.api_version)
            .header("anthropic-beta", "files-api-2025-04-14");

        for (key, value) in &self.headers {
            req_with_headers = req_with_headers.header(key, value);
        }

        let res = req_with_headers.send().await?;

        if res.status().is_success() {
            Ok(())
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }

    /// Downloads a file from the server.
    #[cfg(feature = "files")]
    pub async fn download_file(&self, file_id: &str) -> Result<bytes::Bytes, AnthropicRequestError> {
        let url = format!("{}/{}/{}/content", self.base_url, FILES_URL, file_id);
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
            .header("anthropic-beta", "files-api-2025-04-14");

        for (key, value) in &self.headers {
            req_with_headers = req_with_headers.header(key, value);
        }

        let res = req_with_headers.send().await?;

        if res.status().is_success() {
            Ok(res.bytes().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
    }
}

// Admin API methods
#[cfg(feature = "admin")]
impl Anthropic {
    // Organization Users
    /// Lists the users in the organization.
    pub async fn list_organization_users(&self) -> Result<UserListResponse, AnthropicRequestError> {
        let url = format!("{}/{}/users", self.base_url, ADMIN_ORGANIZATIONS_URL);
        self.get(&url).await
    }

    /// Retrieves a specific user by their ID.
    pub async fn get_organization_user(&self, user_id: &str) -> Result<User, AnthropicRequestError> {
        let url = format!("{}/{}/users/{}", self.base_url, ADMIN_ORGANIZATIONS_URL, user_id);
        self.get(&url).await
    }

    /// Updates a user's role in the organization.
    pub async fn update_organization_user(&self, user_id: &str, role: &crate::admin::users::UserRole) -> Result<User, AnthropicRequestError> {
        let url = format!("{}/{}/users/{}", self.base_url, ADMIN_ORGANIZATIONS_URL, user_id);
        let body = serde_json::json!({ "role": role });
        self.post(&url, &body).await
    }

    /// Removes a user from the organization.
    pub async fn remove_organization_user(&self, user_id: &str) -> Result<(), AnthropicRequestError> {
        let url = format!("{}/{}/users/{}", self.base_url, ADMIN_ORGANIZATIONS_URL, user_id);
        self.delete(&url).await
    }

    // Organization Invites
    /// Lists the pending invitations for the organization.
    pub async fn list_organization_invites(&self) -> Result<InviteListResponse, AnthropicRequestError> {
        let url = format!("{}/{}/invites", self.base_url, ADMIN_ORGANIZATIONS_URL);
        self.get(&url).await
    }

    /// Creates a new invitation to the organization.
    pub async fn create_organization_invite(&self, request: &CreateInviteRequest) -> Result<Invite, AnthropicRequestError> {
        let url = format!("{}/{}/invites", self.base_url, ADMIN_ORGANIZATIONS_URL);
        self.post(&url, request).await
    }

    /// Deletes a pending invitation to the organization.
    pub async fn delete_organization_invite(&self, invite_id: &str) -> Result<(), AnthropicRequestError> {
        let url = format!("{}/{}/invites/{}", self.base_url, ADMIN_ORGANIZATIONS_URL, invite_id);
        self.delete(&url).await
    }

    // Workspaces
    /// Lists the workspaces in the organization.
    pub async fn list_workspaces(&self) -> Result<WorkspaceListResponse, AnthropicRequestError> {
        let url = format!("{}/{}/workspaces", self.base_url, ADMIN_ORGANIZATIONS_URL);
        self.get(&url).await
    }

    /// Retrieves a specific workspace by its ID.
    pub async fn get_workspace(&self, workspace_id: &str) -> Result<Workspace, AnthropicRequestError> {
        let url = format!("{}/{}/workspaces/{}", self.base_url, ADMIN_ORGANIZATIONS_URL, workspace_id);
        self.get(&url).await
    }

    /// Creates a new workspace in the organization.
    pub async fn create_workspace(&self, request: &CreateWorkspaceRequest) -> Result<Workspace, AnthropicRequestError> {
        let url = format!("{}/{}/workspaces", self.base_url, ADMIN_ORGANIZATIONS_URL);
        self.post(&url, request).await
    }

    /// Updates a workspace.
    pub async fn update_workspace(&self, workspace_id: &str, request: &UpdateWorkspaceRequest) -> Result<Workspace, AnthropicRequestError> {
        let url = format!("{}/{}/workspaces/{}", self.base_url, ADMIN_ORGANIZATIONS_URL, workspace_id);
        self.post(&url, request).await
    }

    /// Archives a workspace.
    pub async fn archive_workspace(&self, workspace_id: &str) -> Result<Workspace, AnthropicRequestError> {
        let url = format!("{}/{}/workspaces/{}/archive", self.base_url, ADMIN_ORGANIZATIONS_URL, workspace_id);
        self.post(&url, &serde_json::json!({})).await
    }

    // API Keys
    /// Lists the API keys in the organization.
    pub async fn list_api_keys(&self) -> Result<ApiKeyListResponse, AnthropicRequestError> {
        let url = format!("{}/{}/api_keys", self.base_url, ADMIN_ORGANIZATIONS_URL);
        self.get(&url).await
    }

    /// Retrieves a specific API key by its ID.
    pub async fn get_api_key(&self, api_key_id: &str) -> Result<ApiKey, AnthropicRequestError> {
        let url = format!("{}/{}/api_keys/{}", self.base_url, ADMIN_ORGANIZATIONS_URL, api_key_id);
        self.get(&url).await
    }

    /// Updates an API key.
    pub async fn update_api_key(&self, api_key_id: &str, request: &UpdateApiKeyRequest) -> Result<ApiKey, AnthropicRequestError> {
        let url = format!("{}/{}/api_keys/{}", self.base_url, ADMIN_ORGANIZATIONS_URL, api_key_id);
        self.post(&url, request).await
    }

    // Usage and Cost
    /// Retrieves a usage report for the organization.
    pub async fn get_usage_report(&self) -> Result<UsageReportResponse, AnthropicRequestError> {
        let url = format!("{}/{}/usage_report/messages", self.base_url, ADMIN_ORGANIZATIONS_URL);
        self.get(&url).await
    }

    /// Retrieves a cost report for the organization.
    pub async fn get_cost_report(&self) -> Result<CostReportResponse, AnthropicRequestError> {
        let url = format!("{}/{}/cost_report", self.base_url, ADMIN_ORGANIZATIONS_URL);
        self.get(&url).await
    }

    // Generic helpers for GET and POST requests
    #[doc(hidden)]
    pub async fn get<T: serde::de::DeserializeOwned>(&self, url: &str) -> Result<T, AnthropicRequestError> {
        let mut req = self.client.get(url);
        req = self.add_auth_headers(req);
        let res = req.send().await?;
        self.handle_response(res).await
    }

    #[doc(hidden)]
    pub async fn post<T: serde::de::DeserializeOwned, B: Serialize>(&self, url: &str, body: &B) -> Result<T, AnthropicRequestError> {
        let mut req = self.client.post(url);
        req = self.add_auth_headers(req);
        let res = req.json(body).send().await?;
        self.handle_response(res).await
    }

    async fn delete<T: serde::de::DeserializeOwned>(&self, url: &str) -> Result<T, AnthropicRequestError> {
        let mut req = self.client.delete(url);
        req = self.add_auth_headers(req);
        let res = req.send().await?;
        self.handle_response(res).await
    }

    #[doc(hidden)]
    pub fn add_auth_headers(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut req = req.header("anthropic-version", &self.api_version);
        if let Some(api_key) = &self.api_key {
            req = req.header("x-api-key", api_key);
        }
        // Note: Admin API doesn't use OAuth tokens, so we don't handle them here.
        req
    }

    #[doc(hidden)]
    pub async fn handle_response<T: serde::de::DeserializeOwned>(&self, res: reqwest::Response) -> Result<T, AnthropicRequestError> {
        if res.status().is_success() {
            Ok(res.json::<T>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(error::parse_error_response(status, bytes))
        }
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
