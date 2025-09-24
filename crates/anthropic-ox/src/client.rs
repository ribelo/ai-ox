use bon::Builder;
use core::fmt;
use futures_util::StreamExt;
use futures_util::stream::{self, BoxStream};
#[cfg(feature = "leaky-bucket")]
use leaky_bucket::RateLimiter;
#[cfg(feature = "leaky-bucket")]
use std::sync::Arc;

use crate::{
    error::{self, AnthropicRequestError},
    internal::{AnthropicRequestHelper, Endpoint, HttpMethod},
    request,
    response::{self, StreamEvent},
};

#[cfg(feature = "admin")]
use crate::admin::{
    api_keys::{ApiKey, ApiKeyListResponse, UpdateApiKeyRequest},
    invites::{CreateInviteRequest, Invite, InviteListResponse},
    usage::{CostReportResponse, UsageReportResponse},
    users::{User, UserListResponse},
    workspaces::{
        CreateWorkspaceRequest, UpdateWorkspaceRequest, Workspace, WorkspaceListResponse,
    },
};
#[cfg(feature = "batches")]
use crate::batches::{BatchListResponse, MessageBatch, MessageBatchRequest};
#[cfg(feature = "files")]
use crate::files::{FileInfo, FileListResponse, FileUploadRequest};
#[cfg(feature = "models")]
use crate::models::{ModelInfo, ModelsListResponse};
#[cfg(feature = "tokens")]
use crate::tokens::{TokenCountRequest, TokenCountResponse};

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
    /// Create a request helper instance for this client
    fn request_helper(&self) -> Result<AnthropicRequestHelper, AnthropicRequestError> {
        AnthropicRequestHelper::new(
            self.client.clone(),
            &self.base_url,
            &self.api_key,
            &self.oauth_token,
            &self.api_version,
            &self.headers,
        )
    }

    /// Generic method for API requests that return JSON
    async fn api_request<T: serde::de::DeserializeOwned>(
        &self,
        endpoint: Endpoint,
    ) -> Result<T, AnthropicRequestError> {
        let helper = self.request_helper()?;
        helper.request(&endpoint).await
    }

    /// Generic method for API requests with JSON body
    async fn api_request_with_body<T: serde::de::DeserializeOwned, B: serde::Serialize>(
        &self,
        endpoint: Endpoint,
        body: &B,
    ) -> Result<T, AnthropicRequestError> {
        let helper = self.request_helper()?;
        helper.request_json(&endpoint, Some(body)).await
    }

    /// Generic method for delete requests
    async fn api_delete(&self, endpoint: Endpoint) -> Result<(), AnthropicRequestError> {
        let helper = self.request_helper()?;
        helper.request_unit(&endpoint).await
    }

    /// Generic method for requests that return raw bytes
    async fn api_request_bytes(
        &self,
        endpoint: Endpoint,
    ) -> Result<bytes::Bytes, AnthropicRequestError> {
        let helper = self.request_helper()?;
        helper.request_bytes(&endpoint).await
    }

    /// Generic method for streaming requests
    fn api_stream<T, B>(
        &self,
        endpoint: Endpoint,
        body: Option<&B>,
    ) -> BoxStream<'static, Result<T, AnthropicRequestError>>
    where
        T: serde::de::DeserializeOwned + Send + 'static,
        B: serde::Serialize,
    {
        match self.request_helper() {
            Ok(helper) => helper.stream(&endpoint, body),
            Err(err) => stream::once(async move { Err(err) }).boxed(),
        }
    }

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
        let mut query_params = Vec::new();
        if let Some(limit) = limit {
            query_params.push(("limit".to_string(), limit.to_string()));
        }
        if let Some(before_id) = before_id {
            query_params.push(("before_id".to_string(), before_id.to_string()));
        }
        if let Some(after_id) = after_id {
            query_params.push(("after_id".to_string(), after_id.to_string()));
        }

        let endpoint = Endpoint::new(MODELS_URL, HttpMethod::Get).with_query_params(query_params);

        self.api_request(endpoint).await
    }

    /// Retrieves a specific model by its ID.
    #[cfg(feature = "models")]
    pub async fn get_model(&self, model_id: &str) -> Result<ModelInfo, AnthropicRequestError> {
        let endpoint = Endpoint::new(format!("{}/{}", MODELS_URL, model_id), HttpMethod::Get);
        self.api_request(endpoint).await
    }

    /// Counts the number of tokens in a message.
    #[cfg(feature = "tokens")]
    pub async fn count_tokens(
        &self,
        request: &TokenCountRequest,
    ) -> Result<TokenCountResponse, AnthropicRequestError> {
        let endpoint = Endpoint::new(TOKENS_URL, HttpMethod::Post);
        self.api_request_with_body(endpoint, request).await
    }

    pub async fn send(
        &self,
        request: &request::ChatRequest,
    ) -> Result<response::ChatResponse, AnthropicRequestError> {
        let endpoint = Endpoint::new(CHAT_URL, HttpMethod::Post);
        self.api_request_with_body(endpoint, request).await
    }

    pub fn stream(
        &self,
        request: &request::ChatRequest,
    ) -> BoxStream<'static, Result<StreamEvent, AnthropicRequestError>> {
        let endpoint = Endpoint::new(CHAT_URL, HttpMethod::Post);
        self.api_stream(endpoint, Some(request))
    }

    /// Creates a new message batch.
    #[cfg(feature = "batches")]
    pub async fn create_message_batch(
        &self,
        request: &MessageBatchRequest,
    ) -> Result<MessageBatch, AnthropicRequestError> {
        let endpoint = Endpoint::new(BATCHES_URL, HttpMethod::Post);
        self.api_request_with_body(endpoint, request).await
    }

    /// Retrieves a message batch.
    #[cfg(feature = "batches")]
    pub async fn get_message_batch(
        &self,
        batch_id: &str,
    ) -> Result<MessageBatch, AnthropicRequestError> {
        let endpoint = Endpoint::new(format!("{}/{}", BATCHES_URL, batch_id), HttpMethod::Get);
        self.api_request(endpoint).await
    }

    /// Lists message batches.
    #[cfg(feature = "batches")]
    pub async fn list_message_batches(
        &self,
        limit: Option<u32>,
        after: Option<&str>,
    ) -> Result<BatchListResponse, AnthropicRequestError> {
        let mut query_params = Vec::new();
        if let Some(limit) = limit {
            query_params.push(("limit".to_string(), limit.to_string()));
        }
        if let Some(after) = after {
            query_params.push(("after".to_string(), after.to_string()));
        }

        let endpoint = Endpoint::new(BATCHES_URL, HttpMethod::Get).with_query_params(query_params);

        self.api_request(endpoint).await
    }

    /// Cancels a message batch.
    #[cfg(feature = "batches")]
    pub async fn cancel_message_batch(
        &self,
        batch_id: &str,
    ) -> Result<MessageBatch, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/{}/cancel", BATCHES_URL, batch_id),
            HttpMethod::Post,
        );
        // For POST requests that need empty body
        let empty_body = serde_json::json!({});
        self.api_request_with_body(endpoint, &empty_body).await
    }

    /// Retrieves the results of a message batch.
    #[cfg(feature = "batches")]
    pub fn get_message_batch_results(
        &self,
        batch_id: &str,
    ) -> BoxStream<'static, Result<crate::batches::BatchResult, AnthropicRequestError>> {
        let endpoint = Endpoint::new(
            format!("{}/{}/results", BATCHES_URL, batch_id),
            HttpMethod::Get,
        );
        match self.request_helper() {
            Ok(helper) => helper.stream_jsonl(&endpoint),
            Err(err) => stream::once(async move { Err(err) }).boxed(),
        }
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
        let mut query_params = Vec::new();
        if let Some(limit) = limit {
            query_params.push(("limit".to_string(), limit.to_string()));
        }
        if let Some(before_id) = before_id {
            query_params.push(("before_id".to_string(), before_id.to_string()));
        }
        if let Some(after_id) = after_id {
            query_params.push(("after_id".to_string(), after_id.to_string()));
        }

        let endpoint = Endpoint::new(FILES_URL, HttpMethod::Get)
            .with_beta("files-api-2025-04-14")
            .with_query_params(query_params);
        self.api_request(endpoint).await
    }

    /// Retrieves metadata for a specific file.
    #[cfg(feature = "files")]
    pub async fn get_file(&self, file_id: &str) -> Result<FileInfo, AnthropicRequestError> {
        let endpoint = Endpoint::new(format!("{}/{}", FILES_URL, file_id), HttpMethod::Get)
            .with_beta("files-api-2025-04-14");
        self.api_request(endpoint).await
    }

    /// Deletes a file from the server.
    #[cfg(feature = "files")]
    pub async fn delete_file(&self, file_id: &str) -> Result<(), AnthropicRequestError> {
        let endpoint = Endpoint::new(format!("{}/{}", FILES_URL, file_id), HttpMethod::Delete)
            .with_beta("files-api-2025-04-14");
        self.api_delete(endpoint).await
    }

    /// Downloads a file from the server.
    #[cfg(feature = "files")]
    pub async fn download_file(
        &self,
        file_id: &str,
    ) -> Result<bytes::Bytes, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/{}/content", FILES_URL, file_id),
            HttpMethod::Get,
        )
        .with_beta("files-api-2025-04-14");
        self.api_request_bytes(endpoint).await
    }
}

// Admin API methods
#[cfg(feature = "admin")]
impl Anthropic {
    // Organization Users
    /// Lists the users in the organization.
    pub async fn list_organization_users(&self) -> Result<UserListResponse, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/users", ADMIN_ORGANIZATIONS_URL),
            HttpMethod::Get,
        );
        self.api_request(endpoint).await
    }

    /// Retrieves a specific user by their ID.
    pub async fn get_organization_user(
        &self,
        user_id: &str,
    ) -> Result<User, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/users/{}", ADMIN_ORGANIZATIONS_URL, user_id),
            HttpMethod::Get,
        );
        self.api_request(endpoint).await
    }

    /// Updates a user's role in the organization.
    pub async fn update_organization_user(
        &self,
        user_id: &str,
        role: &crate::admin::users::UserRole,
    ) -> Result<User, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/users/{}", ADMIN_ORGANIZATIONS_URL, user_id),
            HttpMethod::Post,
        );
        let body = serde_json::json!({ "role": role });
        self.api_request_with_body(endpoint, &body).await
    }

    /// Removes a user from the organization.
    pub async fn remove_organization_user(
        &self,
        user_id: &str,
    ) -> Result<(), AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/users/{}", ADMIN_ORGANIZATIONS_URL, user_id),
            HttpMethod::Delete,
        );
        self.api_delete(endpoint).await
    }

    // Organization Invites
    /// Lists the pending invitations for the organization.
    pub async fn list_organization_invites(
        &self,
    ) -> Result<InviteListResponse, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/invites", ADMIN_ORGANIZATIONS_URL),
            HttpMethod::Get,
        );
        self.api_request(endpoint).await
    }

    /// Creates a new invitation to the organization.
    pub async fn create_organization_invite(
        &self,
        request: &CreateInviteRequest,
    ) -> Result<Invite, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/invites", ADMIN_ORGANIZATIONS_URL),
            HttpMethod::Post,
        );
        self.api_request_with_body(endpoint, request).await
    }

    /// Deletes a pending invitation to the organization.
    pub async fn delete_organization_invite(
        &self,
        invite_id: &str,
    ) -> Result<(), AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/invites/{}", ADMIN_ORGANIZATIONS_URL, invite_id),
            HttpMethod::Delete,
        );
        self.api_delete(endpoint).await
    }

    // Workspaces
    /// Lists the workspaces in the organization.
    pub async fn list_workspaces(&self) -> Result<WorkspaceListResponse, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/workspaces", ADMIN_ORGANIZATIONS_URL),
            HttpMethod::Get,
        );
        self.api_request(endpoint).await
    }

    /// Retrieves a specific workspace by its ID.
    pub async fn get_workspace(
        &self,
        workspace_id: &str,
    ) -> Result<Workspace, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/workspaces/{}", ADMIN_ORGANIZATIONS_URL, workspace_id),
            HttpMethod::Get,
        );
        self.api_request(endpoint).await
    }

    /// Creates a new workspace in the organization.
    pub async fn create_workspace(
        &self,
        request: &CreateWorkspaceRequest,
    ) -> Result<Workspace, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/workspaces", ADMIN_ORGANIZATIONS_URL),
            HttpMethod::Post,
        );
        self.api_request_with_body(endpoint, request).await
    }

    /// Updates a workspace.
    pub async fn update_workspace(
        &self,
        workspace_id: &str,
        request: &UpdateWorkspaceRequest,
    ) -> Result<Workspace, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/workspaces/{}", ADMIN_ORGANIZATIONS_URL, workspace_id),
            HttpMethod::Post,
        );
        self.api_request_with_body(endpoint, request).await
    }

    /// Archives a workspace.
    pub async fn archive_workspace(
        &self,
        workspace_id: &str,
    ) -> Result<Workspace, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!(
                "{}/workspaces/{}/archive",
                ADMIN_ORGANIZATIONS_URL, workspace_id
            ),
            HttpMethod::Post,
        );
        let empty_body = serde_json::json!({});
        self.api_request_with_body(endpoint, &empty_body).await
    }

    // API Keys
    /// Lists the API keys in the organization.
    pub async fn list_api_keys(&self) -> Result<ApiKeyListResponse, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/api_keys", ADMIN_ORGANIZATIONS_URL),
            HttpMethod::Get,
        );
        self.api_request(endpoint).await
    }

    /// Retrieves a specific API key by its ID.
    pub async fn get_api_key(&self, api_key_id: &str) -> Result<ApiKey, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/api_keys/{}", ADMIN_ORGANIZATIONS_URL, api_key_id),
            HttpMethod::Get,
        );
        self.api_request(endpoint).await
    }

    /// Updates an API key.
    pub async fn update_api_key(
        &self,
        api_key_id: &str,
        request: &UpdateApiKeyRequest,
    ) -> Result<ApiKey, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/api_keys/{}", ADMIN_ORGANIZATIONS_URL, api_key_id),
            HttpMethod::Post,
        );
        self.api_request_with_body(endpoint, request).await
    }

    // Usage and Cost
    /// Retrieves a usage report for the organization.
    pub async fn get_usage_report(&self) -> Result<UsageReportResponse, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/usage_report/messages", ADMIN_ORGANIZATIONS_URL),
            HttpMethod::Get,
        );
        self.api_request(endpoint).await
    }

    /// Retrieves a cost report for the organization.
    pub async fn get_cost_report(&self) -> Result<CostReportResponse, AnthropicRequestError> {
        let endpoint = Endpoint::new(
            format!("{}/cost_report", ADMIN_ORGANIZATIONS_URL),
            HttpMethod::Get,
        );
        self.api_request(endpoint).await
    }
}

impl fmt::Debug for Anthropic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Anthropic")
            .field("api_key", &"[REDACTED]")
            .field(
                "oauth_token",
                &self.oauth_token.as_ref().map(|_| "[REDACTED]"),
            )
            .field("client", &self.client)
            .field("base_url", &self.base_url)
            .field("api_version", &self.api_version)
            .field("headers", &self.headers)
            .finish_non_exhaustive()
    }
}
