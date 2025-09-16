use crate::{
    content::Content,
    generate_content::usage::UsageMetadata,
    tool::{config::ToolConfig, Tool},
    Gemini, GeminiRequestError,
};
use ai_ox_common::request_builder::{Endpoint, HttpMethod};
use bon::Builder;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A handler for interacting with the caching-related endpoints of the Gemini API.
///
/// Created via the `gemini.caches()` method.
#[derive(Debug, Clone)]
pub struct Caches {
    gemini: Gemini,
}

impl Caches {
    /// Creates a new `Caches` instance.
    /// This is not intended to be called directly, but rather through `gemini.caches()`.
    pub(crate) fn new(gemini: Gemini) -> Self {
        Self { gemini }
    }

    /// Creates a new `CachedContent` resource.
    ///
    /// Returns a builder that can be used to configure and send the request.
    pub fn create(
        &self,
    ) -> CreateCachedContentRequestBuilder<create_cached_content_request_builder::SetGemini> {
        CreateCachedContentRequest::builder().gemini(self.gemini.clone())
    }

    /// Gets a specific `CachedContent` resource by its name.
    ///
    /// # Arguments
    ///
    /// * `name` - The resource name of the cached content to retrieve, e.g., "cachedContents/my-cache-123".
    pub async fn get(&self, name: &str) -> Result<CachedContent, GeminiRequestError> {
        let helper = self.gemini.request_helper()?;
        let endpoint = Endpoint::new(format!("v1beta/{}", name), HttpMethod::Get);
        helper.request(endpoint).await
    }

    /// Lists all `CachedContent` resources.
    ///
    /// # Arguments
    ///
    /// * `page_size` - Optional. The maximum number of cached contents to return.
    /// * `page_token` - Optional. A page token, received from a previous `list` call.
    pub async fn list(
        &self,
        page_size: Option<u32>,
        page_token: Option<String>,
    ) -> Result<ListCachedContentsResponse, GeminiRequestError> {
        let helper = self.gemini.request_helper()?;
        let mut endpoint = Endpoint::new("v1beta/cachedContents", HttpMethod::Get);

        let mut params = Vec::new();
        if let Some(size) = page_size {
            params.push(("pageSize".to_string(), size.to_string()));
        }
        if let Some(token) = page_token {
            params.push(("pageToken".to_string(), token));
        }
        if !params.is_empty() {
            endpoint = endpoint.with_query_params(params);
        }

        helper.request(endpoint).await
    }

    /// Updates a `CachedContent` resource.
    ///
    /// Currently, only the `ttl` or `expire_time` can be updated.
    ///
    /// # Arguments
    ///
    /// * `name` - The resource name of the cached content to update.
    pub fn update<'a>(
        &self,
        name: &'a str,
    ) -> UpdateCachedContentRequestBuilder<
        'a,
        update_cached_content_request_builder::SetName<
            update_cached_content_request_builder::SetGemini,
        >,
    > {
        UpdateCachedContentRequest::builder()
            .gemini(self.gemini.clone())
            .name(name)
    }

    /// Deletes a `CachedContent` resource.
    ///
    /// # Arguments
    ///
    /// * `name` - The resource name of the cached content to delete.
    pub async fn delete(&self, name: &str) -> Result<(), GeminiRequestError> {
        let helper = self.gemini.request_helper()?;
        let endpoint = Endpoint::new(format!("v1beta/{}", name), HttpMethod::Delete);
        helper.request_unit(endpoint).await
    }
}

/// Represents the cached content resource from the Gemini API.
///
/// This struct holds information about a piece of content that has been preprocessed
/// and can be used in subsequent requests to the Gemini API to reduce latency and cost.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CachedContent {
    /// Output only. Identifier. The resource name of the cached content.
    /// Format: `cachedContents/{id}`
    pub name: String,

    /// Optional. Immutable. The user-generated meaningful display name of the cached content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,

    /// Required. Immutable. The name of the `Model` to use for the cached content.
    /// Format: `models/{model}`
    pub model: String,

    /// Optional. Input only. Immutable. The content to cache.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contents: Vec<Content>,

    /// Optional. Input only. Immutable. A list of `Tools` the model may use.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,

    /// Optional. Input only. Immutable. Tool config.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,

    /// Optional. Input only. Immutable. Developer-set system instruction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,

    /// Output only. Creation time of the cache entry.
    pub create_time: String, // Using String for RFC 3339 timestamp

    /// Output only. When the cache entry was last updated.
    pub update_time: String, // Using String for RFC 3339 timestamp

    /// Timestamp in UTC of when this resource is considered expired.
    pub expire_time: String, // Using String for RFC 3339 timestamp

    /// Input only. The time-to-live for this resource.
    /// Format: A duration in seconds with up to nine fractional digits, ending with 's'. Example: "300.5s".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,

    /// Output only. Metadata on the usage of the cached content.
    pub usage_metadata: UsageMetadata,
}

/// The response for a `cachedContents.list` request.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ListCachedContentsResponse {
    /// List of cached contents.
    pub cached_contents: Vec<CachedContent>,

    /// A token, which can be sent as `pageToken` to retrieve the next page.
    /// If this field is omitted, there are no subsequent pages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_page_token: Option<String>,
}

/// A request to update a `CachedContent` resource.
///
/// This struct is used to build the request payload for patching a cache entry.
/// It is typically constructed using the `Caches::update()` builder.
#[derive(Debug, Clone, Serialize, Builder)]
#[serde(rename_all = "camelCase")]
pub struct UpdateCachedContentRequest<'a> {
    /// The new time-to-live for this resource.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,

    /// The new expiration time for this resource in RFC 3339 format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expire_time: Option<String>,

    #[serde(skip)]
    name: &'a str,

    #[serde(skip)]
    gemini: Gemini,
}

impl<'a> UpdateCachedContentRequest<'a> {
    /// Sends the PATCH request to update the `CachedContent`.
    ///
    /// # Errors
    /// Returns a `GeminiRequestError` if the request fails.
    pub async fn send(&self) -> Result<CachedContent, GeminiRequestError> {
        let mut request_body = serde_json::json!({});
        let mut mask_paths = Vec::new();

        if let Some(ttl) = &self.ttl {
            request_body["ttl"] = ttl.clone().into();
            mask_paths.push("ttl");
        }

        if let Some(expire_time) = &self.expire_time {
            request_body["expireTime"] = expire_time.clone().into();
            mask_paths.push("expireTime");
        }

        let helper = self.gemini.request_helper()?;
        let mut endpoint = Endpoint::new(format!("v1beta/{}", self.name), HttpMethod::Patch);
        endpoint =
            endpoint.with_query_params(vec![("updateMask".to_string(), mask_paths.join(","))]);

        helper.request_json(endpoint, Some(&request_body)).await
    }
}

/// A request to create a `CachedContent` resource.
///
/// This struct is used to build the request payload for creating a new cache entry.
/// It is typically constructed using the `Caches::create()` builder.
#[derive(Debug, Clone, Serialize, Builder)]
#[serde(rename_all = "camelCase")]
pub struct CreateCachedContentRequest {
    /// Optional. The content to cache.
    #[builder(field)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub contents: Vec<Content>,

    /// Optional. A list of `Tools` the model may use.
    #[builder(field)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,

    /// Optional. The time-to-live for this resource.
    #[builder(field)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,

    /// Required. The name of the `Model` to use for this cached content.
    /// Format: `models/{model}`
    #[builder(into)]
    pub model: String,

    /// Optional. Tool configuration for any `Tool`s specified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,

    /// Optional. Developer-set system instruction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,

    /// Optional. The user-generated meaningful display name of the cached content.
    #[builder(into)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,

    #[serde(skip)]
    gemini: Gemini,
}

impl<S: create_cached_content_request_builder::State> CreateCachedContentRequestBuilder<S> {
    /// Sets the contents to be cached.
    pub fn contents(mut self, contents: impl IntoIterator<Item = impl Into<Content>>) -> Self {
        self.contents = contents.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the tools that can be used with the cached content.
    pub fn tools(mut self, tools: impl IntoIterator<Item = impl Into<Tool>>) -> Self {
        self.tools = tools.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the time-to-live (TTL) for the cache.
    pub fn ttl(mut self, duration: Duration) -> Self {
        self.ttl = Some(format!("{}s", duration.as_secs_f64()));
        self
    }
}

impl CreateCachedContentRequest {
    /// Sends the request to create a `CachedContent` resource.
    ///
    /// # Errors
    /// Returns a `GeminiRequestError` if the request fails, for example due to
    /// network issues, invalid API key, or an invalid request payload.
    pub async fn send(&self) -> Result<CachedContent, GeminiRequestError> {
        let helper = self.gemini.request_helper()?;
        let endpoint = Endpoint::new("v1beta/cachedContents", HttpMethod::Post);
        helper.request_json(endpoint, Some(self)).await
    }
}
