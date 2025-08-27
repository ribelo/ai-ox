use serde::{Deserialize, Serialize};

/// The status of an API key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApiKeyStatus {
    Active,
    Inactive,
    Archived,
}

/// Information about the actor that created an object.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CreatedBy {
    /// The unique identifier of the actor.
    pub id: String,
    /// The type of the actor.
    #[serde(rename = "type")]
    pub object_type: String,
}

/// Information about an API key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApiKey {
    /// The unique identifier for the API key.
    pub id: String,
    /// The type of the object, which is always "api_key".
    #[serde(rename = "type")]
    pub object_type: String,
    /// The name of the API key.
    pub name: String,
    /// A partially redacted hint for the API key.
    pub partial_key_hint: Option<String>,
    /// The status of the API key.
    pub status: ApiKeyStatus,
    /// The timestamp of when the API key was created.
    pub created_at: String,
    /// The actor that created the API key.
    pub created_by: CreatedBy,
    /// The ID of the workspace associated with the API key.
    pub workspace_id: Option<String>,
}

/// A request to update an API key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateApiKeyRequest {
    /// The new name for the API key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The new status for the API key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<ApiKeyStatus>,
}

/// A response containing a list of API keys.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApiKeyListResponse {
    /// The list of API keys.
    pub data: Vec<ApiKey>,
    /// Indicates if there are more API keys to fetch.
    pub has_more: bool,
    /// The ID of the first API key in the list.
    pub first_id: Option<String>,
    /// The ID of the last API key in the list.
    pub last_id: Option<String>,
}
