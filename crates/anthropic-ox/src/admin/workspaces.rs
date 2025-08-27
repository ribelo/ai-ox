use serde::{Deserialize, Serialize};

/// Information about a workspace.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Workspace {
    /// The unique identifier for the workspace.
    pub id: String,
    /// The type of the object, which is always "workspace".
    #[serde(rename = "type")]
    pub object_type: String,
    /// The name of the workspace.
    pub name: String,
    /// The hex color code for the workspace.
    pub display_color: String,
    /// The timestamp of when the workspace was created.
    pub created_at: String,
    /// The timestamp of when the workspace was archived, if applicable.
    pub archived_at: Option<String>,
}

/// A request to create a new workspace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateWorkspaceRequest {
    /// The name of the workspace.
    pub name: String,
    /// The hex color code for the workspace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_color: Option<String>,
}

/// A request to update a workspace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateWorkspaceRequest {
    /// The new name of the workspace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The new hex color code for the workspace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_color: Option<String>,
}

/// A response containing a list of workspaces.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceListResponse {
    /// The list of workspaces.
    pub data: Vec<Workspace>,
    /// Indicates if there are more workspaces to fetch.
    pub has_more: bool,
    /// The ID of the first workspace in the list.
    pub first_id: Option<String>,
    /// The ID of the last workspace in the list.
    pub last_id: Option<String>,
}
