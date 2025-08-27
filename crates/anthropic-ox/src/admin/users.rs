use serde::{Deserialize, Serialize};

/// The role of a user in the organization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UserRole {
    User,
    Developer,
    Billing,
    Admin,
    ClaudeCodeUser,
}

/// Information about a user in the organization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct User {
    /// The unique identifier for the user.
    pub id: String,
    /// The type of the object, which is always "user".
    #[serde(rename = "type")]
    pub object_type: String,
    /// The user's email address.
    pub email: String,
    /// The user's name.
    pub name: String,
    /// The user's role in the organization.
    pub role: UserRole,
    /// The timestamp of when the user was added to the organization.
    pub added_at: String,
}

/// A response containing a list of users.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UserListResponse {
    /// The list of users.
    pub data: Vec<User>,
    /// Indicates if there are more users to fetch.
    pub has_more: bool,
    /// The ID of the first user in the list.
    pub first_id: Option<String>,
    /// The ID of the last user in the list.
    pub last_id: Option<String>,
}
