use super::users::UserRole;
use serde::{Deserialize, Serialize};

/// The status of an invitation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InviteStatus {
    Accepted,
    Expired,
    Deleted,
    Pending,
}

/// Information about an invitation to the organization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Invite {
    /// The unique identifier for the invite.
    pub id: String,
    /// The type of the object, which is always "invite".
    #[serde(rename = "type")]
    pub object_type: String,
    /// The email address of the user being invited.
    pub email: String,
    /// The timestamp of when the invite expires.
    pub expires_at: String,
    /// The timestamp of when the invite was created.
    pub invited_at: String,
    /// The role of the user being invited.
    pub role: UserRole,
    /// The status of the invite.
    pub status: InviteStatus,
}

/// A request to create a new invitation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateInviteRequest {
    /// The email address of the user to invite.
    pub email: String,
    /// The role to assign to the invited user.
    pub role: UserRole,
}

/// A response containing a list of invitations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InviteListResponse {
    /// The list of invitations.
    pub data: Vec<Invite>,
    /// Indicates if there are more invitations to fetch.
    pub has_more: bool,
    /// The ID of the first invitation in the list.
    pub first_id: Option<String>,
    /// The ID of the last invitation in the list.
    pub last_id: Option<String>,
}
