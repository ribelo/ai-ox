use serde::{Deserialize, Serialize};

/// Information about a single model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInfo {
    /// The unique identifier for the model.
    pub id: String,
    /// The display name of the model.
    pub display_name: String,
    /// The creation date of the model.
    pub created_at: String,
    /// The type of the object, which is always "model".
    #[serde(rename = "type")]
    pub object_type: String, // "model"
}

/// A response containing a list of models.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelsListResponse {
    /// The list of models.
    pub data: Vec<ModelInfo>,
    /// Whether there are more models to fetch.
    pub has_more: bool,
    /// The ID of the first model in the list.
    pub first_id: Option<String>,
    /// The ID of the last model in the list.
    pub last_id: Option<String>,
}
