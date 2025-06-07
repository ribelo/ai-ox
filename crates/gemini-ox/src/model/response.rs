use serde::Deserialize;

/// Represents a Gemini model with its capabilities and configuration
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    /// The unique name identifier for the model
    pub name: String,
    /// The base model this model is built upon
    pub base_model_id: String,
    /// The version string of the model
    pub version: String,
    /// Human-readable display name for the model
    pub display_name: String,
    /// Description of the model's capabilities
    pub description: String,
    /// Maximum number of input tokens the model can process
    pub input_token_limit: u32,
    /// Maximum number of output tokens the model can generate
    pub output_token_limit: u32,
    /// List of generation methods supported by this model
    pub supported_generation_methods: Vec<String>,
    /// Default temperature setting for the model, if available
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Default top-p setting for the model, if available
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Default top-k setting for the model, if available
    #[serde(default)]
    pub top_k: Option<u32>,
}

/// Response from the list models API endpoint
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListModelsResponse {
    /// List of available models
    pub models: Vec<Model>,
    /// Token for retrieving the next page of results, if available
    pub next_page_token: Option<String>,
}
