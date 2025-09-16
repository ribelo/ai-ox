use serde::{Deserialize, Serialize};
use strum::Display;

/// Available models for OpenAI
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Display)]
#[serde(rename_all = "kebab-case")]
pub enum Model {
    #[serde(rename = "gpt-3.5-turbo")]
    Gpt3_5Turbo,

    #[serde(rename = "gpt-4")]
    Gpt4,

    /// Custom model (for models not in this enum)
    #[serde(untagged)]
    Custom(String),
}

impl Model {
    /// Get the string representation of the model
    pub fn as_str(&self) -> &str {
        match self {
            Model::Gpt3_5Turbo => "gpt-3.5-turbo",
            Model::Gpt4 => "gpt-4",
            Model::Custom(s) => s,
        }
    }

    /// Check if this is a chat completion model
    pub fn supports_chat(&self) -> bool {
        match self {
            Model::Gpt3_5Turbo | Model::Gpt4 => true,
            Model::Custom(_) => true, // Assume custom models support chat
        }
    }

    /// Get the maximum context length for this model
    pub fn max_context_length(&self) -> Option<usize> {
        match self {
            Model::Gpt3_5Turbo => Some(4096),
            Model::Gpt4 => Some(8192),
            Model::Custom(_) => None, // Unknown for custom models
        }
    }

    /// Check if this model supports tool/function calling
    pub fn supports_tools(&self) -> bool {
        match self {
            Model::Gpt3_5Turbo | Model::Gpt4 => true,
            Model::Custom(_) => false, // Conservative assumption
        }
    }
}

impl From<String> for Model {
    fn from(s: String) -> Self {
        match s.as_str() {
            "gpt-3.5-turbo" => Model::Gpt3_5Turbo,
            "gpt-4" => Model::Gpt4,
            _ => Model::Custom(s),
        }
    }
}

impl From<&str> for Model {
    fn from(s: &str) -> Self {
        Model::from(s.to_string())
    }
}
