use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

/// Available models for {{Provider}}
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum Model {
    /// {{Model descriptions - customize based on provider}}
    #[serde(rename = "{{model_1}}")]
    #[strum(serialize = "{{model_1}}")]
    {{Model1}},
    
    #[serde(rename = "{{model_2}}")]
    #[strum(serialize = "{{model_2}}")]
    {{Model2}},
    
    /// Custom model (for models not in this enum)
    #[serde(untagged)]
    Custom(String),
}

impl Model {
    /// Get the string representation of the model
    pub fn as_str(&self) -> &str {
        match self {
            Model::{{Model1}} => "{{model_1}}",
            Model::{{Model2}} => "{{model_2}}",
            Model::Custom(s) => s,
        }
    }
    
    /// Check if this is a chat completion model
    pub fn supports_chat(&self) -> bool {
        match self {
            Model::{{Model1}} | Model::{{Model2}} => true,
            Model::Custom(_) => true, // Assume custom models support chat
        }
    }
    
    /// Get the maximum context length for this model
    pub fn max_context_length(&self) -> Option<usize> {
        match self {
            Model::{{Model1}} => Some({{context_1}}),
            Model::{{Model2}} => Some({{context_2}}),
            Model::Custom(_) => None, // Unknown for custom models
        }
    }
    
    /// Check if this model supports tool/function calling
    pub fn supports_tools(&self) -> bool {
        match self {
            Model::{{Model1}} | Model::{{Model2}} => true,
            Model::Custom(_) => false, // Conservative assumption
        }
    }
}

impl From<String> for Model {
    fn from(s: String) -> Self {
        match s.as_str() {
            "{{model_1}}" => Model::{{Model1}},
            "{{model_2}}" => Model::{{Model2}},
            _ => Model::Custom(s),
        }
    }
}

impl From<&str> for Model {
    fn from(s: &str) -> Self {
        Model::from(s.to_string())
    }
}