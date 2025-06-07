use serde::{Deserialize, Serialize};

use super::Tool;

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleSearchRetrieval {
    pub dynamic_retrieval_config: DynamicRetrievalConfig,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct DynamicRetrievalConfig {
    pub mode: Mode,
    pub dynamic_threshold: Option<f64>,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Mode {
    /// Always trigger retrieval.
    ModeUnspecified,
    /// Run retrieval only when system decides it is necessary.
    #[default]
    ModeDynamic,
}

impl From<GoogleSearchRetrieval> for Tool {
    fn from(value: GoogleSearchRetrieval) -> Self {
        Tool::GoogleSearchRetrieval {
            google_search_retrieval: value,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct GoogleSearch {}

impl From<GoogleSearch> for Tool {
    fn from(value: GoogleSearch) -> Self {
        Tool::GoogleSearch(value)
    }
}
