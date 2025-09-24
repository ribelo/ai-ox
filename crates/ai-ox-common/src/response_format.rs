use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Common response-format wrapper shared across OpenAI-compatible providers.
///
/// Providers expect a tagged object with a `type` discriminator.
/// `JsonSchema` mirrors OpenAI/Groq requirement where the schema payload is nested
/// beneath a `json_schema` key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: Value },
}

impl ResponseFormat {
    /// Helper for constructing the `json_schema` variant without repeating the type tag.
    #[must_use]
    pub fn json_schema(json_schema: Value) -> Self {
        Self::JsonSchema { json_schema }
    }

    /// Convert the response format into a raw `serde_json::Value`.
    pub fn into_value(self) -> Value {
        serde_json::to_value(self).expect("ResponseFormat should serialize to JSON value")
    }

    /// Borrowing conversion to `Value` when mutation isn't required.
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("ResponseFormat should serialize to JSON value")
    }
}
