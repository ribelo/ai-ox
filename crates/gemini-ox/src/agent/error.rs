use serde::{Serialize, Serializer, ser::SerializeStructVariant};
use thiserror::Error;

use crate::{GeminiRequestError, tool::FunctionCallError};

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("API request failed: {0}")]
    ApiError(#[from] GeminiRequestError),

    #[error("Agent reached maximum iterations ({limit}) without completing task")]
    MaxIterationsReached { limit: usize },

    #[error("Function call execution failed: {0}")]
    FunctionCallError(#[from] FunctionCallError),
    #[error("Schema generation failed: {0}")]
    SchemaGenerationFailed(String),
    #[error("Failed to parse LLM response: {source}. Response text: '{response_text}'")]
    ResponseParsingFailed {
        #[source]
        source: serde_json::Error,
        response_text: String,
    },
}

impl Serialize for AgentError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            AgentError::ApiError(e) => serializer.serialize_newtype_variant(
                "AgentError",
                0, // Variant index
                "ApiError",
                e, // GeminiRequestError is already Serialize
            ),
            AgentError::MaxIterationsReached { limit } => {
                let mut state = serializer.serialize_struct_variant(
                    "AgentError",
                    1, // Variant index
                    "MaxIterationsReached",
                    1, // Number of fields
                )?;
                state.serialize_field("limit", limit)?;
                state.end()
            }
            AgentError::FunctionCallError(e) => serializer.serialize_newtype_variant(
                "AgentError",
                2, // Variant index
                "FunctionCallError",
                &e, // Serialize the underlying error message
            ),
            AgentError::SchemaGenerationFailed(msg) => serializer.serialize_newtype_variant(
                "AgentError",
                3, // Variant index
                "SchemaGenerationFailed",
                msg, // Serialize the message string
            ),
            AgentError::ResponseParsingFailed {
                source,
                response_text,
            } => {
                let mut state = serializer.serialize_struct_variant(
                    "AgentError",
                    4, // Variant index
                    "ResponseParsingFailed",
                    2, // Number of fields
                )?;
                // Serialize the source error's Display representation for simplicity
                // or decide on a more structured serialization if needed.
                // Here, we serialize its string representation.
                state.serialize_field("source", &source.to_string())?;
                state.serialize_field("response_text", response_text)?;
                state.end()
            }
        }
    }
}
