use serde::{Deserialize, Serialize, Serializer, ser::SerializeStruct};
use thiserror::Error;

/// {{Provider}} API error details
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct {{Provider}}ApiErrorPayload {
    error: Option<{{Provider}}ApiError>,
}

/// Specific error information from {{Provider}} API
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct {{Provider}}ApiError {
    message: String,
    r#type: Option<String>,
    code: Option<String>,
}

/// Errors that can occur when making requests to the {{Provider}} API
#[derive(Debug, Error)]
pub enum {{Provider}}RequestError {
    /// HTTP client errors
    #[error(transparent)]
    ReqwestError(#[from] reqwest::Error),

    /// JSON serialization/deserialization errors
    #[error(transparent)]
    SerdeError(#[from] serde_json::Error),

    /// Invalid request errors from the API
    #[error("Invalid request error: {message}")]
    InvalidRequestError {
        code: Option<String>,
        message: String,
        r#type: Option<String>,
    },

    /// Unexpected response from the API
    #[error("Unexpected response from API: {0}")]
    UnexpectedResponse(String),

    /// Invalid event data in stream
    #[error("Invalid event data: {0}")]
    InvalidEventData(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimit,

    /// Missing API key
    #[error("Missing API key")]
    MissingApiKey,

    /// Invalid model
    #[error("Invalid model: {0}")]
    InvalidModel(String),

    /// I/O errors
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

impl Serialize for {{Provider}}RequestError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            {{Provider}}RequestError::ReqwestError(e) => {
                let mut state = serializer.serialize_struct("{{Provider}}RequestError", 2)?;
                state.serialize_field("type", "ReqwestError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            {{Provider}}RequestError::SerdeError(e) => {
                let mut state = serializer.serialize_struct("{{Provider}}RequestError", 2)?;
                state.serialize_field("type", "SerdeError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            {{Provider}}RequestError::InvalidRequestError {
                code,
                message,
                r#type,
            } => {
                let field_count = 1 // type
                    + if code.is_some() { 1 } else { 0 }
                    + 1 // message
                    + if r#type.is_some() { 1 } else { 0 };
                let mut state = serializer.serialize_struct("{{Provider}}RequestError", field_count)?;
                state.serialize_field("type", "InvalidRequestError")?;
                if let Some(c) = code {
                    state.serialize_field("code", c)?;
                }
                state.serialize_field("message", message)?;
                if let Some(t) = r#type {
                    state.serialize_field("error_type", t)?;
                }
                state.end()
            }
            {{Provider}}RequestError::UnexpectedResponse(response) => {
                let mut state = serializer.serialize_struct("{{Provider}}RequestError", 2)?;
                state.serialize_field("type", "UnexpectedResponse")?;
                state.serialize_field("response", response)?;
                state.end()
            }
            {{Provider}}RequestError::InvalidEventData(message) => {
                let mut state = serializer.serialize_struct("{{Provider}}RequestError", 2)?;
                state.serialize_field("type", "InvalidEventData")?;
                state.serialize_field("message", message)?;
                state.end()
            }
            {{Provider}}RequestError::RateLimit => {
                let mut state = serializer.serialize_struct("{{Provider}}RequestError", 1)?;
                state.serialize_field("type", "RateLimit")?;
                state.end()
            }
            {{Provider}}RequestError::MissingApiKey => {
                let mut state = serializer.serialize_struct("{{Provider}}RequestError", 1)?;
                state.serialize_field("type", "MissingApiKey")?;
                state.end()
            }
            {{Provider}}RequestError::InvalidModel(model) => {
                let mut state = serializer.serialize_struct("{{Provider}}RequestError", 2)?;
                state.serialize_field("type", "InvalidModel")?;
                state.serialize_field("model", model)?;
                state.end()
            }
            {{Provider}}RequestError::IoError(e) => {
                let mut state = serializer.serialize_struct("{{Provider}}RequestError", 2)?;
                state.serialize_field("type", "IoError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
        }
    }
}

/// Parse an error response from the {{Provider}} API
pub(crate) fn parse_error_response(status: reqwest::StatusCode, bytes: bytes::Bytes) -> {{Provider}}RequestError {
    // Try to parse as a structured {{Provider}} API error first
    if let Ok(payload) = serde_json::from_slice::<{{Provider}}ApiErrorPayload>(&bytes) {
        if let Some(error) = payload.error {
            {{Provider}}RequestError::InvalidRequestError {
                code: error.code,
                message: error.message,
                r#type: error.r#type,
            }
        } else {
            let error_text = String::from_utf8_lossy(&bytes).to_string();
            {{Provider}}RequestError::UnexpectedResponse(format!(
                "HTTP status {}: {}",
                status.as_u16(),
                error_text
            ))
        }
    } else {
        // Fall back to text
        let error_text = String::from_utf8_lossy(&bytes).to_string();
        {{Provider}}RequestError::UnexpectedResponse(format!(
            "HTTP status {}: {}",
            status.as_u16(),
            error_text
        ))
    }
}