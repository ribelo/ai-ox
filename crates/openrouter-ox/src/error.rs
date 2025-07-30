use serde::{Deserialize, Serialize, Serializer, ser::SerializeStruct};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OpenRouterRequestError {
    /// Errors from the HTTP client
    #[error(transparent)]
    ReqwestError(#[from] reqwest::Error),

    /// JSON serialization/deserialization errors
    #[error(transparent)]
    SerdeError(#[from] serde_json::Error),

    /// JSON deserialization errors with context
    #[error("Failed to deserialize JSON: {0}")]
    JsonDeserializationError(serde_json::Error),

    /// Invalid request errors from the API
    #[error("Invalid request error: {message}")]
    InvalidRequestError {
        code: Option<String>,
        details: serde_json::Value,
        message: String,
        status: Option<String>,
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

    /// URL building error
    #[error("URL build failed: {0}")]
    UrlBuildError(String),

    /// I/O errors
    #[error(transparent)]
    IoError(#[from] std::io::Error),

    /// Stream error
    #[error("Stream error: {0}")]
    Stream(String),
}

impl Serialize for OpenRouterRequestError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            OpenRouterRequestError::ReqwestError(e) => {
                let mut state = serializer.serialize_struct("OpenRouterRequestError", 2)?;
                state.serialize_field("type", "ReqwestError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            OpenRouterRequestError::SerdeError(e) => {
                let mut state = serializer.serialize_struct("OpenRouterRequestError", 2)?;
                state.serialize_field("type", "SerdeError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            OpenRouterRequestError::JsonDeserializationError(e) => {
                let mut state = serializer.serialize_struct("OpenRouterRequestError", 2)?;
                state.serialize_field("type", "JsonDeserializationError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            OpenRouterRequestError::InvalidRequestError {
                code,
                details,
                message,
                status,
            } => {
                let field_count = 1 // type
                    + if code.is_some() { 1 } else { 0 }
                    + 1 // details
                    + 1 // message
                    + if status.is_some() { 1 } else { 0 };
                let mut state = serializer.serialize_struct("OpenRouterRequestError", field_count)?;
                state.serialize_field("type", "InvalidRequestError")?;
                if let Some(c) = code {
                    state.serialize_field("code", c)?;
                }
                state.serialize_field("details", details)?;
                state.serialize_field("message", message)?;
                if let Some(s) = status {
                    state.serialize_field("status", s)?;
                }
                state.end()
            }
            OpenRouterRequestError::UnexpectedResponse(response) => {
                let mut state = serializer.serialize_struct("OpenRouterRequestError", 2)?;
                state.serialize_field("type", "UnexpectedResponse")?;
                state.serialize_field("response", response)?;
                state.end()
            }
            OpenRouterRequestError::InvalidEventData(message) => {
                let mut state = serializer.serialize_struct("OpenRouterRequestError", 2)?;
                state.serialize_field("type", "InvalidEventData")?;
                state.serialize_field("message", message)?;
                state.end()
            }
            OpenRouterRequestError::RateLimit => {
                let mut state = serializer.serialize_struct("OpenRouterRequestError", 1)?;
                state.serialize_field("type", "RateLimit")?;
                state.end()
            }
            OpenRouterRequestError::UrlBuildError(message) => {
                let mut state = serializer.serialize_struct("OpenRouterRequestError", 2)?;
                state.serialize_field("type", "UrlBuildError")?;
                state.serialize_field("message", message)?;
                state.end()
            }
            OpenRouterRequestError::IoError(e) => {
                let mut state = serializer.serialize_struct("OpenRouterRequestError", 2)?;
                state.serialize_field("type", "IoError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            OpenRouterRequestError::Stream(message) => {
                let mut state = serializer.serialize_struct("OpenRouterRequestError", 2)?;
                state.serialize_field("type", "Stream")?;
                state.serialize_field("message", message)?;
                state.end()
            }
        }
    }
}

/// Parse an error response from the OpenRouter API.
/// This function handles both JSON format errors and plain text errors.
pub(crate) fn parse_error_response(status: reqwest::StatusCode, bytes: Vec<u8>) -> OpenRouterRequestError {
    #[derive(Debug, Deserialize)]
    struct OpenRouterErrorPayload {
        error: OpenRouterErrorDetails,
        user_id: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    struct OpenRouterErrorDetails {
        code: i32,
        message: String,
    }

    // Try to parse as a structured OpenRouter API error first
    if let Ok(payload) = serde_json::from_slice::<OpenRouterErrorPayload>(&bytes) {
        OpenRouterRequestError::InvalidRequestError {
            code: Some(payload.error.code.to_string()),
            message: payload.error.message,
            status: Some(status.as_u16().to_string()),
            details: serde_json::json!({
                "user_id": payload.user_id
            }),
        }
    } else {
        // Fall back to text
        let error_text = String::from_utf8_lossy(&bytes).to_string();
        OpenRouterRequestError::UnexpectedResponse(format!(
            "HTTP status {}: {}",
            status.as_u16(),
            error_text
        ))
    }
}