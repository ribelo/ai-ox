use serde::{Deserialize, Serialize, Serializer, ser::SerializeStruct};
use thiserror::Error;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct GroqApiErrorPayload {
    error: Option<GroqApiError>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct GroqApiError {
    message: String,
    r#type: Option<String>,
    code: Option<String>,
}

#[derive(Debug, Error)]
pub enum GroqRequestError {
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

    /// URL building error
    #[error("URL build failed: {0}")]
    UrlBuildError(String),

    /// I/O errors
    #[error(transparent)]
    IoError(#[from] std::io::Error),

    /// Missing API key
    #[error("Missing API key")]
    MissingApiKey,

    /// Invalid model
    #[error("Invalid model: {0}")]
    InvalidModel(String),
}

impl Serialize for GroqRequestError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            GroqRequestError::ReqwestError(e) => {
                let mut state = serializer.serialize_struct("GroqRequestError", 2)?;
                state.serialize_field("type", "ReqwestError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            GroqRequestError::SerdeError(e) => {
                let mut state = serializer.serialize_struct("GroqRequestError", 2)?;
                state.serialize_field("type", "SerdeError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            GroqRequestError::JsonDeserializationError(e) => {
                let mut state = serializer.serialize_struct("GroqRequestError", 2)?;
                state.serialize_field("type", "JsonDeserializationError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            GroqRequestError::InvalidRequestError {
                code,
                message,
                r#type,
            } => {
                let field_count = 1 // type
                    + if code.is_some() { 1 } else { 0 }
                    + 1 // message
                    + if r#type.is_some() { 1 } else { 0 };
                let mut state = serializer.serialize_struct("GroqRequestError", field_count)?;
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
            GroqRequestError::UnexpectedResponse(response) => {
                let mut state = serializer.serialize_struct("GroqRequestError", 2)?;
                state.serialize_field("type", "UnexpectedResponse")?;
                state.serialize_field("response", response)?;
                state.end()
            }
            GroqRequestError::InvalidEventData(message) => {
                let mut state = serializer.serialize_struct("GroqRequestError", 2)?;
                state.serialize_field("type", "InvalidEventData")?;
                state.serialize_field("message", message)?;
                state.end()
            }
            GroqRequestError::RateLimit => {
                let mut state = serializer.serialize_struct("GroqRequestError", 1)?;
                state.serialize_field("type", "RateLimit")?;
                state.end()
            }
            GroqRequestError::UrlBuildError(message) => {
                let mut state = serializer.serialize_struct("GroqRequestError", 2)?;
                state.serialize_field("type", "UrlBuildError")?;
                state.serialize_field("message", message)?;
                state.end()
            }
            GroqRequestError::IoError(e) => {
                let mut state = serializer.serialize_struct("GroqRequestError", 2)?;
                state.serialize_field("type", "IoError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            GroqRequestError::MissingApiKey => {
                let mut state = serializer.serialize_struct("GroqRequestError", 1)?;
                state.serialize_field("type", "MissingApiKey")?;
                state.end()
            }
            GroqRequestError::InvalidModel(model) => {
                let mut state = serializer.serialize_struct("GroqRequestError", 2)?;
                state.serialize_field("type", "InvalidModel")?;
                state.serialize_field("model", model)?;
                state.end()
            }
        }
    }
}

/// Parse an error response from the Groq API.
/// This function handles both JSON format errors and plain text errors.
pub(crate) fn parse_error_response(status: reqwest::StatusCode, bytes: bytes::Bytes) -> GroqRequestError {
    // Try to parse as a structured Groq API error first
    if let Ok(payload) = serde_json::from_slice::<GroqApiErrorPayload>(&bytes) {
        if let Some(error) = payload.error {
            GroqRequestError::InvalidRequestError {
                code: error.code,
                message: error.message,
                r#type: error.r#type,
            }
        } else {
            let error_text = String::from_utf8_lossy(&bytes).to_string();
            GroqRequestError::UnexpectedResponse(format!(
                "HTTP status {}: {}",
                status.as_u16(),
                error_text
            ))
        }
    } else {
        // Fall back to text
        let error_text = String::from_utf8_lossy(&bytes).to_string();
        GroqRequestError::UnexpectedResponse(format!(
            "HTTP status {}: {}",
            status.as_u16(),
            error_text
        ))
    }
}