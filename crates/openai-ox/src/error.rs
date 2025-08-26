use serde::{Deserialize, Serialize, Serializer, ser::SerializeStruct};
use thiserror::Error;

/// OpenAI API error details
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct OpenAIApiErrorPayload {
    error: Option<OpenAIApiError>,
}

/// Specific error information from OpenAI API
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct OpenAIApiError {
    message: String,
    r#type: Option<String>,
    code: Option<String>,
}

/// Errors that can occur when making requests to the OpenAI API
#[derive(Debug, Error)]
pub enum OpenAIRequestError {
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

impl Serialize for OpenAIRequestError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            OpenAIRequestError::ReqwestError(e) => {
                let mut state = serializer.serialize_struct("OpenAIRequestError", 2)?;
                state.serialize_field("type", "ReqwestError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            OpenAIRequestError::SerdeError(e) => {
                let mut state = serializer.serialize_struct("OpenAIRequestError", 2)?;
                state.serialize_field("type", "SerdeError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            OpenAIRequestError::InvalidRequestError {
                code,
                message,
                r#type,
            } => {
                let field_count = 1 // type
                    + if code.is_some() { 1 } else { 0 }
                    + 1 // message
                    + if r#type.is_some() { 1 } else { 0 };
                let mut state = serializer.serialize_struct("OpenAIRequestError", field_count)?;
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
            OpenAIRequestError::UnexpectedResponse(response) => {
                let mut state = serializer.serialize_struct("OpenAIRequestError", 2)?;
                state.serialize_field("type", "UnexpectedResponse")?;
                state.serialize_field("response", response)?;
                state.end()
            }
            OpenAIRequestError::InvalidEventData(message) => {
                let mut state = serializer.serialize_struct("OpenAIRequestError", 2)?;
                state.serialize_field("type", "InvalidEventData")?;
                state.serialize_field("message", message)?;
                state.end()
            }
            OpenAIRequestError::RateLimit => {
                let mut state = serializer.serialize_struct("OpenAIRequestError", 1)?;
                state.serialize_field("type", "RateLimit")?;
                state.end()
            }
            OpenAIRequestError::MissingApiKey => {
                let mut state = serializer.serialize_struct("OpenAIRequestError", 1)?;
                state.serialize_field("type", "MissingApiKey")?;
                state.end()
            }
            OpenAIRequestError::InvalidModel(model) => {
                let mut state = serializer.serialize_struct("OpenAIRequestError", 2)?;
                state.serialize_field("type", "InvalidModel")?;
                state.serialize_field("model", model)?;
                state.end()
            }
            OpenAIRequestError::IoError(e) => {
                let mut state = serializer.serialize_struct("OpenAIRequestError", 2)?;
                state.serialize_field("type", "IoError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
        }
    }
}

/// Parse an error response from the OpenAI API
pub(crate) fn parse_error_response(status: reqwest::StatusCode, bytes: bytes::Bytes) -> OpenAIRequestError {
    // Try to parse as a structured OpenAI API error first
    if let Ok(payload) = serde_json::from_slice::<OpenAIApiErrorPayload>(&bytes) {
        if let Some(error) = payload.error {
            OpenAIRequestError::InvalidRequestError {
                code: error.code,
                message: error.message,
                r#type: error.r#type,
            }
        } else {
            let error_text = String::from_utf8_lossy(&bytes).to_string();
            OpenAIRequestError::UnexpectedResponse(format!(
                "HTTP status {}: {}",
                status.as_u16(),
                error_text
            ))
        }
    } else {
        // Fall back to text
        let error_text = String::from_utf8_lossy(&bytes).to_string();
        OpenAIRequestError::UnexpectedResponse(format!(
            "HTTP status {}: {}",
            status.as_u16(),
            error_text
        ))
    }
}