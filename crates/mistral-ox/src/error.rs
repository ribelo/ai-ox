use serde::{Deserialize, Serialize, Serializer, ser::SerializeStruct};
use thiserror::Error;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct MistralApiErrorPayload {
    detail: Option<String>,
    message: Option<String>,
    r#type: Option<String>,
    param: Option<String>,
    code: Option<String>,
}

#[derive(Debug, Error)]
pub enum MistralRequestError {
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
        detail: Option<String>,
        message: String,
        param: Option<String>,
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

impl Serialize for MistralRequestError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            MistralRequestError::ReqwestError(e) => {
                let mut state = serializer.serialize_struct("MistralRequestError", 2)?;
                state.serialize_field("type", "ReqwestError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            MistralRequestError::SerdeError(e) => {
                let mut state = serializer.serialize_struct("MistralRequestError", 2)?;
                state.serialize_field("type", "SerdeError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            MistralRequestError::JsonDeserializationError(e) => {
                let mut state = serializer.serialize_struct("MistralRequestError", 2)?;
                state.serialize_field("type", "JsonDeserializationError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            MistralRequestError::InvalidRequestError {
                code,
                detail,
                message,
                param,
                r#type,
            } => {
                let field_count = 1 // type
                    + if code.is_some() { 1 } else { 0 }
                    + if detail.is_some() { 1 } else { 0 }
                    + 1 // message
                    + if param.is_some() { 1 } else { 0 }
                    + if r#type.is_some() { 1 } else { 0 };
                let mut state = serializer.serialize_struct("MistralRequestError", field_count)?;
                state.serialize_field("type", "InvalidRequestError")?;
                if let Some(c) = code {
                    state.serialize_field("code", c)?;
                }
                if let Some(d) = detail {
                    state.serialize_field("detail", d)?;
                }
                state.serialize_field("message", message)?;
                if let Some(p) = param {
                    state.serialize_field("param", p)?;
                }
                if let Some(t) = r#type {
                    state.serialize_field("error_type", t)?;
                }
                state.end()
            }
            MistralRequestError::UnexpectedResponse(response) => {
                let mut state = serializer.serialize_struct("MistralRequestError", 2)?;
                state.serialize_field("type", "UnexpectedResponse")?;
                state.serialize_field("response", response)?;
                state.end()
            }
            MistralRequestError::InvalidEventData(message) => {
                let mut state = serializer.serialize_struct("MistralRequestError", 2)?;
                state.serialize_field("type", "InvalidEventData")?;
                state.serialize_field("message", message)?;
                state.end()
            }
            MistralRequestError::RateLimit => {
                let mut state = serializer.serialize_struct("MistralRequestError", 1)?;
                state.serialize_field("type", "RateLimit")?;
                state.end()
            }
            MistralRequestError::UrlBuildError(message) => {
                let mut state = serializer.serialize_struct("MistralRequestError", 2)?;
                state.serialize_field("type", "UrlBuildError")?;
                state.serialize_field("message", message)?;
                state.end()
            }
            MistralRequestError::IoError(e) => {
                let mut state = serializer.serialize_struct("MistralRequestError", 2)?;
                state.serialize_field("type", "IoError")?;
                state.serialize_field("error", &e.to_string())?;
                state.end()
            }
            MistralRequestError::MissingApiKey => {
                let mut state = serializer.serialize_struct("MistralRequestError", 1)?;
                state.serialize_field("type", "MissingApiKey")?;
                state.end()
            }
            MistralRequestError::InvalidModel(model) => {
                let mut state = serializer.serialize_struct("MistralRequestError", 2)?;
                state.serialize_field("type", "InvalidModel")?;
                state.serialize_field("model", model)?;
                state.end()
            }
        }
    }
}

/// Parse an error response from the Mistral API.
/// This function handles both JSON format errors and plain text errors.
pub(crate) fn parse_error_response(status: reqwest::StatusCode, bytes: bytes::Bytes) -> MistralRequestError {
    // Try to parse as a structured Mistral API error first
    if let Ok(payload) = serde_json::from_slice::<MistralApiErrorPayload>(&bytes) {
        let detail = payload.detail.clone();
        MistralRequestError::InvalidRequestError {
            code: payload.code,
            message: payload.message.or(detail.clone()).unwrap_or_else(|| "Unknown error".to_string()),
            detail,
            param: payload.param,
            r#type: payload.r#type,
        }
    } else {
        // Fall back to text
        let error_text = String::from_utf8_lossy(&bytes).to_string();
        MistralRequestError::UnexpectedResponse(format!(
            "HTTP status {}: {}",
            status.as_u16(),
            error_text
        ))
    }
}