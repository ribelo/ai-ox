use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Categorizes errors for retry logic and handling
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    /// Rate limiting - should retry with backoff
    RateLimit,
    /// Authentication/authorization issues - should not retry
    Auth,
    /// Invalid request format - should not retry
    InvalidRequest,
    /// Server overloaded - may retry
    ServerOverloaded,
    /// Network/connection issues - may retry
    Network,
    /// API temporarily unavailable - may retry
    ServiceUnavailable,
    /// Unknown/other errors
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorInfo {
    pub r#type: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorDetail {
    pub message: String,
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub param: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub error: ApiErrorDetail,
}

#[derive(Debug, Error)]
pub enum AnthropicRequestError {
    /// Errors from the HTTP client
    #[error(transparent)]
    ReqwestError(#[from] reqwest::Error),

    /// JSON serialization/deserialization errors
    #[error(transparent)]
    SerdeError(#[from] serde_json::Error),

    /// Invalid request errors from the API
    #[error("Invalid request error: {message}")]
    InvalidRequestError {
        message: String,
        param: Option<String>,
        code: Option<String>,
    },

    /// Authentication error
    #[error("Authentication error: {0}")]
    Authentication(String),

    /// Permission denied
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// Resource not found
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimit,

    /// API overloaded
    #[error("API overloaded: {0}")]
    Overloaded(String),

    /// Generic API error
    #[error("API error: {0}")]
    Generic(String),

    /// Unexpected response from the API
    #[error("Unexpected response from API: {0}")]
    UnexpectedResponse(String),

    /// Invalid event data in stream
    #[error("Invalid event data: {0}")]
    InvalidEventData(String),

    /// Stream error
    #[error("Stream error: {0}")]
    Stream(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    /// Unknown event type
    #[error("Unknown event type: {0}")]
    UnknownEventType(String),
}

impl AnthropicRequestError {
    /// Returns the error kind for categorizing errors in retry logic
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::RateLimit => ErrorKind::RateLimit,
            Self::Authentication(_) | Self::PermissionDenied(_) => ErrorKind::Auth,
            Self::InvalidRequestError { .. } | Self::NotFound(_) => ErrorKind::InvalidRequest,
            Self::Overloaded(_) => ErrorKind::ServerOverloaded,
            Self::ReqwestError(e) => {
                if e.is_timeout() || e.is_connect() || e.is_request() {
                    ErrorKind::Network
                } else {
                    ErrorKind::Other
                }
            }
            Self::Generic(_) | Self::UnexpectedResponse(_) => ErrorKind::ServiceUnavailable,
            Self::SerdeError(_) | Self::InvalidEventData(_) | Self::Stream(_) 
            | Self::Deserialization(_) | Self::UnknownEventType(_) => ErrorKind::Other,
        }
    }
    
    /// Returns true if this error should be retried
    pub fn is_retryable(&self) -> bool {
        matches!(self.kind(), ErrorKind::RateLimit | ErrorKind::ServerOverloaded | ErrorKind::Network | ErrorKind::ServiceUnavailable)
    }
}

impl From<ErrorInfo> for AnthropicRequestError {
    fn from(error: ErrorInfo) -> Self {
        match error.r#type.as_str() {
            "invalid_request_error" => AnthropicRequestError::InvalidRequestError {
                message: error.message,
                param: None,
                code: None,
            },
            "authentication_error" => AnthropicRequestError::Authentication(error.message),
            "permission_error" => AnthropicRequestError::PermissionDenied(error.message),
            "not_found_error" => AnthropicRequestError::NotFound(error.message),
            "rate_limit_error" => AnthropicRequestError::RateLimit,
            "api_error" => AnthropicRequestError::Generic(error.message),
            "overloaded_error" => AnthropicRequestError::Overloaded(error.message),
            _ => AnthropicRequestError::UnexpectedResponse(format!("Unknown error type: {}", error.r#type)),
        }
    }
}

/// Parse an error response from the Anthropic API.
/// This function handles both JSON format errors and plain text errors.
pub fn parse_error_response(status: reqwest::StatusCode, bytes: bytes::Bytes) -> AnthropicRequestError {
    // Try to parse as a structured Anthropic API error first
    if let Ok(payload) = serde_json::from_slice::<ApiErrorResponse>(&bytes) {
        match payload.error.r#type.as_deref() {
            Some("invalid_request_error") => AnthropicRequestError::InvalidRequestError {
                message: payload.error.message,
                param: payload.error.param,
                code: payload.error.code,
            },
            Some("authentication_error") => AnthropicRequestError::Authentication(payload.error.message),
            Some("permission_error") => AnthropicRequestError::PermissionDenied(payload.error.message),
            Some("not_found_error") => AnthropicRequestError::NotFound(payload.error.message),
            Some("rate_limit_error") => AnthropicRequestError::RateLimit,
            Some("api_error") => AnthropicRequestError::Generic(payload.error.message),
            Some("overloaded_error") => AnthropicRequestError::Overloaded(payload.error.message),
            _ => AnthropicRequestError::UnexpectedResponse(payload.error.message),
        }
    } else {
        // Fall back to text
        let error_text = String::from_utf8_lossy(&bytes).to_string();
        match status.as_u16() {
            429 => AnthropicRequestError::RateLimit,
            401 => AnthropicRequestError::Authentication(error_text),
            403 => AnthropicRequestError::PermissionDenied(error_text),
            404 => AnthropicRequestError::NotFound(error_text),
            _ => AnthropicRequestError::UnexpectedResponse(format!(
                "HTTP status {}: {}",
                status.as_u16(),
                error_text
            )),
        }
    }
}