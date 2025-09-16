use serde::Serialize;
use thiserror::Error;

/// Unified error type for all AI providers
/// Covers 80%+ of error cases across OpenAI, Groq, Mistral, OpenRouter, Anthropic
#[derive(Debug, Error, Serialize)]
#[serde(tag = "type", content = "error")]
pub enum ProviderError {
    /// HTTP client errors (reqwest)
    #[error("HTTP request failed: {0}")]
    Http(String),

    /// JSON serialization/deserialization errors
    #[error("JSON error: {0}")]
    Json(String),

    /// I/O errors (file operations, etc.)
    #[error("I/O error: {0}")]
    Io(String),

    /// Invalid request errors from API with flexible details
    #[error("Invalid request: {message}")]
    InvalidRequest {
        code: Option<String>,
        message: String,
        /// Provider-specific extra fields (param, detail, type, status, etc.)
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
    },

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimit,

    /// Authentication is missing (no API key or OAuth token)
    #[error("Authentication missing")]
    AuthenticationMissing,

    /// Invalid model identifier
    #[error("Invalid model: {0}")]
    InvalidModel(String),

    /// Unexpected response from API
    #[error("Unexpected response: {0}")]
    UnexpectedResponse(String),

    /// Invalid streaming event data
    #[error("Invalid event data: {0}")]
    InvalidEventData(String),

    /// URL building error
    #[error("URL build failed: {0}")]
    UrlBuildError(String),

    /// Stream-specific error
    #[error("Stream error: {0}")]
    Stream(String),

    /// Invalid MIME type for file upload
    #[error("Invalid MIME type: {0}")]
    InvalidMimeType(String),

    /// UTF-8 conversion error
    #[error("UTF-8 conversion error: {0}")]
    Utf8Error(String),

    /// JSON deserialization with more context
    #[error("Failed to deserialize JSON: {0}")]
    JsonDeserializationError(String),
}

/// Convert standard library errors to ProviderError
impl From<reqwest::Error> for ProviderError {
    fn from(err: reqwest::Error) -> Self {
        ProviderError::Http(err.to_string())
    }
}

impl From<serde_json::Error> for ProviderError {
    fn from(err: serde_json::Error) -> Self {
        ProviderError::Json(err.to_string())
    }
}

impl From<std::io::Error> for ProviderError {
    fn from(err: std::io::Error) -> Self {
        ProviderError::Io(err.to_string())
    }
}

impl From<std::string::FromUtf8Error> for ProviderError {
    fn from(err: std::string::FromUtf8Error) -> Self {
        ProviderError::Utf8Error(err.to_string())
    }
}

/// Legacy alias for backward compatibility
pub type CommonRequestError = ProviderError;

/// Unified error parsing for all providers
/// Handles multiple API error response formats (OpenAI, Anthropic, Mistral, OpenRouter, etc.)
pub fn parse_api_error_response(status: reqwest::StatusCode, body: &[u8]) -> ProviderError {
    // Try parsing as structured JSON error first
    if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(body) {
        if let Some(parsed_error) = extract_structured_error(&json_value, status) {
            return parsed_error;
        }
    }

    // Fall back to plain text error
    let body_str = String::from_utf8_lossy(body);
    ProviderError::UnexpectedResponse(format!("HTTP {}: {}", status.as_u16(), body_str))
}

/// Extract structured error from various provider JSON error formats
fn extract_structured_error(
    json: &serde_json::Value,
    status: reqwest::StatusCode,
) -> Option<ProviderError> {
    // OpenAI/Groq format: {"error": {"message": "...", "type": "...", "code": "..."}}
    if let Some(error_obj) = json.get("error") {
        if let Some(message) = error_obj.get("message").and_then(|m| m.as_str()) {
            let code = error_obj
                .get("code")
                .and_then(|c| c.as_str())
                .map(|s| s.to_string());
            let error_type = error_obj.get("type").and_then(|t| t.as_str());

            // Create details object with any extra fields
            let mut details = serde_json::Map::new();
            if let Some(et) = error_type {
                details.insert(
                    "type".to_string(),
                    serde_json::Value::String(et.to_string()),
                );
            }
            // Mistral-specific fields
            if let Some(param) = error_obj.get("param") {
                details.insert("param".to_string(), param.clone());
            }
            if let Some(detail) = error_obj.get("detail") {
                details.insert("detail".to_string(), detail.clone());
            }

            return Some(ProviderError::InvalidRequest {
                code,
                message: message.to_string(),
                details: if details.is_empty() {
                    None
                } else {
                    Some(serde_json::Value::Object(details))
                },
            });
        }
    }

    // OpenRouter format: {"error": {"code": 123, "message": "..."}, "user_id": "..."}
    if let Some(error_obj) = json.get("error") {
        if let Some(code_num) = error_obj.get("code").and_then(|c| c.as_i64()) {
            if let Some(message) = error_obj.get("message").and_then(|m| m.as_str()) {
                let mut details = serde_json::Map::new();
                details.insert(
                    "status".to_string(),
                    serde_json::Value::String(status.as_u16().to_string()),
                );
                if let Some(user_id) = json.get("user_id") {
                    details.insert("user_id".to_string(), user_id.clone());
                }

                return Some(ProviderError::InvalidRequest {
                    code: Some(code_num.to_string()),
                    message: message.to_string(),
                    details: Some(serde_json::Value::Object(details)),
                });
            }
        }
    }

    // Mistral direct format: {"detail": "...", "message": "...", "type": "...", "param": "...", "code": "..."}
    if let Some(message) = json.get("message").and_then(|m| m.as_str()) {
        let code = json
            .get("code")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string());
        let mut details = serde_json::Map::new();

        if let Some(detail) = json.get("detail") {
            details.insert("detail".to_string(), detail.clone());
        }
        if let Some(param) = json.get("param") {
            details.insert("param".to_string(), param.clone());
        }
        if let Some(error_type) = json.get("type") {
            details.insert("type".to_string(), error_type.clone());
        }

        return Some(ProviderError::InvalidRequest {
            code,
            message: message.to_string(),
            details: if details.is_empty() {
                None
            } else {
                Some(serde_json::Value::Object(details))
            },
        });
    }

    // Anthropic format: {"error": {"type": "...", "message": "...", "param": "...", "code": "..."}}
    if let Some(error_obj) = json.get("error") {
        if let Some(message) = error_obj.get("message").and_then(|m| m.as_str()) {
            let code = error_obj
                .get("code")
                .and_then(|c| c.as_str())
                .map(|s| s.to_string());
            let mut details = serde_json::Map::new();

            if let Some(param) = error_obj.get("param") {
                details.insert("param".to_string(), param.clone());
            }
            if let Some(error_type) = error_obj.get("type") {
                details.insert("type".to_string(), error_type.clone());
            }

            return Some(ProviderError::InvalidRequest {
                code,
                message: message.to_string(),
                details: if details.is_empty() {
                    None
                } else {
                    Some(serde_json::Value::Object(details))
                },
            });
        }
    }

    // Generic top-level message
    if let Some(message) = json.get("message").and_then(|m| m.as_str()) {
        return Some(ProviderError::InvalidRequest {
            code: None,
            message: message.to_string(),
            details: None,
        });
    }

    None
}

/// Legacy function name for compatibility
pub fn parse_error_response(status: reqwest::StatusCode, body: bytes::Bytes) -> ProviderError {
    parse_api_error_response(status, &body)
}
