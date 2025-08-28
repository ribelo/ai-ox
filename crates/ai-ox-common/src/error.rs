use thiserror::Error;

/// Common errors that can occur in AI provider HTTP requests
#[derive(Error, Debug)]
pub enum CommonRequestError {
    /// HTTP request failed
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization/deserialization failed
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid event data in streaming response
    #[error("Invalid event data: {0}")]
    InvalidEventData(String),

    /// Authentication is missing (no API key or OAuth token)
    #[error("Authentication missing: no API key or OAuth token provided")]
    AuthenticationMissing,

    /// Invalid MIME type for file upload
    #[error("Invalid MIME type: {0}")]
    InvalidMimeType(String),

    /// UTF-8 conversion error
    #[error("UTF-8 conversion error: {0}")]
    Utf8Error(#[from] std::string::FromUtf8Error),

    /// Error originating from the request builder
    #[error("Request builder error: {0}")]
    RequestBuilder(String),
}

/// Parse error response from HTTP status and body
pub fn parse_error_response(status: reqwest::StatusCode, body: &bytes::Bytes) -> CommonRequestError {
    let body_str = String::from_utf8_lossy(body);
    
    // Try to parse as JSON error first
    if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(body) {
        if let Some(error_message) = extract_error_message(&json_value) {
            return CommonRequestError::InvalidEventData(format!(
                "HTTP {}: {}",
                status.as_u16(),
                error_message
            ));
        }
    }
    
    // Fall back to raw body
    CommonRequestError::InvalidEventData(format!(
        "HTTP {}: {}",
        status.as_u16(),
        body_str
    ))
}

/// Extract error message from various provider JSON error formats
fn extract_error_message(json: &serde_json::Value) -> Option<String> {
    // OpenAI/Groq/Mistral format: {"error": {"message": "..."}}
    if let Some(error_obj) = json.get("error") {
        if let Some(message) = error_obj.get("message") {
            if let Some(msg_str) = message.as_str() {
                return Some(msg_str.to_string());
            }
        }
    }
    
    // Anthropic format: {"error": {"type": "...", "message": "..."}}
    if let Some(error_obj) = json.get("error") {
        if let Some(message) = error_obj.get("message") {
            if let Some(msg_str) = message.as_str() {
                return Some(msg_str.to_string());
            }
        }
    }
    
    // Gemini format: {"error": {"code": 400, "message": "...", "status": "..."}}
    if let Some(error_obj) = json.get("error") {
        if let Some(message) = error_obj.get("message") {
            if let Some(msg_str) = message.as_str() {
                return Some(msg_str.to_string());
            }
        }
    }
    
    // Generic message field
    if let Some(message) = json.get("message") {
        if let Some(msg_str) = message.as_str() {
            return Some(msg_str.to_string());
        }
    }
    
    None
}