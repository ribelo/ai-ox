// Re-export shared error types from ai-ox-common
pub use ai_ox_common::error::{ProviderError, parse_api_error_response};

/// Type alias for backward compatibility with existing Groq code
pub type GroqRequestError = ProviderError;

/// Parse an error response from the Groq API (backward compatibility wrapper)
pub(crate) fn parse_error_response(
    status: reqwest::StatusCode,
    bytes: bytes::Bytes,
) -> GroqRequestError {
    parse_api_error_response(status, &bytes)
}
