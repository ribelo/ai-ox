// Re-export shared error types from ai-ox-common
pub use ai_ox_common::error::{ProviderError, parse_api_error_response};

/// Type alias for backward compatibility with existing Mistral code
pub type MistralRequestError = ProviderError;

/// Parse an error response from the Mistral API (backward compatibility wrapper)
pub(crate) fn parse_error_response(status: reqwest::StatusCode, bytes: bytes::Bytes) -> MistralRequestError {
    parse_api_error_response(status, &bytes)
}