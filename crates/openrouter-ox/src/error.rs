// Re-export shared error types from ai-ox-common
pub use ai_ox_common::error::{ProviderError, parse_api_error_response};

/// Type alias for backward compatibility with existing OpenRouter code
pub type OpenRouterRequestError = ProviderError;

/// Parse an error response from the OpenRouter API (backward compatibility wrapper)
pub(crate) fn parse_error_response(status: reqwest::StatusCode, bytes: Vec<u8>) -> OpenRouterRequestError {
    parse_api_error_response(status, &bytes)
}