//! Legacy conversion module - use conversion-ox crate for direct conversions
//!
//! The complex From trait implementations have been moved to conversion-ox::anthropic_openrouter
//! module. This module is kept for backwards compatibility but is now deprecated.
//!
//! ## Migration Guide
//!
//! Replace From trait usage with explicit conversion functions from the `conversion-ox` crate:
//!
//! - Use `conversion_ox::anthropic_openrouter::anthropic_to_openrouter_request()`
//! - Use `conversion_ox::anthropic_openrouter::openrouter_to_anthropic_response()`
//! - Use `conversion_ox::anthropic_openrouter::streaming::AnthropicOpenRouterStreamConverter`

/// Error type for conversion failures - kept for backwards compatibility
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("Unable to convert content: {0}")]
    ContentConversion(String),
    #[error("Missing required data: {0}")]
    MissingData(String),
    #[error("Unsupported conversion: {0}")]
    UnsupportedConversion(String),
}