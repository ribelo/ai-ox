use thiserror::Error;

use ai_ox_common::error::ProviderError;

/// Errors that can occur during speech-to-text operations
#[derive(Debug, Error)]
pub enum SttError {
    /// Provider-specific error
    #[error("Provider error: {0}")]
    Provider(Box<dyn std::error::Error + Send + Sync>),

    /// Unsupported audio format
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    /// Transcription failed
    #[error("Transcription failed: {0}")]
    TranscriptionFailed(String),

    /// Missing API key
    #[error("Missing API key")]
    MissingApiKey,

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// No provider specified
    #[error("No provider specified")]
    NoProviderSpecified,

    /// Model not found
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Audio file too large
    #[error("Audio file too large: {0} bytes (max: {1} bytes)")]
    AudioTooLarge(usize, usize),

    /// Audio duration too long
    #[error("Audio duration too long: {0:?} (max: {1:?})")]
    AudioTooLong(std::time::Duration, std::time::Duration),

    /// Network/IO error
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// HTTP request error
    #[error("HTTP error: {0}")]
    Http(String),

    /// Authentication failed
    #[error("Authentication failed")]
    AuthenticationFailed,

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    /// Service unavailable
    #[error("Service unavailable")]
    ServiceUnavailable,

    /// Invalid audio data
    #[error("Invalid audio data: {0}")]
    InvalidAudioData(String),

    /// Provider error
    #[error("Provider error: {0}")]
    ProviderError(ProviderError),
}



// Conversion implementations for provider-specific errors

#[cfg(feature = "groq")]
impl From<groq_ox::GroqRequestError> for SttError {
    fn from(error: groq_ox::GroqRequestError) -> Self {
        SttError::ProviderError(error)
    }
}

#[cfg(feature = "gemini")]
impl From<crate::errors::GenerateContentError> for SttError {
    fn from(error: crate::errors::GenerateContentError) -> Self {
        SttError::Provider(Box::new(error))
    }
}

// Note: reqwest errors are handled through provider-specific error types
