use crate::errors::GenerateContentError;
use thiserror::Error;

/// Errors that can occur when interacting with Groq models
#[derive(Debug, Error)]
pub enum GroqError {
    /// Missing API key error
    #[error("Missing GROQ_API_KEY environment variable")]
    MissingApiKey,

    /// API error from Groq service
    #[error("Groq API error: {0}")]
    Api(#[from] groq_ox::GroqRequestError),

    /// Error parsing response from Groq
    #[error("Response parsing error: {0}")]
    ResponseParsing(String),

    /// Conversion error between formats
    #[error("Conversion error: {0}")]
    Conversion(String),
}

impl From<GroqError> for GenerateContentError {
    fn from(error: GroqError) -> Self {
        match error {
            GroqError::MissingApiKey => GenerateContentError::configuration(error.to_string()),
            GroqError::Api(e) => GenerateContentError::provider_error("groq", e.to_string()),
            GroqError::ResponseParsing(msg) => GenerateContentError::response_parsing(msg),
            GroqError::Conversion(msg) => GenerateContentError::message_conversion(msg),
        }
    }
}