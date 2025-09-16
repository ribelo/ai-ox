use thiserror::Error;

use crate::errors::GenerateContentError;

#[derive(Debug, Error)]
pub enum AnthropicError {
    #[error("Missing API key")]
    MissingApiKey,

    #[error("API error: {0}")]
    Api(#[from] anthropic_ox::AnthropicRequestError),

    #[error("Response parsing error: {0}")]
    ResponseParsing(String),

    #[error("Message conversion error: {0}")]
    MessageConversion(String),

    #[error("Invalid schema: {0}")]
    InvalidSchema(String),
}

impl From<AnthropicError> for GenerateContentError {
    fn from(error: AnthropicError) -> Self {
        match error {
            AnthropicError::MissingApiKey => GenerateContentError::configuration(
                "Missing ANTHROPIC_API_KEY environment variable",
            ),
            AnthropicError::Api(e) => {
                GenerateContentError::provider_error("anthropic", e.to_string())
            }
            AnthropicError::ResponseParsing(e) => GenerateContentError::response_parsing(e),
            AnthropicError::MessageConversion(e) => GenerateContentError::message_conversion(e),
            AnthropicError::InvalidSchema(e) => {
                GenerateContentError::configuration(format!("Invalid schema: {}", e))
            }
        }
    }
}
