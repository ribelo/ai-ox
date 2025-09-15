#[cfg(feature = "bedrock")]
use thiserror::Error;

use crate::errors::GenerateContentError;

#[derive(Error, Debug)]
pub enum BedrockError {
    #[error("AWS Converse API error: {0}")]
    Converse(String),

    #[error("AWS Converse Stream API error: {0}")]
    ConverseStream(String),

    #[error("Error reading from the response stream: {0}")]
    StreamRead(String),

    #[error("The model returned no response content")]
    NoResponse,

    #[error("Failed to build a request for the Bedrock API: {0}")]
    RequestBuilder(String),

    #[error("Failed to parse the tool input arguments from the model: {0}")]
    ToolInputParse(String),

    #[error("Failed to convert an ai-ox message to the Bedrock format: {0}")]
    MessageConversion(String),
}

impl From<BedrockError> for GenerateContentError {
    fn from(error: BedrockError) -> Self {
        match error {
            BedrockError::NoResponse => GenerateContentError::NoResponse,
            BedrockError::MessageConversion(msg) => GenerateContentError::MessageConversion(msg),
            BedrockError::ToolInputParse(msg) => GenerateContentError::ResponseParsing(msg),
            BedrockError::RequestBuilder(msg) => GenerateContentError::Configuration(msg),
            _ => GenerateContentError::ResponseParsing(error.to_string()),
        }
    }
}

impl From<GenerateContentError> for BedrockError {
    fn from(error: GenerateContentError) -> Self {
        match error {
            GenerateContentError::NoResponse => BedrockError::NoResponse,
            GenerateContentError::MessageConversion(msg) => BedrockError::MessageConversion(msg),
            GenerateContentError::ResponseParsing(msg) => BedrockError::ToolInputParse(msg),
            GenerateContentError::Configuration(msg) => BedrockError::RequestBuilder(msg),
            _ => BedrockError::MessageConversion(error.to_string()),
        }
    }
}