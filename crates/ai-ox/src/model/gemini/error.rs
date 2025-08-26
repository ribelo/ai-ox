use crate::errors::GenerateContentError;
use gemini_ox::GeminiRequestError;
use std::env::VarError;

/// Gemini-specific error types that properly wrap the underlying errors
#[derive(Debug, thiserror::Error)]
pub enum GeminiError {
    #[error("Gemini API error: {0}")]
    Api(#[from] GeminiRequestError),

    #[error("Missing API key. Please set it via the builder or the GOOGLE_AI_API_KEY environment variable.")]
    MissingApiKey,

    #[error("Missing model name. Please set it via the builder.")]
    MissingModel,

    #[error("Environment variable error: {0}")]
    Env(#[from] VarError),

    #[error("Failed to build request: {0}")]
    RequestBuilder(String),

    #[error("Response parsing error: {0}")]
    ResponseParsing(String),

    #[error("Invalid schema: {0}")]
    InvalidSchema(String),

    #[error("Tool conversion error: {0}")]
    ToolConversion(String),

    #[error("Message conversion error: {0}")]
    MessageConversion(String),
}

impl From<GeminiError> for GenerateContentError {
    fn from(error: GeminiError) -> Self {
        match error {
            GeminiError::Api(api_error) => {
                match api_error {
                    GeminiRequestError::ReqwestError(reqwest_err) => {
                        GenerateContentError::provider_error("gemini", format!("Network error: {}", reqwest_err))
                    }
                    GeminiRequestError::SerdeError(serde_err) => {
                        GenerateContentError::response_parsing(format!("Gemini JSON parsing error: {}", serde_err))
                    }
                    GeminiRequestError::JsonDeserializationError(json_err) => {
                        GenerateContentError::response_parsing(format!("Gemini JSON deserialization error: {}", json_err))
                    }
                    GeminiRequestError::InvalidRequestError { message, code, .. } => {
                        let error_msg = match code {
                            Some(code) => format!("API error {}: {}", code, message),
                            None => format!("API error: {}", message),
                        };
                        GenerateContentError::provider_error("gemini", error_msg)
                    }
                    GeminiRequestError::UnexpectedResponse(response) => {
                        GenerateContentError::response_parsing(format!("Gemini unexpected response: {}", response))
                    }
                    GeminiRequestError::InvalidEventData(event_error) => {
                        GenerateContentError::response_parsing(format!("Gemini invalid event data: {}", event_error))
                    }
                    GeminiRequestError::RateLimit => {
                        GenerateContentError::provider_error("gemini", "Rate limit exceeded")
                    }
                    GeminiRequestError::UrlBuildError(url_error) => {
                        GenerateContentError::configuration(format!("Gemini URL build error: {}", url_error))
                    }
                    GeminiRequestError::IoError(io_error) => {
                        GenerateContentError::provider_error("gemini", format!("IO error: {}", io_error))
                    }
                    GeminiRequestError::AuthenticationMissing => {
                        GenerateContentError::configuration("Gemini authentication is missing: no API key or OAuth token provided")
                    }
                }
            }
            GeminiError::MissingApiKey => {
                GenerateContentError::configuration("Missing Gemini API key".to_string())
            }
            GeminiError::MissingModel => {
                GenerateContentError::configuration("Missing Gemini model name".to_string())
            }
            GeminiError::Env(var_error) => {
                GenerateContentError::configuration(format!("Gemini environment variable error: {}", var_error))
            }
            GeminiError::RequestBuilder(msg) => {
                GenerateContentError::configuration(format!("Gemini request builder: {}", msg))
            }
            GeminiError::ResponseParsing(msg) => {
                GenerateContentError::response_parsing(format!("Gemini response parsing: {}", msg))
            }
            GeminiError::InvalidSchema(msg) => {
                GenerateContentError::configuration(format!("Gemini invalid schema: {}", msg))
            }
            GeminiError::ToolConversion(msg) => {
                GenerateContentError::configuration(format!("Gemini tool conversion: {}", msg))
            }
            GeminiError::MessageConversion(msg) => {
                GenerateContentError::message_conversion(format!("Gemini message conversion: {}", msg))
            }
        }
    }
}