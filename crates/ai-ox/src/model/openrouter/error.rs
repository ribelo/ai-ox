use crate::errors::GenerateContentError;
use openrouter_ox::OpenRouterRequestError;
use std::env::VarError;

/// OpenRouter-specific error types that properly wrap the underlying errors
#[derive(Debug, thiserror::Error)]
pub enum OpenRouterError {
    #[error("OpenRouter API error: {0}")]
    Api(#[from] OpenRouterRequestError),

    #[error("Environment variable error: {0}")]
    Env(#[from] VarError),

    #[error("Request builder error: {0}")]
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

impl From<OpenRouterError> for GenerateContentError {
    fn from(error: OpenRouterError) -> Self {
        match error {
            OpenRouterError::Api(api_error) => {
                match api_error {
                    OpenRouterRequestError::ReqwestError(reqwest_err) => {
                        GenerateContentError::provider_error("openrouter", format!("Network error: {}", reqwest_err))
                    }
                    OpenRouterRequestError::SerdeError(serde_err) => {
                        GenerateContentError::response_parsing(format!("OpenRouter JSON parsing error: {}", serde_err))
                    }
                    OpenRouterRequestError::InvalidRequestError { code, message, .. } => {
                        let code_str = code.as_deref().unwrap_or("unknown");
                        GenerateContentError::provider_error("openrouter", format!("API error {}: {}", code_str, message))
                    }
                    OpenRouterRequestError::UnexpectedResponse(response) => {
                        GenerateContentError::response_parsing(format!("OpenRouter unexpected response: {}", response))
                    }
                    OpenRouterRequestError::Stream(stream_error) => {
                        GenerateContentError::provider_error("openrouter", format!("Stream error: {}", stream_error))
                    }
                    OpenRouterRequestError::JsonDeserializationError(json_err) => {
                        GenerateContentError::response_parsing(format!("OpenRouter JSON deserialization error: {}", json_err))
                    }
                    OpenRouterRequestError::InvalidEventData(event_error) => {
                        GenerateContentError::response_parsing(format!("OpenRouter invalid event data: {}", event_error))
                    }
                    OpenRouterRequestError::RateLimit => {
                        GenerateContentError::provider_error("openrouter", "Rate limit exceeded".to_string())
                    }
                    OpenRouterRequestError::UrlBuildError(url_error) => {
                        GenerateContentError::configuration(format!("OpenRouter URL build error: {}", url_error))
                    }
                    OpenRouterRequestError::IoError(io_error) => {
                        GenerateContentError::provider_error("openrouter", format!("I/O error: {}", io_error))
                    }
                }
            }
            OpenRouterError::Env(var_error) => {
                GenerateContentError::configuration(format!("OpenRouter environment variable error: {}", var_error))
            }
            OpenRouterError::RequestBuilder(msg) => {
                GenerateContentError::configuration(format!("OpenRouter request builder: {}", msg))
            }
            OpenRouterError::ResponseParsing(msg) => {
                GenerateContentError::response_parsing(format!("OpenRouter response parsing: {}", msg))
            }
            OpenRouterError::InvalidSchema(msg) => {
                GenerateContentError::configuration(format!("OpenRouter invalid schema: {}", msg))
            }
            OpenRouterError::ToolConversion(msg) => {
                GenerateContentError::configuration(format!("OpenRouter tool conversion: {}", msg))
            }
            OpenRouterError::MessageConversion(msg) => {
                GenerateContentError::message_conversion(format!("OpenRouter message conversion: {}", msg))
            }
        }
    }
}