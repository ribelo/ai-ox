use thiserror::Error;

#[derive(Error, Debug)]
pub enum MistralError {
    #[error("Missing API key")]
    MissingApiKey,

    #[error("Mistral API error: {0}")]
    Api(#[from] mistral_ox::MistralRequestError),

    #[error("Response parsing error: {0}")]
    ResponseParsing(String),

    #[error("Invalid schema: {0}")]
    InvalidSchema(String),
}

impl From<MistralError> for crate::errors::GenerateContentError {
    fn from(err: MistralError) -> Self {
        match err {
            MistralError::MissingApiKey => crate::errors::GenerateContentError::configuration(
                "Missing MISTRAL_API_KEY environment variable",
            ),
            MistralError::Api(api_err) => {
                crate::errors::GenerateContentError::provider_error("mistral", api_err.to_string())
            }
            MistralError::ResponseParsing(msg) => {
                crate::errors::GenerateContentError::response_parsing(msg)
            }
            MistralError::InvalidSchema(msg) => {
                crate::errors::GenerateContentError::configuration(msg)
            }
        }
    }
}
