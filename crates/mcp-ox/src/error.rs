use thiserror::Error;

#[derive(Error, Debug)]
pub enum McpConversionError {
    #[error("Unsupported content type: {0}")]
    UnsupportedContentType(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid data format: {0}")]
    InvalidFormat(String),

    #[error("Conversion not supported: {0}")]
    ConversionNotSupported(String),
}
