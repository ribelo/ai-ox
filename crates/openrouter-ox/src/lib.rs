use bon::Builder;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod message;
pub mod models;
pub mod provider_preference;
pub mod request;
pub mod response;
pub mod router;
pub mod tool;
const BASE_URL: &str = "https://openrouter.ai";

#[cfg(feature = "leaky-bucket")]
pub use leaky_bucket::RateLimiter;
#[cfg(feature = "leaky-bucket")]
use std::sync::Arc;
use std::{collections::HashMap, fmt};

#[derive(Clone, Builder)]
pub struct OpenRouter {
    #[builder(into)]
    api_key: String,
    #[builder(default)]
    #[allow(dead_code)]
    headers: HashMap<String, String>,
    #[builder(default)]
    client: reqwest::Client,
    #[cfg(feature = "leaky-bucket")]
    #[allow(dead_code)]
    leaky_bucket: Option<Arc<RateLimiter>>,
}

impl OpenRouter {
    pub fn new_from_env() -> Result<Self, std::env::VarError> {
        let api_key = std::env::var("OPENROUTER_API_KEY")?;
        Ok(Self::builder().api_key(api_key).build())
    }
}

impl fmt::Debug for OpenRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAi")
            .field("api_key", &"[REDACTED]")
            .field("client", &self.client)
            .finish()
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ErrorDetail {
    pub code: i32,
    pub message: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
    pub user_id: Option<String>,
}

impl std::fmt::Display for ErrorResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Error {}: {} (user_id: {:?})",
            self.error.code, self.error.message, self.user_id
        )
    }
}

#[derive(Debug, Error)]
pub enum ApiRequestError {
    #[error(transparent)]
    ReqwestError(#[from] reqwest::Error),
    #[error(transparent)]
    SerdeError(#[from] serde_json::Error),

    #[error("Invalid request error: {0}")]
    InvalidRequestError(ErrorResponse),
    #[error("Unexpected response from API: {response}")]
    UnexpectedResponse { response: String },
    #[error("Stream error: {0}")]
    Stream(String),
}

/// `ApiRequest` trait allows sending any prepared request by explicitly providing OpenAI client.
///
/// This trait is useful to abstract the details about API request, response, and error handling.
///
/// # Associated Types
///
/// - `Response`: A type that implements the `DeserializeOwned` trait from Serde. This type
/// represents the deserialized response that you expect back from the API call.
#[async_trait::async_trait]
pub trait ApiRequest {
    type Response: serde::de::DeserializeOwned;
    /// An async function that takes in an `OpenAi` object reference and returns a `Result` with
    /// deserialized `Response` type or an `ApiRequestError`. This function sends off the API
    /// request with given OpenAi client.
    async fn send_with(&self, open_ai: &OpenRouter) -> Result<Self::Response, ApiRequestError>;
}

/// `ApiRequestWithClient` trait allows sending any prepared request which internally uses OpenAI
/// client.
///
/// This trait is useful when the client does not want to externally manage or provide the `OpenAi`
/// client for making requests.
#[async_trait::async_trait]
pub trait ApiRequestWithClient: ApiRequest {
    /// An async function that takes no parameters. It internally uses the API client and so
    /// returns a `Result` with deserialized `Response` type or an `ApiRequestError`. This function
    /// sends off the API request.
    async fn send(&self) -> Result<Self::Response, ApiRequestError>;
}
