//! Response types for authentication token operations.

use serde::Deserialize;

/// Response from the Gemini API when creating an authentication token.
///
/// This structure represents the response received after successfully creating
/// an authentication token that can be used for bidirectional streaming
/// connections with the Gemini API.
///
/// # Examples
///
/// ```ignore
/// use gemini_ox::Gemini;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let gemini = Gemini::new("your-api-key");
///
/// let response = gemini
///     .create_auth_token()
///     .uses(10)
///     .build()
///     .send()
///     .await?;
///
/// println!("Token: {}", response.token);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AuthTokenResponse {
    /// The authentication token string that can be used for API calls.
    ///
    /// This token should be included in subsequent API requests that require
    /// authentication for bidirectional streaming operations.
    pub token: String,
    // Add other fields if the API returns them, e.g., expire_time
}
