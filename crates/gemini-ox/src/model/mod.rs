pub mod response;

use self::response::{ListModelsResponse, Model};
use crate::{Gemini, GeminiRequestError};
use ai_ox_common::request_builder::{Endpoint, HttpMethod};

impl Gemini {
    /// Get information about a specific model by name
    ///
    /// # Arguments
    /// * `name` - The name of the model to retrieve (e.g., "gemini-1.5-flash-latest")
    ///
    /// # Returns
    /// Returns a `Result` containing the model information or a `GeminiRequestError`
    ///
    /// # Example
    /// ```rust,no_run
    /// # use gemini_ox::Gemini;
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Gemini::new("your-api-key");
    /// let model = client.get_model("gemini-1.5-flash-latest").await?;
    /// println!("Model: {}", model.display_name);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_model(&self, name: &str) -> Result<Model, GeminiRequestError> {
        let helper = self.request_helper()?;
        let endpoint = Endpoint::new(
            format!("{}/models/{}", self.api_version, name),
            HttpMethod::Get,
        );

        helper.request(endpoint).await
    }

    /// List available models with optional pagination
    ///
    /// # Arguments
    /// * `page_size` - Maximum number of models to return per page (optional)
    /// * `page_token` - Token for retrieving a specific page of results (optional)
    ///
    /// # Returns
    /// Returns a `Result` containing the list of models and pagination info or a `GeminiRequestError`
    ///
    /// # Example
    /// ```rust,no_run
    /// # use gemini_ox::Gemini;
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Gemini::new("your-api-key");
    /// let response = client.list_models(Some(10), None).await?;
    /// for model in response.models {
    ///     println!("Model: {}", model.display_name);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_models(
        &self,
        page_size: Option<u32>,
        page_token: Option<&str>,
    ) -> Result<ListModelsResponse, GeminiRequestError> {
        let helper = self.request_helper()?;
        let mut endpoint = Endpoint::new(format!("{}/models", self.api_version), HttpMethod::Get);

        let mut params = Vec::new();
        if let Some(size) = page_size {
            params.push(("pageSize".to_string(), size.to_string()));
        }
        if let Some(token) = page_token {
            params.push(("pageToken".to_string(), token.to_string()));
        }
        if !params.is_empty() {
            endpoint = endpoint.with_query_params(params);
        }

        helper.request(endpoint).await
    }
}
