pub mod response;

use self::response::{ListModelsResponse, Model};
use crate::{BASE_URL, Gemini, GeminiRequestError, parse_error_response};

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
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Gemini::new("your-api-key");
    /// let model = client.get_model("gemini-1.5-flash-latest").await?;
    /// println!("Model: {}", model.display_name);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_model(&self, name: &str) -> Result<Model, GeminiRequestError> {
        let url = format!("{}/{}/models/{}", BASE_URL, self.api_version, name);
        let res = self
            .client
            .get(url)
            .query(&[("key", &self.api_key)])
            .send()
            .await?;

        let status = res.status();
        let body_bytes = res.bytes().await?;

        if status.is_success() {
            serde_json::from_slice::<Model>(&body_bytes)
                .map_err(GeminiRequestError::JsonDeserializationError)
        } else {
            Err(parse_error_response(status, body_bytes))
        }
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
    /// # #[tokio::main]
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
        let url = format!("{}/{}/models", BASE_URL, self.api_version);

        let mut query_params = vec![("key", self.api_key.as_str())];

        let page_size_string;
        if let Some(size) = page_size {
            page_size_string = size.to_string();
            query_params.push(("pageSize", &page_size_string));
        }

        if let Some(token) = page_token {
            query_params.push(("pageToken", token));
        }

        let res = self.client.get(url).query(&query_params).send().await?;

        let status = res.status();
        let body_bytes = res.bytes().await?;

        if status.is_success() {
            serde_json::from_slice::<ListModelsResponse>(&body_bytes)
                .map_err(GeminiRequestError::JsonDeserializationError)
        } else {
            Err(parse_error_response(status, body_bytes))
        }
    }
}
