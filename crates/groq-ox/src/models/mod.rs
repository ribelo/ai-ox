pub mod response;

use self::response::{ListModelsResponse, ModelInfo};
use crate::{Groq, GroqRequestError};

const MODELS_URL: &str = "openai/v1/models";

impl Groq {
    /// Get information about a specific model by ID
    ///
    /// # Arguments
    /// * `model_id` - The ID of the model to retrieve (e.g., "llama-3.3-70b-versatile")
    ///
    /// # Returns
    /// Returns a `Result` containing the model information or a `GroqRequestError`
    ///
    /// # Example
    /// ```rust,no_run
    /// # use groq_ox::Groq;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Groq::load_from_env()?;
    /// let model = client.get_model("llama-3.3-70b-versatile").await?;
    /// println!("Model: {}", model.id);
    /// println!("Context window: {}", model.context_window);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_model(&self, model_id: &str) -> Result<ModelInfo, GroqRequestError> {
        let url = format!("{}/{}/{}", self.base_url, MODELS_URL, model_id);

        let res = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        if res.status().is_success() {
            Ok(res.json::<ModelInfo>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(crate::error::parse_error_response(status, bytes))
        }
    }

    /// List all available models
    ///
    /// # Returns
    /// Returns a `Result` containing the list of models or a `GroqRequestError`
    ///
    /// # Example
    /// ```rust,no_run
    /// # use groq_ox::Groq;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Groq::load_from_env()?;
    /// let response = client.list_models().await?;
    /// for model in response.data {
    ///     println!("Model: {} - Context: {}", model.id, model.context_window);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_models(&self) -> Result<ListModelsResponse, GroqRequestError> {
        let url = format!("{}/{}", self.base_url, MODELS_URL);

        let res = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        if res.status().is_success() {
            Ok(res.json::<ListModelsResponse>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(crate::error::parse_error_response(status, bytes))
        }
    }
}
