//! # Gemini Embeddings API
//!
//! This module provides support for generating text embeddings using the Google Gemini API.
//! Embeddings are numerical representations of text that can be used for various machine learning
//! tasks such as semantic search, clustering, classification, and similarity comparison.
//!
//! ## Features
//!
//! - Generate embeddings for text content using Gemini embedding models
//! - Support for different task types to optimize embeddings for specific use cases
//! - Configurable output dimensionality to control embedding size
//! - Builder pattern for easy request construction
//! - Comprehensive error handling
//!
//! ## Models
//!
//! The Gemini API supports various embedding models optimized for different use cases.
//! Check the [official documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for the most up-to-date list of available models.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use gemini_ox::{Gemini, Model, generate_content::Content};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let gemini = Gemini::new("your-api-key");
//!
//! let response = gemini
//!     .embed_content()
//!     .model(Model::TextEmbedding004)
//!     .content(Content::from("Hello, world!"))
//!     .build()
//!     .send()
//!     .await?;
//!
//! println!("Generated embedding with {} dimensions", response.embedding.values.len());
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced Usage
//!
//! ### Using Task Types
//!
//! Task types optimize embeddings for specific use cases:
//!
//! ```rust,no_run
//! use gemini_ox::{Gemini, Model, generate_content::Content, embedding::TaskType};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let gemini = Gemini::new("your-api-key");
//!
//! // For document indexing
//! let doc_response = gemini
//!     .embed_content()
//!     .model(Model::TextEmbedding004)
//!     .content(Content::from("This is a document to be indexed."))
//!     .task_type(TaskType::RetrievalDocument)
//!     .title("Document Title")
//!     .build()
//!     .send()
//!     .await?;
//!
//! // For search queries
//! let query_response = gemini
//!     .embed_content()
//!     .model(Model::TextEmbedding004)
//!     .content(Content::from("search query"))
//!     .task_type(TaskType::RetrievalQuery)
//!     .build()
//!     .send()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Controlling Output Dimensions
//!
//! Reduce embedding size for storage efficiency:
//!
//! ```rust,no_run
//! use gemini_ox::{Gemini, Model, generate_content::Content};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let gemini = Gemini::new("your-api-key");
//!
//! let response = gemini
//!     .embed_content()
//!     .model(Model::TextEmbedding004)
//!     .content(Content::from("Text to embed"))
//!     .output_dimensionality(256) // Reduce from 768 to 256 dimensions
//!     .build()
//!     .send()
//!     .await?;
//!
//! assert_eq!(response.embedding.values.len(), 256);
//! # Ok(())
//! # }
//! ```
//!
//! ## Error Handling
//!
//! All API calls return `Result<EmbedContentResponse, GeminiRequestError>`. Common errors include:
//!
//! - `GeminiRequestError::RateLimit` - Too many requests
//! - `GeminiRequestError::InvalidRequestError` - Invalid parameters or model name
//! - `GeminiRequestError::JsonDeserializationError` - Unexpected response format
//!
//! ## Performance Considerations
//!
//! - Use appropriate task types to improve embedding quality
//! - Consider using `output_dimensionality` to reduce storage requirements
//! - Batch processing is not supported by this endpoint; use multiple requests for multiple texts

use crate::{BASE_URL, GeminiRequestError, parse_error_response};

pub mod request;
pub mod response;

pub use request::{EmbedContentRequest, TaskType};
pub use response::{ContentEmbedding, EmbedContentResponse};

impl EmbedContentRequest {
    /// Sends an embed content request to the Gemini API
    ///
    /// This method makes a request to the embedContent endpoint and returns the embedding response.
    ///
    /// # Errors
    ///
    /// This function can return the following error variants:
    /// - `GeminiRequestError::ReqwestError` - if the HTTP request fails
    /// - `GeminiRequestError::RateLimit` - if the API rate limit is exceeded (HTTP 429)
    /// - `GeminiRequestError::InvalidRequestError` - if the API returns a 4xx/5xx error with structured error data
    /// - `GeminiRequestError::JsonDeserializationError` - if the API response cannot be parsed as JSON
    /// - `GeminiRequestError::UnexpectedResponse` - if the API returns an unexpected response format or error
    pub async fn send(&self) -> Result<EmbedContentResponse, GeminiRequestError> {
        // Construct the API URL properly using reqwest::Url
        let mut url = reqwest::Url::parse(BASE_URL)
            .map_err(|e| GeminiRequestError::UrlBuildError(e.to_string()))?;

        // Join the path elements properly
        url.path_segments_mut()
            .map_err(|_| GeminiRequestError::UrlBuildError("URL cannot be a base URL".to_string()))?
            .push(&self.gemini.api_version)
            .push("models")
            .push(&format!("{}:embedContent", self.model));

        // Add the API key as a query parameter
        url.query_pairs_mut()
            .append_pair("key", &self.gemini.api_key);

        // Send the HTTP request
        let res = self.gemini.client.post(url).json(self).send().await?;
        let status = res.status();

        // Read the response body once
        let body_bytes = res.bytes().await?;

        match status.as_u16() {
            // Success responses
            200 | 201 => {
                // Try to deserialize the successful response
                match serde_json::from_slice::<EmbedContentResponse>(&body_bytes) {
                    Ok(data) => Ok(data),
                    Err(e) => Err(GeminiRequestError::JsonDeserializationError(e)),
                }
            }
            // Rate limit response
            429 => Err(GeminiRequestError::RateLimit),
            // All other status codes are errors
            _ => Err(parse_error_response(status, body_bytes)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Gemini, Model, generate_content::content::Content};

    #[tokio::test]
    #[ignore = "Requires GOOGLE_AI_API_KEY environment variable and makes actual API calls"]
    async fn test_single_embedding_success() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    {
        let api_key = match std::env::var("GOOGLE_AI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!("GOOGLE_AI_API_KEY not set, skipping test_single_embedding_success");
                return Ok(());
            }
        };

        let gemini = Gemini::new(api_key);
        let request = gemini
            .embed_content()
            .model(Model::TextEmbedding004)
            .content(Content::from("Hello, world!"))
            .build();

        let response = request.send().await?;

        // Verify the response structure
        assert!(
            !response.embedding.values.is_empty(),
            "Embedding should contain values"
        );
        assert!(
            response.embedding.values.len() > 100,
            "Embedding should have reasonable dimension (>100)"
        );

        // Verify the values are normalized (typical for embedding models)
        let magnitude: f32 = response
            .embedding
            .values
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(magnitude > 0.0, "Embedding magnitude should be positive");

        Ok(())
    }

    #[tokio::test]
    #[ignore = "Requires GOOGLE_AI_API_KEY environment variable and makes actual API calls"]
    async fn test_embedding_with_task_type() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    {
        let api_key = match std::env::var("GOOGLE_AI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!("GOOGLE_AI_API_KEY not set, skipping test_embedding_with_task_type");
                return Ok(());
            }
        };

        let gemini = Gemini::new(api_key);
        let request = gemini
            .embed_content()
            .model(Model::TextEmbedding004)
            .content(Content::from("This is a document for retrieval."))
            .task_type(TaskType::RetrievalDocument)
            .title("Test Document")
            .build();

        let response = request.send().await?;

        // Verify the response structure
        assert!(
            !response.embedding.values.is_empty(),
            "Embedding should contain values"
        );
        assert!(
            response.embedding.values.len() > 100,
            "Embedding should have reasonable dimension (>100)"
        );

        Ok(())
    }

    #[tokio::test]
    #[ignore = "Requires GOOGLE_AI_API_KEY environment variable and makes actual API calls"]
    async fn test_embedding_with_output_dimensionality()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let api_key = match std::env::var("GOOGLE_AI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!(
                    "GOOGLE_AI_API_KEY not set, skipping test_embedding_with_output_dimensionality"
                );
                return Ok(());
            }
        };

        let target_dimension = 256;
        let gemini = Gemini::new(api_key);
        let request = gemini
            .embed_content()
            .model(Model::TextEmbedding004)
            .content(Content::from("This text will get a truncated embedding."))
            .output_dimensionality(target_dimension)
            .build();

        let response = request.send().await?;

        // Verify the response has the expected dimension
        assert_eq!(
            response.embedding.values.len(),
            target_dimension as usize,
            "Embedding should have exactly {} dimensions",
            target_dimension
        );

        Ok(())
    }
}
