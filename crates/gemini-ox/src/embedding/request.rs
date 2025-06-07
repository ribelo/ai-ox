use crate::content::Content;
use bon::Builder;
use serde::Serialize;

/// Task type for embedding generation, optimized for specific use cases
///
/// Different task types result in embeddings that are optimized for different downstream tasks.
/// Choosing the appropriate task type can improve the quality of your embeddings for specific applications.
#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TaskType {
    /// Unspecified task type (default)
    ///
    /// Use this when you're unsure which task type to use, or for general-purpose embeddings.
    #[default]
    TaskTypeUnspecified,

    /// For queries in a retrieval/search setting
    ///
    /// Use this for short text that will be used as search queries to find relevant documents.
    /// Examples: search queries, questions.
    RetrievalQuery,

    /// For documents in a retrieval/search setting  
    ///
    /// Use this for longer text that will be stored in a corpus and searched against.
    /// Examples: articles, documents, web pages that will be indexed.
    RetrievalDocument,

    /// For semantic similarity tasks
    ///
    /// Use this when you want to compare the semantic similarity between different pieces of text.
    /// Examples: finding similar documents, duplicate detection.
    SemanticSimilarity,

    /// For classification tasks
    ///
    /// Use this when you plan to use the embeddings as features for a classification model.
    /// Examples: sentiment analysis, topic classification.
    Classification,

    /// For clustering tasks
    ///
    /// Use this when you plan to cluster the embeddings to find groups of similar content.
    /// Examples: topic modeling, content organization.
    Clustering,
}

/// Request to generate embeddings for content using the Gemini API
///
/// This struct represents a request to the `embedContent` endpoint of the Gemini API.
/// Use the builder pattern to construct requests with optional parameters.
///
/// # Examples
///
/// Basic usage:
/// ```rust,no_run
/// # use gemini_ox::{Gemini, Model, content::Content};
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let gemini = Gemini::new("your-api-key");
/// let response = gemini.embed_content()
///     .model(Model::TextEmbedding004)
///     .content(Content::from("Hello, world!"))
///     .build()
///     .send()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
/// With task type and configuration:
/// ```rust,no_run
/// # use gemini_ox::{Gemini, Model, content::Content, embedding::TaskType};
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let gemini = Gemini::new("your-api-key");
/// let response = gemini.embed_content()
///     .model(Model::TextEmbedding004)
///     .content(Content::from("This is a document to be indexed."))
///     .task_type(TaskType::RetrievalDocument)
///     .title("Document Title")
///     .output_dimensionality(512)
///     .build()
///     .send()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Serialize, Builder)]
#[serde(rename_all = "camelCase")]
pub struct EmbedContentRequest {
    /// The model to use for embedding generation
    ///
    /// Use one of the available Gemini embedding models. Check the official
    /// documentation for the most up-to-date list of supported models.
    #[builder(into)]
    pub model: String,

    /// The content to generate embeddings for
    ///
    /// This can be text content or other supported content types.
    pub content: Content,

    /// Optional task type for optimized embedding generation
    ///
    /// Specifying a task type can improve embedding quality for specific use cases.
    /// If not specified, the model will use its default behavior.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<TaskType>,

    /// Optional title for RETRIEVAL_DOCUMENT task types
    ///
    /// When using `TaskType::RetrievalDocument`, you can provide a title that gives
    /// additional context about the document being embedded.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub title: Option<String>,

    /// Optional output dimensionality to truncate embeddings
    ///
    /// If specified, the embedding vector will be truncated to this size.
    /// This can be useful for reducing storage requirements or API costs.
    /// The value must be less than or equal to the model's native dimension.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimensionality: Option<u32>,

    /// The Gemini client instance (not serialized)
    #[serde(skip)]
    pub(crate) gemini: crate::Gemini,
}

impl crate::Gemini {
    /// Create an embed content request builder
    ///
    /// Returns an `EmbedContentRequestBuilder` that can be configured to generate embeddings
    /// for text content using the Gemini embedding models.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use gemini_ox::{Gemini, Model, content::Content};
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let gemini = Gemini::new("your-api-key");
    /// let response = gemini.embed_content()
    ///     .model(Model::TextEmbedding004)
    ///     .content(Content::from("Hello, world!"))
    ///     .build()
    ///     .send()
    ///     .await?;
    ///
    /// println!("Embedding dimension: {}", response.embedding.values.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed_content(
        &self,
    ) -> EmbedContentRequestBuilder<embed_content_request_builder::SetGemini> {
        EmbedContentRequest::builder().gemini(self.clone())
    }
}
