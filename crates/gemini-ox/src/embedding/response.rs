use serde::{Deserialize, Serialize};

/// Represents a single embedding vector from the Gemini API
///
/// An embedding is a numerical representation of content (typically text) as a vector
/// of floating-point numbers. These vectors can be used for various machine learning
/// tasks such as semantic search, clustering, and classification.
///
/// # Examples
///
/// ```rust,no_run
/// # use gemini_ox::embedding::ContentEmbedding;
/// let embedding = ContentEmbedding {
///     values: vec![0.1, 0.2, -0.3, 0.4],
/// };
///
/// // Calculate the magnitude (L2 norm) of the embedding
/// let magnitude: f32 = embedding.values.iter()
///     .map(|x| x * x)
///     .sum::<f32>()
///     .sqrt();
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentEmbedding {
    /// The embedding values as a vector of floating-point numbers
    ///
    /// The length of this vector depends on the model used and can be reduced by
    /// specifying `output_dimensionality` in the request. The values are typically
    /// normalized to unit length for embedding models.
    pub values: Vec<f32>,
}

/// Response from the embedContent API endpoint
///
/// This struct contains the embedding generated for the input content.
/// The response always contains a single embedding since the embedContent
/// endpoint processes one piece of content at a time.
///
/// # Examples
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
/// println!("Generated embedding with {} dimensions", response.embedding.values.len());
///
/// // Access the first few dimensions
/// if let Some(first_values) = response.embedding.values.get(0..5) {
///     println!("First 5 dimensions: {:?}", first_values);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbedContentResponse {
    /// The generated embedding for the input content
    ///
    /// This contains the numerical representation of the input content as a vector
    /// of floating-point numbers that can be used for downstream machine learning tasks.
    pub embedding: ContentEmbedding,
}
