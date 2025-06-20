pub mod content;
pub mod gemini;

use crate::{
    content::{
        delta::MessageStreamEvent,
        response::{GenerateContentResponse, GenerateContentStructuredResponse},
    },
    errors::GenerateContentError,
};
use futures_util::stream::BoxStream;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::content::message::Message;

/// A trait for interacting with a large language model.
///
/// This trait provides a standardized interface for sending requests and streaming
/// responses from various LLM providers.
pub trait Model: Send + Sync {
    /// Returns the model name/identifier.
    ///
    /// # Returns
    ///
    /// A string slice containing the model name or identifier.
    fn model(&self) -> &str;

    /// Sends a single, non-streaming request to the model.
    ///
    /// # Arguments
    ///
    /// * `messages` - An iterator of items that can be converted into `Message`.
    ///                This allows passing single messages, vectors of messages, or other collections.
    ///
    /// # Returns
    ///
    /// A `Result` containing either the `GenerateContentResponse` from the model or
    /// a `GenerateContentError` if something went wrong.
    async fn request(
        &self,
        messages: impl IntoIterator<Item = impl Into<Message>>,
    ) -> Result<GenerateContentResponse, GenerateContentError>;

    /// Initiates a streaming request to the model.
    ///
    /// This method returns a stream of `MessageStreamEvent`s, allowing for real-time
    /// processing of the model's output as it is generated.
    ///
    /// # Arguments
    ///
    /// * `messages` - An iterator of items that can be converted into `Message`.
    ///
    /// # Returns
    ///
    /// A `BoxStream` that yields `Result<MessageStreamEvent, GenerateContentError>` items.
    fn request_stream<'a>(
        &'a self,
        messages: impl IntoIterator<Item = impl Into<Message>>,
    ) -> BoxStream<'a, Result<MessageStreamEvent, GenerateContentError>>;

    /// Generates structured content that conforms to a specific schema.
    ///
    /// This method takes a collection of messages and returns a structured response
    /// parsed into the specified type `O`. The model is guided by the JSON schema
    /// automatically derived from `O` to ensure the response matches the expected format.
    ///
    /// # Type Parameters
    ///
    /// * `O` - The target type to deserialize the response into. Must implement:
    ///   - `DeserializeOwned` - For JSON deserialization
    ///   - `JsonSchema` - For schema generation to guide the model
    ///   - `Send` - For async compatibility
    ///
    /// # Arguments
    ///
    /// * `messages` - An iterator of items that can be converted into `Message`.
    ///
    /// # Returns
    ///
    /// A `Result` containing either:
    /// - `Ok(GenerateContentStructuredResponse<O>)` - The successfully parsed structured response
    /// - `Err(GenerateContentError)` - An error that occurred during:
    ///   - Model communication (`ClientError`)
    ///   - Response parsing (`ResponseParsing`)
    ///   - Missing response (`NoResponse`)
    ///   - Unsupported feature (`UnsupportedFeature`)
    ///
    /// # Default Implementation
    ///
    /// The default implementation returns `UnsupportedFeature` error, which should be
    /// overridden by models that support structured output generation.
    async fn request_structured<O>(
        &self,
        _messages: impl IntoIterator<Item = impl Into<Message>>,
    ) -> Result<GenerateContentStructuredResponse<O>, GenerateContentError>
    where
        O: DeserializeOwned + JsonSchema + Send,
    {
        Err(GenerateContentError::unsupported_feature(
            "structured response",
        ))
    }
}
