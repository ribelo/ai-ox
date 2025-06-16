pub mod gemini;
pub mod content;

use crate::{
    content::{delta::MessageStreamEvent, response::GenerateContentResponse}, errors::GenerateContentError
};
use futures_util::stream::BoxStream;

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
}
