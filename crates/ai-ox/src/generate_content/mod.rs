pub mod gemini;

use std::fmt;

use futures_util::{future::BoxFuture, stream::BoxStream};

use crate::{
    content::{
        delta::MessageStreamEvent, requests::GenerateContentRequest,
        response::GenerateContentResponse,
    },
    errors::GenerateContentError,
};

/// A trait for interacting with a large language model.
///
/// This trait provides a standardized interface for sending requests and streaming
/// responses from various LLM providers.
pub trait GenerateContent: Send + Sync + fmt::Debug {
    /// Sends a single, non-streaming request to the model.
    ///
    /// # Arguments
    ///
    /// * `request` - A reference to the `GenerateContentRequest` containing the messages
    /// and other parameters for the model.
    ///
    /// # Returns
    ///
    /// A `Result` containing either the `GenerateContentResponse` from the model or
    /// a `ModelError` if something went wrong.
    fn request<'a>(
        &'a self,
        request: GenerateContentRequest,
    ) -> BoxFuture<'a, Result<GenerateContentResponse, GenerateContentError>>;

    /// Initiates a streaming request to the model.
    ///
    /// This method returns a stream of `MessageStreamEvent`s, allowing for real-time
    /// processing of the model's output as it is generated.
    ///
    /// # Arguments
    ///
    /// * `request` - A reference to the `GenerateContentRequest`.
    ///
    /// # Returns
    ///
    /// A `BoxStream` that yields `Result<MessageStreamEvent, ModelError>` items.
    fn request_stream<'a>(
        &'a self,
        request: &'a GenerateContentRequest,
    ) -> BoxStream<'a, Result<MessageStreamEvent, GenerateContentError>>;
}
