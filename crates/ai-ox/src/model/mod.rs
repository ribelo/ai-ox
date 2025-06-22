pub mod gemini;
pub mod request;
pub mod response;

use futures_util::{future::BoxFuture, stream::BoxStream};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::{
    StructuredResponse,
    content::{delta::MessageStreamEvent, message::Message},
    errors::GenerateContentError,
    model::{
        request::ModelRequest,
        response::{ModelResponse, RawStructuredResponse},
    },
};

/// The primary trait for interacting with a large language model.
///
/// This trait provides a standardized interface for sending requests, streaming
/// responses, and generating structured content. It is designed to be object-safe
/// (`dyn Model`) for its core, non-generic methods. Generic helper methods are
/// provided for a more ergonomic developer experience.
pub trait Model: Send + Sync + 'static + std::fmt::Debug {
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
    /// * `request` - A ModelRequest containing messages, system instruction, and tools.
    ///
    /// # Returns
    ///
    /// A `Result` containing either the `ModelResponse` from the model or
    /// a `GenerateContentError` if something went wrong.
    fn request(
        &self,
        request: ModelRequest,
    ) -> BoxFuture<'_, Result<ModelResponse, GenerateContentError>>;

    /// Initiates a streaming request to the model.
    ///
    /// # Arguments
    ///
    /// * `request` - A ModelRequest containing messages, system instruction, and tools.
    ///
    /// # Returns
    ///
    /// A `BoxStream` that yields `Result<MessageStreamEvent, GenerateContentError>` items.
    fn request_stream(
        &self,
        request: ModelRequest,
    ) -> BoxStream<'_, Result<MessageStreamEvent, GenerateContentError>>;

    /// The internal, object-safe method for handling structured content requests.
    ///
    /// This method is intended to be implemented by concrete model providers. It takes
    /// a request and a JSON schema string, and is responsible for interacting with
    /// the model to get a response that conforms to the schema.
    ///
    /// This should not typically be called directly by end-users. Prefer the generic
    /// `request_structured` helper method instead.
    ///
    /// # Arguments
    ///
    /// * `request` - A `ModelRequest` containing the messages.
    /// * `schema` - A string representation of the JSON schema for the desired output.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `RawStructuredContent` (raw JSON and usage data) or a
    /// `GenerateContentError`.
    fn request_structured_internal(
        &self,
        request: ModelRequest,
        schema: String,
    ) -> BoxFuture<'_, Result<RawStructuredResponse, GenerateContentError>>;

    /// Generates structured content that conforms to a specific schema.
    ///
    /// This is a high-level helper method that takes a collection of messages and returns
    /// a structured response parsed into the specified type `O`. The model is guided
    /// by the JSON schema automatically derived from `O`.
    ///
    /// This method is only available on sized types (`where Self: Sized`) and is not
    /// part of the `dyn Model` vtable. It acts as a generic wrapper around the required
    /// `request_structured_internal` method.
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
    /// - `Ok(StructuredResponse<O>)` - The successfully parsed structured response
    /// - `Err(GenerateContentError)` - An error that occurred during model interaction or parsing.
    fn request_structured<O>(
        &self,
        messages: impl IntoIterator<Item = impl Into<Message>>,
    ) -> BoxFuture<'_, Result<StructuredResponse<O>, GenerateContentError>>
    where
        O: DeserializeOwned + JsonSchema + Send,
        Self: Sized,
    {
        use futures_util::FutureExt;
        use schemars::schema_for;

        let msgs: Vec<Message> = messages.into_iter().map(Into::into).collect();
        let request = ModelRequest {
            messages: msgs,
            system_message: None,
            tools: None,
        };
        let schema = serde_json::to_string(&schema_for!(O)).unwrap_or_default();

        async move {
            let structured_content = self.request_structured_internal(request, schema).await?;

            let data: O = serde_json::from_value(structured_content.json.clone())
                .map_err(|e| GenerateContentError::response_parsing(e.to_string()))?;

            Ok(StructuredResponse {
                data,
                model_name: structured_content.model_name,
                usage: structured_content.usage,
                vendor_name: structured_content.vendor_name,
            })
        }
        .boxed()
    }
}
