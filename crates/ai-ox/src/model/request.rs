//! Defines the canonical `ModelRequest` used for all model interactions.
//!
//! This module introduces the `ModelRequest` struct, which encapsulates all
//! possible parameters for a request to a large language model. Using a single
//! request struct provides a stable, vendor-agnostic API.

use bon::Builder;

use crate::{content::Message, tool::Tool};

/// Represents a single, canonical request to a large language model.
///
/// This struct uses the builder pattern for ergonomic construction. It is the
/// single source of truth for all parameters that can be sent to a model,
/// including messages, system instructions, and tools.
#[derive(Debug, Clone, Builder)]
pub struct ModelRequest {
    /// The messages that form the core of the request.
    #[builder(field)]
    pub messages: Vec<Message>,
    /// A list of tools the model can use to respond to the request.
    #[builder(field)]
    pub tools: Option<Vec<Tool>>,
    /// An optional system instruction to guide the model's behavior.
    #[builder(into)]
    pub system_message: Option<Message>,
}

impl<S: model_request_builder::State> ModelRequestBuilder<S> {
    pub fn messages(mut self, messages: impl IntoIterator<Item = impl Into<Message>>) -> Self {
        self.messages = messages.into_iter().map(Into::into).collect();
        self
    }
    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.get_or_insert_default().push(tool);
        self
    }
    pub fn tools(mut self, tools: impl IntoIterator<Item = Tool>) -> Self {
        self.tools.get_or_insert_default().extend(tools);
        self
    }
}

impl<T> From<T> for ModelRequest
where
    T: IntoIterator,
    T::Item: Into<Message>,
{
    fn from(messages: T) -> Self {
        Self {
            messages: messages.into_iter().map(Into::into).collect(),
            system_message: None,
            tools: None,
        }
    }
}
