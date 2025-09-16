mod conversion;
mod error;

pub use error::MistralError;

use crate::{
    ModelResponse,
    content::delta::StreamEvent,
    errors::GenerateContentError,
    model::{Model, ModelInfo, ModelRequest, Provider, response::RawStructuredResponse},
    usage::Usage,
};
use async_stream::try_stream;
use bon::Builder;
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream::BoxStream};
use mistral_ox::{Mistral, tool::ToolChoice};

/// Returns the default tool choice for Mistral models.
/// Defaults to Auto, allowing the model to decide when to use tools.
fn default_tool_choice() -> ToolChoice {
    ToolChoice::Auto
}

/// Represents a model from the Mistral AI family.
#[derive(Debug, Clone, Builder)]
pub struct MistralModel {
    /// Mistral client
    #[builder(field)]
    client: Mistral,
    /// The specific model name (e.g., "mistral-large-latest").
    #[builder(into)]
    model: String,
    /// System instruction if provided
    #[builder(into)]
    system_instruction: Option<String>,
    /// Tool choice configuration
    #[builder(default = default_tool_choice())]
    tool_choice: ToolChoice,
}

impl<S: mistral_model_builder::State> MistralModelBuilder<S> {
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.client = Mistral::new(api_key);
        self
    }
}

impl MistralModel {
    /// Create a new MistralModel from environment variables.
    ///
    /// This function reads the MISTRAL_API_KEY from the environment and returns an error if missing.
    pub async fn new(model: impl Into<String>) -> Result<Self, MistralError> {
        let api_key = std::env::var("MISTRAL_API_KEY").map_err(|_| MistralError::MissingApiKey)?;

        let client = Mistral::new(&api_key);

        Ok(Self {
            client,
            model: model.into(),
            system_instruction: None,
            tool_choice: default_tool_choice(),
        })
    }
}

impl Model for MistralModel {
    fn info(&self) -> ModelInfo<'_> {
        ModelInfo(Provider::Mistral, &self.model)
    }

    fn name(&self) -> &str {
        &self.model
    }

    /// Sends a request to the Mistral API and returns the response.
    fn request(
        &self,
        request: ModelRequest,
    ) -> BoxFuture<'_, Result<ModelResponse, GenerateContentError>> {
        async move {
            let mistral_request = conversion::convert_request_to_mistral(
                request,
                self.model.clone(),
                self.system_instruction.clone(),
                Some(mistral_ox::tool::ToolChoice::Auto),
            )?;
            let response = self
                .client
                .send(&mistral_request)
                .await
                .map_err(MistralError::Api)?;
            conversion::convert_mistral_response_to_ai_ox(response, self.model.clone())
        }
        .boxed()
    }

    /// Returns a stream of events for a streaming request.
    fn request_stream(
        &self,
        request: ModelRequest,
    ) -> BoxStream<'_, Result<StreamEvent, GenerateContentError>> {
        let client = self.client.clone();

        let stream = try_stream! {
            let mistral_request = conversion::convert_request_to_mistral(
                request,
                self.model.clone(),
                self.system_instruction.clone(),
                Some(mistral_ox::tool::ToolChoice::Auto),
            )?;
            let mut response_stream = client.stream(&mistral_request);

            while let Some(response) = response_stream.next().await {
                let response = response.map_err(MistralError::Api)?;
                let events = conversion::convert_response_to_stream_events(response);
                for event in events {
                    yield event?;
                }
            }
        };

        Box::pin(stream)
    }

    fn request_structured_internal(
        &self,
        request: ModelRequest,
        _schema: String,
    ) -> BoxFuture<'_, Result<RawStructuredResponse, GenerateContentError>> {
        async move {
            // For Mistral, we use response_format with json_object type
            let mut mistral_request = conversion::convert_request_to_mistral(
                request,
                self.model.clone(),
                self.system_instruction.clone(),
                Some(self.tool_choice.clone()),
            )?;

            // Set response format to JSON
            mistral_request.response_format = Some(serde_json::json!({
                "type": "json_object"
            }));

            let response = self
                .client
                .send(&mistral_request)
                .await
                .map_err(MistralError::Api)?;

            // Extract the text content from the first choice
            let text = response
                .choices
                .first()
                .and_then(|choice| {
                    choice
                        .message
                        .content
                        .0
                        .iter()
                        .find_map(|p| p.as_text().map(|t| t.text.clone()))
                })
                .ok_or_else(|| MistralError::ResponseParsing("No response content".to_string()))?;

            // Parse the text as JSON
            let json = serde_json::from_str(&text)
                .map_err(|e| MistralError::ResponseParsing(e.to_string()))?;

            let usage = response
                .usage
                .map(|u| {
                    let mut usage = Usage::new();
                    usage.requests = 1;
                    usage
                        .input_tokens_by_modality
                        .insert(crate::usage::Modality::Text, u.prompt_tokens as u64);
                    usage
                        .output_tokens_by_modality
                        .insert(crate::usage::Modality::Text, u.completion_tokens as u64);
                    usage
                })
                .unwrap_or_else(Usage::new);

            Ok(RawStructuredResponse {
                json,
                usage,
                model_name: self.model.clone(),
                vendor_name: "mistral".to_string(),
            })
        }
        .boxed()
    }
}
