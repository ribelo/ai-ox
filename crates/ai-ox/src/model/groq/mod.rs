mod conversion;
mod error;

pub use error::GroqError;

use crate::{
    ModelResponse,
    content::delta::StreamEvent,
    errors::GenerateContentError,
    model::{Model, ModelInfo, ModelRequest, Provider, response::RawStructuredResponse},
    usage::Usage,
};
use ai_ox_common::openai_format::ToolChoice;
use async_stream::try_stream;
use bon::Builder;
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream::BoxStream};
use groq_ox::{Groq, request::ResponseFormat};

/// Returns the default tool choice for Groq models.
/// Defaults to Auto, allowing the model to decide when to use tools.
fn default_tool_choice() -> ToolChoice {
    ToolChoice::Auto
}

/// Represents a model from the Groq family.
#[derive(Debug, Clone, Builder)]
pub struct GroqModel {
    /// Groq client
    #[builder(field)]
    client: Groq,
    /// The specific model name (e.g., "llama-3.1-70b-versatile").
    #[builder(into)]
    model: String,
    /// System instruction if provided
    #[builder(into)]
    system_instruction: Option<String>,
    /// Tool choice configuration
    #[builder(default = default_tool_choice())]
    tool_choice: ToolChoice,
}

impl<S: groq_model_builder::State> GroqModelBuilder<S> {
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.client = Groq::new(api_key);
        self
    }
}

impl GroqModel {
    /// Create a new GroqModel from environment variables.
    ///
    /// This function reads the GROQ_API_KEY from the environment and returns an error if missing.
    pub async fn new(model: impl Into<String>) -> Result<Self, GroqError> {
        let api_key = std::env::var("GROQ_API_KEY").map_err(|_| GroqError::MissingApiKey)?;

        let client = Groq::new(&api_key);

        Ok(Self {
            client,
            model: model.into(),
            system_instruction: None,
            tool_choice: default_tool_choice(),
        })
    }
}

impl Model for GroqModel {
    fn info(&self) -> ModelInfo<'_> {
        ModelInfo(Provider::Groq, &self.model)
    }

    fn name(&self) -> &str {
        &self.model
    }

    /// Sends a request to the Groq API and returns the response.
    fn request(
        &self,
        request: ModelRequest,
    ) -> BoxFuture<'_, Result<ModelResponse, GenerateContentError>> {
        async move {
            let groq_request = conversion::convert_request_to_groq(
                request,
                self.model.clone(),
                self.system_instruction.clone(),
                Some(self.tool_choice.clone()),
            )?;
            let response = self
                .client
                .send(&groq_request)
                .await
                .map_err(GroqError::Api)?;
            conversion::convert_groq_response_to_ai_ox(response, self.model.clone())
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
            let groq_request = conversion::convert_request_to_groq(
                request,
                self.model.clone(),
                self.system_instruction.clone(),
                Some(self.tool_choice.clone()),
            )?;
            let mut response_stream = client.stream(&groq_request);

            while let Some(response) = response_stream.next().await {
                let response = response.map_err(GroqError::Api)?;
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
        schema: String,
    ) -> BoxFuture<'_, Result<RawStructuredResponse, GenerateContentError>> {
        async move {
            // For Groq, we use response_format with json_schema type
            let mut groq_request = conversion::convert_request_to_groq(
                request,
                self.model.clone(),
                self.system_instruction.clone(),
                Some(self.tool_choice.clone()),
            )?;

            // Set response format to JSON schema
            let schema_value: serde_json::Value = serde_json::from_str(&schema)
                .map_err(|e| GroqError::ResponseParsing(format!("Invalid schema: {}", e)))?;

            groq_request.response_format = Some(ResponseFormat::JsonSchema {
                r#type: "json_schema".to_string(),
                json_schema: serde_json::json!({
                    "name": "response",
                    "schema": schema_value
                }),
            });

            let response = self
                .client
                .send(&groq_request)
                .await
                .map_err(GroqError::Api)?;

            // Extract the text content from the first choice
            let text = response
                .choices
                .first()
                .and_then(|choice| choice.message.content.as_deref())
                .ok_or_else(|| GroqError::ResponseParsing("No response content".to_string()))?;

            // Parse the text as JSON
            let json: serde_json::Value = serde_json::from_str(text)
                .map_err(|e| GroqError::ResponseParsing(e.to_string()))?;

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
                vendor_name: "groq".to_string(),
            })
        }
        .boxed()
    }
}
