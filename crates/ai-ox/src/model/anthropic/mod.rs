mod conversion;
mod error;

pub use error::AnthropicError;

use crate::{
    content::delta::StreamEvent,
    errors::GenerateContentError,
    model::{Model, ModelInfo, ModelRequest, Provider, response::RawStructuredResponse},
    usage::Usage,
    ModelResponse,
};
use anthropic_ox::{
    message::Content,
    tool::{CustomTool, Tool, ToolChoice},
    Anthropic,
};
use async_stream::try_stream;
use bon::Builder;
use futures_util::{future::BoxFuture, FutureExt, StreamExt};

/// Default maximum tokens for Anthropic models
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Represents a model from the Anthropic family (Claude).
#[derive(Debug, Clone, Builder)]
pub struct AnthropicModel {
    /// Anthropic client
    #[builder(field)]
    client: Anthropic,
    /// The specific model name (e.g., "claude-3-5-sonnet-20241022").
    #[builder(into)]
    model: String,
    /// System instruction if provided
    #[builder(into)]
    system_instruction: Option<String>,
    /// Maximum tokens for response
    #[builder(default = DEFAULT_MAX_TOKENS)]
    max_tokens: u32,
}

impl<S: anthropic_model_builder::State> AnthropicModelBuilder<S> {
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.client = Anthropic::new(api_key);
        self
    }

    pub fn oauth_token(mut self, oauth_token: impl Into<String>) -> Self {
        self.client = anthropic_ox::Anthropic::builder()
            .oauth_token(oauth_token)
            .build();
        self
    }
}

impl AnthropicModel {
    /// Create a new AnthropicModel from environment variables.
    ///
    /// This function reads the ANTHROPIC_API_KEY from the environment and returns an error if missing.
    pub async fn new(model: impl Into<String>) -> Result<Self, AnthropicError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| AnthropicError::MissingApiKey)?;

        let client = Anthropic::new(&api_key);

        Ok(Self {
            client,
            model: model.into(),
            system_instruction: None,
            max_tokens: 4096,
        })
    }
}

use futures_util::stream::BoxStream;

impl Model for AnthropicModel {
    fn info(&self) -> ModelInfo<'_> {
        ModelInfo(Provider::Anthropic, &self.model)
    }

    fn name(&self) -> &str {
        &self.model
    }

    /// Sends a request to the Anthropic API and returns the response.
    fn request(
        &self,
        request: ModelRequest,
    ) -> BoxFuture<'_, Result<ModelResponse, GenerateContentError>> {
        async move {
            let anthropic_request = conversion::convert_request_to_anthropic(
                request,
                self.model.clone(),
                self.system_instruction.clone(),
                self.max_tokens,
                None, // No tools for standard request
            )?;
            let response = self.client
                .send(&anthropic_request)
                .await
                .map_err(|e| AnthropicError::Api(e))
                .map_err(|e| {
                    GenerateContentError::provider_error(
                        "anthropic",
                        format!("Failed to send request for model {}: {}", self.model, e),
                    )
                })?;
            conversion::convert_anthropic_response_to_ai_ox(response, self.model.clone())
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
            let anthropic_request = conversion::convert_request_to_anthropic(
                request,
                self.model.clone(),
                self.system_instruction.clone(),
                self.max_tokens,
                None, // No tools for streaming request
            )?;
            let mut response_stream = client.stream(&anthropic_request);

            while let Some(response) = response_stream.next().await {
                let response = response
                    .map_err(|e| AnthropicError::Api(e))
                    .map_err(|e| GenerateContentError::provider_error(
                        "anthropic",
                        format!("Stream error for model {}: {}", self.model, e)
                    ))?;
                let events = conversion::convert_stream_event_to_ai_ox(response);
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
        const TOOL_NAME: &str = "json_data";

        async move {
            let schema_json: serde_json::Value =
                serde_json::from_str(&schema).map_err(|e| AnthropicError::InvalidSchema(e.to_string()))?;

            let tool = Tool::Custom(CustomTool {
                object_type: "custom".to_string(),
                name: TOOL_NAME.to_string(),
                description: "Function call with a JSON schema for structured data extraction."
                    .to_string(),
                input_schema: schema_json,
            });

            let tool_choice = ToolChoice::Tool {
                name: TOOL_NAME.to_string(),
            };

            let anthropic_request = conversion::convert_request_to_anthropic(
                request,
                self.model.clone(),
                self.system_instruction.clone(),
                self.max_tokens,
                Some((vec![tool], Some(tool_choice))),
            )?;

            let response = self.client
                .send(&anthropic_request)
                .await
                .map_err(|e| AnthropicError::Api(e))?;

            let tool_use = response.content.iter().find_map(|c| match c {
                Content::ToolUse(tool_use) => Some(tool_use),
                _ => None,
            }).ok_or_else(|| {
                AnthropicError::ResponseParsing("No tool use content found in response".to_string())
            })?;

            Ok(RawStructuredResponse {
                json: tool_use.input.clone(),
                usage: response.usage.into(),
                model_name: self.model.clone(),
                vendor_name: "anthropic".to_string(),
            })
        }
        .boxed()
    }
}