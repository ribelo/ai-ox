mod conversion;
mod error;

pub use error::OpenRouterError;

use async_stream::try_stream;
use bon::Builder;
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream::BoxStream};
use openrouter_ox::{OpenRouter, request::Request as OpenRouterRequest};
use serde_json::Value;

use crate::{
    content::delta::StreamEvent,
    errors::GenerateContentError,
    model::{
        Model, ModelInfo, Provider,
        request::ModelRequest,
        response::{ModelResponse, RawStructuredResponse},
    },
};

/// OpenRouter model implementation that adapts OpenRouter API to the ai-ox Model trait.
#[derive(Debug, Clone, Builder)]
pub struct OpenRouterModel {
    client: OpenRouter,
    #[builder(into)]
    model: String,
    #[builder(default = default_tool_choice())]
    tool_choice: openrouter_ox::tool::ToolChoice,
}

/// Returns the default tool choice for OpenRouter models.
/// Defaults to Auto, allowing the model to decide when to use tools.
fn default_tool_choice() -> openrouter_ox::tool::ToolChoice {
    openrouter_ox::tool::ToolChoice::Auto
}

impl<S: open_router_model_builder::State> OpenRouterModelBuilder<S> 
where 
    <S as open_router_model_builder::State>::Client: open_router_model_builder::IsUnset
{
    pub fn api_key(self, api_key: impl Into<String>) -> OpenRouterModelBuilder<open_router_model_builder::SetClient<S>> {
        self.client(OpenRouter::new(api_key))
    }
}

impl OpenRouterModel {
    /// Create a new OpenRouterModel from environment variables.
    ///
    /// This function reads the OPENROUTER_API_KEY from the environment and returns an error if missing.
    pub async fn new(model: impl Into<String>) -> Result<Self, OpenRouterError> {
        let model_name = model.into();
        let client = OpenRouter::load_from_env()?;

        Ok(Self {
            model: model_name,
            client,
            tool_choice: default_tool_choice(),
        })
    }

    /// Helper function to build OpenRouter requests
    fn build_openrouter_request(
        request: ModelRequest,
        model: &str,
        tool_choice: &openrouter_ox::tool::ToolChoice,
        response_format: Option<Value>,
    ) -> Result<OpenRouterRequest, OpenRouterError> {
        // Convert messages using the conversion module
        let messages = conversion::build_openrouter_messages(&request)?;

        // Convert tools using the conversion module
        let tools = conversion::convert_tools_to_openrouter(request.tools)?;

        // Build request based on whether we have tools or not
        let mut request = if !tools.is_empty() {
            let openrouter_tools: Vec<openrouter_ox::tool::Tool> = tools
                .into_iter()
                .map(|func| openrouter_ox::tool::Tool {
                    tool_type: "function".to_string(),
                    function: func,
                })
                .collect();
            
            OpenRouterRequest::builder()
                .model(model)
                .messages(messages)
                .tools(openrouter_tools)
                .tool_choice(tool_choice.clone())
                .build()
        } else {
            OpenRouterRequest::builder()
                .model(model)
                .messages(messages)
                .build()
        };

        // Set response_format if provided
        if let Some(format) = response_format {
            request.response_format = Some(format);
        }

        Ok(request)
    }
}

impl Model for OpenRouterModel {
    fn info(&self) -> ModelInfo<'_> {
        ModelInfo(Provider::OpenRouter, &self.model)
    }

    fn name(&self) -> &str {
        &self.model
    }

    fn request(
        &self,
        request: ModelRequest,
    ) -> BoxFuture<'_, Result<ModelResponse, GenerateContentError>> {
        async move {
            // Build the request using the helper function
            let chat_request = Self::build_openrouter_request(request, &self.model, &self.tool_choice, None)?;

            let response = self.client
                .send(&chat_request)
                .await
                .map_err(OpenRouterError::Api)?;

            // Convert response using the conversion module to properly handle tool calls
            let choice = response.choices.first().ok_or_else(|| {
                OpenRouterError::ResponseParsing("No choices in response".to_string())
            })?;

            // Convert the OpenRouter message to ai-ox Message using the From trait
            let openrouter_message = openrouter_ox::message::Message::Assistant(choice.message.clone());
            let message = openrouter_message.into();

            // Extract usage data using conversion module
            let usage = conversion::extract_usage_from_response(Some(&response.usage));

            Ok(ModelResponse {
                message,
                model_name: self.model.clone(),
                vendor_name: "openrouter".to_string(),
                usage,
            })
        }
        .boxed()
    }

    fn request_stream(
        &self,
        request: ModelRequest,
    ) -> BoxStream<'_, Result<StreamEvent, GenerateContentError>> {
        let client = self.client.clone();
        let model_name = self.model.clone();
        let tool_choice = self.tool_choice.clone();

        let stream = try_stream! {
            // Build the request using the helper function
            let chat_request = Self::build_openrouter_request(request, &model_name, &tool_choice, None)?;

            let mut chunk_stream = client.stream(&chat_request);

            while let Some(chunk_result) = chunk_stream.next().await {
                let chunk = chunk_result.map_err(OpenRouterError::Api)?;

                // Convert chunk to stream events - yield each event individually
                let events = conversion::convert_chunk_to_stream_events(chunk);

                for event in events {
                    // Check for stream end
                    let is_stream_stop = matches!(&event, StreamEvent::StreamStop(_));
                    yield event;

                    // End the stream after yielding StreamStop
                    if is_stream_stop {
                        return;
                    }
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
            // Parse and validate the schema
            let schema_value: Value = serde_json::from_str(&schema)
                .map_err(|e| OpenRouterError::InvalidSchema(e.to_string()))?;

            // Format the schema in the expected OpenRouter format
            let response_format = serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "Response",
                    "schema": schema_value
                }
            });

            // Build the request using the helper function
            let chat_request =
                Self::build_openrouter_request(request, &self.model, &self.tool_choice, Some(response_format))?;

            let response = self.client
                .send(&chat_request)
                .await
                .map_err(OpenRouterError::Api)?;

            // Convert response to structured format
            let choice = response.choices.first().ok_or_else(|| {
                OpenRouterError::ResponseParsing("No choices in response".to_string())
            })?;

            let content = choice
                .message
                .content
                .0
                .first()
                .and_then(|part| part.as_text())
                .map(|text| text.text.clone())
                .unwrap_or_else(|| "{}".to_string());


            let json: Value = serde_json::from_str(&content).map_err(|e| {
                OpenRouterError::ResponseParsing(format!(
                    "Failed to parse JSON: {}. Response content: '{}'", 
                    e, 
                    content
                ))
            })?;

            // Extract usage data using conversion module
            let usage = conversion::extract_usage_from_response(Some(&response.usage));

            Ok(RawStructuredResponse {
                json,
                model_name: self.model.clone(),
                vendor_name: "openrouter".to_string(),
                usage,
            })
        }
        .boxed()
    }
}
