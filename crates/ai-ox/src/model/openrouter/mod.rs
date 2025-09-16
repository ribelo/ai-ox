mod conversion;
mod error;

pub use error::OpenRouterError;

use async_stream::try_stream;
use bon::Builder;
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream::BoxStream};
use openrouter_ox::{OpenRouter, request::ChatRequest as OpenRouterRequest};
use serde_json::Value;

// Import the correct OpenAI format types
use ai_ox_common::openai_format::{
    Function as OaiFunction, Tool as OaiTool, ToolChoice as OaiToolChoice,
};

use crate::{
    content::delta::StreamEvent,
    errors::GenerateContentError,
    model::{
        Model, ModelInfo, Provider,
        request::ModelRequest,
        response::{ModelResponse, RawStructuredResponse},
    },
    tool::ToolUse,
};
use std::collections::HashMap;

/// Simple buffer for accumulating partial tool calls from OpenRouter streaming
#[derive(Debug, Default)]
struct OpenRouterStreamProcessor {
    partial_calls: HashMap<String, PartialCall>,
}

#[derive(Debug)]
struct PartialCall {
    id: String,
    name: Option<String>,
    args: String,
}

impl OpenRouterStreamProcessor {
    fn new() -> Self {
        Self::default()
    }

    fn is_valid_json(s: &str) -> bool {
        !s.trim().is_empty() && serde_json::from_str::<Value>(s).is_ok()
    }

    fn process_chunk(
        &mut self,
        chunk: openrouter_ox::response::ChatCompletionChunk,
    ) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        // Process each choice in the chunk
        for choice in chunk.choices {
            // Handle text content
            if let Some(content) = choice.delta.content {
                if !content.is_empty() {
                    events.push(StreamEvent::TextDelta(content));
                }
            }

            // Handle tool calls - just accumulate by ID until complete
            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    if let Some(id) = tool_call.id {
                        // New tool call - create entry
                        self.partial_calls.insert(
                            id.clone(),
                            PartialCall {
                                id: id.clone(),
                                name: tool_call.function.name,
                                args: tool_call.function.arguments,
                            },
                        );
                    } else {
                        // Continue existing tool call - append to the most recent incomplete one
                        // NOTE: This assumes OpenRouter streams one tool at a time (which it does in practice).
                        // If multiple tools streamed simultaneously, we'd need tool call indexing.
                        if let Some(partial) = self
                            .partial_calls
                            .values_mut()
                            .find(|p| p.name.is_some() && !Self::is_valid_json(&p.args))
                        {
                            partial.args.push_str(&tool_call.function.arguments);
                        }
                    }

                    // Check for completed tool calls
                    let mut completed_ids = Vec::new();
                    for (id, partial) in &self.partial_calls {
                        if partial.name.is_some() && Self::is_valid_json(&partial.args) {
                            completed_ids.push(id.clone());
                        }
                    }

                    // Emit completed tool calls
                    for id in completed_ids {
                        if let Some(partial) = self.partial_calls.remove(&id) {
                            let args = match serde_json::from_str(&partial.args) {
                                Ok(json) => json,
                                Err(e) => {
                                    eprintln!(
                                        "Warning: Failed to parse tool call args as JSON: {}. Args: '{}'",
                                        e, partial.args
                                    );
                                    serde_json::Value::Object(Default::default())
                                }
                            };
                            events.push(StreamEvent::ToolCall(ToolUse {
                                id: partial.id,
                                name: partial.name.unwrap(),
                                args,
                                ext: None,
                            }));
                        }
                    }
                }
            }

            // Handle finish reason and usage
            if let Some(finish_reason) = choice.finish_reason {
                // Emit any remaining complete tool calls
                let remaining_complete: Vec<String> = self
                    .partial_calls
                    .iter()
                    .filter(|(_, p)| p.name.is_some() && Self::is_valid_json(&p.args))
                    .map(|(id, _)| id.clone())
                    .collect();

                for id in remaining_complete {
                    if let Some(partial) = self.partial_calls.remove(&id) {
                        let args = match serde_json::from_str(&partial.args) {
                            Ok(json) => json,
                            Err(e) => {
                                eprintln!(
                                    "Warning: Failed to parse final tool call args as JSON: {}. Args: '{}'",
                                    e, partial.args
                                );
                                serde_json::Value::Object(Default::default())
                            }
                        };
                        events.push(StreamEvent::ToolCall(ToolUse {
                            id: partial.id,
                            name: partial.name.unwrap(),
                            args,
                            ext: None,
                        }));
                    }
                }

                // Clear any remaining incomplete calls
                self.partial_calls.clear();

                // Add usage and stop events
                if let Some(usage_data) = &chunk.usage {
                    let usage = conversion::extract_usage_from_response(Some(usage_data));
                    events.push(StreamEvent::Usage(usage.clone()));
                    events.push(StreamEvent::StreamStop(crate::content::delta::StreamStop {
                        finish_reason: conversion::convert_finish_reason(finish_reason),
                        usage,
                    }));
                } else {
                    events.push(StreamEvent::StreamStop(crate::content::delta::StreamStop {
                        finish_reason: conversion::convert_finish_reason(finish_reason),
                        usage: crate::usage::Usage::default(),
                    }));
                }
            }
        }

        events
    }
}

/// OpenRouter model implementation that adapts OpenRouter API to the ai-ox Model trait.
#[derive(Debug, Clone, Builder)]
pub struct OpenRouterModel {
    client: OpenRouter,
    #[builder(into)]
    model: String,
    #[builder(default = default_tool_choice())]
    tool_choice: OaiToolChoice,
}

/// Returns the default tool choice for OpenRouter models.
/// Defaults to Auto, allowing the model to decide when to use tools.
fn default_tool_choice() -> OaiToolChoice {
    OaiToolChoice::Auto
}

impl<S: open_router_model_builder::State> OpenRouterModelBuilder<S>
where
    <S as open_router_model_builder::State>::Client: open_router_model_builder::IsUnset,
{
    pub fn api_key(
        self,
        api_key: impl Into<String>,
    ) -> OpenRouterModelBuilder<open_router_model_builder::SetClient<S>> {
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
        tool_choice: &OaiToolChoice,
        response_format: Option<Value>,
    ) -> Result<OpenRouterRequest, OpenRouterError> {
        // Convert messages using the conversion module
        let messages = conversion::build_openrouter_messages(&request, model)?;

        // Convert tools using the conversion module
        let tools = conversion::convert_tools_to_openrouter(request.tools)?;

        // Build request based on whether we have tools or not
        let mut request = if !tools.is_empty() {
            let openrouter_tools: Vec<OaiTool> = tools
                .into_iter()
                .map(|func| OaiTool {
                    r#type: "function".to_string(),
                    function: OaiFunction {
                        name: func.name,
                        description: func.description,
                        parameters: Some(func.parameters),
                    },
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
            let chat_request =
                Self::build_openrouter_request(request, &self.model, &self.tool_choice, None)?;

            let response = self
                .client
                .send(&chat_request)
                .await
                .map_err(OpenRouterError::Api)?;

            // Convert response using the conversion module to properly handle tool calls
            let choice = response.choices.first().ok_or_else(|| {
                OpenRouterError::ResponseParsing("No choices in response".to_string())
            })?;

            // Convert the OpenRouter message to ai-ox Message using the From trait
            let openrouter_message =
                openrouter_ox::message::Message::Assistant(choice.message.clone());
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
            let mut processor = OpenRouterStreamProcessor::new();

            while let Some(chunk_result) = chunk_stream.next().await {
                let chunk = chunk_result.map_err(OpenRouterError::Api)?;

                // Process chunk with stateful processor to handle tool calls properly
                let events = processor.process_chunk(chunk);

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
            let chat_request = Self::build_openrouter_request(
                request,
                &self.model,
                &self.tool_choice,
                Some(response_format),
            )?;

            let response = self
                .client
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
                    e, content
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
