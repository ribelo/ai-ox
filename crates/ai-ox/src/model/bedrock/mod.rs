mod error;
mod conversion;

pub use error::BedrockError;

use crate::{
    content::{delta::StreamEvent, part::Part},
    errors::GenerateContentError,
    model::{request::ModelRequest, response::ModelResponse, Model, ModelInfo, Provider},
    tool::ToolUse,
};
use async_stream::try_stream;
use aws_config::{meta::region::RegionProviderChain, BehaviorVersion};
use aws_sdk_bedrockruntime::{
    Client as BedrockRuntimeClient,
    types::ConverseStreamOutput,
};
use bon::Builder;
use futures_util::{future::BoxFuture, FutureExt, StreamExt};
use serde_json;

#[derive(Debug, Clone, Builder)]
pub struct BedrockModel {
    client: BedrockRuntimeClient,
    #[builder(into)]
    model_id: String,
}

impl BedrockModel {
    /// Create a new BedrockModel from environment AWS configuration.
    ///
    /// This function uses aws_config::load_from_env().await to get AWS credentials and region.
    pub async fn new(model_id: String) -> Result<Self, BedrockError> {
        let region_provider = RegionProviderChain::default_provider().or_else("us-east-1");
        let config = aws_config::defaults(BehaviorVersion::latest())
            .region(region_provider)
            .load()
            .await;

        let client = BedrockRuntimeClient::new(&config);
        Ok(BedrockModel { client, model_id })
    }

    // Note: builder() method is provided by the bon::Builder derive macro

    /// Create a BedrockModel with a custom client.
    pub fn with_client(client: BedrockRuntimeClient, model_id: impl Into<String>) -> Self {
        Self {
            client,
            model_id: model_id.into(),
        }
    }
}

/// Private trait to abstract over request builder types
trait BedrockRequestBuilder {
    fn set_messages(self, messages: Option<Vec<aws_sdk_bedrockruntime::types::Message>>) -> Self;
    fn system(self, system: aws_sdk_bedrockruntime::types::SystemContentBlock) -> Self;
    fn tool_config(self, config: aws_sdk_bedrockruntime::types::ToolConfiguration) -> Self;
}

impl BedrockRequestBuilder for aws_sdk_bedrockruntime::operation::converse::builders::ConverseFluentBuilder {
    fn set_messages(self, messages: Option<Vec<aws_sdk_bedrockruntime::types::Message>>) -> Self {
        self.set_messages(messages)
    }

    fn system(self, system: aws_sdk_bedrockruntime::types::SystemContentBlock) -> Self {
        self.system(system)
    }

    fn tool_config(self, config: aws_sdk_bedrockruntime::types::ToolConfiguration) -> Self {
        self.tool_config(config)
    }
}

impl BedrockRequestBuilder for aws_sdk_bedrockruntime::operation::converse_stream::builders::ConverseStreamFluentBuilder {
    fn set_messages(self, messages: Option<Vec<aws_sdk_bedrockruntime::types::Message>>) -> Self {
        self.set_messages(messages)
    }

    fn system(self, system: aws_sdk_bedrockruntime::types::SystemContentBlock) -> Self {
        self.system(system)
    }

    fn tool_config(self, config: aws_sdk_bedrockruntime::types::ToolConfiguration) -> Self {
        self.tool_config(config)
    }
}

impl BedrockModel {
    fn build_request<B: BedrockRequestBuilder>(
        mut builder: B,
        request: ModelRequest,
    ) -> Result<B, BedrockError> {
        // Convert ai-ox messages to Bedrock format
        let bedrock_messages = conversion::convert_ai_ox_messages_to_bedrock(request.messages)?;
        builder = builder.set_messages(Some(bedrock_messages));

        // Add system message if present
        if let Some(system_message) = request.system_message {
            if !system_message.content.is_empty() {
                let system_text = system_message.content
                    .into_iter()
                    .filter_map(|part| match part {
                        Part::Text { text, .. } => Some(text),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                if !system_text.is_empty() {
                    builder = builder.system(
                        aws_sdk_bedrockruntime::types::SystemContentBlock::Text(system_text)
                    );
                }
            }
        }

        // Add tools if present
        if let Some(tools) = request.tools {
            let tool_specs = conversion::convert_ai_ox_tools_to_bedrock(tools)?;

            if !tool_specs.is_empty() {
                let tool_config = aws_sdk_bedrockruntime::types::ToolConfiguration::builder()
                    .set_tools(Some(tool_specs))
                    .build()
                    .map_err(|e| BedrockError::RequestBuilder(
                        format!("Failed to build tool configuration: {}", e)
                    ))?;

                builder = builder.tool_config(tool_config);
            }
        }

        Ok(builder)
    }

}

impl Model for BedrockModel {
    fn info(&self) -> ModelInfo<'_> {
        ModelInfo(Provider::Bedrock, &self.model_id)
    }

    fn name(&self) -> &str {
        &self.model_id
    }

    fn request(&self, request: ModelRequest) -> BoxFuture<'_, Result<ModelResponse, GenerateContentError>> {
        async move {
            // Build the ConverseRequest from the ModelRequest
            let converse_request = self.client
                .converse()
                .model_id(&self.model_id);

            // Build the complete request using helper function
            let converse_request = Self::build_request(converse_request, request)?;

            // Send the request and handle SDK errors
            let response = converse_request.send().await.map_err(|sdk_error| {
                let error_message = format!("AWS SDK error: {}", sdk_error);
                GenerateContentError::provider_error("bedrock", error_message)
            })?;

            // Extract and convert the response
            let output = response.output.ok_or_else(|| GenerateContentError::response_parsing("Bedrock response missing output"))?;
            let usage = response.usage.ok_or_else(|| GenerateContentError::response_parsing("Bedrock response missing usage info"))?;

            conversion::convert_bedrock_response_to_ai_ox(output, self.model_id.clone(), usage)
                .map_err(GenerateContentError::from)
        }.boxed()
    }

    fn request_stream(&self, request: ModelRequest) -> futures_util::stream::BoxStream<'_, Result<StreamEvent, GenerateContentError>> {
        let client = self.client.clone();
        let model_id = self.model_id.clone();

        let stream = try_stream! {
            // Build the initial request using the existing helper
            let converse_request = client
                .converse_stream()
                .model_id(&model_id);

            // Build the complete request using helper function
            let converse_request = Self::build_request(converse_request, request)?;

            // Send the streaming request
            let response = converse_request.send().await.map_err(|sdk_error| {
                let error_message = format!("AWS SDK streaming error: {}", sdk_error);
                GenerateContentError::provider_error("bedrock", error_message)
            })?;

            let mut response_stream = response.stream;

            // State management for tool call reassembly
            let mut current_tool_name: Option<String> = None;
            let mut current_tool_id: Option<String> = None;
            let mut current_tool_input = String::new();

            // Process the event stream
            loop {
                match response_stream.recv().await {
                    Ok(Some(event)) => {
                        match event {
                    ConverseStreamOutput::ContentBlockStart(start) => {
                        if let Some(content_block_start) = &start.start {
                            if let aws_sdk_bedrockruntime::types::ContentBlockStart::ToolUse(tool_use) = content_block_start {
                                // Capture tool information at the start of a tool call
                                current_tool_name = Some(tool_use.name().to_string());
                                current_tool_id = Some(tool_use.tool_use_id().to_string());
                                current_tool_input.clear();
                            }
                        }
                    },
                    ConverseStreamOutput::ContentBlockDelta(delta) => {
                        if let Some(delta_content) = delta.delta {
                            match delta_content {
                                aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(text_delta) => {
                                    // Yield text deltas immediately
                                    yield StreamEvent::TextDelta(text_delta);
                                },
                                aws_sdk_bedrockruntime::types::ContentBlockDelta::ToolUse(tool_delta) => {
                                    // Accumulate tool input (don't yield yet)
                                    current_tool_input.push_str(tool_delta.input());
                                },
                                _ => {
                                    // Handle other delta types if needed
                                }
                            }
                        }
                    },
                    ConverseStreamOutput::ContentBlockStop(_) => {
                        // If we have a complete tool call, finalize and yield it
                        if let (Some(name), Some(id)) = (current_tool_name.take(), current_tool_id.take()) {
                            // Parse the accumulated tool input as JSON
                            let args = serde_json::from_str(&current_tool_input)
                                .map_err(|e| GenerateContentError::response_parsing(
                                    format!("Failed to parse tool arguments: {}", e)
                                ))?;

                            // Yield the complete tool call
                            yield StreamEvent::ToolCall(ToolUse::new(id, name, args));

                            // Reset state for next tool call
                            current_tool_input.clear();
                        }
                    },
                    ConverseStreamOutput::MessageStop(stop) => {
                        let finish_reason = conversion::convert_bedrock_finish_reason(stop.stop_reason().clone());

                        // CORRECTLY check for usage here.
                        let mut usage = crate::usage::Usage::default();
                        if let Some(additional_fields) = stop.additional_model_response_fields {
                            // Convert AWS Document to serde_json::Value using our helper function
                            let fields_json = crate::model::bedrock::conversion::document_to_json(&additional_fields);
                            if let Some(usage_val) = fields_json.get("usage") {
                                // Manually extract usage fields since TokenUsage doesn't implement Deserialize
                                if let Some(input_tokens) = usage_val.get("inputTokens").and_then(|v| v.as_i64()) {
                                    usage.input_tokens_by_modality.insert(crate::usage::Modality::Text, input_tokens as u64);
                                }
                                if let Some(output_tokens) = usage_val.get("outputTokens").and_then(|v| v.as_i64()) {
                                    usage.output_tokens_by_modality.insert(crate::usage::Modality::Text, output_tokens as u64);
                                }
                                usage.requests = 1;
                            }
                        }

                        yield StreamEvent::StreamStop(crate::content::delta::StreamStop {
                            finish_reason,
                            usage,
                        });
                    },
                    ConverseStreamOutput::Metadata(metadata) => {
                        // Handle metadata events with usage information
                        if let Some(usage_info) = metadata.usage() {
                            let usage = conversion::convert_token_usage_to_ai_ox(usage_info.clone());
                            yield StreamEvent::Usage(usage);
                        }
                    },
                            _ => {
                                // Handle any other event types
                            }
                        }
                    },
                    Ok(None) => {
                        // Stream ended
                        break;
                    },
                    Err(e) => {
                        // This correctly handles errors *from* the stream itself.
                        let error_message = format!("AWS SDK stream event error: {}", e);
                        Err(GenerateContentError::provider_error("bedrock", error_message))?;
                    }
                }
            }
        };

        stream.boxed()
    }

    fn request_structured_internal(
        &self,
        _request: ModelRequest,
        _schema: String,
    ) -> BoxFuture<'_, Result<crate::model::response::RawStructuredResponse, GenerateContentError>> {
        async move {
            Err(GenerateContentError::UnsupportedFeature(
                "BedrockModel does not currently support structured generation.".to_string(),
            ))
        }.boxed()
    }
}
