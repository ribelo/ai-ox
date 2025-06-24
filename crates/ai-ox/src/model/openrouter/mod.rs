mod conversion;

use bon::Builder;
use futures_util::{future::BoxFuture, stream::BoxStream, FutureExt};
use openrouter_ox::{
    message::{Message as OpenRouterMessage, Messages as OpenRouterMessages, SystemMessage},
    request::Request as OpenRouterRequest,
    response::{ChatCompletionResponse, ChatCompletionChunk, Delta as OpenRouterDelta, FinishReason as OpenRouterFinishReason},
    tool::ToolSchema,
    OpenRouter,
};
use serde_json::Value;

use crate::{
    content::{
        delta::{FinishReason, StreamEvent, StreamStop},
        message::{Message, MessageRole},
        part::Part,
    },
    errors::GenerateContentError,
    model::{
        request::ModelRequest,
        response::{ModelResponse, RawStructuredResponse},
        Model,
    },
    tool::{Tool, ToolCall, ToolSet},
    usage::Usage,
};

/// OpenRouter model implementation that adapts OpenRouter API to the ai-ox Model trait.
#[derive(Clone, Builder)]
pub struct OpenRouterModel {
    #[builder(into)]
    model: String,
    #[builder(default)]
    client: OpenRouter,
}

impl std::fmt::Debug for OpenRouterModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenRouterModel")
            .field("model", &self.model)
            .field("client", &"[REDACTED]")
            .finish()
    }
}

impl OpenRouterModel {
    /// Create a new OpenRouter model with API key from environment
    pub fn new_from_env(model: impl Into<String>) -> Result<Self, std::env::VarError> {
        let client = OpenRouter::new_from_env()?;
        Ok(Self::builder().model(model).client(client).build())
    }

    /// Convert OpenRouter finish reason to ai-ox finish reason
    fn convert_finish_reason(reason: OpenRouterFinishReason) -> FinishReason {
        match reason {
            OpenRouterFinishReason::Stop => FinishReason::Stop,
            OpenRouterFinishReason::Length | OpenRouterFinishReason::Limit => FinishReason::Length,
            OpenRouterFinishReason::ContentFilter => FinishReason::ContentFilter,
            OpenRouterFinishReason::ToolCalls => FinishReason::ToolCalls,
        }
    }

    /// Convert OpenRouter chunk to ai-ox StreamEvents with 1-to-1 mapping
    fn convert_chunk_to_stream_events(chunk: ChatCompletionChunk) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        // Process each choice in the chunk
        for choice in chunk.choices {
            // Handle text content
            if let Some(content) = choice.delta.content {
                if !content.is_empty() {
                    events.push(StreamEvent::TextDelta(content));
                }
            }

            // Handle tool calls
            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    if let (Some(id), Some(name)) = (tool_call.id, tool_call.function.name) {
                        let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
                            .unwrap_or(serde_json::Value::Object(Default::default()));

                        let ai_tool_call = ToolCall {
                            id,
                            name,
                            args,
                        };
                        events.push(StreamEvent::ToolCall(ai_tool_call));
                    }
                }
            }

            // Handle finish reason and usage
            if let Some(finish_reason) = choice.finish_reason {
                // Add usage event if available
                if let Some(usage_data) = &chunk.usage {
                    let usage = Usage::new()
                        .with_input_tokens(usage_data.prompt_tokens as u32)
                        .with_output_tokens(usage_data.completion_tokens as u32);
                    events.push(StreamEvent::Usage(usage.clone()));

                    // Add stream stop event
                    events.push(StreamEvent::StreamStop(StreamStop {
                        finish_reason: Self::convert_finish_reason(finish_reason),
                        usage,
                    }));
                } else {
                    // Add stream stop event with empty usage
                    events.push(StreamEvent::StreamStop(StreamStop {
                        finish_reason: Self::convert_finish_reason(finish_reason),
                        usage: Usage::new(),
                    }));
                }
            }
        }

        events
    }
}

impl Model for OpenRouterModel {
    fn model(&self) -> &str {
        &self.model
    }

    fn request(
        &self,
        request: ModelRequest,
    ) -> BoxFuture<'_, Result<ModelResponse, GenerateContentError>> {
        async move {
            let mut messages = Vec::new();

            // Add system message if present
            if let Some(system_msg) = request.system_message {
                let system_text = system_msg.content.iter()
                    .filter_map(|part| match part {
                        Part::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                messages.push(OpenRouterMessage::System(SystemMessage::text(system_text)));
            }

            // Convert regular messages using From trait
            for message in request.messages {
                messages.push(message.into());
            }

            // Convert tools using From trait
            let tools = request.tools
                .map(|tool_set| {
                    tool_set.tools()
                        .iter()
                        .flat_map(|tool| -> Vec<ToolSchema> { tool.into() })
                        .collect()
                })
                .unwrap_or_default();

            let chat_request = OpenRouterRequest::builder()
                .model(&self.model)
                .messages(OpenRouterMessages(messages))
                .open_router(self.client.clone())
                .tools(tools)
                .build();
            
            let response = chat_request
                .send()
                .await
                .map_err(|e| GenerateContentError::request_failed(e.to_string()))?;

            // Convert response
            let choice = response.choices.first()
                .ok_or_else(|| GenerateContentError::response_parsing("No choices in response".to_string()))?;

            let content = choice.message.content.clone()
                .unwrap_or_else(|| "".to_string());

            let message = Message {
                role: MessageRole::Assistant,
                content: vec![Part::Text { text: content }],
                timestamp: chrono::Utc::now(),
            };

            let usage = Usage::new(); // TODO: Convert from OpenRouter usage if available

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
        use futures_util::StreamExt;
        use async_stream::try_stream;

        let model_name = self.model.clone();
        let client = self.client.clone();

        let stream = try_stream! {
            let mut messages = Vec::new();

            // Add system message if present
            if let Some(system_msg) = request.system_message {
                let system_text = system_msg.content.iter()
                    .filter_map(|part| match part {
                        Part::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                messages.push(OpenRouterMessage::System(SystemMessage::text(system_text)));
            }

            // Convert regular messages using From trait
            for message in request.messages {
                messages.push(message.into());
            }

            // Convert tools using From trait
            let tools = request.tools
                .map(|tool_set| {
                    tool_set.tools()
                        .iter()
                        .flat_map(|tool| -> Vec<ToolSchema> { tool.into() })
                        .collect()
                })
                .unwrap_or_default();

            let chat_request = OpenRouterRequest::builder()
                .model(&model_name)
                .messages(OpenRouterMessages(messages))
                .open_router(client)
                .tools(tools)
                .build();

            let mut chunk_stream = chat_request.stream(&client);
            
            while let Some(chunk_result) = chunk_stream.next().await {
                let chunk = chunk_result
                    .map_err(|e| GenerateContentError::request_failed(e.to_string()))?;

                // Convert chunk to stream events
                let events = Self::convert_chunk_to_stream_events(chunk);
                
                for event in events {
                    if matches!(&event, StreamEvent::StreamStop(_)) {
                        yield event;
                        return; // End the stream
                    }
                    yield event;
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
            let mut messages = Vec::new();

            // Add system message if present
            if let Some(system_msg) = request.system_message {
                let system_text = system_msg.content.iter()
                    .filter_map(|part| match part {
                        Part::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                messages.push(OpenRouterMessage::System(SystemMessage::text(system_text)));
            }

            // Convert regular messages using From trait
            for message in request.messages {
                messages.push(message.into());
            }

            // Convert tools using From trait
            let tools = request.tools
                .map(|tool_set| {
                    tool_set.tools()
                        .iter()
                        .flat_map(|tool| -> Vec<ToolSchema> { tool.into() })
                        .collect()
                })
                .unwrap_or_default();

            // Add JSON schema formatting instruction
            let schema_value: Value = serde_json::from_str(&schema)
                .map_err(|e| GenerateContentError::request_failed(format!("Invalid schema: {}", e)))?;

            let mut chat_request = OpenRouterRequest::builder()
                .model(&self.model)
                .messages(OpenRouterMessages(messages))
                .open_router(self.client.clone())
                .tools(tools)
                .build();
                
            chat_request.response_format = Some(schema_value);

            let response = chat_request
                .send()
                .await
                .map_err(|e| GenerateContentError::request_failed(e.to_string()))?;

            // Convert response to structured format
            let choice = response.choices.first()
                .ok_or_else(|| GenerateContentError::response_parsing("No choices in response".to_string()))?;

            let content = choice.message.content.clone()
                .unwrap_or_else(|| "{}".to_string());

            let json: Value = serde_json::from_str(&content)
                .map_err(|e| GenerateContentError::response_parsing(format!("Failed to parse JSON: {}", e)))?;

            let usage = Usage::new(); // TODO: Convert from OpenRouter usage if available

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

