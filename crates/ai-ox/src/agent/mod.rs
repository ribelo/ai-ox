use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::{
    content::{
        Part,
        delta::StreamEvent,
        message::{Message, MessageRole},
    },
    errors::GenerateContentError,
    model::{
        Model,
        request::ModelRequest,
        response::{ModelResponse, StructuredResponse},
    },
    tool::{ToolBox, ToolError, ToolSet, ToolUse},
    usage::Usage,
};

pub mod error;
pub mod events;

use bon::Builder;
use error::AgentError;

/// Configuration for the agent's behavior.
#[derive(Debug, Clone, Builder)]
pub struct Agent {
    /// A toolbox for executing tool calls from the model.
    #[builder(field)]
    tools: ToolSet,
    /// The AI model to use for generation.
    model: Arc<dyn Model>,
    /// An optional system instruction to guide the model's behavior.
    #[builder(into)]
    system_instruction: Option<String>,
    /// Maximum number of iterations for tool execution loops.
    #[builder(default = 12)]
    max_iterations: u32,
}

impl Agent {
    /// Creates a new agent builder with the specified model.
    pub fn model<M: Model + 'static>(model: M) -> AgentBuilder<agent_builder::SetModel> {
        Agent::builder().model(Arc::new(model))
    }
}

impl<S: agent_builder::State> AgentBuilder<S> {
    /// Adds a toolbox to the agent's `ToolSet`.
    ///
    /// This is a convenience method that simplifies adding toolboxes
    /// during the build process.
    pub fn tools(mut self, tools: impl ToolBox) -> Self {
        self.tools.add_toolbox(tools);
        self
    }
}

impl Agent {
    /// Returns the system instruction if set.
    pub fn system_instruction(&self) -> Option<&str> {
        self.system_instruction.as_deref()
    }

    /// Returns the maximum number of iterations.
    pub fn max_iterations(&self) -> u32 {
        self.max_iterations
    }

    /// Sets the system instruction.
    pub fn set_system_instruction(&mut self, instruction: impl Into<String>) {
        self.system_instruction = Some(instruction.into());
    }

    /// Clears the system instruction.
    pub fn clear_system_instruction(&mut self) {
        self.system_instruction = None;
    }

    /// Generates a response without tool execution.
    ///
    /// This method sends the messages to the model and returns the response
    /// without handling any tool calls that might be included.
    pub async fn generate(
        &self,
        messages: impl IntoIterator<Item = impl Into<Message>> + Send,
    ) -> Result<ModelResponse, AgentError> {
        let conversation = self.build_messages(messages)?;
        let request = self.build_request(conversation);

        self.model.request(request).await.map_err(AgentError::Api)
    }

    /// Executes a conversation with automatic tool handling.
    ///
    /// This method manages a multi-turn conversation, automatically executing
    /// any tool calls requested by the model until either:
    /// - The model provides a response without tool calls
    /// - The maximum number of iterations is reached
    pub async fn run(
        &self,
        messages: impl IntoIterator<Item = impl Into<Message>> + Send,
    ) -> Result<ModelResponse, AgentError> {
        let mut conversation = self.build_messages(messages)?;
        let mut iteration = 0;

        loop {
            if iteration >= self.max_iterations {
                return Err(AgentError::max_iterations_reached(self.max_iterations));
            }

            let request = self.build_request(conversation.clone());
            let response = self.model.request(request).await?;

            conversation.push(response.message.clone());

            if let Some(tool_calls) = response.to_tool_calls() {
                if self.tools.get_all_tools().is_empty() {
                    return Err(AgentError::ToolCallsWithoutTools);
                }

                let mut join_set = tokio::task::JoinSet::new();

                for call in tool_calls {
                    let tools = self.tools.clone();
                    let call_clone = call.clone();

                    join_set.spawn(async move {
                        let result = tools.invoke(call_clone.clone()).await;
                        (call_clone, result)
                    });
                }

                while let Some(join_result) = join_set.join_next().await {
                    let (_call, tool_result) = join_result
                        .map_err(|e| AgentError::Tool(ToolError::internal("Task join error", e)))?;

                    match tool_result {
                        Ok(result) => {
                            conversation.push(Message::new(MessageRole::Assistant, vec![result]));
                        }
                        Err(e) => {
                            return Err(AgentError::Tool(e));
                        }
                    }
                }
            } else {
                return Ok(response);
            }

            iteration += 1;
        }
    }

    /// Generates a structured response of type `O`.
    ///
    /// This method constrains the model to return a JSON response that conforms
    /// to the schema of type `O`, then deserializes it and returns it along
    /// with response metadata.
    pub async fn generate_typed<O>(
        &self,
        messages: impl IntoIterator<Item = impl Into<Message>> + Send,
    ) -> Result<StructuredResponse<O>, AgentError>
    where
        O: DeserializeOwned + JsonSchema + Send,
    {
        let conversation = self.build_messages(messages)?;
        let request = self.build_request(conversation.clone());

        // For structured requests, we use the schema of the target type.
        let schema = crate::tool::schema_for_type::<O>();

        let schema_string = schema.to_string();
        match self
            .model
            .request_structured_internal(request.clone(), schema_string.clone())
            .await
        {
            Ok(raw_structured_content) => {
                let response_text = raw_structured_content.json.to_string();
                let data: O = serde_json::from_value(raw_structured_content.json).map_err(|e| {
                    AgentError::response_parsing_failed(e, response_text, schema_string.clone())
                })?;
                Ok(StructuredResponse {
                    data,
                    model_name: raw_structured_content.model_name,
                    usage: raw_structured_content.usage,
                    vendor_name: raw_structured_content.vendor_name,
                })
            }
            Err(GenerateContentError::UnsupportedFeature(_)) => {
                // Fallback to regular generation and manual parsing if the model doesn't support structured output.
                let response = self.model.request(request).await?;
                let data = parse_response_as_typed(&response)?;
                Ok(StructuredResponse {
                    data,
                    model_name: response.model_name,
                    usage: response.usage,
                    vendor_name: response.vendor_name,
                })
            }
            Err(e) => Err(AgentError::Api(e)),
        }
    }

    /// Executes a conversation with tool handling and returns a structured response.
    ///
    /// This method combines the multi-turn tool execution of `run()` with
    /// the structured output parsing of `generate_typed()`.
    pub async fn execute_typed<O>(
        &self,
        messages: impl IntoIterator<Item = impl Into<Message>> + Send,
    ) -> Result<StructuredResponse<O>, AgentError>
    where
        O: DeserializeOwned + JsonSchema + Send,
    {
        let messages_vec: Vec<Message> = messages.into_iter().map(|m| m.into()).collect();

        if self.tools.get_all_tools().is_empty() {
            // No tools, so just run a direct structured generation.
            return self.generate_typed(messages_vec).await;
        }

        // Execute the conversation with tools to get the final text response.
        let final_response = self.run(messages_vec).await?;

        // Parse the final response from the model as the structured type.
        let data = parse_response_as_typed(&final_response)?;
        Ok(StructuredResponse {
            data,
            model_name: final_response.model_name,
            usage: final_response.usage,
            vendor_name: final_response.vendor_name,
        })
    }

    /// Streams agent execution events for real-time processing.
    ///
    /// This method provides a stream of `AgentEvent`s that implements the full
    /// multi-turn conversation loop with tool execution, streaming each step
    /// of the agentic process in real-time.

    pub fn stream(
        &self,
        messages: impl IntoIterator<Item = impl Into<Message>> + Send,
    ) -> futures_util::stream::BoxStream<'_, Result<events::AgentEvent, AgentError>> {
        use async_stream::try_stream;
        use futures_util::StreamExt;

        let conversation = match self.build_messages(messages) {
            Ok(msgs) => msgs,
            Err(e) => return Box::pin(futures_util::stream::once(async move { Err(e) })),
        };

        let stream = try_stream! {
            yield events::AgentEvent::Started;

            let mut conversation = conversation;
            let mut iteration = 0;

            loop {
                if iteration >= self.max_iterations {
                    yield events::AgentEvent::Failed(format!("Agent reached maximum iterations ({})", self.max_iterations));
                    break;
                }

                let request = self.build_request(conversation.clone());
                let mut model_stream = self.model.request_stream(request);
                let mut accumulator = StreamAccumulator::new();
                let mut response_complete = false;

                while let Some(stream_event_result) = model_stream.next().await {
                    let stream_event = stream_event_result.map_err(AgentError::Api)?;

                    match &stream_event {
                        StreamEvent::TextDelta(_) => {
                            yield events::AgentEvent::StreamEvent(stream_event.clone());
                        }
                        StreamEvent::StreamStop(_) => {
                            response_complete = true;
                            break;
                        }
                        _ => {
                            yield events::AgentEvent::StreamEvent(stream_event.clone());
                        }
                    }

                    accumulator.accumulate(&stream_event);
                }

                if !response_complete {
                    yield events::AgentEvent::Failed("Model stream ended without completion".to_string());
                    break;
                }

                let final_usage = accumulator.get_usage();
                let (assistant_message, tool_calls) = accumulator.finalize();
                conversation.push(assistant_message.clone());

                if !tool_calls.is_empty() {
                    if self.tools.get_all_tools().is_empty() {
                        yield events::AgentEvent::Completed(ModelResponse {
                            message: assistant_message,
                            model_name: self.model.name().to_string(),
                            vendor_name: self.model.info().to_string(),
                            usage: final_usage.clone(),
                        });
                        yield events::AgentEvent::Failed("Model generated tool calls but no tools are available".to_string());
                        break;
                    }

                    let mut join_set = tokio::task::JoinSet::new();

                    for tool_call in &tool_calls {
                        yield events::AgentEvent::ToolExecution(tool_call.clone());

                        let tools = self.tools.clone();
                        let call_clone = tool_call.clone();

                        join_set.spawn(async move {
                            tools.invoke(call_clone).await
                        });
                    }

                    while let Some(join_result) = join_set.join_next().await {
                        let tool_result = match join_result {
                            Ok(result) => result,
                            Err(e) => {
                                yield events::AgentEvent::Failed(format!("Task join error: {e}"));
                                return;
                            }
                        };

                        match tool_result {
                            Ok(tool_part) => {
                                if let crate::content::Part::ToolResult { parts, .. } = &tool_part {
                                    let messages: Vec<crate::content::Message> = parts.iter()
                                        .filter_map(|part| {
                                            if let crate::content::Part::Text { .. } = part {
                                                Some(crate::content::Message::new(
                                                    crate::content::MessageRole::Assistant,
                                                    vec![part.clone()]
                                                ))
                                            } else {
                                                None
                                            }
                                        })
                                        .collect();

                                    if messages.is_empty() {
                                        let messages = vec![crate::content::Message::new(
                                            crate::content::MessageRole::Assistant,
                                            parts.clone()
                                        )];
                                        yield events::AgentEvent::ToolResult(messages.clone());
                                        conversation.extend(messages);
                                    } else {
                                        yield events::AgentEvent::ToolResult(messages.clone());
                                        conversation.extend(messages);
                                    }
                                } else {
                                    yield events::AgentEvent::Failed("Invalid tool result format".to_string());
                                    return;
                                }
                            }
                            Err(tool_error) => {
                                yield events::AgentEvent::Failed(format!("Tool execution failed: {tool_error}"));
                                return;
                            }
                        }
                    }

                    iteration += 1;
                    continue;
                } else {
                    yield events::AgentEvent::Completed(ModelResponse {
                        message: assistant_message,
                        model_name: self.model.name().to_string(),
                        vendor_name: self.model.info().to_string(),
                        usage: final_usage,
                    });
                    break;
                }
            }
        };

        Box::pin(stream)
    }
}

/// Helper struct for accumulating streaming events into a final Message.
struct StreamAccumulator {
    text: String,
    tool_calls: Vec<ToolUse>,
    usage: Option<Usage>,
}

impl StreamAccumulator {
    fn new() -> Self {
        Self {
            text: String::new(),
            tool_calls: Vec::new(),
            usage: None,
        }
    }

    fn accumulate(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::TextDelta(text) => {
                self.text.push_str(text);
            }
            StreamEvent::ToolCall(tool_call) => {
                self.tool_calls.push(tool_call.clone());
            }
            StreamEvent::Usage(usage) => {
                self.usage = Some(usage.clone());
            }
            _ => {
                // Other events don't affect message construction
            }
        }
    }

    fn get_usage(&self) -> Usage {
        self.usage.clone().unwrap_or_default()
    }

    fn finalize(self) -> (Message, Vec<ToolUse>) {
        let mut content = vec![];
        if !self.text.is_empty() {
            content.push(Part::Text {
                text: self.text,
                ext: std::collections::BTreeMap::new(),
            });
        }

        content.extend(
            self.tool_calls
                .iter()
                .cloned()
                .map(|tool_use| Part::ToolUse {
                    id: tool_use.id,
                    name: tool_use.name,
                    args: tool_use.args,
                    ext: tool_use.ext.unwrap_or_default(),
                }),
        );

        let message = Message::new(MessageRole::Assistant, content);
        (message, self.tool_calls)
    }
}

impl Agent {
    // Helper methods

    fn build_request(&self, messages: Vec<Message>) -> ModelRequest {
        let mut request = ModelRequest {
            messages,
            system_message: None,
            tools: None,
        };

        if let Some(ref system_instruction) = self.system_instruction {
            request.system_message = Some(Message::new(
                MessageRole::System,
                vec![Part::Text {
                    text: system_instruction.clone(),
                    ext: std::collections::BTreeMap::new(),
                }],
            ));
        }

        let available_tools = self.tools.get_all_tools();
        if !available_tools.is_empty() {
            request.tools = Some(available_tools);
        }

        request
    }

    fn build_messages(
        &self,
        messages: impl IntoIterator<Item = impl Into<Message>>,
    ) -> Result<Vec<Message>, AgentError> {
        Ok(messages.into_iter().map(Into::into).collect())
    }
}

// Helper functions

/// Parses the text from a model's response into a structured type `O`.
fn parse_response_as_typed<O>(response: &ModelResponse) -> Result<O, AgentError>
where
    O: DeserializeOwned + JsonSchema,
{
    let text = response.to_string().ok_or(AgentError::NoResponse)?;
    let schema = crate::tool::schema_for_type::<O>();
    serde_json::from_str(&text)
        .map_err(|e| AgentError::response_parsing_failed(e, &text, schema.to_string()))
}
