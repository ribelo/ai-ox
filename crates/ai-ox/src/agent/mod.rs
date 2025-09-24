use std::{collections::HashSet, sync::Arc};

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
    tool::{ApprovalRequest, ToolBox, ToolError, ToolHooks, ToolSet, ToolUse},
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
    /// Pre-approved dangerous tools that won't require individual approval.
    #[builder(default)]
    approved_dangerous_tools: HashSet<String>,
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

    /// Pre-approve specific dangerous tools for this agent.
    ///
    /// These tools will execute without requesting approval through hooks.
    /// Use this for session-based approval where the user has already
    /// granted permission for certain operations.
    pub fn approve_dangerous_tools(&mut self, tool_names: &[&str]) {
        self.approved_dangerous_tools
            .extend(tool_names.iter().map(|s| s.to_string()));
    }

    /// Remove approval for specific dangerous tools.
    ///
    /// These tools will once again require approval through hooks.
    pub fn revoke_dangerous_tools(&mut self, tool_names: &[&str]) {
        for name in tool_names {
            self.approved_dangerous_tools.remove(*name);
        }
    }

    /// Pre-approve ALL dangerous tools for this agent (trust mode).
    ///
    /// This allows the agent to execute any dangerous operation without
    /// requesting approval. Use with caution.
    pub fn approve_all_dangerous_tools(&mut self) {
        self.approved_dangerous_tools.extend(
            self.tools
                .get_all_dangerous_functions()
                .iter()
                .map(|s| s.to_string()),
        );
    }

    /// Clear all pre-approved dangerous tools.
    ///
    /// All dangerous tools will once again require approval through hooks.
    pub fn clear_approved_dangerous_tools(&mut self) {
        self.approved_dangerous_tools.clear();
    }

    /// Get the list of currently approved dangerous tools.
    pub fn get_approved_dangerous_tools(&self) -> &HashSet<String> {
        &self.approved_dangerous_tools
    }

    /// Check if a specific dangerous tool is pre-approved.
    pub fn is_dangerous_tool_approved(&self, tool_name: &str) -> bool {
        self.approved_dangerous_tools.contains(tool_name)
    }

    /// Execute a tool call with dangerous tool approval logic.
    async fn execute_tool_call(
        tools: &ToolSet,
        approved_dangerous_tools: &HashSet<String>,
        call: ToolUse,
        hooks: Option<&ToolHooks>,
    ) -> Result<Part, ToolError> {
        let call_name = &call.name;
        if tools.is_dangerous_function(call_name) {
            if approved_dangerous_tools.contains(call_name) {
                return tools.invoke(call).await;
            }
            if let Some(h) = hooks {
                let req = ApprovalRequest {
                    tool_name: call_name.clone(),
                    args: call.args.clone(),
                };
                if h.request_approval(req).await {
                    return tools.invoke(call).await;
                }
                return Err(ToolError::execution(
                    call_name,
                    std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "User denied execution of dangerous operation",
                    ),
                ));
            }
            return Err(ToolError::execution(
                call_name,
                std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "Dangerous operation requires approval but no hooks provided",
                ),
            ));
        }
        tools.invoke(call).await
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
        self.run_with_hooks(messages, None).await
    }

    /// Executes a conversation with automatic tool handling and optional hooks.
    ///
    /// This method is like `run()` but allows passing ToolHooks for dangerous
    /// operations that need approval or progress reporting.
    pub async fn run_with_hooks(
        &self,
        messages: impl IntoIterator<Item = impl Into<Message>> + Send,
        hooks: Option<ToolHooks>,
    ) -> Result<ModelResponse, AgentError> {
        let mut conversation = self.build_messages(messages)?;
        let mut iteration = 0;

        loop {
            if iteration >= self.max_iterations {
                return Err(AgentError::max_iterations_reached(self.max_iterations));
            }

            // Create a request with the current conversation history.
            let request = self.build_request(conversation.clone());

            // Generate a response from the model.
            let response = self.model.request(request).await?;

            // Add the assistant's response (which may contain tool calls) to the conversation.
            // This is crucial for maintaining the context of the conversation.
            conversation.push(response.message.clone());

            // Check if the response contains any tool calls.
            if let Some(tool_calls) = response.to_tool_calls() {
                if self.tools.get_all_tools().is_empty() {
                    // Model generated tool calls but we have no tools available.
                    // This is an error condition that should not occur.
                    return Err(AgentError::ToolCallsWithoutTools);
                }

                // Execute all tool calls in parallel
                let mut join_set = tokio::task::JoinSet::new();

                // Clone hooks once outside the loop for better performance
                let hooks_clone = hooks.clone();
                let approved_tools = self.approved_dangerous_tools.clone();

                // Start all tool calls concurrently
                for call in tool_calls {
                    let tools = self.tools.clone();
                    let call_clone = call.clone();
                    let hooks_for_task = hooks_clone.clone();
                    let approved_tools_for_task = approved_tools.clone();

                    join_set.spawn(async move {
                        let call_name = call_clone.name.clone();
                        let result = if tools.is_dangerous_function(&call_name) {
                            // Check if pre-approved first
                            if approved_tools_for_task.contains(&call_name) {
                                // Pre-approved dangerous tool - execute without asking
                                tools.invoke(call_clone.clone()).await
                            } else if let Some(hooks) = hooks_for_task {
                                // Not pre-approved - ask for approval via hooks
                                let approval_request = ApprovalRequest {
                                    tool_name: call_name.clone(),
                                    args: call_clone.args.clone(),
                                };

                                if hooks.request_approval(approval_request).await {
                                    // Approved this time - execute the tool
                                    tools.invoke(call_clone.clone()).await
                                } else {
                                    // Denied - return error
                                    Err(crate::tool::ToolError::execution(
                                        &call_name,
                                        std::io::Error::new(
                                            std::io::ErrorKind::PermissionDenied,
                                            "User denied execution of dangerous operation"
                                        )
                                    ))
                                }
                            } else {
                                // Dangerous tool, not pre-approved, no hooks - deny
                                Err(crate::tool::ToolError::execution(
                                    &call_name,
                                    std::io::Error::new(
                                        std::io::ErrorKind::PermissionDenied,
                                        "Dangerous operation requires approval but no hooks provided"
                                    )
                                ))
                            }
                        } else {
                            // Safe function - execute normally
                            tools.invoke(call_clone.clone()).await
                        };
                        (call_clone, result)
                    });
                }

                // Collect all results
                while let Some(join_result) = join_set.join_next().await {
                    let (_call, tool_result) = join_result.map_err(|e| {
                        AgentError::Tool(crate::tool::ToolError::internal("Task join error", e))
                    })?;
                    match tool_result {
                        Ok(result) => {
                            // The tool result is a Part that should be added to the conversation history.
                            conversation.push(crate::content::Message::new(
                                crate::content::MessageRole::Assistant,
                                vec![result],
                            ));
                        }
                        Err(e) => {
                            // If any tool fails, abort the execution.
                            return Err(AgentError::Tool(e));
                        }
                    }
                }
            } else {
                // No tool calls in the response, so the conversation is complete.
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
        self.stream_with_hooks(messages, None)
    }

    /// Streams agent execution events with optional hooks for dangerous operations.
    ///
    /// This method is like `stream()` but allows passing ToolHooks for dangerous
    /// operations that need approval or progress reporting.
    pub fn stream_with_hooks(
        &self,
        messages: impl IntoIterator<Item = impl Into<Message>> + Send,
        hooks: Option<ToolHooks>,
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

                // Create a request with the current conversation history
                let request = self.build_request(conversation.clone());

                // Stream the model response
                let mut model_stream = self.model.request_stream(request);
                let mut accumulator = StreamAccumulator::new();
                let mut response_complete = false;

                // Process the model's streaming response
                while let Some(stream_event_result) = model_stream.next().await {
                    let stream_event = stream_event_result.map_err(AgentError::Api)?;

                    // First, yield the raw event for real-time streaming
                    match &stream_event {
                        StreamEvent::TextDelta(_) => {
                            yield events::AgentEvent::StreamEvent(stream_event.clone());
                        }
                        StreamEvent::StreamStop(_) => {
                            response_complete = true;
                            break;
                        }
                        _ => {
                            // Forward other events as deltas
                            yield events::AgentEvent::StreamEvent(stream_event.clone());
                        }
                    }

                    // Second, accumulate the event for message construction
                    accumulator.accumulate(&stream_event);
                }

                if !response_complete {
                    yield events::AgentEvent::Failed("Model stream ended without completion".to_string());
                    break;
                }

                // Build the assistant's response message
                let _final_usage = accumulator.get_usage();
                let (assistant_message, tool_calls) = accumulator.finalize();
                conversation.push(assistant_message.clone());

                if !tool_calls.is_empty() {
                    if self.tools.get_all_tools().is_empty() {
                        // Model generated tool calls but we have no tools available
                        let _final_response = ModelResponse {
                            message: assistant_message,
                            model_name: self.model.name().to_string(),
                            vendor_name: format!("{}", self.model.info().0),
                            usage: _final_usage.clone(),
                        };
                        yield events::AgentEvent::Completed(_final_response);
                        yield events::AgentEvent::Failed("Model generated tool calls but no tools are available".to_string());
                        break;
                    }

                    // Execute all tool calls in parallel
                    let mut join_set = tokio::task::JoinSet::new();

                    // Clone hooks once outside the loop for better performance
                    let hooks_clone = hooks.clone();
                    let approved_tools = self.approved_dangerous_tools.clone();

                    // Emit tool execution events and start all tool calls concurrently
                    for tool_call in &tool_calls {
                        yield events::AgentEvent::ToolExecution(tool_call.clone());

                        let tools = self.tools.clone();
                        let call_clone = tool_call.clone();
                        let hooks_for_task = hooks_clone.clone();
                        let approved_tools_for_task = approved_tools.clone();
                        join_set.spawn(async move {
                            let call_name = call_clone.name.clone();
                            let result = if tools.is_dangerous_function(&call_name) {
                                // Check if pre-approved first
                                if approved_tools_for_task.contains(&call_name) {
                                    // Pre-approved dangerous tool - execute without asking
                                    tools.invoke(call_clone.clone()).await
                                } else if let Some(hooks) = hooks_for_task {
                                    // Not pre-approved - ask for approval via hooks
                                    let approval_request = ApprovalRequest {
                                        tool_name: call_name.clone(),
                                        args: call_clone.args.clone(),
                                    };

                                    if hooks.request_approval(approval_request).await {
                                        // Approved this time - execute the tool
                                        tools.invoke(call_clone.clone()).await
                                    } else {
                                        // Denied - return error
                                        Err(crate::tool::ToolError::execution(
                                            &call_name,
                                            std::io::Error::new(
                                                std::io::ErrorKind::PermissionDenied,
                                                "User denied execution of dangerous operation"
                                            )
                                        ))
                                    }
                                } else {
                                    // Dangerous tool, not pre-approved, no hooks - deny
                                    Err(crate::tool::ToolError::execution(
                                        &call_name,
                                        std::io::Error::new(
                                            std::io::ErrorKind::PermissionDenied,
                                            "Dangerous operation requires approval but no hooks provided"
                                        )
                                    ))
                                }
                            } else {
                                // Safe function - execute normally
                                tools.invoke(call_clone.clone()).await
                            };
                            result
                        });
                    }

                    // Collect all results as they complete
                    while let Some(join_result) = join_set.join_next().await {
                        let tool_result = match join_result {
                            Ok(result) => result,
                            Err(e) => {
                                yield events::AgentEvent::Failed(format!("Task join error: {e}"));
                                return;
                            }
                        };

                        match tool_result {
                            Ok(tool_result) => {
                                // Extract messages from Part::ToolResult
                                if let crate::content::Part::ToolResult { parts, .. } = &tool_result {
                                    // Try to extract messages from parts
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
                                        // If no text parts found, create a single message with all parts
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
                                    // This shouldn't happen, but handle it gracefully
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

                    // Continue to next iteration for the model to respond to tool results
                    iteration += 1;
                    continue;
                } else {
                    // No tool calls, conversation is complete
                    let _final_response = ModelResponse {
                        message: assistant_message,
                        model_name: self.model.name().to_string(),
                        vendor_name: format!("{}", self.model.info().0),
                        usage: _final_usage,
                    };
                    yield events::AgentEvent::Completed(_final_response);
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
