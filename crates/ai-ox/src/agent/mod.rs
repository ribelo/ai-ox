use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::{
    content::{
        message::{Message, MessageRole},
        part::Part,
    },
    errors::GenerateContentError,
    model::{
        Model,
        request::ModelRequest,
        response::{ModelResponse, StructuredResponse},
    },
    tool::{ToolBox, ToolSet},
};

pub mod error;
pub mod events;

use bon::Builder;
use error::AgentError;

/// Configuration for the agent's behavior.
#[derive(Debug, Clone, Builder)]
pub struct Agent {
    /// The AI model to use for generation.
    #[builder(start_fn)]
    model: Arc<dyn Model>,
    /// A toolbox for executing tool calls from the model.
    #[builder(field)]
    tools: ToolSet,
    /// An optional system instruction to guide the model's behavior.
    #[builder(into)]
    system_instruction: Option<String>,
    /// Maximum number of iterations for tool execution loops.
    #[builder(default = 12)]
    max_iterations: u32,
}

impl<S: agent_builder::State> AgentBuilder<S> {
    /// Adds a toolbox to the agent's `ToolSet`.
    ///
    /// This is a convenience method that simplifies adding toolboxes
    /// during the build process.
    pub fn toolbox(mut self, toolbox: impl ToolBox) -> Self {
        self.tools.add_toolbox(toolbox);
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
        let mut request = ModelRequest {
            messages: conversation,
            system_message: None,
            tools: None,
        };

        if let Some(ref system_instruction) = self.system_instruction {
            request.system_message = Some(Message::new(
                MessageRole::User,
                vec![Part::Text {
                    text: system_instruction.clone(),
                }],
            ));
        }

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

            // Create a request with the current conversation history.
            let mut request = ModelRequest {
                messages: conversation.clone(),
                system_message: None,
                tools: None,
            };

            if let Some(ref system_instruction) = self.system_instruction {
                request.system_message = Some(Message::new(
                    MessageRole::User,
                    vec![Part::Text {
                        text: system_instruction.clone(),
                    }],
                ));
            }
            // Inform the model about the available tools for this turn.
            let available_tools = self.tools.get_all_tools();
            if !available_tools.is_empty() {
                request.tools = Some(available_tools);
            }

            // Generate a response from the model.
            let response = self.model.request(request).await?;

            // Add the assistant's response (which may contain tool calls) to the conversation.
            // This is crucial for maintaining the context of the conversation.
            conversation.push(response.message.clone());

            // Check if the response contains any tool calls.
            if let Some(tool_calls) = response.to_tool_calls() {
                if self.tools.get_all_tools().is_empty() {
                    // The model is hallucinating tool calls, but we have no tools.
                    // Return the current response as the final answer.
                    return Ok(response);
                }

                // Execute each tool call and collect the results.
                for call in tool_calls {
                    match self.tools.invoke(call).await {
                        Ok(result) => {
                            // The ToolResult contains one or more messages (e.g., the tool output)
                            // that should be added to the conversation history.
                            conversation.extend(result.response);
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
        let mut request = ModelRequest {
            messages: conversation.clone(),
            system_message: None,
            tools: None,
        };

        if let Some(ref system_instruction) = self.system_instruction {
            request.system_message = Some(Message::new(
                MessageRole::User,
                vec![Part::Text {
                    text: system_instruction.clone(),
                }],
            ));
        }

        // For structured requests, we use the schema of the target type.
        let schema = crate::tool::schema_for_type::<O>();

        match self
            .model
            .request_structured_internal(request.clone(), schema.to_string())
            .await
        {
            Ok(raw_structured_content) => {
                let data: O = serde_json::from_value(raw_structured_content.json)
                    .map_err(|e| AgentError::response_parsing_failed(e, "structured response"))?;
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

    // Helper methods

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
    O: DeserializeOwned,
{
    let text = response.to_string().ok_or(AgentError::NoResponse)?;
    serde_json::from_str(&text).map_err(|e| AgentError::response_parsing_failed(e, &text))
}
