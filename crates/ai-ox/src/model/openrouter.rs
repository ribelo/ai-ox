use std::collections::HashMap;

use bon::Builder;
use futures_util::{future::BoxFuture, stream::BoxStream, FutureExt};
use openrouter_ox::{
    message::{Message as OpenRouterMessage, Messages as OpenRouterMessages, Role as OpenRouterRole},
    request::Request as OpenRouterRequest,
    response::ChatCompletionResponse,
    tool::{FunctionMetadata, ToolSchema},
    OpenRouter,
};
use serde_json::Value;

use crate::{
    content::{
        delta::MessageStreamEvent,
        message::{Message, MessageRole},
        part::Part,
    },
    errors::GenerateContentError,
    model::{
        request::ModelRequest,
        response::{ModelResponse, RawStructuredResponse},
        Model,
    },
    tool::{Tool, ToolSet},
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
            let chat_request = convert_model_request_to_openrouter_request(request, &self.model, &self.client)?;
            
            let response = chat_request
                .send()
                .await
                .map_err(|e| GenerateContentError::request_failed(e.to_string()))?;

            convert_chat_response_to_model_response(response, &self.model)
        }
        .boxed()
    }

    fn request_stream(
        &self,
        _request: ModelRequest,
    ) -> BoxStream<'_, Result<MessageStreamEvent, GenerateContentError>> {
        // For now, streaming is not implemented for OpenRouter
        // This could be added later by implementing OpenRouter's streaming API
        futures_util::stream::once(async {
            Err(GenerateContentError::unsupported_feature(
                "Streaming is not yet supported for OpenRouter models".to_string(),
            ))
        })
        .boxed()
    }

    fn request_structured_internal(
        &self,
        request: ModelRequest,
        schema: String,
    ) -> BoxFuture<'_, Result<RawStructuredResponse, GenerateContentError>> {
        async move {
            let mut chat_request = convert_model_request_to_openrouter_request(request, &self.model, &self.client)?;
            
            // Add JSON schema formatting instruction
            let schema_value: Value = serde_json::from_str(&schema)
                .map_err(|e| GenerateContentError::request_failed(format!("Invalid schema: {}", e)))?;
                
            chat_request.response_format = Some(schema_value);

            let response = chat_request
                .send()
                .await
                .map_err(|e| GenerateContentError::request_failed(e.to_string()))?;

            convert_chat_response_to_raw_structured_response(response, &self.model)
        }
        .boxed()
    }
}

/// Convert ai-ox ModelRequest to OpenRouter Request
fn convert_model_request_to_openrouter_request(
    request: ModelRequest,
    model: &str,
    client: &OpenRouter,
) -> Result<OpenRouterRequest, GenerateContentError> {
    let mut messages = Vec::new();

    // Add system message if present
    if let Some(system_msg) = request.system_message {
        messages.push(OpenRouterMessage {
            role: OpenRouterRole::System,
            content: convert_message_parts_to_content(&system_msg.content)?,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Convert regular messages
    for message in request.messages {
        messages.push(convert_ai_ox_message_to_openrouter(message)?);
    }

    let tools = request.tools.map(|tool_set| convert_tools_to_openrouter_tools(tool_set));

    Ok(OpenRouterRequest::builder()
        .model(model)
        .messages(OpenRouterMessages(messages))
        .open_router(client.clone())
        .tools(tools.unwrap_or_default())
        .build())
}

/// Convert ai-ox Message to OpenRouter Message
fn convert_ai_ox_message_to_openrouter(
    message: Message,
) -> Result<OpenRouterMessage, GenerateContentError> {
    let role = match message.role {
        MessageRole::User => OpenRouterRole::User,
        MessageRole::Assistant => OpenRouterRole::Assistant,
        MessageRole::System => OpenRouterRole::System,
    };

    let content = convert_message_parts_to_content(&message.content)?;

    Ok(OpenRouterMessage {
        role,
        content,
        name: None,
        tool_calls: None,
        tool_call_id: None,
    })
}

/// Convert message parts to string content (simplified for now)
fn convert_message_parts_to_content(parts: &[Part]) -> Result<String, GenerateContentError> {
    let mut content_parts = Vec::new();
    
    for part in parts {
        match part {
            Part::Text { text } => content_parts.push(text.clone()),
            Part::FunctionCall { .. } => {
                // For now, we'll serialize function calls as JSON
                let serialized = serde_json::to_string(part)
                    .map_err(|e| GenerateContentError::request_failed(e.to_string()))?;
                content_parts.push(serialized);
            }
            Part::FunctionResponse { .. } => {
                // For now, we'll serialize function responses as JSON
                let serialized = serde_json::to_string(part)
                    .map_err(|e| GenerateContentError::request_failed(e.to_string()))?;
                content_parts.push(serialized);
            }
            _ => {
                // For other part types, we'll convert to string representation
                let serialized = serde_json::to_string(part)
                    .map_err(|e| GenerateContentError::request_failed(e.to_string()))?;
                content_parts.push(serialized);
            }
        }
    }
    
    Ok(content_parts.join("\n"))
}

/// Convert ai-ox ToolSet to OpenRouter tools
fn convert_tools_to_openrouter_tools(tool_set: ToolSet) -> Vec<ToolSchema> {
    let mut openrouter_tools = Vec::new();
    
    for tool in tool_set.tools() {
        match tool {
            Tool::FunctionDeclarations(functions) => {
                for func in functions {
                    openrouter_tools.push(ToolSchema {
                        tool_type: "function".to_string(),
                        function: FunctionMetadata {
                            name: func.name,
                            description: func.description,
                            parameters: func.parameters,
                        },
                    });
                }
            }
            Tool::GeminiTool(_) => {
                // Gemini-specific tools aren't directly compatible with OpenRouter
                // This would need custom handling if needed
            }
        }
    }
    
    openrouter_tools
}

/// Convert OpenRouter ChatCompletionResponse to ai-ox ModelResponse
fn convert_chat_response_to_model_response(
    response: ChatCompletionResponse,
    model_name: &str,
) -> Result<ModelResponse, GenerateContentError> {
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
        model_name: model_name.to_string(),
        vendor_name: "openrouter".to_string(),
        usage,
    })
}

/// Convert OpenRouter ChatCompletionResponse to RawStructuredResponse for structured requests
fn convert_chat_response_to_raw_structured_response(
    response: ChatCompletionResponse,
    model_name: &str,
) -> Result<RawStructuredResponse, GenerateContentError> {
    let choice = response.choices.first()
        .ok_or_else(|| GenerateContentError::response_parsing("No choices in response".to_string()))?;

    let content = choice.message.content.clone()
        .unwrap_or_else(|| "{}".to_string());

    let json: Value = serde_json::from_str(&content)
        .map_err(|e| GenerateContentError::response_parsing(format!("Failed to parse JSON: {}", e)))?;

    let usage = Usage::new(); // TODO: Convert from OpenRouter usage if available

    Ok(RawStructuredResponse {
        json,
        model_name: model_name.to_string(),
        vendor_name: "openrouter".to_string(),
        usage,
    })
}