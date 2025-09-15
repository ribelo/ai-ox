use mistral_ox::{
    message::{AssistantMessage as MistralAssistantMessage, AudioContent as MistralAudioContent, ContentPart as MistralContentPart, 
             Message as MistralMessage, SystemMessage as MistralSystemMessage, TextContent as MistralTextContent, 
             ToolMessage as MistralToolMessage, UserMessage as MistralUserMessage},
    request::ChatRequest,
    response::{ChatResponse, ChatCompletionChunk},
    tool::{Tool as MistralTool, ToolCall as MistralToolCall, ToolChoice as MistralToolChoice, ToolFunction},
};

use crate::{
    ModelResponse,
    content::{delta::{StreamEvent, StreamStop, FinishReason}, message::{Message, MessageRole}, part::{Part, DataRef}},
    errors::GenerateContentError,
    model::ModelRequest,
    tool::{ToolUse, encode_tool_result_parts, decode_tool_result_parts},
    usage::Usage,
};
use std::collections::BTreeMap;

use super::MistralError;

/// Convert from ai-ox ModelRequest to Mistral ChatRequest
pub fn convert_request_to_mistral(
    request: ModelRequest,
    model: String,
    system_instruction: Option<String>,
    tool_choice: Option<mistral_ox::tool::ToolChoice>,
) -> Result<ChatRequest, GenerateContentError> {
    let mut mistral_messages = Vec::new();
    
    // Add system instruction if provided
    if let Some(system_msg) = system_instruction {
        mistral_messages.push(MistralMessage::System(MistralSystemMessage::text(system_msg)));
    }
    
    // Convert messages
    for message in request.messages {
        let mistral_msgs = convert_message_to_mistral(message)?;
        mistral_messages.extend(mistral_msgs);
    }

    // Convert tools if present
    if let Some(tools) = request.tools {
        let common_tools = convert_tools_to_mistral(tools)?;
        Ok(ChatRequest::builder()
            .model(model)
            .messages(mistral_messages)
            .tools(common_tools)
            .tool_choice(ai_ox_common::openai_format::ToolChoice::Auto)
            .build())
    } else {
        Ok(ChatRequest::builder()
            .model(model)
            .messages(mistral_messages)
            .build())
    }
}

/// Convert from ai-ox Message to Mistral Message(s)
/// Returns Vec<MistralMessage> to handle cases where one ai-ox message
/// needs to expand into multiple Mistral messages (e.g., multiple tool results)
fn convert_message_to_mistral(message: Message) -> Result<Vec<MistralMessage>, GenerateContentError> {
    match message.role {
        MessageRole::User => {
            let mut messages = Vec::new();
            let mut text_parts = Vec::new();
            
            for part in message.content {
                match part {
                    Part::Text { text, .. } => {
                        text_parts.push(MistralContentPart::Text(MistralTextContent::new(text)));
                    }
                      Part::Blob { data_ref, mime_type, .. } => {
                          if mime_type.starts_with("audio/") {
                              match data_ref {
                                  DataRef::Uri { uri } => {
                                      text_parts.push(MistralContentPart::Audio(MistralAudioContent::new(uri)));
                                  }
                                  DataRef::Base64 { .. } => {
                                      return Err(GenerateContentError::message_conversion(
                                          format!("Mistral does not support base64-encoded audio. Upload the audio and provide a URI instead. (MIME type: {})", mime_type)
                                      ));
                                  }
                              }
                          } else {
                              return Err(GenerateContentError::message_conversion(
                                  format!("Mistral does not support {} content in messages. Only audio content is supported.", mime_type)
                              ));
                          }
                      }
                     Part::ToolUse { .. } => {
                         return Err(GenerateContentError::message_conversion(
                             "Tool calls should not appear in user messages",
                         ));
                     }
                     Part::ToolResult { id, name, parts, .. } => {
                         // If we have accumulated text, flush it as a user message first
                         if !text_parts.is_empty() {
                             messages.push(MistralMessage::User(MistralUserMessage::new(
                                 std::mem::take(&mut text_parts)
                             )));
                         }

                         // Each tool result becomes a separate tool message
                         let content_str = encode_tool_result_parts(&name, &parts)?;
                         messages.push(MistralMessage::Tool(MistralToolMessage::new(id, content_str)));
                     }
                     Part::Opaque { provider, kind, .. } => {
                         return Err(GenerateContentError::message_conversion(
                             format!("Cannot convert opaque content from provider '{}' of type '{}' to Mistral format", provider, kind)
                         ));
                     }
                 }
            }
            
            // If we still have text parts, add them as a final user message
            if !text_parts.is_empty() {
                messages.push(MistralMessage::User(MistralUserMessage::new(text_parts)));
            }
            
            // If we didn't create any messages, return an empty user message
            if messages.is_empty() {
                messages.push(MistralMessage::User(MistralUserMessage::new(Vec::<MistralContentPart>::new())));
            }
            
            Ok(messages)
        }
        MessageRole::Assistant => {
            let mut text_content = String::new();
            let mut tool_calls = Vec::new();

            for part in message.content {
                match part {
                    Part::Text { text, .. } => {
                        text_content.push_str(&text);
                    }
                     Part::ToolUse { id, name, args, .. } => {
                          let args_str = serde_json::to_string(&args)
                              .map_err(|e| GenerateContentError::message_conversion(e.to_string()))?;
                          tool_calls.push(MistralToolCall {
                              id,
                              r#type: "function".to_string(),
                              function: mistral_ox::tool::FunctionCall {
                                  name,
                                  arguments: args_str,
                              },
                              index: Some(tool_calls.len() as u32),
                          });
                      }
                        Part::Blob { mime_type, .. } => {
                            return Err(GenerateContentError::message_conversion(
                                format!("Mistral does not support {} content in assistant messages", mime_type)
                            ));
                        }
                      Part::ToolResult { .. } => {
                          return Err(GenerateContentError::message_conversion(
                              "ToolResult should not appear in assistant messages - use a separate message".to_string()
                          ));
                      }
                      Part::Opaque { provider, kind, .. } => {
                          return Err(GenerateContentError::message_conversion(
                              format!("Cannot convert opaque content from provider '{}' of type '{}' to Mistral format", provider, kind)
                          ));
                      }
                }
            }

            let mut assistant_msg = MistralAssistantMessage::text(text_content);
            if !tool_calls.is_empty() {
                assistant_msg.tool_calls = Some(tool_calls);
            }

            Ok(vec![MistralMessage::Assistant(assistant_msg)])
        }
        MessageRole::System => {
            return Err(GenerateContentError::message_conversion("System messages not supported in Mistral"));
        }
        MessageRole::Unknown(role) => {
            return Err(GenerateContentError::message_conversion(&format!("Unknown role: {}", role)));
        }
    }
}

/// Convert from ai-ox Tools to Mistral Tools
fn convert_tools_to_mistral(tools: Vec<crate::tool::Tool>) -> Result<Vec<ai_ox_common::openai_format::Tool>, GenerateContentError> {
    let mut common_tools = Vec::new();

    for tool in tools {
        match tool {
            crate::tool::Tool::FunctionDeclarations(functions) => {
                for function in functions {
                    // Defensive fix: if the schema object is {} (tool with no params)
                    // Mistral rejects it unless you omit the field or supply null
                    let parameters = if function.parameters == serde_json::json!({}) {
                        None
                    } else {
                        Some(function.parameters)
                    };

                    common_tools.push(ai_ox_common::openai_format::Tool {
                        r#type: "function".to_string(),
                        function: ai_ox_common::openai_format::Function {
                            name: function.name,
                            description: function.description,
                            parameters,
                        },
                    });
                }
            }
            #[cfg(feature = "gemini")]
            crate::tool::Tool::GeminiTool(_) => {
                // Skip Gemini tools for Mistral
            }
        }
    }

    Ok(common_tools)
}

/// Convert from Mistral ChatResponse to ai-ox ModelResponse
pub fn convert_mistral_response_to_ai_ox(
    response: ChatResponse,
    model_name: String,
) -> Result<ModelResponse, GenerateContentError> {
    let choice = response.choices.first()
        .ok_or_else(|| MistralError::ResponseParsing("No choices in response".to_string()))?;
    
    let mut parts = Vec::new();
    
    // Convert content
    if let Some(content) = &choice.message.content {
        parts.push(Part::Text {
            text: content.clone(),
            ext: BTreeMap::new(),
        });
    }
    
    // Convert tool calls
    if let Some(tool_calls) = &choice.message.tool_calls {
        for tc in tool_calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Null);
            
            parts.push(Part::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                args,
                ext: BTreeMap::new(),
            });
        }
    }
    
    let message = Message {
        role: MessageRole::Assistant,
        content: parts,
        timestamp: Some(chrono::Utc::now()),
        ext: Some(BTreeMap::new()),
    };
    
    let usage = response.usage.map(|u| {
        let mut usage = Usage::new();
        usage.requests = 1;
        usage.input_tokens_by_modality.insert(crate::usage::Modality::Text, u.prompt_tokens as u64);
        usage.output_tokens_by_modality.insert(crate::usage::Modality::Text, u.completion_tokens as u64);
        usage
    });
    
    Ok(ModelResponse {
        message,
        usage: usage.unwrap_or_else(Usage::new),
        model_name,
        vendor_name: "mistral".to_string(),
    })
}

/// Convert streaming response to stream events
pub fn convert_response_to_stream_events(
    chunk: ChatCompletionChunk,
) -> Vec<Result<StreamEvent, GenerateContentError>> {
    let mut events = Vec::new();
    
    if let Some(choice) = chunk.choices.first() {
        let delta = &choice.delta;
        
        // Handle content delta
        if let Some(content) = &delta.content {
            if !content.is_empty() {
                events.push(Ok(StreamEvent::TextDelta(content.clone())));
            }
        }
        
        // Handle tool call deltas
        if let Some(tool_calls) = &delta.tool_calls {
            for tc_delta in tool_calls {
                if let (Some(id), Some(name), Some(args)) = 
                    (&tc_delta.id, &tc_delta.function.as_ref().and_then(|f| f.name.as_ref()), 
                     &tc_delta.function.as_ref().and_then(|f| f.arguments.as_ref())) {
                    
                    if let Ok(args_value) = serde_json::from_str::<serde_json::Value>(args) {
                        events.push(Ok(StreamEvent::ToolCall(ToolUse {
                            id: id.clone(),
                            name: name.to_string(),
                            args: args_value,
                            ext: Some(BTreeMap::new()),
                        })));
                    }
                }
            }
        }
        
        // Handle finish reason
        if let Some(finish_reason) = &choice.finish_reason {
            if finish_reason == "stop" {
                let usage = chunk.usage.map(|u| {
                    let mut usage = Usage::new();
                    usage.requests = 1;
                    usage.input_tokens_by_modality.insert(crate::usage::Modality::Text, u.prompt_tokens as u64);
                    usage.output_tokens_by_modality.insert(crate::usage::Modality::Text, u.completion_tokens as u64);
                    usage
                }).unwrap_or_else(Usage::new);
                
                events.push(Ok(StreamEvent::StreamStop(StreamStop {
                    usage,
                    finish_reason: FinishReason::Stop,
                })));
            }
        }
    }
    
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;

    #[test]
    fn test_multiple_tool_results_expand_to_multiple_messages() {
        // Create an ai-ox message with multiple tool results
        let ai_msg = Message {
            role: MessageRole::User,
            content: vec![
                Part::Text { text: "Here are the results:".into(), ext: BTreeMap::new() },
                Part::ToolResult {
                    id: "tool_call_1".into(),
                    name: "get_weather".into(),
                    parts: vec![Part::text(serde_json::to_string(&json!({"temperature": 72, "condition": "sunny"})).unwrap())],
                    ext: BTreeMap::new(),
                },
                Part::ToolResult {
                    id: "tool_call_2".into(),
                    name: "get_news".into(),
                    parts: vec![Part::text(serde_json::to_string(&json!({"headlines": ["News 1", "News 2"]})).unwrap())],
                    ext: BTreeMap::new(),
                },
            ],
            timestamp: Some(Utc::now()),
            ext: None,
        };

        // Convert to Mistral messages
        let result = convert_message_to_mistral(ai_msg).unwrap();
        
        // Should create 3 messages: 1 user message with text + 2 tool messages
        assert_eq!(result.len(), 3);
        
        // First message should be user message with text
        match &result[0] {
            MistralMessage::User(user_msg) => {
                assert_eq!(user_msg.content.len(), 1);
                if let Some(MistralContentPart::Text(text)) = user_msg.content.first() {
                    assert_eq!(text.text, "Here are the results:");
                } else {
                    panic!("Expected text content in first message");
                }
            }
            _ => panic!("Expected user message first"),
        }
        
        // Second message should be tool message for first tool result
        match &result[1] {
            MistralMessage::Tool(tool_msg) => {
                assert_eq!(tool_msg.tool_call_id, "tool_call_1");
                let (name, parts) = decode_tool_result_parts(&tool_msg.content).unwrap();
                assert_eq!(name, "get_weather");
                assert_eq!(parts.len(), 1);
                if let Part::Text { text, .. } = &parts[0] {
                    let content: serde_json::Value = serde_json::from_str(text).unwrap();
                    assert_eq!(content["temperature"], 72);
                    assert_eq!(content["condition"], "sunny");
                } else {
                    panic!("Expected text part in tool result");
                }
            }
            _ => panic!("Expected tool message second"),
        }

        // Third message should be tool message for second tool result
        match &result[2] {
            MistralMessage::Tool(tool_msg) => {
                assert_eq!(tool_msg.tool_call_id, "tool_call_2");
                let (name, parts) = decode_tool_result_parts(&tool_msg.content).unwrap();
                assert_eq!(name, "get_news");
                assert_eq!(parts.len(), 1);
                if let Part::Text { text, .. } = &parts[0] {
                    let content: serde_json::Value = serde_json::from_str(text).unwrap();
                    assert!(content["headlines"].is_array());
                } else {
                    panic!("Expected text part in tool result");
                }
            }
            _ => panic!("Expected tool message third"),
        }
    }

    #[test]
    fn test_tool_results_only_no_text() {
        // Create an ai-ox message with only tool results (no text)
        let ai_msg = Message {
            role: MessageRole::User,
            content: vec![
                Part::ToolResult {
                    id: "call_1".into(),
                    name: "func1".into(),
                    parts: vec![Part::text(serde_json::to_string(&json!({"result": "A"})).unwrap())],
                    ext: BTreeMap::new(),
                },
                Part::ToolResult {
                    id: "call_2".into(),
                    name: "func2".into(),
                    parts: vec![Part::text(serde_json::to_string(&json!({"result": "B"})).unwrap())],
                    ext: BTreeMap::new(),
                },
            ],
            timestamp: Some(Utc::now()),
            ext: None,
        };

        let result = convert_message_to_mistral(ai_msg).unwrap();
        
        // Should create 2 tool messages only
        assert_eq!(result.len(), 2);
        
        // Both should be tool messages
        assert!(matches!(&result[0], MistralMessage::Tool(_)));
        assert!(matches!(&result[1], MistralMessage::Tool(_)));
    }

    #[test]
    fn test_interleaved_text_and_tool_results() {
        // Create an ai-ox message with interleaved text and tool results
        let ai_msg = Message {
            role: MessageRole::User,
            content: vec![
                Part::Text { text: "First text".into(), ext: BTreeMap::new() },
                Part::ToolResult {
                    id: "call_1".into(),
                    name: "func1".into(),
                    parts: vec![Part::text(serde_json::to_string(&json!({"data": 1})).unwrap())],
                    ext: BTreeMap::new(),
                },
                Part::Text { text: "Second text".into(), ext: BTreeMap::new() },
                Part::ToolResult {
                    id: "call_2".into(),
                    name: "func2".into(),
                    parts: vec![Part::text(serde_json::to_string(&json!({"data": 2})).unwrap())],
                    ext: BTreeMap::new(),
                },
                Part::Text { text: "Third text".into(), ext: BTreeMap::new() },
            ],
            timestamp: Some(Utc::now()),
            ext: None,
        };

        let result = convert_message_to_mistral(ai_msg).unwrap();
        
        // Should create 5 messages alternating between user and tool
        assert_eq!(result.len(), 5);
        
        // Check the pattern: User, Tool, User, Tool, User
        assert!(matches!(&result[0], MistralMessage::User(_)));
        assert!(matches!(&result[1], MistralMessage::Tool(_)));
        assert!(matches!(&result[2], MistralMessage::User(_)));
        assert!(matches!(&result[3], MistralMessage::Tool(_)));
        assert!(matches!(&result[4], MistralMessage::User(_)));
    }

    #[test]
    fn test_assistant_message_with_tool_calls_returns_single_message() {
        // Assistant messages should still return a single message
        let ai_msg = Message {
            role: MessageRole::Assistant,
            content: vec![
                Part::Text { text: "I'll help you with that.".into(), ext: BTreeMap::new() },
                Part::ToolUse {
                    id: "call_123".into(),
                    name: "get_weather".into(),
                    args: json!({"location": "NYC"}),
                    ext: BTreeMap::new(),
                },
            ],
            timestamp: Some(Utc::now()),
            ext: None,
        };

        let result = convert_message_to_mistral(ai_msg).unwrap();
        
        // Should create only 1 assistant message
        assert_eq!(result.len(), 1);
        
        match &result[0] {
            MistralMessage::Assistant(assist_msg) => {
                // Check the text content
                assert_eq!(assist_msg.content.len(), 1);
                if let Some(MistralContentPart::Text(text)) = assist_msg.content.first() {
                    assert_eq!(text.text, "I'll help you with that.");
                } else {
                    panic!("Expected text content in assistant message");
                }
                
                // Check tool calls
                assert!(assist_msg.tool_calls.is_some());
                let tool_calls = assist_msg.tool_calls.as_ref().unwrap();
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, "call_123");
            }
            _ => panic!("Expected assistant message"),
        }
    }

    #[test]
    fn test_user_message_with_audio_content() {
        // Create an ai-ox message with audio content
        let ai_msg = Message {
            role: MessageRole::User,
            content: vec![
                Part::Text { text: "What is being said in this audio?".into(), ext: BTreeMap::new() },
                 Part::Blob {
                     data_ref: DataRef::Uri {
                         uri: "https://example.com/audio.mp3".into()
                     },
                     mime_type: "audio/mp3".into(),
                     name: None,
                     description: None,
                     ext: BTreeMap::new(),
                 },
             ],
             timestamp: Some(Utc::now()),
             ext: None,
         };

        // Convert to Mistral messages
        let result = convert_message_to_mistral(ai_msg).unwrap();
        
        // Should create 1 user message with both text and audio
        assert_eq!(result.len(), 1);
        
        match &result[0] {
            MistralMessage::User(user_msg) => {
                assert_eq!(user_msg.content.len(), 2);
                
                // Check text content
                if let Some(MistralContentPart::Text(text)) = user_msg.content.get(0) {
                    assert_eq!(text.text, "What is being said in this audio?");
                } else {
                    panic!("Expected text content first");
                }
                
                // Check audio content
                if let Some(MistralContentPart::Audio(audio)) = user_msg.content.get(1) {
                    assert_eq!(audio.audio_url, "https://example.com/audio.mp3");
                } else {
                    panic!("Expected audio content second");
                }
            }
            _ => panic!("Expected user message"),
        }
    }

    #[test]
    fn test_user_message_with_unsupported_image_returns_error() {
        // Create an ai-ox message with image content (unsupported by Mistral)
        let ai_msg = Message {
            role: MessageRole::User,
            content: vec![
                Part::Text { text: "What do you see in this image?".into(), ext: BTreeMap::new() },
                Part::Blob {
                    data_ref: DataRef::Uri {
                        uri: "https://example.com/image.jpg".into()
                    },
                    mime_type: "image/jpeg".into(),
                    name: None,
                    description: None,
                    ext: BTreeMap::new(),
                },
            ],
            timestamp: Some(Utc::now()),
            ext: None,
        };

        // Convert should fail with proper error
        let result = convert_message_to_mistral(ai_msg);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Mistral does not currently support images"));
    }

    #[test]
    fn test_user_message_with_base64_audio_returns_error() {
        // Create an ai-ox message with base64 audio content (unsupported by Mistral)
        let ai_msg = Message {
            role: MessageRole::User,
            content: vec![
                Part::Text { text: "What is being said?".into(), ext: BTreeMap::new() },
                Part::Blob {
                    data_ref: DataRef::Base64 {
                        data: "base64data".into(),
                    },
                    mime_type: "audio/mp3".into(),
                    name: None,
                    description: None,
                    ext: BTreeMap::new(),
                },
            ],
            timestamp: Some(Utc::now()),
            ext: None,
        };

        // Convert should fail with proper error
        let result = convert_message_to_mistral(ai_msg);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Mistral does not support base64-encoded audio"));
        assert!(error.to_string().contains("audio/mp3"));
    }

    #[test]
    fn test_assistant_message_with_unsupported_image_returns_error() {
        // Create an ai-ox assistant message with image content (unsupported)
        let ai_msg = Message {
            role: MessageRole::Assistant,
            content: vec![
                Part::Text { text: "I see an image".into(), ext: BTreeMap::new() },
                Part::Blob {
                    data_ref: DataRef::Uri {
                        uri: "https://example.com/image.jpg".into()
                    },
                    mime_type: "image/jpeg".into(),
                    name: None,
                    description: None,
                    ext: BTreeMap::new(),
                },
            ],
            timestamp: Some(Utc::now()),
            ext: Some(BTreeMap::new()),
        };

        // Convert should fail with proper error
        let result = convert_message_to_mistral(ai_msg);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Mistral does not support images in assistant messages"));
    }
}