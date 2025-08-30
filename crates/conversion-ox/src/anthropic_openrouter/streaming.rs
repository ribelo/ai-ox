//! Streaming conversion utilities for OpenRouter ↔ Anthropic
//!
//! OpenRouter sends tool calls fragmented across dozens of JSON chunks,
//! so we need stateful conversion to reassemble them correctly.

use anthropic_ox::response::{StreamEvent as AnthropicStreamEvent, StreamMessage, ContentBlockDelta};
use anthropic_ox::message::ContentBlock;
use openrouter_ox::response::ChatCompletionChunk;

/// Stateful converter for OpenRouter chunks to Anthropic stream events
/// 
/// This is required because OpenRouter fragments tool calls and content
/// across multiple chunks that need to be reassembled.
pub struct AnthropicOpenRouterStreamConverter {
    message_id: Option<String>,
    content_index: usize,
    is_first_chunk: bool,
    accumulated_content: String,
}

impl AnthropicOpenRouterStreamConverter {
    /// Create a new stream converter
    pub fn new() -> Self {
        Self {
            message_id: None,
            content_index: 0,
            is_first_chunk: true,
            accumulated_content: String::new(),
        }
    }

    /// Convert an OpenRouter ChatCompletionChunk to Anthropic StreamEvent
    /// 
    /// Returns a Vec because some chunks may generate multiple events
    /// (e.g., MessageStart + ContentBlockStart + ContentBlockDelta)
    /// 
    /// This is a complex, stateful operation because OpenRouter fragments
    /// tool calls and content across multiple chunks that need reassembly.
    pub fn convert_chunk(&mut self, chunk: ChatCompletionChunk) -> Vec<AnthropicStreamEvent> {
        let mut events = Vec::new();
        
        // Set message ID if not set
        if self.message_id.is_none() {
            self.message_id = Some(chunk.id.clone());
        }

        if let Some(choice) = chunk.choices.first() {
            // Handle first chunk - send MessageStart event
            if self.is_first_chunk {
                self.is_first_chunk = false;
                
                let stream_message = StreamMessage {
                    id: chunk.id.clone(),
                    r#type: "message".to_string(),
                    role: anthropic_ox::message::Role::Assistant,
                    content: Vec::new(),
                    model: chunk.model.clone(),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: anthropic_ox::response::Usage {
                        input_tokens: None,
                        output_tokens: None,
                    },
                };
                
                events.push(AnthropicStreamEvent::MessageStart {
                    message: stream_message,
                });

                // Send ContentBlockStart if there's content
                if choice.delta.content.is_some() {
                    events.push(AnthropicStreamEvent::ContentBlockStart {
                        index: self.content_index,
                        content_block: ContentBlock::Text { 
                            text: String::new() 
                        },
                    });
                }
            }

            // Handle content delta
            if let Some(content) = &choice.delta.content {
                if !content.is_empty() {
                    self.accumulated_content.push_str(content);
                    
                    events.push(AnthropicStreamEvent::ContentBlockDelta {
                        index: self.content_index,
                        delta: ContentBlockDelta::TextDelta {
                            text: content.clone(),
                        },
                    });
                }
            }

            // Handle finish reason (final chunk)
            if let Some(finish_reason) = &choice.finish_reason {
                // Send ContentBlockStop
                events.push(AnthropicStreamEvent::ContentBlockStop {
                    index: self.content_index,
                });

                // Send MessageDelta with stop reason
                events.push(AnthropicStreamEvent::MessageDelta {
                    delta: anthropic_ox::response::MessageDelta {
                        stop_reason: Some(match finish_reason {
                            openrouter_ox::response::FinishReason::Stop => "end_turn".to_string(),
                            openrouter_ox::response::FinishReason::Length => "max_tokens".to_string(),
                            openrouter_ox::response::FinishReason::Limit => "max_tokens".to_string(),
                            openrouter_ox::response::FinishReason::ContentFilter => "end_turn".to_string(),
                            openrouter_ox::response::FinishReason::ToolCalls => "tool_use".to_string(),
                        }),
                        stop_sequence: None,
                    },
                    usage: chunk.usage.map(|usage| anthropic_ox::response::Usage {
                        input_tokens: Some(usage.prompt_tokens),
                        output_tokens: Some(usage.completion_tokens),
                    }),
                });

                // Send MessageStop
                events.push(AnthropicStreamEvent::MessageStop);
            }
        }

        events
    }
}

impl Default for AnthropicOpenRouterStreamConverter {
    fn default() -> Self {
        Self::new()
    }
}