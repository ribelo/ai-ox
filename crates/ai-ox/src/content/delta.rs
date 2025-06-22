//! Defines the events and deltas used for streaming model responses.

use crate::usage::Usage;
use serde::{Deserialize, Serialize};

/// Represents a delta for a single content block within a message.
///
/// A content block can be a piece of text or a tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    /// A delta for a text block.
    Text { text: String },
    /// A delta for a tool call block.
    ToolCall {
        /// The unique ID for the tool call. Sent once per tool call.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// The name of the function being called. Sent once per tool call.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// A delta of the JSON arguments for the function.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args_delta: Option<String>,
    },
}

/// Represents an event in the message stream from the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageStreamEvent {
    /// Occurs when a new message is created. This is the first event in a stream.
    MessageStart,

    /// Occurs when a new content block is started.
    ContentBlockStart {
        /// The index of the content block.
        index: usize,
    },

    /// A delta for a content block.
    ContentBlockDelta {
        /// The index of the content block being updated.
        index: usize,
        /// The delta for the content block.
        delta: ContentBlockDelta,
    },

    /// Occurs when a content block is finished.
    ContentBlockStop {
        /// The index of the content block.
        index: usize,
    },

    /// Occurs when the message is complete. This is the final event in a stream.
    MessageStop {
        /// The token usage statistics for the request.
        usage: Usage,
    },
}