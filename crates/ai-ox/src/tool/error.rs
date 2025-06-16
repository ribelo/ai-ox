use std::error::Error as StdError;
use thiserror::Error;

/// A type alias for a boxed error that is thread-safe.
type BoxedError = Box<dyn StdError + Send + Sync>;

/// Represents errors that can occur during tool invocation.
///
/// This error enum is designed for use within the framework's logic.
/// It preserves the original source errors where applicable, allowing for
/// detailed logging and debugging. It is not intended to be serialized
/// or sent across process boundaries directly.
#[derive(Debug, Error)]
pub enum ToolError {
    /// The requested tool was not found.
    #[error("Tool not found: {name}")]
    NotFound { name: String },

    /// Failed to deserialize input arguments from the provided data.
    /// This indicates a problem with the input data format or structure.
    #[error("Input deserialization failed for tool '{name}'")]
    InputDeserialization {
        name: String,
        /// The underlying deserialization error.
        #[source]
        error: BoxedError,
    },

    /// The tool executed but failed with a specific, tool-defined error.
    /// This represents a failure in the tool's business logic.
    #[error("Tool execution failed for tool '{name}'")]
    Execution {
        name: String,
        /// The underlying tool-specific error.
        #[source]
        error: BoxedError,
    },

    /// Failed to serialize the output of a successful tool execution.
    /// This indicates a problem with the tool's output type or the serialization process.
    #[error("Output serialization failed for tool '{name}'")]
    OutputSerialization {
        name: String,
        /// The underlying serialization error.
        #[source]
        error: BoxedError,
    },

    /// An internal error occurred within the tool-handling framework itself.
    /// This points to a bug or unexpected state in the framework, not the tool.
    #[error("Internal tool error: {context}")]
    Internal {
        context: String,
        /// The underlying framework error.
        #[source]
        error: BoxedError,
    },
}

impl ToolError {
    /// Creates an internal error, capturing the context and the source error.
    pub fn internal(
        context: impl Into<String>,
        error: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::Internal {
            context: context.into(),
            error: Box::new(error),
        }
    }

    /// Creates a "not found" error.
    pub fn not_found(name: impl Into<String>) -> Self {
        Self::NotFound { name: name.into() }
    }

    /// Creates an "input deserialization" error, wrapping the source error.
    pub fn input_deserialization(
        name: impl Into<String>,
        error: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::InputDeserialization {
            name: name.into(),
            error: Box::new(error),
        }
    }

    /// Creates a "tool execution" error, wrapping the tool's specific error.
    pub fn execution(
        name: impl Into<String>,
        error: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::Execution {
            name: name.into(),
            error: Box::new(error),
        }
    }

    /// Creates an "output serialization" error, wrapping the source error.
    pub fn output_serialization(
        name: impl Into<String>,
        error: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::OutputSerialization {
            name: name.into(),
            error: Box::new(error),
        }
    }
}