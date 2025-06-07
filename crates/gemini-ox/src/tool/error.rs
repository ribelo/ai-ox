//! Error types for tool usage and invocation.

use crate::generate_content::content::Content; // Assuming this path
use serde::{Deserialize, Serialize}; // Add Deserialize import
use thiserror::Error;

/// Errors that can occur during the processing or execution of a function call.
#[derive(Error, Debug, Clone, PartialEq, Serialize, Deserialize)] // Added PartialEq, Deserialize
#[serde(tag = "error_type", content = "details")] // Makes serialization structured
pub enum FunctionCallError {
    /// The LLM did not provide arguments when the tool expected them.
    /// Note: This is typically reported back *within* Ok(Content), but defined here for completeness.
    #[error("Function call is missing required arguments")]
    MissingArguments,

    /// Failed to deserialize the arguments provided by the LLM into the tool's input type.
    /// Note: This is typically reported back *within* Ok(Content), but defined here for completeness.
    #[error("Failed to deserialize input arguments: {0}")]
    InputDeserializationFailed(String),

    /// Failed to serialize the tool's successful output or an error response into the required format.
    /// This usually indicates an internal issue.
    #[error("Failed to serialize output/error response: {0}")]
    OutputSerializationFailed(String),

    /// The tool itself encountered an error during its execution.
    #[error("Tool execution failed: {0}")]
    ExecutionFailed(String),

    /// The tool panicked during execution.
    #[error("Tool panicked: {0}")]
    ToolPanic(String),

    /// The requested tool could not be found in the ToolBox.
    /// Note: This is typically reported back *within* Ok(Content), but defined here for completeness.
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// Failed to serialize the error response itself (a meta-error).
    #[error("Failed to serialize the error structure: {0}")]
    ErrorSerializationFailed(String),
}

impl FunctionCallError {
    /// Helper to create a [`FunctionResponse`] content object representing this error.
    /// Returns `Ok(Content)` if serialization succeeds.
    ///
    /// # Errors
    ///
    /// Returns `Err(FunctionCallError::OutputSerializationFailed)` if serialization
    /// of the error response itself fails. This typically indicates an internal issue
    /// with the `serde` implementation for `FunctionCallError` or the underlying
    /// serialization format (e.g., JSON).
    pub fn into_error_response_content(
        self,
        tool_name: String,
    ) -> Result<Content, FunctionCallError> {
        // Pass a clone of self, as it implements Serialize and Content::function_response
        // likely takes `impl Serialize`. Cloning allows `self` to still be available
        // in the `map_err` closure below if serialization fails.
        Content::function_response(tool_name, self.clone()).map_err(|e| {
            // If we can't even serialize the error message, that's a critical failure.
            // Use the `Display` representation of the original error (`self`) in the message.
            FunctionCallError::OutputSerializationFailed(format!(
                "Failed to serialize error response for error \'{self}\': {e}"
            ))
        })
    }
}
