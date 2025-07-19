use std::error::Error as StdError;
use thiserror::Error;

/// Represents errors that can occur during workflow execution.
#[derive(Debug, Error)]
pub enum WorkflowError {
    /// An error that occurs within the business logic of a node.
    #[error("Node execution failed")]
    NodeExecutionFailed {
        /// The underlying, node-specific error.
        #[source]
        source: Box<dyn StdError + Send + Sync>,
    },
}

impl WorkflowError {
    /// Creates a new node execution error, capturing the source.
    pub fn node_execution_failed(error: impl StdError + Send + Sync + 'static) -> Self {
        Self::NodeExecutionFailed {
            source: Box::new(error),
        }
    }
}

