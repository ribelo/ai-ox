use crate::{
    content::delta::MessageStreamEvent,
    ModelResponse,
    tool::{ToolCall, ToolResult},
};

/// Events that can occur during agent execution.
///
/// These events provide visibility into the agent's internal state
/// and can be used for logging, debugging, or streaming interfaces.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Agent started processing a request.
    Started {
        /// The iteration number (0-based).
        iteration: u32,
    },

    /// Agent is making a request to the model.
    ModelRequest {
        /// The iteration number (0-based).
        iteration: u32,
        /// Number of messages in the conversation history.
        message_count: usize,
    },

    /// Agent received a streaming event from the model.
    ModelStreamEvent {
        /// The iteration number (0-based).
        iteration: u32,
        /// The stream event from the model.
        event: MessageStreamEvent,
    },

    /// Agent received a complete response from the model.
    ModelResponse {
        /// The iteration number (0-based).
        iteration: u32,
        /// The complete response from the model.
        response: ModelResponse,
    },

    /// Agent is executing a tool call.
    ToolCallStarted {
        /// The iteration number (0-based).
        iteration: u32,
        /// The tool call being executed.
        call: ToolCall,
    },

    /// Agent completed a tool call.
    ToolCallCompleted {
        /// The iteration number (0-based).
        iteration: u32,
        /// The tool call that was executed.
        call: ToolCall,
        /// The result of the tool execution.
        result: ToolResult,
    },

    /// Agent failed to execute a tool call.
    ToolCallFailed {
        /// The iteration number (0-based).
        iteration: u32,
        /// The tool call that failed.
        call: ToolCall,
        /// The error message.
        error: String,
    },

    /// Agent completed processing and has a final response.
    Completed {
        /// Total number of iterations executed.
        iterations: u32,
        /// The final response.
        response: ModelResponse,
    },

    /// Agent failed with an error.
    Failed {
        /// The iteration number where the failure occurred.
        iteration: u32,
        /// The error message.
        error: String,
    },
}

impl AgentEvent {
    /// Returns the iteration number for this event.
    pub fn iteration(&self) -> u32 {
        match self {
            AgentEvent::Started { iteration } => *iteration,
            AgentEvent::ModelRequest { iteration, .. } => *iteration,
            AgentEvent::ModelStreamEvent { iteration, .. } => *iteration,
            AgentEvent::ModelResponse { iteration, .. } => *iteration,
            AgentEvent::ToolCallStarted { iteration, .. } => *iteration,
            AgentEvent::ToolCallCompleted { iteration, .. } => *iteration,
            AgentEvent::ToolCallFailed { iteration, .. } => *iteration,
            AgentEvent::Completed { iterations, .. } => *iterations,
            AgentEvent::Failed { iteration, .. } => *iteration,
        }
    }
}
