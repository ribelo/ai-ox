use crate::{
    ModelResponse,
    content::delta::StreamEvent,
    tool::{ToolCall, ToolResult},
};

/// Events that can occur during agent execution.
///
/// These events provide visibility into the agent's internal state
/// and can be used for logging, debugging, or streaming interfaces.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Agent started processing a request.
    Started,

    /// Agent received a streaming event from the model.
    StreamEvent(StreamEvent),

    /// Agent is executing a tool call.
    ToolExecution(ToolCall),

    /// Agent completed a tool call execution.
    ToolResult(ToolResult),

    /// Agent completed processing and has a final response.
    Completed(ModelResponse),

    /// Agent failed with an error.
    Failed(String),
}

impl AgentEvent {
    /// Returns a descriptive name for this event type.
    pub fn event_type(&self) -> &'static str {
        match self {
            AgentEvent::Started => "Started",
            AgentEvent::StreamEvent(_) => "StreamEvent",
            AgentEvent::ToolExecution(_) => "ToolExecution",
            AgentEvent::ToolResult(_) => "ToolResult",
            AgentEvent::Completed(_) => "Completed",
            AgentEvent::Failed(_) => "Failed",
        }
    }
}
