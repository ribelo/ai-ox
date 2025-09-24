# User-Side Approval Example

`ai-ox` no longer ships with a `ToolHooks` type. If you still want interactive approval, implement it in your application code. The snippet below illustrates one way to intercept tool calls before they run.

```rust
use ai_ox::agent::Agent;
use ai_ox::tool::{ToolSet, ToolUse, ToolError};
use ai_ox::content::Part;
use std::io;

async fn execute_with_prompt(tools: &ToolSet, call: ToolUse) -> Result<Part, ToolError> {
    if is_sensitive(&call) {
        if !user_approves(&call).await {
            return Err(ToolError::execution(
                &call.name,
                io::Error::new(io::ErrorKind::PermissionDenied, "user denied"),
            ));
        }
    }

    tools.invoke(call).await
}

async fn run_agent_with_guard(agent: &Agent, messages: Vec<Message>) -> Result<ModelResponse, AgentError> {
    // Pseudocode: copy the agent loop, but replace the tool execution line with execute_with_prompt.
    // This gives your application complete control over when a tool call is allowed to run.
}
```

Ways to tailor this pattern:

1. Persist approvals in your own store to support "approve once" behaviour.
2. Instrument logging to create an audit trail for every sensitive invocation.
3. Build richer prompts (e.g. GUI, HTTP) for approval requests.

Because the approval flow now lives in your application, you can adjust it without waiting for library changes.
