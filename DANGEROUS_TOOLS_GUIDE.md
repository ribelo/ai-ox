# Dangerous Tools Security Guide

The ai-ox crate no longer includes built-in approval hooks. Library consumers are responsible for deciding when a tool call is safe, asking for approval, or blocking the call entirely. This guide outlines two patterns you can adopt today.

## Guard Inside The Tool

Inspect the arguments inside your `ToolBox::invoke` implementation and bail out before running a dangerous operation. This keeps the check close to the code that performs the action.

```rust
use ai_ox::tool::{FunctionMetadata, Tool, ToolBox, ToolError, ToolUse};
use ai_ox::content::part::Part;
use futures_util::future::BoxFuture;
use serde_json::json;
use std::io;

struct BashTool;

impl ToolBox for BashTool {
    fn tools(&self) -> Vec<Tool> {
        vec![Tool::FunctionDeclarations(vec![FunctionMetadata {
            name: "execute".into(),
            description: Some("Execute a bash command".into()),
            parameters: json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
        }])]
    }

    fn invoke(&self, call: ToolUse) -> BoxFuture<'_, Result<Part, ToolError>> {
        Box::pin(async move {
            let command = call.args
                .get("command")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::input_deserialization(
                    "execute",
                    io::Error::new(io::ErrorKind::InvalidInput, "missing command"),
                ))?;

            if command.contains("rm -rf") {
                return Err(ToolError::execution(
                    "execute",
                    io::Error::new(io::ErrorKind::PermissionDenied, "dangerous command blocked"),
                ));
            }

            // Call into your real implementation here
            run_command(command).await
        })
    }
}
```

## Wrap Agent::run For Custom Approval

If you want a reusable approval experience, wrap calls to `Agent::run` and prompt the user whenever a dangerous command is about to execute. You can inspect the `ToolUse` payload, approve or deny it, and only call `tools.invoke` once you are satisfied.

```rust
async fn execute_with_approval(agent: &Agent, call: ToolUse) -> Result<Part, ToolError> {
    if is_sensitive(&call) && !user_confirms(&call).await {
        return Err(ToolError::execution(
            &call.name,
            io::Error::new(io::ErrorKind::PermissionDenied, "user denied request"),
        ));
    }

    agent.tools().invoke(call).await
}
```

You can build any UI on top of this pattern: CLI prompts, HTTP endpoints, or GUI dialogs. Keeping the logic outside the library means you decide how strict the policy should be for each deployment.

## Key Takeaways

- There is no automatic approval pipeline in the ai-ox runtime anymore.
- Tools should defensively validate their arguments before executing.
- Application code can wrap `Agent` methods to provide bespoke approval or auditing flows.
- Revisit existing tools to ensure they fail closed when the user or policy declines an action.

With these patterns you retain full control over how (or whether) potentially dangerous tools execute.
