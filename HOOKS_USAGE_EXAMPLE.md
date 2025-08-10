# ToolHooks Usage Guide

## How ApprovalRequest Works (FIXED DESIGN)

**You were 100% correct** - ApprovalRequest now contains:
- `tool_name`: Automatically filled with the function name
- `args`: Automatically filled with the tool arguments (as JSON)

**The Agent creates ApprovalRequest automatically** - you don't have to!

## Basic Usage

```rust
use ai_ox::{
    tool::{ToolHooks, ApprovalRequest},
    toolbox, dangerous,
};

struct MyTool;

#[toolbox]
impl MyTool {
    /// Safe operation - no approval needed
    pub fn get_info(&self) -> String {
        "System info".to_string()
    }
    
    /// Dangerous operation - will trigger approval
    #[dangerous]
    pub fn delete_file(&self, path: String) -> String {
        format!("Would delete: {}", path)
    }
}

#[tokio::main]
async fn main() {
    // Create hooks with smart approval logic
    let hooks = ToolHooks::new()
        .with_approval(|request| {
            Box::pin(async move {
                println!("🚨 Tool: {}", request.tool_name);
                println!("📝 Args: {:?}", request.args);
                
                // Your approval logic here:
                match request.tool_name.as_str() {
                    "delete_file" => {
                        // Only allow deleting temp files
                        if let Some(path) = request.args.as_str() {
                            path.starts_with("/tmp/")
                        } else {
                            false
                        }
                    },
                    _ => true // Allow other dangerous operations
                }
            })
        });
    
    // Use with Agent (recommended)
    let agent = Agent::model(your_model)
        .tools(MyTool)
        .build();
    
    let messages = vec![
        Message::user("Please delete the file /tmp/test.txt")
    ];
    
    // Agent automatically:
    // 1. Sees delete_file is dangerous
    // 2. Creates ApprovalRequest with tool_name="delete_file", args="/tmp/test.txt"
    // 3. Calls your approval callback
    // 4. Executes or denies based on your response
    let response = agent.run_with_hooks(messages, Some(hooks)).await?;
}
```

## Smart Approval Patterns

```rust
let hooks = ToolHooks::new()
    .with_approval(|request| {
        Box::pin(async move {
            match request.tool_name.as_str() {
                // File operations
                "delete_file" | "write_file" => {
                    if let Some(path) = request.args.as_str() {
                        // Only allow operations in safe directories
                        path.starts_with("/tmp/") || path.starts_with("/var/tmp/")
                    } else {
                        false
                    }
                },
                
                // Shell commands
                "execute_command" => {
                    if let Some(cmd) = request.args.as_str() {
                        // Block dangerous commands
                        !cmd.contains("rm -rf") && 
                        !cmd.contains("sudo") &&
                        !cmd.contains("format")
                    } else {
                        false
                    }
                },
                
                // Network operations
                "send_email" | "post_webhook" => {
                    // Ask user for network operations
                    println!("Allow network operation {}? (y/N)", request.tool_name);
                    // Get user input...
                    user_approves()
                },
                
                _ => true // Allow other dangerous operations by default
            }
        })
    });
```

## Key Benefits

✅ **Automatic ApprovalRequest creation** - Agent fills tool_name and args  
✅ **Surgical control** - Only dangerous tools trigger approval  
✅ **Rich context** - Full access to tool name and arguments for decisions  
✅ **Zero tool changes** - Tools just declare `#[dangerous]`  
✅ **Backward compatible** - Safe tools work unchanged  

## What Happens

1. **Safe tools** → Execute immediately, no approval
2. **Dangerous tools** → Agent creates ApprovalRequest automatically
3. **Your callback** → Receives tool_name and args  
4. **Returns true** → Tool executes
5. **Returns false** → Tool returns "User denied execution" error

## Migration from Old Design

**Before (manual):**
```rust
// Tools had to create ApprovalRequest manually
let request = ApprovalRequest {
    operation: "Delete file".to_string(),
    details: format!("path: {}", path),
};
```

**After (automatic):**
```rust
// Agent creates this automatically:
// ApprovalRequest {
//     tool_name: "delete_file",
//     args: serde_json::json!("/tmp/test.txt"),
// }
```

## Session-Based Approval (No Repeated Prompts!)

You can pre-approve dangerous tools to avoid repetitive prompts:

```rust
// Pre-approve specific tools for the session
let mut agent = Agent::model(your_model)
    .tools(MyTool)
    .build();

// User clicks "approve for session" - add tools to pre-approved list
agent.approve_dangerous_tools(&["delete_file", "execute_command"]);

// Now these tools execute without asking hooks (until revoked)
let response = agent.run_with_hooks(messages, Some(hooks)).await?;

// Later, revoke approval if needed
agent.revoke_dangerous_tools(&["execute_command"]);
```

### Trust Mode (Approve All)
```rust
// For trusted environments - approve ALL dangerous operations
agent.approve_all_dangerous_tools();

// Or clear all approvals
agent.clear_approved_dangerous_tools();
```

### How It Works
1. **Pre-approved tools** → Execute immediately, no hooks called
2. **Not pre-approved** → Normal approval flow via hooks
3. **No hooks provided** → Dangerous tools are denied

## Argument-Based Danger Detection

For tools where danger depends on the arguments (like bash commands), you can override `invoke_with_hooks`:

```rust
struct BashTool;

impl ToolBox for BashTool {
    fn invoke_with_hooks(&self, call: ToolCall, hooks: ToolHooks) -> BoxFuture<'_, Result<ToolResult, ToolError>> {
        Box::pin(async move {
            // Check the ACTUAL command being executed
            if let Some(cmd) = call.args.get("command").and_then(|v| v.as_str()) {
                if is_dangerous_command(cmd) {
                    // Only dangerous commands need approval
                    let request = ApprovalRequest {
                        tool_name: call.name.clone(),
                        args: call.args.clone(),
                    };
                    
                    if !hooks.request_approval(request).await {
                        return Err(/* denied */);
                    }
                }
            }
            // Safe commands execute without approval
            self.invoke(call).await
        })
    }
    
    fn dangerous_functions(&self) -> &[&str] {
        &[] // Not globally dangerous - we check per invocation
    }
}
```

### Important: Session Approval Limitation

⚠️ **WARNING**: When you use `agent.approve_dangerous_tools(&["execute"])`, you're approving ALL invocations of that function:

```rust
// User approves bash for session (tired of approving "ls")
agent.approve_dangerous_tools(&["execute"]);

// Now ALL these work without asking:
bash.execute("ls")        // Safe
bash.execute("rm -rf /")  // DANGEROUS - but still executes!
```

For tools like bash where danger varies by argument, consider:
1. Never using session approval (always ask)
2. Implementing smart detection with `invoke_with_hooks` 
3. Using a whitelist of allowed commands

See `examples/bash_tool_safe.rs` and `DANGEROUS_TOOLS_GUIDE.md` for complete implementations.

Perfect! The system now works exactly as you suggested.