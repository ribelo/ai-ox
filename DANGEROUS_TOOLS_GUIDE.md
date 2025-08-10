# Dangerous Tools Security Guide

## The Session Approval Problem

**WARNING**: When you approve a dangerous tool for a session using `approve_dangerous_tools()`, you're approving ALL invocations of that function, not specific commands.

### Example of the Problem

```rust
// User tired of approving "ls" every time
agent.approve_dangerous_tools(&["execute"]);

// Now these ALL work without asking:
bash.execute("ls")           // ✅ Safe command
bash.execute("cat README")    // ✅ Probably safe
bash.execute("rm -rf /")      // ⚠️ DESTRUCTIVE - but still executes!
```

Once `execute` is approved for the session, the agent can run ANY command without further approval.

## Three Security Models

### 1. Always Ask (Maximum Security)

Mark the tool as dangerous, never use session approval:

```rust
#[toolbox]
impl BashTool {
    #[dangerous]
    pub async fn execute(&self, command: String) -> Result<String, Error> {
        // Execute command
        run_command(&command)
    }
}

// Never call approve_dangerous_tools for bash
// Every command will request approval
```

**Pros**: Maximum security, user sees every command  
**Cons**: Annoying for repetitive safe commands like `ls`

### 2. Trust the Agent (Convenience over Security)

Approve bash for the session, trust the AI to behave:

```rust
// User decides to trust the AI
agent.approve_dangerous_tools(&["execute"]);

// Now all bash commands execute without approval
```

**Pros**: Convenient, no repeated prompts  
**Cons**: AI can execute ANY command without asking

### 3. Smart Detection (Advanced)

Override `invoke_with_hooks` to inspect arguments and decide per-invocation:

```rust
use ai_ox::{toolbox, tool::*};
use futures_util::future::BoxFuture;

struct BashTool;

#[toolbox]
impl BashTool {
    pub async fn execute(&self, command: String) -> Result<String, Error> {
        run_command(&command)
    }
}

// Custom implementation for smart detection
impl ToolBox for BashTool {
    fn tools(&self) -> Vec<Tool> {
        // Use the macro-generated implementation
        vec![Tool::FunctionDeclarations(vec![
            FunctionMetadata {
                name: "execute".to_string(),
                description: Some("Execute a bash command".to_string()),
                parameters: /* ... */,
            }
        ])]
    }
    
    fn invoke(&self, call: ToolCall) -> BoxFuture<'_, Result<ToolResult, ToolError>> {
        // Use the macro-generated implementation for actual execution
        Box::pin(async move {
            // Deserialize args and call execute()
            // ... macro-generated code ...
        })
    }
    
    fn invoke_with_hooks(&self, call: ToolCall, hooks: ToolHooks) -> BoxFuture<'_, Result<ToolResult, ToolError>> {
        Box::pin(async move {
            // Extract command from arguments
            if let Some(command) = call.args.get("command").and_then(|v| v.as_str()) {
                // Check if THIS SPECIFIC command is dangerous
                if is_dangerous_command(command) {
                    // Request approval for dangerous commands
                    let request = ApprovalRequest {
                        tool_name: call.name.clone(),
                        args: call.args.clone(),
                    };
                    
                    if !hooks.request_approval(request).await {
                        return Err(ToolError::execution(
                            &call.name,
                            std::io::Error::new(
                                std::io::ErrorKind::PermissionDenied,
                                format!("Dangerous command '{}' was denied", command)
                            )
                        ));
                    }
                }
                // Safe or approved - continue
            }
            
            // Execute using the standard invoke
            self.invoke(call).await
        })
    }
    
    fn dangerous_functions(&self) -> &[&str] {
        // Return empty - we handle danger per-invocation
        &[]
    }
}

fn is_dangerous_command(cmd: &str) -> bool {
    // Commands that are ALWAYS safe
    const SAFE_COMMANDS: &[&str] = &[
        "ls", "pwd", "echo", "date", "whoami", "hostname",
        "ps", "df", "du", "uptime", "uname", "which"
    ];
    
    // Patterns that indicate danger
    const DANGEROUS_PATTERNS: &[&str] = &[
        "rm ", "rmdir", "del ",           // Deletion
        "sudo", "su ",                    // Privilege escalation
        "chmod", "chown", "chgrp",        // Permission changes  
        "kill", "pkill", "killall",        // Process control
        "shutdown", "reboot", "halt",      // System control
        "format", "mkfs", "fdisk", "dd",   // Disk operations
        "curl", "wget", "nc",              // Network operations
        ">", ">>",                         // File overwriting
        "eval", "exec", "source",          // Code execution
        "/etc/passwd", "/etc/shadow",      // Sensitive files
        ".ssh/", ".aws/", ".config/",      // Sensitive directories
    ];
    
    let cmd_lower = cmd.to_lowercase();
    let first_word = cmd_lower.split_whitespace().next().unwrap_or("");
    
    // Check if explicitly safe
    if SAFE_COMMANDS.contains(&first_word) {
        // But still check for dangerous patterns in arguments
        for pattern in DANGEROUS_PATTERNS {
            if cmd_lower.contains(pattern) {
                return true; // e.g., "ls /etc/passwd" is suspicious
            }
        }
        return false;
    }
    
    // Check for dangerous patterns
    DANGEROUS_PATTERNS.iter().any(|&pattern| cmd_lower.contains(pattern))
}
```

**Pros**: Fine-grained control, safe commands don't prompt  
**Cons**: Complex to implement correctly, may miss edge cases

## Important Security Considerations

### The Limits of Pattern Matching

No pattern matching is perfect. Consider these "safe-looking" but dangerous commands:

```bash
# Looks safe but isn't
echo "rm -rf /" > script.sh && bash script.sh
cat /dev/zero > /dev/sda
:(){ :|:& };:                    # Fork bomb
python -c "import os; os.system('rm -rf /')"
```

Pattern matching will NEVER catch all dangerous commands.

### Agent vs Hooks Approval Flow

Understanding the approval flow is critical:

```rust
// When agent executes dangerous tool:
if tool in approved_dangerous_tools {
    // Pre-approved: Execute immediately, hooks NOT called
    execute()
} else if hooks.on_approval_needed {
    // Not pre-approved: Ask via hooks
    if hooks.request_approval() {
        execute()
    } else {
        deny()
    }
} else {
    // Dangerous, not approved, no hooks: Deny
    deny()
}
```

### Recommendations

1. **For Production Systems**: Use Model 1 (Always Ask) or don't provide bash access at all
2. **For Development**: Model 2 (Trust) is acceptable if you monitor the AI
3. **For Mixed Use**: Model 3 with a conservative `is_dangerous_command` that errs on the side of caution

## Command Whitelist Pattern

For maximum security with good UX, consider a whitelist approach:

```rust
struct SafeBashTool {
    allowed_commands: HashSet<String>,
}

impl SafeBashTool {
    pub fn new() -> Self {
        let mut allowed = HashSet::new();
        allowed.insert("ls".to_string());
        allowed.insert("pwd".to_string());
        allowed.insert("echo".to_string());
        allowed.insert("cat".to_string());
        allowed.insert("grep".to_string());
        // Add more as needed
        Self { allowed_commands: allowed }
    }
}

#[toolbox]
impl SafeBashTool {
    pub async fn execute(&self, command: String) -> Result<String, Error> {
        let cmd = command.split_whitespace().next().unwrap_or("");
        
        if !self.allowed_commands.contains(cmd) {
            return Err(Error::new(format!(
                "Command '{}' not in whitelist. Allowed: {:?}",
                cmd, self.allowed_commands
            )));
        }
        
        run_command(&command)
    }
}
```

This ensures ONLY whitelisted commands can run, regardless of approval status.

## Summary

- **Session approval is all-or-nothing per function**
- **There's no perfect solution for variable-danger tools like bash**
- **Choose your security model based on your trust level and use case**
- **When in doubt, be more restrictive**
- **Document your security model clearly for your users**

Remember: The tool author is responsible for implementing appropriate security measures. The framework provides the hooks, but YOU decide when to use them.