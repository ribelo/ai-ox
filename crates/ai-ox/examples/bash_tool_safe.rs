//! Example of a BashTool with argument-based danger detection.
//!
//! This example shows how to implement fine-grained control over
//! which bash commands are allowed based on their content.

use ai_ox::{
    content::part::Part,
    tool::{FunctionMetadata, Tool, ToolBox, ToolError, ToolUse},
};
use futures_util::future::BoxFuture;
use serde_json::json;
use std::collections::{BTreeMap, HashSet};
use std::process::Command;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BashError {
    #[error("Command execution failed: {0}")]
    ExecutionFailed(String),
}

/// A bash tool that intelligently determines which commands are dangerous.
pub struct SmartBashTool;

// Don't use the toolbox macro - we're implementing manually
impl SmartBashTool {
    /// Execute a bash command with smart danger detection.
    pub async fn execute(&self, command: String) -> Result<String, BashError> {
        // Actually run the command
        match Command::new("sh").arg("-c").arg(&command).output() {
            Ok(output) => {
                if output.status.success() {
                    Ok(String::from_utf8_lossy(&output.stdout).to_string())
                } else {
                    Err(BashError::ExecutionFailed(
                        String::from_utf8_lossy(&output.stderr).to_string(),
                    ))
                }
            }
            Err(e) => Err(BashError::ExecutionFailed(format!(
                "Failed to execute command: {}",
                e
            ))),
        }
    }
}

// Override ToolBox implementation for smart danger detection
impl ToolBox for SmartBashTool {
    fn tools(&self) -> Vec<Tool> {
        vec![Tool::FunctionDeclarations(vec![FunctionMetadata {
            name: "execute".to_string(),
            description: Some("Execute a bash command".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    }
                },
                "required": ["command"]
            }),
        }])]
    }

    fn invoke(&self, call: ToolUse) -> BoxFuture<'_, Result<Part, ToolError>> {
        Box::pin(async move {
            // Extract command from args
            let command = call
                .args
                .get("command")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ToolError::input_deserialization(
                        "execute",
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Missing 'command' parameter",
                        ),
                    )
                })?;

            let danger_level = analyze_command_danger(command);

            match danger_level {
                DangerLevel::Safe => {
                    println!("✅ Executing safe command: {}", command);
                }
                DangerLevel::Suspicious(reason) => {
                    println!("⚠️  Suspicious command blocked ({}): {}", reason, command);
                    return Err(ToolError::execution(
                        "execute",
                        std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            format!("Command denied ({}): {}", reason, command),
                        ),
                    ));
                }
                DangerLevel::Dangerous(reason) => {
                    println!("🚨 Dangerous command blocked ({}): {}", reason, command);
                    return Err(ToolError::execution(
                        "execute",
                        std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            format!("Dangerous command denied: {}", command),
                        ),
                    ));
                }
            }

            // Execute the command
            match self.execute(command.to_string()).await {
                Ok(output) => Ok(Part::ToolResult {
                    id: call.id.clone(),
                    name: "execute".to_string(),
                    parts: vec![Part::Text {
                        text: output,
                        ext: BTreeMap::new(),
                    }],
                    ext: BTreeMap::new(),
                }),
                Err(e) => Err(ToolError::execution("execute", e)),
            }
        })
    }
}

#[derive(Debug)]
enum DangerLevel {
    Safe,
    Suspicious(&'static str),
    Dangerous(&'static str),
}

/// Analyze a command to determine its danger level.
/// This is where the "smart" detection happens.
fn analyze_command_danger(cmd: &str) -> DangerLevel {
    let cmd_lower = cmd.trim().to_lowercase();
    let parts: Vec<&str> = cmd_lower.split_whitespace().collect();

    if parts.is_empty() {
        return DangerLevel::Safe;
    }

    let base_command = parts[0];

    // Explicitly safe commands (with no dangerous arguments)
    match base_command {
        "ls" | "pwd" | "date" | "whoami" | "hostname" | "uptime" | "uname" | "echo" | "printf" => {
            // Check for suspicious arguments even in "safe" commands
            if cmd_lower.contains("/etc/passwd")
                || cmd_lower.contains("/etc/shadow")
                || cmd_lower.contains(".ssh")
                || cmd_lower.contains(".aws")
            {
                return DangerLevel::Suspicious("accessing sensitive files");
            }
            DangerLevel::Safe
        }

        // Reading commands - check what they're reading
        "cat" | "head" | "tail" | "less" | "more" => {
            if cmd_lower.contains("/etc/passwd")
                || cmd_lower.contains("/etc/shadow")
                || cmd_lower.contains("private")
                || cmd_lower.contains("secret")
                || cmd_lower.contains(".env")
            {
                DangerLevel::Suspicious("reading potentially sensitive files")
            } else {
                DangerLevel::Safe
            }
        }

        // Search commands - generally safe but check for password hunting
        "grep" | "find" | "locate" => {
            if cmd_lower.contains("password")
                || cmd_lower.contains("secret")
                || cmd_lower.contains("token")
                || cmd_lower.contains("api_key")
            {
                DangerLevel::Suspicious("searching for credentials")
            } else {
                DangerLevel::Safe
            }
        }

        // File operations - always dangerous
        "rm" | "rmdir" | "del" | "unlink" => {
            if cmd_lower.contains("-rf") || cmd_lower.contains("-r") {
                DangerLevel::Dangerous("recursive deletion")
            } else {
                DangerLevel::Dangerous("file deletion")
            }
        }

        // Permission changes - dangerous
        "chmod" | "chown" | "chgrp" => DangerLevel::Dangerous("permission modification"),

        // Privilege escalation - always dangerous
        "sudo" | "su" | "doas" => DangerLevel::Dangerous("privilege escalation"),

        // System control - always dangerous
        "shutdown" | "reboot" | "halt" | "poweroff" | "init" => {
            DangerLevel::Dangerous("system control")
        }

        // Process control - suspicious
        "kill" | "pkill" | "killall" => DangerLevel::Suspicious("process termination"),

        // Network operations - suspicious
        "curl" | "wget" | "nc" | "netcat" | "telnet" | "ssh" | "scp" => {
            DangerLevel::Suspicious("network operation")
        }

        // Package management - dangerous
        "apt" | "apt-get" | "yum" | "dnf" | "pacman" | "brew" | "npm" | "pip" => {
            DangerLevel::Dangerous("package management")
        }

        // Disk operations - extremely dangerous
        "dd" | "format" | "mkfs" | "fdisk" | "parted" => DangerLevel::Dangerous("disk operation"),

        // Script execution - dangerous
        "sh" | "bash" | "zsh" | "python" | "ruby" | "perl" | "node" => {
            DangerLevel::Dangerous("script execution")
        }

        // Default for unknown commands
        _ => {
            // Check for dangerous patterns in any command
            if cmd_lower.contains(">") || cmd_lower.contains(">>") {
                DangerLevel::Suspicious("file redirection")
            } else if cmd_lower.contains("|") {
                DangerLevel::Suspicious("command piping")
            } else if cmd_lower.contains("eval") || cmd_lower.contains("exec") {
                DangerLevel::Dangerous("code evaluation")
            } else if cmd_lower.contains("&") || cmd_lower.contains(";") {
                DangerLevel::Suspicious("command chaining")
            } else {
                // Unknown command - be cautious
                DangerLevel::Suspicious("unknown command")
            }
        }
    }
}

/// Alternative: A whitelist-only bash tool for maximum security.
pub struct WhitelistBashTool {
    allowed_commands: HashSet<String>,
}

impl WhitelistBashTool {
    pub fn new() -> Self {
        let mut allowed = HashSet::new();
        // Only these exact commands are allowed
        allowed.insert("ls".to_string());
        allowed.insert("ls -la".to_string());
        allowed.insert("pwd".to_string());
        allowed.insert("date".to_string());
        allowed.insert("whoami".to_string());
        allowed.insert("echo hello".to_string());
        // Add more as needed
        Self {
            allowed_commands: allowed,
        }
    }

    /// Execute only whitelisted commands.
    pub async fn execute(&self, command: String) -> Result<String, BashError> {
        if !self.allowed_commands.contains(&command) {
            return Err(BashError::ExecutionFailed(format!(
                "Command '{}' not in whitelist. Allowed commands: {:?}",
                command, self.allowed_commands
            )));
        }

        // Run the whitelisted command
        match Command::new("sh").arg("-c").arg(&command).output() {
            Ok(output) => Ok(String::from_utf8_lossy(&output.stdout).to_string()),
            Err(e) => Err(BashError::ExecutionFailed(format!(
                "Execution failed: {}",
                e
            ))),
        }
    }
}

#[tokio::main]
async fn main() {
    println!("Smart Bash Tool Example\n");

    // Test the danger analyzer
    let test_commands = vec![
        "ls -la",
        "pwd",
        "cat /etc/passwd",
        "rm -rf /tmp/test",
        "sudo apt update",
        "echo hello world",
        "curl https://example.com",
        "grep password /var/log/*",
        "python -c 'print(42)'",
        "unknown_command --flag",
    ];

    println!("Command Danger Analysis:");
    println!("{:-<60}", "");
    for cmd in test_commands {
        let danger = analyze_command_danger(cmd);
        println!("{:40} -> {:?}", cmd, danger);
    }

    println!("\n{:-<60}", "");
    println!("To use in an agent:");
    println!("{:-<60}\n", "");

    println!("```rust");
    println!("let agent = Agent::model(your_model)");
    println!("    .tools(SmartBashTool)");
    println!("    .build();");
    println!("");
    println!("// Safe commands execute without approval");
    println!("// Suspicious commands request approval");
    println!("// Dangerous commands always request approval");
    println!("```");
}
