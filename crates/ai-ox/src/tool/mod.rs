pub mod encoding;
pub mod error;
#[cfg(feature = "gemini")]
pub mod gemini;
pub mod set;
pub mod types;

pub use encoding::{decode_tool_result_parts, encode_tool_result_parts};
pub use error::ToolError;
pub use set::ToolSet;
pub use types::ToolUse;

use futures_util::future::BoxFuture;
use schemars::{JsonSchema, generate::SchemaSettings};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

/// Request for user approval of a potentially dangerous operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// Name of the tool function being called
    pub tool_name: String,
    /// Arguments passed to the tool (as JSON)
    pub args: serde_json::Value,
}

/// Collection of optional hooks that tools can use during execution.
#[derive(Default, Clone)]
pub struct ToolHooks {
    /// Called when tool needs approval for a dangerous operation
    pub on_approval_needed: Option<Arc<dyn Fn(ApprovalRequest) -> BoxFuture<'static, bool> + Send + Sync>>,
    /// Called for progress updates during tool execution
    pub on_progress: Option<Arc<dyn Fn(String) -> BoxFuture<'static, ()> + Send + Sync>>,
}

impl ToolHooks {
    /// Create new empty ToolHooks
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the approval callback
    pub fn with_approval<F>(mut self, callback: F) -> Self 
    where
        F: Fn(ApprovalRequest) -> BoxFuture<'static, bool> + Send + Sync + 'static,
    {
        self.on_approval_needed = Some(Arc::new(callback));
        self
    }
    
    /// Set the progress callback
    pub fn with_progress<F>(mut self, callback: F) -> Self 
    where
        F: Fn(String) -> BoxFuture<'static, ()> + Send + Sync + 'static,
    {
        self.on_progress = Some(Arc::new(callback));
        self
    }
    
    /// Request approval if callback is available
    pub async fn request_approval(&self, request: ApprovalRequest) -> bool {
        match &self.on_approval_needed {
            Some(callback) => callback(request).await,
            None => false, // Default to deny if no callback
        }
    }
    
    /// Report progress if callback is available
    pub async fn report_progress(&self, message: String) {
        if let Some(callback) = &self.on_progress {
            callback(message).await;
        }
    }
}

impl std::fmt::Debug for ToolHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolHooks")
            .field("has_approval_callback", &self.on_approval_needed.is_some())
            .field("has_progress_callback", &self.on_progress.is_some())
            .finish()
    }
}

/// Metadata for a tool function.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FunctionMetadata {
    /// Name of the function
    pub name: String,

    /// Optional description of what the function does
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON schema for the function's input parameters
    pub parameters: Value,
}

/// Represents different types of tools that can be used.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Tool {
    /// Function declarations that can be called
    FunctionDeclarations(Vec<FunctionMetadata>),
    /// Vendor-specific tool with opaque metadata
    #[cfg(feature = "gemini")]
    GeminiTool(gemini_ox::tool::Tool),
}

/// Trait for objects that provide tool functionality.
///
/// This trait allows objects to expose their available tools and handle
/// tool invocations in a standardized way.
/// 
/// # Security Model
/// 
/// The ToolBox trait supports two levels of danger detection:
/// 
/// 1. **Function-level** (via `dangerous_functions()`): Marks entire functions as always dangerous
/// 2. **Invocation-level** (via `invoke_with_hooks()`): Allows runtime inspection of arguments
/// 
/// ## Session Approval Limitation
/// 
/// When using `Agent::approve_dangerous_tools()`, you're approving ALL invocations of that
/// function. For tools where danger varies by argument (like bash commands), this means:
/// 
/// ```rust
/// // User approves "execute" for session (example - agent would be an Agent instance)
/// // agent.approve_dangerous_tools(&["execute"]);
///
/// // Now ALL commands work without asking:
/// // bash.execute("ls")        // Safe command
/// // bash.execute("rm -rf /")  // DANGEROUS - but still executes!
/// ```
/// 
/// ## Implementing Argument-Based Detection
/// 
/// For fine-grained control, override `invoke_with_hooks()` to inspect arguments:
/// 
/// ```rust
/// # use ai_ox::tool::{ToolUse, ToolHooks, ToolError, ApprovalRequest};
/// # use ai_ox::content::part::Part;
/// # use futures_util::future::BoxFuture;
/// # use std::io::{Error, ErrorKind};
/// fn invoke_with_hooks(call: ToolUse, hooks: ToolHooks) -> BoxFuture<'static, Result<Part, ToolError>> {
///     Box::pin(async move {
///         // Check if THIS SPECIFIC invocation is dangerous
///         if call.name == "dangerous_command" {
///             // Request approval for this specific call
///             let request = ApprovalRequest {
///                 tool_name: call.name.clone(),
///                 args: call.args.clone(),
///             };
///
///             if !hooks.request_approval(request).await {
///                 return Err(ToolError::execution(&call.name, Error::new(ErrorKind::PermissionDenied, "Permission denied")));
///             }
///         }
///         // Safe or approved - execute (placeholder)
///         Ok(Part::Text { text: "executed".to_string(), ext: std::collections::BTreeMap::new() })
///     })
/// }
/// ```
/// 
/// See `DANGEROUS_TOOLS_GUIDE.md` for complete security documentation.
pub trait ToolBox: Send + Sync + 'static {
    /// Returns the list of tools provided by this toolbox.
    fn tools(&self) -> Vec<Tool>;

    /// Invokes a tool function with the given call parameters.
    ///
    /// Returns a boxed future that resolves to either a Part::ToolResult on success
    /// or a ToolError on failure.
    fn invoke(&self, call: ToolUse) -> BoxFuture<'_, Result<crate::content::Part, ToolError>>;

    /// Invokes a tool function with hooks for dangerous operations.
    ///
    /// Default implementation just calls invoke(), ignoring hooks.
    /// 
    /// Override this method to implement argument-based danger detection.
    /// You can inspect the `call.args` to determine if this specific invocation
    /// is dangerous and should request approval.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// # use ai_ox::tool::{ToolUse, ToolHooks, ToolError, ApprovalRequest};
    /// # use ai_ox::content::part::Part;
    /// # use futures_util::future::BoxFuture;
    /// # use std::io::{Error, ErrorKind};
    /// fn invoke_with_hooks(call: ToolUse, hooks: ToolHooks) -> BoxFuture<'static, Result<Part, ToolError>> {
    ///     Box::pin(async move {
    ///         // Check command for danger patterns
    ///         if let Some(cmd) = call.args.get("command").and_then(|v| v.as_str()) {
    ///             if cmd.contains("rm") || cmd.contains("sudo") {
    ///                 // Request approval for dangerous command
    ///                 let request = ApprovalRequest {
    ///                     tool_name: call.name.clone(),
    ///                     args: call.args.clone(),
    ///                 };
    ///                 if !hooks.request_approval(request).await {
    ///                     return Err(ToolError::execution(&call.name, Error::new(ErrorKind::PermissionDenied, "Permission denied")));
    ///                 }
    ///             }
    ///         }
    ///         // Execute (placeholder)
    ///         Ok(Part::Text { text: "executed".to_string(), ext: std::collections::BTreeMap::new() })
    ///     })
    /// }
    /// ```
    fn invoke_with_hooks(&self, call: ToolUse, _hooks: ToolHooks) -> BoxFuture<'_, Result<crate::content::Part, ToolError>> {
        self.invoke(call)
    }

    /// Returns the names of functions that require approval before execution.
    ///
    /// Functions listed here are considered ALWAYS dangerous - every invocation
    /// will require approval (unless pre-approved via `Agent::approve_dangerous_tools()`).
    /// 
    /// For tools where danger depends on arguments (like bash commands), return
    /// an empty slice here and implement argument-based detection in `invoke_with_hooks()`.
    /// 
    /// # Session Approval Warning
    /// 
    /// When a function is in this list and gets approved via `approve_dangerous_tools()`,
    /// ALL invocations are approved for the session. This may not be appropriate for
    /// tools with variable danger levels.
    fn dangerous_functions(&self) -> &[&str] {
        &[]
    }

    /// Checks if this toolbox has a function with the given name.
    fn has_function(&self, name: &str) -> bool {
        self.tools().iter().any(|tool| match tool {
            Tool::FunctionDeclarations(functions) => functions.iter().any(|func| func.name == name),
            #[cfg(feature = "gemini")]
            Tool::GeminiTool(_) => false,
        })
    }
}

impl<T: ToolBox + ?Sized> ToolBox for Arc<T> {
    fn tools(&self) -> Vec<Tool> {
        self.as_ref().tools()
    }

    fn invoke(&self, call: ToolUse) -> BoxFuture<'_, Result<crate::content::Part, ToolError>> {
        self.as_ref().invoke(call)
    }

    fn invoke_with_hooks(&self, call: ToolUse, hooks: ToolHooks) -> BoxFuture<'_, Result<crate::content::Part, ToolError>> {
        self.as_ref().invoke_with_hooks(call, hooks)
    }

    fn dangerous_functions(&self) -> &[&str] {
        self.as_ref().dangerous_functions()
    }

    fn has_function(&self, name: &str) -> bool {
        self.as_ref().has_function(name)
    }
}

impl From<Box<dyn ToolBox>> for Vec<Tool> {
    fn from(toolbox: Box<dyn ToolBox>) -> Self {
        toolbox.tools()
    }
}

/// Generates a JSON schema for the given type using schemars.
///
/// This function configures schemars to generate schemas compatible with
/// JSON Schema Draft 2020-12 and optimizes them for use in AI tool definitions.
///
/// # Panics
///
/// Panics if the schema cannot be serialized to JSON or if the resulting
/// value is not a JSON object.
#[must_use]
pub fn schema_for_type<T: JsonSchema>() -> Value {
    let settings = SchemaSettings::openapi3().with(|s| {
        s.inline_subschemas = true;
        s.meta_schema = None;
    });
    let generator = schemars::generate::SchemaGenerator::new(settings);
    let root_schema = generator.into_root_schema_for::<T>();
    let mut schema_value =
        serde_json::to_value(root_schema).expect("Failed to serialize schema to JSON");

    // Remove the title field if present
    if let Some(obj) = schema_value.as_object_mut() {
        obj.remove("title");
    }

    schema_value
}
