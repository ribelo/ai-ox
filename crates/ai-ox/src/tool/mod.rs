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
/// tool invocations in a standardized way. Any additional safety or approval
/// flows should now be implemented by library consumers around these calls.
pub trait ToolBox: Send + Sync + 'static {
    /// Returns the list of tools provided by this toolbox.
    fn tools(&self) -> Vec<Tool>;

    /// Invokes a tool function with the given call parameters.
    ///
    /// Returns a boxed future that resolves to either a Part::ToolResult on success
    /// or a ToolError on failure.
    fn invoke(&self, call: ToolUse) -> BoxFuture<'_, Result<crate::content::Part, ToolError>>;

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
