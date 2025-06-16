pub mod call;
pub mod error;
pub mod result;
pub mod set;

pub use call::ToolCall;
pub use error::ToolError;
pub use result::ToolResult;
pub use set::ToolSet;

use futures_util::future::BoxFuture;
use schemars::{JsonSchema, generate::SchemaSettings};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    // Other tool types can be added here in the future
    // (e.g., GoogleSearchRetrieval, CodeExecution, etc.)
}

/// Trait for objects that provide tool functionality.
///
/// This trait allows objects to expose their available tools and handle
/// tool invocations in a standardized way.
pub trait ToolBox: Send + Sync + std::fmt::Debug {
    /// Returns the list of tools provided by this toolbox.
    fn tools(&self) -> Vec<Tool>;

    /// Invokes a tool function with the given call parameters.
    ///
    /// Returns a boxed future that resolves to either a ToolResult on success
    /// or a ToolError on failure.
    fn invoke(&self, call: ToolCall) -> BoxFuture<Result<ToolResult, ToolError>>;

    /// Checks if this toolbox has a function with the given name.
    fn has_function(&self, name: &str) -> bool {
        self.tools().iter().any(|tool| {
            match tool {
                Tool::FunctionDeclarations(functions) => {
                    functions.iter().any(|func| func.name == name)
                } // Add other tool types here as needed
            }
        })
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
    let settings = SchemaSettings::draft2020_12().with(|s| {
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

    // Ensure we have proper properties or return empty object schema
    if schema_value
        .get("properties")
        .is_some_and(|p| !p.is_null() && p.as_object().is_some_and(|o| !o.is_empty()))
    {
        schema_value
    } else {
        // Return an empty object schema for no-input functions
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }
}
