pub mod config;
pub mod error;
pub mod google;

// Integration tests for macros were moved inside mod tests below
// #[cfg(test)]
// mod macros_integration_test;

use crate::content::{FunctionCall, FunctionResponse};
use crate::tool::error::FunctionCallError;

use futures_util::future::BoxFuture;
use google::{GoogleSearch, GoogleSearchRetrieval};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use std::fmt::Debug;

// Re-export commonly used types

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FunctionMetadata {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Tool {
    FunctionDeclarations(Vec<FunctionMetadata>),
    GoogleSearchRetrieval {
        google_search_retrieval: GoogleSearchRetrieval,
    },
    CodeExecution {
        #[serde(flatten)]
        inner: Value,
    },
    GoogleSearch(GoogleSearch),
}

/// Generates a `serde_json::Map` representing the JSON schema for type `T`.
///
/// # Panics
///
/// Panics if the generated schema cannot be serialized to a `serde_json::Value`
/// or if the resulting value is not a JSON object.
#[must_use]
pub fn schema_for_type<T: JsonSchema>() -> Value {
    let settings = schemars::generate::SchemaSettings::openapi3().with(|s| {
        s.inline_subschemas = true;
        s.meta_schema = None;
    });
    let r#gen = schemars::generate::SchemaGenerator::new(settings);
    let json_schema = r#gen.into_root_schema_for::<T>();
    let mut input_schema = serde_json::to_value(json_schema).unwrap();
    // Don't panic if title is already removed or doesn't exist
    if let Some(obj) = input_schema.as_object_mut() {
        obj.remove("title");
    }
    // Ensure properties exist before checking if it's empty or null
    if input_schema.get("properties").map_or_else(
        || false,
        |p| !p.is_null() && p.as_object().map_or_else(|| false, |o| !o.is_empty()),
    ) {
        input_schema
    } else {
        // Return an empty object schema if there are no properties,
        // matching common JSON Schema practices for no-input functions.
        serde_json::json!({ "type": "object", "properties": {} })
    }
}

// /// Generates and caches (thread-locally) the JSON schema for type `T`.
// ///
// /// Subsequent calls for the same type `T` within the same thread will return
// /// a cloned `Arc` to the cached schema.
// ///
// /// # Panics
// ///
// /// - Panics if the schema generation or serialization fails (see `schema_for_type`).
// /// - Panics if the thread-local cache's `RwLock` is poisoned.
// #[must_use]
// pub fn cached_schema_for_type<T: JsonSchema + std::any::Any>() -> Arc<serde_json::Map<String, Value>>
// {
//     thread_local! {
//         static CACHE_FOR_TYPE: std::sync::RwLock<HashMap<TypeId, Arc<serde_json::Map<String, Value>>>> = std::sync::RwLock::default();
//     };
//     CACHE_FOR_TYPE.with(|cache| {
//         if let Some(x) = cache
//             .read()
//             .expect("schema cache lock poisoned")
//             .get(&TypeId::of::<T>())
//         {
//             Arc::clone(x)
//         } else {
//             let schema = schema_for_type::<T>();
//             let schema = Arc::new(schema);
//             cache
//                 .write()
//                 .expect("schema cache lock poisoned")
//                 .insert(TypeId::of::<T>(), Arc::clone(&schema));
//             schema
//         }
//     })
// }

pub trait ToolBox: Send + Sync + 'static {
    // Use BoxFuture for async method in trait
    fn invoke(
        &self,
        fn_call: FunctionCall,
    ) -> BoxFuture<'_, Result<FunctionResponse, FunctionCallError>>;

    fn has_function(&self, name: &str) -> bool {
        // Check through all tool declarations to find matching function name
        self.tools().iter().any(|declaration| {
            match declaration {
                Tool::FunctionDeclarations(function_declarations) => {
                    // Check if function name exists in this declaration's functions
                    function_declarations.iter().any(|info| info.name == name)
                }
                // Other tool declaration types don't contain function declarations
                _ => false,
            }
        })
    }

    #[must_use]
    fn tools(&self) -> Vec<Tool>;
}

pub trait AsTools {
    fn as_tools(&self) -> Vec<Tool>;
}

impl<T> AsTools for T
where
    T: ToolBox,
{
    fn as_tools(&self) -> Vec<Tool> {
        self.tools()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*; // Import items from the outer module (tool)
//     use crate::toolbox; // Import the re-exported macro
//     use futures_util::future::BoxFuture; // Added for manual async impl
//     use std::sync::{Arc, Mutex};

//     // --- #[toolbox] Integration Tests ---

//     // Helper structs and types for macro tests
//     #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
//     struct SimpleInput {
//         value: i32,
//         label: String,
//     }

//     #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
//     struct OptionalInput {
//         data: Option<String>,
//     }

//     #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
//     struct SimpleOutput {
//         result: String,
//     }

//     #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, thiserror::Error)]
//     enum TestToolError {
//         #[error("Something went wrong: {0}")]
//         Processing(String),
//         #[error("Invalid input: {0}")]
//         BadInput(String),
//         // Ensure the From<FunctionCallError> variant can be serialized/deserialized if needed,
//         // although typically only E needs to be Serialize for ExecutionFailed.
//         // FunctionCallError itself isn't typically serialized directly by the user part.
//         #[error("Framework error: {source}")]
//         Framework {
//             #[from]
//             source: FunctionCallError,
//         },
//     }

//     // The service struct that will have tools via the macro
//     #[derive(Clone, Debug)]
//     struct MyToolService {
//         prefix: String,
//         call_count: Arc<Mutex<usize>>, // Example stateful component
//     }

//     impl MyToolService {
//         #[allow(dead_code)]
//         fn new(prefix: &str) -> Self {
//             MyToolService {
//                 prefix: prefix.to_string(),
//                 call_count: Arc::new(Mutex::new(0)),
//             }
//         }
//         // Helper to increment count safely
//         fn increment_call_count(&self) {
//             let mut count = self.call_count.lock().expect("Mutex poisoned");
//             *count += 1;
//         }
//         // Helper to read count safely
//         #[allow(dead_code)]
//         fn get_call_count(&self) -> usize {
//             *self.call_count.lock().expect("Mutex poisoned")
//         }
//     }

//     // Apply the toolbox macro (use crate::toolbox because we are inside the crate)
//     #[toolbox] // Using the re-exported macro
//     impl MyToolService {
//         /// Adds a prefix and label to the input value (async).
//         #[allow(clippy::unused_async)]
//         pub async fn async_tool(&self, input: SimpleInput) -> Result<SimpleOutput, TestToolError> {
//             self.increment_call_count();
//             if input.value < 0 {
//                 Err(TestToolError::BadInput(
//                     "Value cannot be negative".to_string(),
//                 ))
//             } else {
//                 let result = format!(
//                     "{}: Async processed value {} with label '{}'",
//                     self.prefix, input.value, input.label
//                 );
//                 Ok(SimpleOutput { result })
//             }
//         }
//         /// Processes optional input synchronously.
//         #[allow(clippy::unnecessary_wraps)]
//         pub fn sync_optional_tool(
//             &self,
//             input: Option<OptionalInput>,
//         ) -> Result<SimpleOutput, TestToolError> {
//             self.increment_call_count();
//             // Correctly handle Option<OptionalInput> which might contain Option<String>
//             let data_str = input
//                 .and_then(|opt_input| opt_input.data)
//                 .unwrap_or_else(|| "default".to_string());
//             let result = format!(
//                 "{}: Sync processed optional data '{}'",
//                 self.prefix, data_str
//             );
//             Ok(SimpleOutput { result })
//         }

//         /// A synchronous tool.
//         #[allow(clippy::unnecessary_wraps)]
//         pub fn sync_tool(&self, input: SimpleInput) -> Result<SimpleOutput, TestToolError> {
//             self.increment_call_count();
//             let result = format!(
//                 "{}: Sync processed value {} with label '{}'",
//                 self.prefix, input.value, input.label
//             );
//             Ok(SimpleOutput { result })
//         }

//         /// A tool that takes no input.
//         #[allow(clippy::unnecessary_wraps)]
//         pub fn no_input_tool(&self) -> Result<SimpleOutput, TestToolError> {
//             self.increment_call_count();
//             Ok(SimpleOutput {
//                 result: format!("{}: No input tool called", self.prefix),
//             })
//         }

//         // This is private - should be ignored by macro
//         #[allow(dead_code)]
//         async fn private_helper(&self, _input: SimpleInput) -> Result<(), TestToolError> {
//             self.increment_call_count(); // Should not be called
//             Ok(())
//         }
//     }

//     // #[tokio::test]
//     // async fn test_toolbox_direct_invocation() {
//     //     // Test invocation using the higher-level `toolbox.invoke` method
//     //     let service = MyToolService::new("InvokeDirect");
//     //     // Get the toolbox using the trait impl generated by the macro
//     //     let initial_call_count = service.get_call_count();
//     //     assert_eq!(initial_call_count, 0);

//     //     // --- Test async_tool via invoke ---
//     //     // Successful call
//     //     let result_async_ok = service
//     //         .invoke(FunctionCall {
//     //             name: "async_tool".to_string(),
//     //             args: Some(json!({ "value": 20, "label": "TestInvoke" })),
//     //         })
//     //         .await;
//     //     assert!(
//     //         result_async_ok.is_ok(),
//     //         "async_tool invoke failed: {:?}",
//     //         result_async_ok
//     //     );
//     //     let output_async_ok: SimpleOutput =
//     //         serde_json::from_value(result_async_ok.unwrap()).unwrap();
//     //     assert_eq!(
//     //         output_async_ok.result,
//     //         "InvokeDirect: Async processed value 20 with label 'TestInvoke'"
//     //     );
//     //     assert_eq!(
//     //         service.get_call_count(),
//     //         initial_call_count + 1,
//     //         "Call count mismatch after async_tool invoke success"
//     //     );

//     //     // Call resulting in user error
//     //     let result_async_user_err = service
//     //         .invoke(FunctionCall {
//     //             name: "async_tool".to_string(),
//     //             args: Some(json!({ "value": -1, "label": "FailInvoke" })),
//     //         })
//     //         .await;
//     //     dbg!(&result_async_user_err);
//     //     assert!(
//     //         result_async_user_err.is_err(),
//     //         "async_tool invoke should fail with user error"
//     //     );
//     //     assert_eq!(
//     //         service.get_call_count(),
//     //         initial_call_count + 2,
//     //         "Call count mismatch after async_tool invoke user error"
//     //     );
//     //     match result_async_user_err.unwrap_err() {
//     //         FunctionCallError::ExecutionFailed(e_str) => {
//     //             // The error string includes context about which function failed.
//     //             assert_eq!(
//     //                 e_str,
//     //                 "Function 'async_tool' execution failed: Invalid input: Value cannot be negative"
//     //             );
//     //         }
//     //         other_error => panic!("Unexpected error type: {:?}", other_error),
//     //     }

//     //     // Call with invalid input structure (wrong type)
//     //     let result_async_invalid_type = service
//     //         .invoke(FunctionCall {
//     //             name: "async_tool".to_string(),
//     //             args: Some(json!("not an object")),
//     //         })
//     //         .await;
//     //     assert!(
//     //         result_async_invalid_type.is_err(),
//     //         "async_tool invoke should fail with invalid input type"
//     //     );
//     //     assert_eq!(
//     //         service.get_call_count(),
//     //         initial_call_count + 2,
//     //         "Call count should not increment on deserialization error"
//     //     );
//     //     assert!(matches!(
//     //         result_async_invalid_type.unwrap_err(),
//     //         FunctionCallError::InputDeserializationFailed(_)
//     //     ));

//     //     // Call with missing required arguments (value field)
//     //     let result_async_missing_field = service
//     //         .invoke(FunctionCall {
//     //             name: "async_tool".to_string(),
//     //             args: Some(json!({"label": "MissingValue"})), // Missing 'value'
//     //         })
//     //         .await;
//     //     assert!(
//     //         result_async_missing_field.is_err(),
//     //         "async_tool invoke should fail with missing required field"
//     //     );
//     //     assert_eq!(
//     //         service.get_call_count(),
//     //         initial_call_count + 2,
//     //         "Call count should not increment on deserialization error (missing field)"
//     //     );
//     //     assert!(matches!(
//     //         result_async_missing_field.unwrap_err(),
//     //         FunctionCallError::InputDeserializationFailed(_)
//     //     ));

//     //     // Call with args: None when input is required
//     //     let result_async_none_args = service
//     //         .invoke(FunctionCall {
//     //             name: "async_tool".to_string(),
//     //             args: None, // SimpleInput is required, so None should fail deserialization
//     //         })
//     //         .await;
//     //     dbg!(&result_async_none_args);
//     //     assert!(
//     //         result_async_none_args.is_err(),
//     //         "async_tool invoke should fail when args is None for required input"
//     //     );
//     //     assert_eq!(
//     //         service.get_call_count(),
//     //         initial_call_count + 2,
//     //         "Call count should not increment on deserialization error (None args)"
//     //     );
//     //     assert!(matches!(
//     //         result_async_none_args.unwrap_err(),
//     //         // The tool should fail during input deserialization when args is None but input is required.
//     //         FunctionCallError::InputDeserializationFailed(_)
//     //     ));

//     //     // --- Test sync_optional_tool via invoke ---
//     //     // Call with Some(valid_input)
//     //     let result_sync_opt_some = service
//     //         .invoke(FunctionCall {
//     //             name: "sync_optional_tool".to_string(),
//     //             args: Some(json!({ "data": "OptionalData" })),
//     //         })
//     //         .await;
//     //     assert!(
//     //         result_sync_opt_some.is_ok(),
//     //         "sync_optional_tool invoke with Some failed: {:?}",
//     //         result_sync_opt_some
//     //     );
//     //     let output_sync_opt_some: SimpleOutput =
//     //         serde_json::from_value(result_sync_opt_some.unwrap()).unwrap();
//     //     assert_eq!(
//     //         output_sync_opt_some.result,
//     //         "InvokeDirect: Sync processed optional data 'OptionalData'"
//     //     );
//     //     assert_eq!(
//     //         service.get_call_count(),
//     //         initial_call_count + 3,
//     //         "Call count mismatch after sync_opt_tool invoke Some"
//     //     );

//     //     // Call with Some({}) representing OptionalInput { data: None }
//     //     let result_sync_opt_some_none_data = service
//     //         .invoke(FunctionCall {
//     //             name: "sync_optional_tool".to_string(),
//     //             args: Some(json!({})), // Valid OptionalInput, data will be None
//     //         })
//     //         .await;
//     //     assert!(
//     //         result_sync_opt_some_none_data.is_ok(),
//     //         "sync_optional_tool invoke with Some {{}} failed: {:?}",
//     //         result_sync_opt_some_none_data
//     //     );
//     //     let output_sync_opt_some_none_data: SimpleOutput =
//     //         serde_json::from_value(result_sync_opt_some_none_data.unwrap()).unwrap();
//     //     assert_eq!(
//     //         output_sync_opt_some_none_data.result,
//     //         "InvokeDirect: Sync processed optional data 'default'"
//     //     );
//     //     assert_eq!(
//     //         service.get_call_count(),
//     //         initial_call_count + 4,
//     //         "Call count mismatch after sync_opt_tool invoke Some{{}}"
//     //     );

//     //     // Call with args: None (tool takes Option<OptionalInput>, so this is valid)
//     //     let result_sync_opt_none_args = service
//     //         .invoke(FunctionCall {
//     //             name: "sync_optional_tool".to_string(),
//     //             args: None, // Valid because the function takes Option<I>
//     //         })
//     //         .await;
//     //     assert!(
//     //         result_sync_opt_none_args.is_ok(),
//     //         "sync_optional_tool invoke with None args failed: {:?}",
//     //         result_sync_opt_none_args
//     //     );
//     //     let output_sync_opt_none_args: SimpleOutput =
//     //         serde_json::from_value(result_sync_opt_none_args.unwrap()).unwrap();
//     //     assert_eq!(
//     //         output_sync_opt_none_args.result,
//     //         "InvokeDirect: Sync processed optional data 'default'"
//     //     );
//     //     assert_eq!(
//     //         service.get_call_count(),
//     //         initial_call_count + 5,
//     //         "Call count mismatch after sync_opt_tool invoke None args"
//     //     );

//     //     // Call with invalid internal type
//     //     let result_sync_opt_invalid_type = service
//     //         .invoke(FunctionCall {
//     //             name: "sync_optional_tool".to_string(),
//     //             args: Some(json!({ "data": 123 })), // Invalid type for data
//     //         })
//     //         .await;
//     //     assert!(
//     //         result_sync_opt_invalid_type.is_err(),
//     //         "sync_optional_tool invoke should fail for invalid internal type"
//     //     );
//     //     assert_eq!(
//     //         service.get_call_count(),
//     //         initial_call_count + 5,
//     //         "Call count should not increment on deserialization error (optional)"
//     //     );
//     //     assert!(matches!(
//     //         result_sync_opt_invalid_type.unwrap_err(),
//     //         FunctionCallError::InputDeserializationFailed(_)
//     //     ));

//     //     // --- Test sync_tool via invoke --- (Should behave like async_tool)
//     //     let result_sync_ok = service
//     //         .invoke(FunctionCall {
//     //             name: "sync_tool".to_string(),
//     //             args: Some(json!({ "value": 30, "label": "TestSyncInvoke" })),
//     //         })
//     //         .await;
//     //     assert!(
//     //         result_sync_ok.is_ok(),
//     //         "sync_tool invoke failed: {:?}",
//     //         result_sync_ok
//     //     );
//     //     let output_sync_ok: SimpleOutput = serde_json::from_value(result_sync_ok.unwrap()).unwrap();
//     //     assert_eq!(
//     //         output_sync_ok.result,
//     //         "InvokeDirect: Sync processed value 30 with label 'TestSyncInvoke'"
//     //     );
//     //     assert_eq!(
//     //         service.get_call_count(),
//     //         initial_call_count + 6,
//     //         "Call count mismatch after sync_tool invoke success"
//     //     );

//     //     // Test sync_tool with args: None (should fail deserialization)
//     //     let result_sync_none_args = service
//     //         .invoke(FunctionCall {
//     //             name: "sync_tool".to_string(),
//     //             args: None,
//     //         })
//     //         .await;
//     //     assert!(
//     //         result_sync_none_args.is_err(),
//     //         "sync_tool invoke should fail when args is None for required input"
//     //     );
//     //     assert_eq!(
//     //         service.get_call_count(),
//     //         initial_call_count + 6,
//     //         "Call count mismatch after sync_tool invoke None args"
//     //     );
//     //     assert!(matches!(
//     //         result_sync_none_args.unwrap_err(),
//     //         // The tool should fail during input deserialization when args is None but input is required.
//     //         FunctionCallError::InputDeserializationFailed(_)
//     //     ));

//     //     // --- Test no_input_tool via invoke ---
//     //     // Call with args: None (Valid, as function takes no args, wrapper expects Option<()>)
//     //     let result_no_input_none = service
//     //         .invoke(FunctionCall {
//     //             name: "no_input_tool".to_string(),
//     //             args: None,
//     //         })
//     //         .await;
//     //     assert!(
//     //         result_no_input_none.is_ok(),
//     //         "no_input_tool invoke with None args failed: {:?}",
//     //         result_no_input_none
//     //     );
//     //     let output_no_input_none: SimpleOutput =
//     //         serde_json::from_value(result_no_input_none.unwrap()).unwrap();
//     //     assert_eq!(
//     //         output_no_input_none.result,
//     //         "InvokeDirect: No input tool called"
//     //     );
//     //     assert_eq!(
//     //         service.get_call_count(),
//     //         initial_call_count + 7,
//     //         "Call count mismatch after no_input_tool invoke None args"
//     //     );

//     //     // --- Test tool not found via invoke ---
//     //     let result_not_found = service
//     //         .invoke(FunctionCall {
//     //             name: "non_existent_tool".to_string(),
//     //             args: None,
//     //         })
//     //         .await;
//     //     assert!(
//     //         result_not_found.is_err(),
//     //         "Invoke should fail for non-existent tool"
//     //     );
//     //     assert_eq!(
//     //         service.get_call_count(),
//     //         initial_call_count + 7, // Updated expected count after removing a test case
//     //         "Call count should not increment for tool not found"
//     //     );
//     //     assert!(matches!(
//     //         result_not_found.unwrap_err(),
//     //         FunctionCallError::ToolNotFound(name) if name == "non_existent_tool"
//     //     ));
//     // }
// } // End of mod tests
