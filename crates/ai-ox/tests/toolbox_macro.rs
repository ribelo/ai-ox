use ai_ox::content::part::Part;
use ai_ox::tool::{Tool, ToolBox, ToolCall, ToolError};
use ai_ox::toolbox;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

// Test data types for toolbox testing

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct SimpleInput {
    value: i32,
    label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct OptionalInput {
    data: Option<String>,
    count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct ComplexInput {
    metadata: SimpleInput,
    tags: Vec<String>,
    enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct SimpleOutput {
    result: String,
    timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct ComplexOutput {
    processed_data: SimpleOutput,
    status: String,
    count: i32,
}

// Test error type
#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
pub enum TestToolError {
    #[error("Processing failed: {message}")]
    ProcessingFailed { message: String },

    #[error("Invalid input: {field} - {reason}")]
    InvalidInput { field: String, reason: String },

    #[error("Resource not found: {resource}")]
    NotFound { resource: String },
}

// Test service struct with state tracking
#[derive(Debug, Clone)]
struct TestToolService {
    name: String,
    call_count: Arc<Mutex<usize>>,
    last_operation: Arc<Mutex<String>>,
}

impl TestToolService {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            call_count: Arc::new(Mutex::new(0)),
            last_operation: Arc::new(Mutex::new(String::new())),
        }
    }

    fn increment_counter(&self, operation: &str) {
        let mut count = self.call_count.lock().unwrap();
        *count += 1;

        let mut last_op = self.last_operation.lock().unwrap();
        *last_op = operation.to_string();
    }

    fn get_call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }

    fn get_last_operation(&self) -> String {
        self.last_operation.lock().unwrap().clone()
    }
}

// Apply the toolbox macro to the test service
#[toolbox]
impl TestToolService {
    pub fn side_effect_tool(&self) {
        println!("Side effect tool called");
    }
    pub fn infailable_tool(&self) -> String {
        "Success".to_string()
    }
    /// A simple synchronous tool that processes basic input
    pub fn simple_sync_tool(&self, input: SimpleInput) -> Result<SimpleOutput, TestToolError> {
        self.increment_counter("simple_sync_tool");

        if input.value < 0 {
            return Err(TestToolError::InvalidInput {
                field: "value".to_string(),
                reason: "cannot be negative".to_string(),
            });
        }

        Ok(SimpleOutput {
            result: format!(
                "{}: Processed {} with label '{}'",
                self.name, input.value, input.label
            ),
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// An asynchronous tool that processes input with delay simulation
    pub async fn simple_async_tool(
        &self,
        input: SimpleInput,
    ) -> Result<SimpleOutput, TestToolError> {
        self.increment_counter("simple_async_tool");

        // Simulate async work
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;

        if input.label.is_empty() {
            return Err(TestToolError::InvalidInput {
                field: "label".to_string(),
                reason: "cannot be empty".to_string(),
            });
        }

        Ok(SimpleOutput {
            result: format!(
                "{}: Async processed {} with label '{}'",
                self.name, input.value, input.label
            ),
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// A tool that takes optional input parameters
    pub fn optional_input_tool(
        &self,
        input: Option<OptionalInput>,
    ) -> Result<SimpleOutput, TestToolError> {
        self.increment_counter("optional_input_tool");

        let data = match input {
            Some(opt_input) => {
                let data_str = opt_input.data.unwrap_or_else(|| "default".to_string());
                format!("data: '{}', count: {}", data_str, opt_input.count)
            }
            None => "no input provided".to_string(),
        };

        Ok(SimpleOutput {
            result: format!("{}: Optional tool result - {}", self.name, data),
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// A tool that takes no input parameters
    pub fn no_input_tool(&self) -> Result<SimpleOutput, TestToolError> {
        self.increment_counter("no_input_tool");

        Ok(SimpleOutput {
            result: format!("{}: No input tool executed", self.name),
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// A tool with complex input/output types
    pub async fn complex_tool(&self, input: ComplexInput) -> Result<ComplexOutput, TestToolError> {
        self.increment_counter("complex_tool");

        if !input.enabled {
            return Err(TestToolError::ProcessingFailed {
                message: "tool is disabled".to_string(),
            });
        }

        let simple_result = SimpleOutput {
            result: format!(
                "Complex processing: {} tags, metadata value: {}",
                input.tags.len(),
                input.metadata.value
            ),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        Ok(ComplexOutput {
            processed_data: simple_result,
            status: "completed".to_string(),
            count: input.tags.len() as i32,
        })
    }

    /// A tool that always fails for testing error handling
    pub fn failing_tool(&self, _input: SimpleInput) -> Result<SimpleOutput, TestToolError> {
        self.increment_counter("failing_tool");

        Err(TestToolError::NotFound {
            resource: "required_data".to_string(),
        })
    }

    // Private method - should not be included in toolbox
    #[allow(dead_code)]
    fn private_helper(&self, _input: SimpleInput) -> Result<(), TestToolError> {
        Ok(())
    }
}

// Basic functionality tests
#[tokio::test]
async fn test_toolbox_sync_tool_success() {
    let service = TestToolService::new("TestService");
    let initial_count = service.get_call_count();

    let call = ToolCall::new(
        "test_call_1",
        "simple_sync_tool",
        json!({
            "value": 42,
            "label": "test_label"
        }),
    );

    let result = service.invoke(call).await;
    assert!(result.is_ok(), "Tool invocation should succeed");

    let tool_result = result.unwrap();
    assert_eq!(tool_result.id, "test_call_1");
    assert_eq!(tool_result.name, "simple_sync_tool");
    assert_eq!(tool_result.response.len(), 1);

    // Verify state was updated
    assert_eq!(service.get_call_count(), initial_count + 1);
    assert_eq!(service.get_last_operation(), "simple_sync_tool");
}

#[tokio::test]
async fn test_toolbox_async_tool_success() {
    let service = TestToolService::new("AsyncTest");

    let call = ToolCall::new(
        "async_call_1",
        "simple_async_tool",
        json!({
            "value": 100,
            "label": "async_test"
        }),
    );

    let result = service.invoke(call).await;
    assert!(result.is_ok());

    let tool_result = result.unwrap();
    assert_eq!(tool_result.id, "async_call_1");
    assert_eq!(tool_result.name, "simple_async_tool");
    assert_eq!(service.get_last_operation(), "simple_async_tool");
}

#[tokio::test]
async fn test_toolbox_no_input_tool() {
    let service = TestToolService::new("NoInputTest");

    let call = ToolCall::new("no_input_call", "no_input_tool", json!({}));

    let result = service.invoke(call).await;
    assert!(result.is_ok());

    let tool_result = result.unwrap();
    assert_eq!(tool_result.name, "no_input_tool");
    assert_eq!(service.get_last_operation(), "no_input_tool");
}

#[tokio::test]
async fn test_toolbox_optional_input_with_data() {
    let service = TestToolService::new("OptionalTest");

    let call = ToolCall::new(
        "opt_call_1",
        "optional_input_tool",
        json!({
            "data": "test_data",
            "count": 5
        }),
    );

    let result = service.invoke(call).await;
    assert!(result.is_ok());

    let tool_result = result.unwrap();
    assert_eq!(tool_result.name, "optional_input_tool");
}

#[tokio::test]
async fn test_toolbox_optional_input_none() {
    let service = TestToolService::new("OptionalNoneTest");

    let call = ToolCall::new("opt_none_call", "optional_input_tool", Value::Null);

    let result = service.invoke(call).await;
    assert!(result.is_ok());

    let tool_result = result.unwrap();
    assert_eq!(tool_result.name, "optional_input_tool");
}

#[tokio::test]
async fn test_toolbox_complex_tool_success() {
    let service = TestToolService::new("ComplexTest");

    let call = ToolCall::new(
        "complex_call",
        "complex_tool",
        json!({
            "metadata": {
                "value": 123,
                "label": "metadata_label"
            },
            "tags": ["tag1", "tag2", "tag3"],
            "enabled": true
        }),
    );

    let result = service.invoke(call).await;
    assert!(result.is_ok());

    let tool_result = result.unwrap();
    assert_eq!(tool_result.name, "complex_tool");
}

// Error handling tests
#[tokio::test]
async fn test_toolbox_user_error_handling() {
    let service = TestToolService::new("ErrorTest");

    let call = ToolCall::new(
        "error_call",
        "simple_sync_tool",
        json!({
            "value": -1,  // This should trigger an error
            "label": "error_test"
        }),
    );

    let result = service.invoke(call).await;
    assert!(result.is_err());

    let error = result.unwrap_err();
    match error {
        ToolError::Execution { name, error } => {
            assert_eq!(name, "simple_sync_tool");
            assert!(error.to_string().contains("Invalid input"));
        }
        _ => panic!("Expected execution error"),
    }
}

#[tokio::test]
async fn test_toolbox_input_deserialization_error() {
    let service = TestToolService::new("DeserializeTest");

    let call = ToolCall::new(
        "bad_input_call",
        "simple_sync_tool",
        json!("invalid_input_type"), // String instead of object
    );

    let result = service.invoke(call).await;
    assert!(result.is_err());

    let error = result.unwrap_err();
    match error {
        ToolError::InputDeserialization { name, .. } => {
            assert_eq!(name, "simple_sync_tool");
        }
        _ => panic!("Expected input deserialization error"),
    }
}

#[tokio::test]
async fn test_toolbox_missing_field_error() {
    let service = TestToolService::new("MissingFieldTest");

    let call = ToolCall::new(
        "missing_field_call",
        "simple_sync_tool",
        json!({
            "value": 42
            // Missing "label" field
        }),
    );

    let result = service.invoke(call).await;
    assert!(result.is_err());

    match result.unwrap_err() {
        ToolError::InputDeserialization { .. } => {
            // Expected
        }
        _ => panic!("Expected input deserialization error for missing field"),
    }
}

#[tokio::test]
async fn test_toolbox_tool_not_found() {
    let service = TestToolService::new("NotFoundTest");

    let call = ToolCall::new("not_found_call", "non_existent_tool", json!({}));

    let result = service.invoke(call).await;
    assert!(result.is_err());

    let error = result.unwrap_err();
    match error {
        ToolError::NotFound { name } => {
            assert_eq!(name, "non_existent_tool");
        }
        _ => panic!("Expected not found error"),
    }
}

#[tokio::test]
async fn test_toolbox_failing_tool() {
    let service = TestToolService::new("FailTest");

    let call = ToolCall::new(
        "fail_call",
        "failing_tool",
        json!({
            "value": 1,
            "label": "test"
        }),
    );

    let result = service.invoke(call).await;
    assert!(result.is_err());

    match result.unwrap_err() {
        ToolError::Execution { name, error } => {
            assert_eq!(name, "failing_tool");
            assert!(error.to_string().contains("Resource not found"));
        }
        _ => panic!("Expected execution error"),
    }
}

// ToolBox trait implementation tests
#[tokio::test]
async fn test_toolbox_tools_method() {
    let service = TestToolService::new("ToolsTest");
    let tools = service.tools();

    assert_eq!(tools.len(), 1);

    if let Tool::FunctionDeclarations(functions) = &tools[0] {
        assert_eq!(functions.len(), 8); // 8 public methods should be included

        let function_names: Vec<&str> = functions.iter().map(|f| f.name.as_str()).collect();
        assert!(function_names.contains(&"side_effect_tool"));
        assert!(function_names.contains(&"infailable_tool"));
        assert!(function_names.contains(&"simple_sync_tool"));
        assert!(function_names.contains(&"simple_async_tool"));
        assert!(function_names.contains(&"optional_input_tool"));
        assert!(function_names.contains(&"no_input_tool"));
        assert!(function_names.contains(&"complex_tool"));
        assert!(function_names.contains(&"failing_tool"));

        // Private method should not be included
        assert!(!function_names.contains(&"private_helper"));

        // Check that descriptions are included
        let sync_tool = functions
            .iter()
            .find(|f| f.name == "simple_sync_tool")
            .unwrap();
        assert!(sync_tool.description.is_some());
        assert!(
            sync_tool
                .description
                .as_ref()
                .unwrap()
                .contains("synchronous tool")
        );
    } else {
        panic!("Expected FunctionDeclarations tool type");
    }
}

#[tokio::test]
async fn test_toolbox_has_function() {
    let service = TestToolService::new("HasFunctionTest");

    // Should find existing functions
    assert!(service.has_function("side_effect_tool"));
    assert!(service.has_function("infailable_tool"));
    assert!(service.has_function("simple_sync_tool"));
    assert!(service.has_function("simple_async_tool"));
    assert!(service.has_function("optional_input_tool"));
    assert!(service.has_function("no_input_tool"));
    assert!(service.has_function("complex_tool"));
    assert!(service.has_function("failing_tool"));

    // Should not find non-existent or private functions
    assert!(!service.has_function("non_existent_tool"));
    assert!(!service.has_function("private_helper"));
}

// Schema generation tests
#[tokio::test]
async fn test_toolbox_schema_generation() {
    let service = TestToolService::new("SchemaTest");
    let tools = service.tools();

    if let Tool::FunctionDeclarations(functions) = &tools[0] {
        // Test simple input schema
        let simple_sync_tool = functions
            .iter()
            .find(|f| f.name == "simple_sync_tool")
            .unwrap();
        let schema = &simple_sync_tool.parameters;

        // Should have properties for SimpleInput
        assert!(schema.get("properties").is_some());
        let properties = schema.get("properties").unwrap().as_object().unwrap();
        assert!(properties.contains_key("value"));
        assert!(properties.contains_key("label"));

        // Test no-input schema
        let no_input_tool = functions
            .iter()
            .find(|f| f.name == "no_input_tool")
            .unwrap();
        let no_input_schema = &no_input_tool.parameters;

        // Should be empty object schema
        assert_eq!(no_input_schema.get("type").unwrap(), "object");
        let empty_properties = no_input_schema
            .get("properties")
            .unwrap()
            .as_object()
            .unwrap();
        assert!(empty_properties.is_empty());

        // Test complex input schema
        let complex_tool = functions.iter().find(|f| f.name == "complex_tool").unwrap();
        let complex_schema = &complex_tool.parameters;

        let complex_properties = complex_schema
            .get("properties")
            .unwrap()
            .as_object()
            .unwrap();
        assert!(complex_properties.contains_key("metadata"));
        assert!(complex_properties.contains_key("tags"));
        assert!(complex_properties.contains_key("enabled"));
    }
}

// Integration tests combining multiple aspects
#[tokio::test]
async fn test_toolbox_full_workflow() {
    let service = TestToolService::new("WorkflowTest");
    let initial_count = service.get_call_count();

    // Test multiple tool invocations
    let calls = vec![
        ToolCall::new(
            "call1",
            "simple_sync_tool",
            json!({"value": 1, "label": "first"}),
        ),
        ToolCall::new(
            "call2",
            "simple_async_tool",
            json!({"value": 2, "label": "second"}),
        ),
        ToolCall::new("call3", "no_input_tool", json!({})),
        ToolCall::new(
            "call4",
            "optional_input_tool",
            json!({"data": "test", "count": 3}),
        ),
    ];

    let mut results = Vec::new();
    for call in calls {
        let result = service.invoke(call).await;
        assert!(result.is_ok());
        results.push(result.unwrap());
    }

    // Verify all calls were successful
    assert_eq!(results.len(), 4);
    assert_eq!(service.get_call_count(), initial_count + 4);

    // Verify correct tool names in results
    assert_eq!(results[0].name, "simple_sync_tool");
    assert_eq!(results[1].name, "simple_async_tool");
    assert_eq!(results[2].name, "no_input_tool");
    assert_eq!(results[3].name, "optional_input_tool");
}

#[tokio::test]
async fn test_toolbox_concurrent_invocations() {
    let service = Arc::new(TestToolService::new("ConcurrentTest"));
    let initial_count = service.get_call_count();

    // Create multiple concurrent invocations
    let mut handles = Vec::new();

    for i in 0..5 {
        let service_clone = Arc::clone(&service);
        let handle = tokio::spawn(async move {
            let call = ToolCall::new(
                format!("concurrent_call_{i}"),
                "simple_async_tool",
                json!({
                    "value": i,
                    "label": format!("concurrent_{}", i)
                }),
            );
            service_clone.invoke(call).await
        });
        handles.push(handle);
    }

    // Wait for all to complete
    let results: Vec<_> = futures_util::future::join_all(handles).await;

    // Verify all succeeded
    for result in results {
        let tool_result = result.unwrap().unwrap();
        assert_eq!(tool_result.name, "simple_async_tool");
    }

    // Verify call count increased by 5
    assert_eq!(service.get_call_count(), initial_count + 5);
}

#[tokio::test]
async fn test_toolbox_infallible_tool() {
    let service = TestToolService::new("InfallibleTest");

    let call = ToolCall::new("infallible_call", "infailable_tool", json!({}));

    let result = service.invoke(call).await;
    assert!(result.is_ok(), "Infallible tool invocation should succeed");

    let tool_result = result.unwrap();
    assert_eq!(tool_result.id, "infallible_call");
    assert_eq!(tool_result.name, "infailable_tool");
    assert_eq!(tool_result.response.len(), 1);

    // Check that the response contains the expected success message
    if let Some(message) = tool_result.response.first()
        && let Some(content) = message.content.first() {
            match content {
                Part::ToolResult { content, .. } => {
                    assert_eq!(*content, json!("Success"));
                }
                _ => panic!("Expected tool result content"),
            }
        }
}

#[tokio::test]
async fn test_toolbox_side_effect_tool() {
    let service = TestToolService::new("SideEffectTest");

    let call = ToolCall::new("side_effect_call", "side_effect_tool", json!({}));

    let result = service.invoke(call).await;
    assert!(result.is_ok(), "Side effect tool invocation should succeed");

    let tool_result = result.unwrap();
    assert_eq!(tool_result.id, "side_effect_call");
    assert_eq!(tool_result.name, "side_effect_tool");
    assert_eq!(tool_result.response.len(), 1);

    // Check that the response contains null (since it's a side-effect tool)
    if let Some(message) = tool_result.response.first()
        && let Some(content) = message.content.first() {
            match content {
                Part::ToolResult { content, .. } => {
                    assert_eq!(*content, serde_json::Value::Null);
                }
                _ => panic!("Expected tool result content"),
            }
        }
}
