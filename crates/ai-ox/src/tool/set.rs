use super::{Tool, ToolBox, ToolCall, ToolError, ToolResult};
use futures_util::future::BoxFuture;
use std::sync::Arc;

/// A container that holds multiple toolboxes and provides a unified interface
/// for tool discovery and invocation.
#[derive(Debug, Clone, Default)]
pub struct ToolSet {
    toolboxes: Vec<Arc<dyn ToolBox>>,
}

impl ToolSet {
    /// Creates a new empty ToolSet.
    pub fn new() -> Self {
        Self {
            toolboxes: Vec::new(),
        }
    }
    
    /// Adds a toolbox to this set.
    pub fn add_toolbox(&mut self, toolbox: Arc<dyn ToolBox>) {
        self.toolboxes.push(toolbox);
    }
    
    /// Adds a toolbox to this set using a builder pattern.
    pub fn with_toolbox(mut self, toolbox: Arc<dyn ToolBox>) -> Self {
        self.add_toolbox(toolbox);
        self
    }
    
    /// Returns all tools from all toolboxes in this set.
    pub fn get_all_tools(&self) -> Vec<Tool> {
        let mut all_tools = Vec::new();
        
        for toolbox in &self.toolboxes {
            all_tools.extend(toolbox.tools());
        }
        
        all_tools
    }
    
    /// Finds the toolbox that contains the function with the given name.
    fn find_toolbox_for_function(&self, name: &str) -> Option<&Arc<dyn ToolBox>> {
        self.toolboxes
            .iter()
            .find(|toolbox| toolbox.has_function(name))
    }
    
    /// Checks if any toolbox in this set has a function with the given name.
    pub fn has_function(&self, name: &str) -> bool {
        self.find_toolbox_for_function(name).is_some()
    }
    
    /// Invokes a tool function by finding the appropriate toolbox and
    /// delegating the call to it.
    pub async fn invoke(&self, call: ToolCall) -> Result<ToolResult, ToolError> {
        let toolbox = self
            .find_toolbox_for_function(&call.name)
            .ok_or_else(|| ToolError::not_found(&call.name))?;
        
        toolbox.invoke(call).await
    }
}

impl ToolBox for ToolSet {
    fn tools(&self) -> Vec<Tool> {
        self.get_all_tools()
    }
    
    fn invoke(&self, call: ToolCall) -> BoxFuture<Result<ToolResult, ToolError>> {
        Box::pin(async move { self.invoke(call).await })
    }
    
    fn has_function(&self, name: &str) -> bool {
        self.has_function(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{FunctionMetadata, Tool};
    use futures_util::FutureExt;
    use serde_json::json;
    use std::sync::Arc;
    
    // Mock toolbox for testing
    #[derive(Debug)]
    struct MockToolBox {
        function_name: String,
    }
    
    impl MockToolBox {
        fn new(function_name: impl Into<String>) -> Self {
            Self {
                function_name: function_name.into(),
            }
        }
    }
    
    impl ToolBox for MockToolBox {
        fn tools(&self) -> Vec<Tool> {
            vec![Tool::FunctionDeclarations(vec![FunctionMetadata {
                name: self.function_name.clone(),
                description: Some(format!("Mock function {}", self.function_name)),
                parameters: json!({"type": "object", "properties": {}}),
            }])]
        }
        
        fn invoke(&self, call: ToolCall) -> BoxFuture<Result<ToolResult, ToolError>> {
            let function_name = self.function_name.clone();
            async move {
                if call.name == function_name {
                    Ok(ToolResult::new(
                        call.id,
                        call.name,
                        vec![], // Empty response for mock
                    ))
                } else {
                    Err(ToolError::not_found(call.name))
                }
            }.boxed()
        }
    }
    
    #[tokio::test]
    async fn test_empty_toolset() {
        let toolset = ToolSet::new();
        assert!(!toolset.has_function("any_function"));
        assert!(toolset.get_all_tools().is_empty());
        
        let call = ToolCall::new("1", "missing_function", json!({}));
        let result = toolset.invoke(call).await;
        assert!(matches!(result, Err(ToolError::NotFound { .. })));
    }
    
    #[tokio::test]
    async fn test_single_toolbox() {
        let mut toolset = ToolSet::new();
        let mock_toolbox = Arc::new(MockToolBox::new("test_function"));
        toolset.add_toolbox(mock_toolbox);
        
        assert!(toolset.has_function("test_function"));
        assert!(!toolset.has_function("missing_function"));
        
        let tools = toolset.get_all_tools();
        assert_eq!(tools.len(), 1);
        
        // Test successful invocation
        let call = ToolCall::new("1", "test_function", json!({}));
        let result = toolset.invoke(call).await;
        assert!(result.is_ok());
        
        // Test missing function
        let call = ToolCall::new("2", "missing_function", json!({}));
        let result = toolset.invoke(call).await;
        assert!(matches!(result, Err(ToolError::NotFound { .. })));
    }
    
    #[tokio::test]
    async fn test_multiple_toolboxes() {
        let toolset = ToolSet::new()
            .with_toolbox(Arc::new(MockToolBox::new("function_a")))
            .with_toolbox(Arc::new(MockToolBox::new("function_b")));
        
        assert!(toolset.has_function("function_a"));
        assert!(toolset.has_function("function_b"));
        assert!(!toolset.has_function("function_c"));
        
        let tools = toolset.get_all_tools();
        assert_eq!(tools.len(), 2);
        
        // Test invocation of function from first toolbox
        let call = ToolCall::new("1", "function_a", json!({}));
        let result = toolset.invoke(call).await;
        assert!(result.is_ok());
        
        // Test invocation of function from second toolbox
        let call = ToolCall::new("2", "function_b", json!({}));
        let result = toolset.invoke(call).await;
        assert!(result.is_ok());
    }
}